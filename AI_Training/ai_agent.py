import os
import json
import time
import uuid
from datetime import datetime
import requests
import torch
import torch.nn as nn
import numpy as np
from colorama import init, Fore, Style

from data_pipeline import StateEncoder

init(autoreset=True)

PORT = 15526
API_URL = f"http://localhost:{PORT}/api/v1/singleplayer"
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL_PATH = os.path.join(os.path.dirname(__file__), "control_state.json")
AI_LOGIC_PATH = os.path.join(os.path.dirname(__file__), "ai_logic_state.json")
AI_LOG_DIR = os.path.join(WORKSPACE_DIR, "RL_Datasets", "AI_Combat")

DEFAULT_CONTROL = {
    "ai_enabled": True,
    "record_ai_actions": True,
    "include_ai_in_training": False,
}

class CombatBCModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CombatBCModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def load_agent(processed_dir):
    vocab_path = os.path.join(processed_dir, 'vocab.json')
    model_path = os.path.join(processed_dir, 'bc_model_best.pth')
    
    encoded = StateEncoder(vocab_path)
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    id_to_action = {v: k for k, v in vocab['actions'].items()}
    
    input_dim = 61
    output_dim = len(vocab['actions'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CombatBCModel(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    print(Fore.CYAN + f"[*] AI Brain Loaded! Action space: {output_dim} | Device: {device}")
    return encoded, id_to_action, model, device


def get_enemy_hp(enemy):
    """兼容 STS2MCP 不同版本的敌人血量字段。"""
    hp = enemy.get("hp", enemy.get("current_hp", 0))
    try:
        return int(hp)
    except (TypeError, ValueError):
        return 0


def get_alive_enemies(enemies):
    return [e for e in enemies if get_enemy_hp(e) > 0]


def get_enemy_target_id(enemy):
    return enemy.get("entity_id") or enemy.get("id") or enemy.get("name") or ""


def parse_card_cost(card, energy):
    cost = card.get("cost", 0)
    if cost == "X":
        return energy
    try:
        return int(cost)
    except (TypeError, ValueError):
        return 99


def is_enemy_target_card(card):
    return card.get("target_type") == "AnyEnemy"


def is_playable_with_energy(card, energy):
    return card.get("can_play", False) and parse_card_cost(card, energy) <= energy


def build_play_card_payload_for_card(card, enemies):
    target_id = ""
    needs_enemy_target = is_enemy_target_card(card)
    if needs_enemy_target and enemies:
        target = min(enemies, key=get_enemy_hp)
        target_id = get_enemy_target_id(target)

    payload = {
        "action": "play_card",
        "card_index": card.get("index", 0),
    }
    if needs_enemy_target:
        if not target_id:
            return None
        payload["target"] = target_id

    return payload


def build_play_card_payload(card_id, hand_cards, enemies):
    """
    核心翻译器：将 AI 模型输出的 card_id (如 "BASH") 
    转换为 MCP API 需要的 card_index + target 格式。
    
    MCP API 要求:
      - card_index: 手牌列表中的数字下标 (0, 1, 2...)
      - target: 敌人的 entity_id (如 "NIBBIT_0"), 仅攻击牌需要
    """
    # 1. 在手牌中找到这张牌的下标
    selected_card = None
    card_index = None
    for i, card in enumerate(hand_cards):
        if card.get("id") == card_id and card.get("can_play", True):
            selected_card = card
            card_index = i
            break
    
    if card_index is None:
        return None  # 手里没有这张牌
    
    return build_play_card_payload_for_card(selected_card, enemies)


def choose_card_to_play(sorted_indices, id_to_action, hand_cards, energy):
    """用模型排序选牌，但禁止还有可打牌时过早 end_turn。"""
    affordable = [c for c in hand_cards if is_playable_with_energy(c, energy)]
    if not affordable:
        return None

    # 0 费牌先打，避免模型把它们留在手里。
    zero_cost = [c for c in affordable if parse_card_cost(c, energy) == 0]
    if zero_cost:
        return zero_cost[0]

    playable_by_id = {}
    for card in affordable:
        playable_by_id.setdefault(card.get("id"), card)

    for idx in sorted_indices:
        action_name = id_to_action.get(idx.item(), "UNKNOWN")
        if not action_name.startswith("play_card_"):
            continue
        card_id = action_name.replace("play_card_", "")
        if card_id in playable_by_id:
            return playable_by_id[card_id]

    # 模型词表里没有的新牌，兜底打第一张可支付的牌。
    return affordable[0]


def fetch_game_state():
    try:
        resp = requests.get(API_URL, timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None


def load_control():
    try:
        with open(CONTROL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    control = DEFAULT_CONTROL.copy()
    control.update(data)
    return control


def append_ai_action_log(session_id, action_payload, state_before, state_after, ok):
    control = load_control()
    if not control.get("record_ai_actions", True):
        return

    os.makedirs(AI_LOG_DIR, exist_ok=True)
    path = os.path.join(AI_LOG_DIR, f"ai_combat_run_{datetime.now():%Y-%m-%d}.jsonl")
    record = {
        "type": "action",
        "run_id": session_id,
        "timestamp": int(time.time() * 1000),
        "source": "ai",
        "action_type": action_payload.get("action"),
        "action_data": action_payload,
        "ok": bool(ok),
        "state_before": state_before,
        "state_after": state_after,
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def write_ai_logic_snapshot(snapshot):
    try:
        with open(AI_LOGIC_PATH, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def send_action(action_payload):
    try:
        resp = requests.post(API_URL, json=action_payload, timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            if "error" in data:
                print(Fore.RED + f"   [Game Error] {data['error']}")
                return False
            msg = data.get("message", "")
            if msg:
                print(Fore.MAGENTA + f"   [Game OK] {msg}")
            return True
        else:
            print(Fore.RED + f"   [HTTP Error] status={resp.status_code}")
            return False
    except Exception as e:
        print(Fore.RED + f"   [Connection Error] {e}")
        return False


def set_data_source(source):
    try:
        requests.post(API_URL, json={"action": "set_data_source", "source": source}, timeout=1.0)
    except:
        pass


def run_agent():
    processed_dir = os.path.join(os.path.dirname(__file__), "ProcessedParams")
    encoder, id_to_action, model, device = load_agent(processed_dir)
    session_id = f"ai_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    
    print(Fore.GREEN + Style.BRIGHT + "\n====== STS2 AI Control System v2 ======")
    print(Fore.WHITE + "  Enter any combat encounter, AI will auto-play when it's your turn.")
    print(Fore.WHITE + "  Press Ctrl+C to stop.\n")
    
    last_status_print = 0
    last_data_source = None

    while True:
        time.sleep(0.8)
        
        state = fetch_game_state()
        if not state:
            now = time.time()
            if now - last_status_print > 5:
                print(Fore.RED + "[Waiting] Game API not reachable... Is the game running?")
                last_status_print = now
            continue

        control = load_control()
        if not control.get("ai_enabled", True):
            if last_data_source != "human":
                set_data_source("human")
                last_data_source = "human"
            now = time.time()
            if now - last_status_print > 5:
                print(Fore.YELLOW + "[Paused] AI disabled from control panel.")
                last_status_print = now
            continue
            
        battle = state.get("battle", {})
        state_type = state.get("state_type", "unknown")
        is_play = battle.get("is_play_phase", False)
        turn = battle.get("turn", "unknown")
        player_disabled = battle.get("player_actions_disabled", False)
        
        # 每5秒打印一次当前状态摘要（非战斗时）
        now = time.time()
        if now - last_status_print > 5:
            print(Fore.WHITE + f"[Status] state={state_type} | turn={turn} | play_phase={is_play} | disabled={player_disabled}")
            last_status_print = now
        
        # 只在：战斗中 + 出牌阶段 + 轮到玩家 + 没有锁定 时才行动
        if not (state_type in ("monster", "elite", "boss") and is_play and turn == "player"):
            if state_type not in ("monster", "elite", "boss") and last_data_source != "human":
                set_data_source("human")
                last_data_source = "human"
            continue
        if player_disabled:
            continue
            
        player_state = state.get("player", {})
        hand_cards = player_state.get("hand", [])
        available_card_ids = [c.get("id") for c in hand_cards]
        playable_card_ids = [c.get("id") for c in hand_cards if c.get("can_play", False)]
        energy = player_state.get("energy", 0)
        enemies = battle.get("enemies", [])
        alive_enemies = get_alive_enemies(enemies)
            
        # 特征编码 + 模型推理
        state_vec = encoder.encode(state)
        state_tensor = torch.tensor([state_vec], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(state_tensor)
            probs = torch.softmax(outputs[0], dim=0)
            sorted_indices = torch.argsort(outputs[0], descending=True)
            
            # === 打印 AI 思考过程 ===
            hp = player_state.get("hp", "?")
            block = player_state.get("block", 0)
            print(Fore.YELLOW + f"\n{'='*50}")
            print(Fore.YELLOW + f"  [AI TURN] HP:{hp} | Energy:{energy} | Hand:{len(hand_cards)} | Block:{block}")
            print(Fore.YELLOW + f"  Hand: {available_card_ids}")
            if alive_enemies:
                enemy_info = ", ".join([f"{e.get('name','?')}(HP:{get_enemy_hp(e)})" for e in alive_enemies])
                print(Fore.YELLOW + f"  Enemies: {enemy_info}")
            print(Fore.CYAN + "  [Brain] Top 5 probabilities:")
            
            top_printed = 0
            top_actions = []
            for idx in sorted_indices:
                if top_printed >= 5: break
                act = id_to_action.get(idx.item(), "UNKNOWN")
                if act not in ["UNKNOWN", "PAD"]:
                    conf = probs[idx].item() * 100
                    marker = ""
                    if act.startswith("play_card_"):
                        cid = act.replace("play_card_", "")
                        if cid in playable_card_ids:
                            marker = Fore.GREEN + " [AVAILABLE]"
                        elif cid in available_card_ids:
                            marker = Fore.YELLOW + " [IN HAND, CANT PLAY]"
                        else:
                            marker = Fore.RED + " [NOT IN HAND]"
                    print(f"    {act:25s}  {conf:5.1f}%{marker}")
                    top_actions.append({"action": act, "confidence": round(conf, 2), "marker": marker.replace("\x1b[32m", "").replace("\x1b[33m", "").replace("\x1b[31m", "")})
                    top_printed += 1
            
            # === 选择最佳合法动作 ===
            chosen_card = choose_card_to_play(sorted_indices, id_to_action, hand_cards, energy)

            if chosen_card:
                chosen_action = f"play_card_{chosen_card.get('id')}"
                print(Fore.GREEN + Style.BRIGHT + f"  >>> EXECUTE: {chosen_action}")
                payload = build_play_card_payload_for_card(chosen_card, alive_enemies)
                write_ai_logic_snapshot({
                    "timestamp": int(time.time() * 1000),
                    "session_id": session_id,
                    "state_type": state_type,
                    "hp": hp,
                    "block": block,
                    "energy": energy,
                    "hand": available_card_ids,
                    "playable": playable_card_ids,
                    "enemies": [{"id": get_enemy_target_id(e), "name": e.get("name"), "hp": get_enemy_hp(e)} for e in alive_enemies],
                    "top_actions": top_actions,
                    "chosen_action": chosen_action,
                    "payload": payload,
                    "reason": "zero-cost first, otherwise model-ranked affordable card",
                })
                
                if payload:
                    print(Fore.WHITE + f"  Sending: {json.dumps(payload)}")
                    if last_data_source != "ai":
                        set_data_source("ai")
                        last_data_source = "ai"
                    success = send_action(payload)
                    state_after = fetch_game_state()
                    append_ai_action_log(session_id, payload, state, state_after, success)
                    if success:
                        time.sleep(1.5)  # 等动画播完
                    else:
                        time.sleep(0.5)  # 失败了也稍微等一下再重试
                else:
                    print(Fore.RED + "  [Bug] Could not build payload")
            else:
                print(Fore.RED + "  [No affordable playable card found, ending turn]")
                payload = {"action": "end_turn"}
                write_ai_logic_snapshot({
                    "timestamp": int(time.time() * 1000),
                    "session_id": session_id,
                    "state_type": state_type,
                    "hp": hp,
                    "block": block,
                    "energy": energy,
                    "hand": available_card_ids,
                    "playable": playable_card_ids,
                    "enemies": [{"id": get_enemy_target_id(e), "name": e.get("name"), "hp": get_enemy_hp(e)} for e in alive_enemies],
                    "top_actions": top_actions,
                    "chosen_action": "end_turn",
                    "payload": payload,
                    "reason": "no affordable playable card",
                })
                if last_data_source != "ai":
                    set_data_source("ai")
                    last_data_source = "ai"
                success = send_action(payload)
                state_after = fetch_game_state()
                append_ai_action_log(session_id, payload, state, state_after, success)
                time.sleep(1.5)
                        
if __name__ == "__main__":
    run_agent()
