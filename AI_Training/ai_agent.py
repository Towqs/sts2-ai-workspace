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

from combat_actions import choose_candidate_for_card, enumerate_combat_actions, public_candidate_catalog
from data_pipeline import StateEncoder
from macro_data_pipeline import Vocab as MacroVocab, encode_record as encode_macro_record
from train_macro_bc import MacroBCModel

init(autoreset=True)

PORT = 15526
API_URL = f"http://localhost:{PORT}/api/v1/singleplayer"
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL_PATH = os.path.join(os.path.dirname(__file__), "control_state.json")
AI_LOGIC_PATH = os.path.join(os.path.dirname(__file__), "ai_logic_state.json")
AI_LOG_DIR = os.path.join(WORKSPACE_DIR, "RL_Datasets", "AI_Combat")

DEFAULT_CONTROL = {
    "ai_enabled": True,
    "macro_enabled": False,
    "macro_shop_enabled": False,
    "record_ai_actions": True,
    "include_ai_in_training": False,
}

from train_bc import CombatBCModel

def load_agent(processed_dir):
    vocab_path = os.path.join(processed_dir, 'vocab.json')
    model_path = os.path.join(processed_dir, 'bc_model_best.pth')
    metadata_path = os.path.join(processed_dir, 'metadata.json')
    
    encoded = StateEncoder(vocab_path)
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    id_to_action = {v: k for k, v in vocab['actions'].items()}
    
    metadata = {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        pass
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    weight_input_dim = int(state_dict.get("net.0.weight").shape[1])
    input_dim = int(metadata.get("features") or weight_input_dim)
    if input_dim != weight_input_dim:
        input_dim = weight_input_dim
    output_dim = len(vocab['actions'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CombatBCModel(input_dim, output_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(Fore.CYAN + f"[*] AI Brain Loaded! Action space: {output_dim} | Features: {input_dim} | Device: {device}")
    return encoded, id_to_action, model, device


def align_feature_vector(vector, expected_dim):
    if len(vector) == expected_dim:
        return vector
    if len(vector) > expected_dim:
        return vector[:expected_dim]
    return np.pad(vector, (0, expected_dim - len(vector)), mode="constant")


def load_macro_agent(processed_dir):
    vocab_path = os.path.join(processed_dir, "vocab.json")
    model_path = os.path.join(processed_dir, "macro_bc_model_best.pth")
    metadata_path = os.path.join(processed_dir, "metadata.json")
    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        print(Fore.YELLOW + "[Macro] Macro model not found. Run macro_data_pipeline.py and train_macro_bc.py first.")
        return None

    vocab = MacroVocab.load(vocab_path)
    id_to_action = {v: k for k, v in vocab.tables["actions"].items()}
    metadata = {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        pass
    input_dim = int(metadata.get("features") or 115)
    output_dim = len(vocab.tables["actions"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MacroBCModel(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(Fore.CYAN + f"[*] Macro Brain Loaded! Action space: {output_dim} | Features: {input_dim} | Device: {device}")
    return {"vocab": vocab, "id_to_action": id_to_action, "model": model, "device": device, "input_dim": input_dim}


def room_type_for_state(state_type):
    mapping = {
        "map": "MapRoom",
        "rewards": "CombatRoom",
        "card_reward": "CombatRoom",
        "event": "EventRoom",
        "rest_site": "RestSiteRoom",
        "shop": "MerchantRoom",
        "fake_merchant": "MerchantRoom",
        "treasure": "TreasureRoom",
    }
    return mapping.get(str(state_type or "").lower(), "unknown")


def build_macro_record_from_state(state):
    state_type = state.get("state_type", "unknown")
    run = state.get("run") or {}
    player = state.get("player") or {}
    relics = player.get("relics") or []
    potions = player.get("potions") or []
    macro_state = {
        "act": run.get("act", 0),
        "floor": run.get("floor", 0),
        "ascension": run.get("ascension", 0),
        "character": player.get("character"),
        "hp": player.get("hp", 0),
        "max_hp": player.get("max_hp", 1),
        "gold": player.get("gold", 0),
        "deck_size": player.get("deck_size") or len(player.get("deck", []) or []),
        "relic_count": player.get("relic_count") or len(relics),
        "potion_slots_filled": player.get("potion_slots_filled") or len(potions),
        "relics": relics,
        "potions": potions,
        "room_type": state.get("room_type") or room_type_for_state(state_type),
    }

    screen = {"state_type": state_type}
    for key in ("map", "rewards", "card_reward", "event", "rest_site", "shop", "fake_merchant", "treasure"):
        if isinstance(state.get(key), dict):
            screen[key] = state.get(key)
    if "fake_merchant" in screen and "shop" not in screen:
        fake_shop = (screen["fake_merchant"] or {}).get("shop")
        if isinstance(fake_shop, dict):
            screen["shop"] = fake_shop

    return {
        "type": "macro_action",
        "action_type": "macro_inference",
        "action_data": {},
        "state": macro_state,
        "screen_state": screen,
    }


def get_items_for_state(state, state_type):
    state_type = str(state_type or "").lower()
    if state_type == "rewards":
        return (state.get("rewards") or {}).get("items", []) or []
    if state_type == "shop":
        return (state.get("shop") or {}).get("items", []) or []
    if state_type == "fake_merchant":
        return ((state.get("fake_merchant") or {}).get("shop") or {}).get("items", []) or []
    return []


def get_can_proceed(state, state_type):
    state_type = str(state_type or "").lower()
    if state_type in ("rewards", "shop", "rest_site"):
        return bool((state.get(state_type) or {}).get("can_proceed"))
    if state_type == "fake_merchant":
        return bool(((state.get("fake_merchant") or {}).get("shop") or {}).get("can_proceed"))
    if state_type == "treasure":
        return bool((state.get("treasure") or {}).get("can_proceed"))
    return False


def shop_item_matches(item, wanted):
    wanted = str(wanted or "").lower()
    fields = (
        item.get("category"),
        item.get("type"),
        item.get("item_id"),
        item.get("item_name"),
        item.get("card_id"),
        item.get("card_name"),
        item.get("relic_id"),
        item.get("relic_name"),
        item.get("potion_id"),
        item.get("potion_name"),
    )
    return any(str(value or "").lower() == wanted for value in fields)


def choose_shop_item(state, state_type, wanted):
    items = get_items_for_state(state, state_type)
    available = [
        (fallback_index, item)
        for fallback_index, item in enumerate(items)
        if item.get("is_stocked", True) and item.get("can_afford", True)
    ]
    if wanted not in ("?", "unknown", ""):
        for fallback_index, item in available:
            if shop_item_matches(item, wanted):
                return item, fallback_index, "available"
        return None, None, "shop_item_unavailable"

    # Ambiguous shop labels come from weak training data. If shop automation is
    # explicitly enabled, pick a conservative non-removal purchase.
    priorities = {"relic": 0, "card": 1, "potion": 2}
    candidates = []
    for fallback_index, item in available:
        category = str(item.get("category") or item.get("type") or "").lower()
        if category == "card_removal":
            continue
        if category not in priorities:
            continue
        candidates.append((
            priorities[category],
            0 if item.get("on_sale") else 1,
            int(item.get("price") or item.get("cost") or 9999),
            fallback_index,
            item,
        ))
    if not candidates:
        return None, None, "no_safe_shop_purchase"
    _, _, _, fallback_index, item = sorted(candidates)[0]
    return item, fallback_index, "fallback_safe_purchase"


def macro_label_to_payload(label, state, allow_shop=False):
    state_type = str(state.get("state_type") or "").lower()

    if label.startswith("select_map_node:index_"):
        if state_type != "map":
            return None, "not_map"
        options = ((state.get("map") or {}).get("next_options") or [])
        try:
            index = int(label.rsplit("_", 1)[1])
        except ValueError:
            return None, "bad_map_index"
        if 0 <= index < len(options):
            return {"action": "choose_map_node", "index": index}, "available"
        return None, "map_index_out_of_range"

    if label.startswith("select_map_node:type_"):
        if state_type != "map":
            return None, "not_map"
        wanted = label.split("type_", 1)[1].lower()
        options = ((state.get("map") or {}).get("next_options") or [])
        for fallback_index, option in enumerate(options):
            if str(option.get("type") or "").lower() == wanted:
                return {"action": "choose_map_node", "index": int(option.get("index", fallback_index))}, "available"
        return None, "map_type_unavailable"

    if label.startswith("claim_reward:"):
        if state_type != "rewards":
            return None, "not_rewards"
        reward_type = label.split(":", 1)[1].lower()
        for fallback_index, item in enumerate(get_items_for_state(state, state_type)):
            item_type = str(item.get("type") or item.get("category") or "").lower()
            if item_type == reward_type or (reward_type == "card" and item_type == "special_card"):
                return {"action": "claim_reward", "index": int(item.get("index", fallback_index))}, "available"
        return None, f"reward_{reward_type}_unavailable"

    if label.startswith("choose_card:index_"):
        if state_type != "card_reward":
            return None, "not_card_reward"
        cards = ((state.get("card_reward") or {}).get("cards") or [])
        try:
            index = int(label.rsplit("_", 1)[1])
        except ValueError:
            return None, "bad_card_index"
        if 0 <= index < len(cards):
            return {"action": "select_card_reward", "card_index": index}, "available"
        return None, "card_index_out_of_range"

    if label == "skip_reward":
        if state_type == "card_reward" and (state.get("card_reward") or {}).get("can_skip"):
            return {"action": "skip_card_reward"}, "available"
        return None, "skip_unavailable"

    if label.startswith("choose_event_option:"):
        if state_type != "event":
            return None, "not_event"
        event = state.get("event") or {}
        if event.get("in_dialogue"):
            return {"action": "advance_dialogue"}, "available"
        options = event.get("options") or []
        available = [o for o in options if not o.get("is_locked") and not o.get("was_chosen")]
        if label.endswith(":unknown"):
            if len(available) == 1:
                return {"action": "choose_event_option", "index": int(available[0].get("index", 0))}, "single_event_option"
            return None, "ambiguous_event_option"
        try:
            index = int(label.rsplit("_", 1)[1])
        except ValueError:
            return None, "bad_event_index"
        valid = {int(o.get("index", i)) for i, o in enumerate(available)}
        if index in valid:
            return {"action": "choose_event_option", "index": index}, "available"
        return None, "event_index_unavailable"

    if label.startswith("choose_rest_option:"):
        if state_type != "rest_site":
            return None, "not_rest_site"
        options = (state.get("rest_site") or {}).get("options") or []
        enabled = [o for o in options if o.get("is_enabled", True)]
        if not enabled:
            return None, "no_rest_option_enabled"
        return {"action": "choose_rest_option", "index": int(enabled[0].get("index", 0))}, "first_enabled_rest_option"

    if label.startswith("buy_item:"):
        if state_type not in ("shop", "fake_merchant"):
            return None, "not_shop"
        if not allow_shop:
            return None, "shop_protected"
        wanted = label.split(":", 1)[1].lower()
        if wanted == "card_removal":
            return None, "card_removal_requires_manual_selection"
        item, fallback_index, status = choose_shop_item(state, state_type, wanted)
        if item:
            return {"action": "shop_purchase", "index": int(item.get("index", fallback_index))}, status
        return None, status

    if label == "proceed":
        if state_type in ("shop", "fake_merchant") and not allow_shop:
            return None, "shop_protected"
        if get_can_proceed(state, state_type):
            return {"action": "proceed"}, "available"
        return None, "proceed_unavailable"

    return None, "unsupported_macro_label"


def macro_fallback_payload(state):
    state_type = str(state.get("state_type") or "").lower()
    if state_type == "rewards":
        rewards = state.get("rewards") or {}
        if not rewards.get("items") and rewards.get("can_proceed"):
            return {"action": "proceed"}, "rewards_empty_proceed"
    if state_type in ("rest_site", "treasure") and get_can_proceed(state, state_type):
        return {"action": "proceed"}, f"{state_type}_proceed"
    return None, ""


def safe_num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def route_type_score(node_type, hp_ratio, gold):
    node_type = str(node_type or "").lower()
    if "boss" in node_type:
        return 0.0
    if "elite" in node_type:
        if hp_ratio >= 0.70:
            return 5.0
        if hp_ratio >= 0.55:
            return 1.5
        return -6.0
    if "rest" in node_type or "camp" in node_type:
        return 5.0 if hp_ratio < 0.55 else 1.0
    if "shop" in node_type or "merchant" in node_type:
        if gold >= 180:
            return 3.5
        if gold >= 110:
            return 1.0
        return -3.0
    if "treasure" in node_type:
        return 2.5
    if "event" in node_type or "unknown" in node_type or "ancient" in node_type:
        return 1.5
    if "monster" in node_type:
        return 2.0
    return 0.5


def choose_map_route_action(state):
    options = ((state.get("map") or {}).get("next_options") or [])
    if not options:
        return None, {
            "top_actions": [],
            "chosen_action": None,
            "payload": None,
            "reason": "no_map_options",
        }
    if len(options) == 1:
        option = options[0]
        return {"action": "choose_map_node", "index": int(option.get("index", 0))}, {
            "top_actions": [{"action": "route:index_0", "confidence": 100.0, "marker": "only_option"}],
            "chosen_action": "route:index_0",
            "payload": {"action": "choose_map_node", "index": int(option.get("index", 0))},
            "reason": "only_option",
        }

    player = state.get("player") or {}
    run = state.get("run") or {}
    hp = safe_num(player.get("hp"))
    max_hp = max(safe_num(player.get("max_hp"), 1.0), 1.0)
    hp_ratio = hp / max_hp
    gold = safe_num(player.get("gold"))
    floor = int(safe_num(run.get("floor")))

    scored = []
    for fallback_index, option in enumerate(options):
        node_type = option.get("type")
        score = route_type_score(node_type, hp_ratio, gold)
        leads = option.get("leads_to") or []
        score += min(len(leads), 3) * 0.25
        for lead in leads[:4]:
            score += route_type_score(lead.get("type"), hp_ratio, gold) * 0.35
        # Avoid deterministic left-edge drift on close scores.
        score += (safe_num(option.get("col")) % 2) * 0.05
        scored.append({
            "fallback_index": fallback_index,
            "index": int(option.get("index", fallback_index)),
            "type": node_type,
            "col": option.get("col"),
            "row": option.get("row"),
            "score": round(score, 3),
            "leads": len(leads),
        })

    best_score = max(item["score"] for item in scored)
    tied = [item for item in scored if best_score - item["score"] <= 0.25]
    if len(tied) > 1:
        center = sum(safe_num(o.get("col")) for o in options) / len(options)
        if floor % 2:
            best = max(tied, key=lambda item: (item["leads"], safe_num(item["col"]) - center))
        else:
            best = max(tied, key=lambda item: (item["leads"], center - safe_num(item["col"])))
    else:
        best = max(scored, key=lambda item: item["score"])

    payload = {"action": "choose_map_node", "index": best["index"]}
    top_actions = [
        {
            "action": f"route:index_{item['index']}:{item['type']}",
            "confidence": item["score"],
            "marker": f"col={item['col']} leads={item['leads']}",
        }
        for item in sorted(scored, key=lambda item: item["score"], reverse=True)[:6]
    ]
    return payload, {
        "top_actions": top_actions,
        "chosen_action": f"route:index_{best['index']}:{best['type']}",
        "payload": payload,
        "reason": f"route_score hp={hp_ratio:.2f} gold={int(gold)}",
    }


def macro_state_signature(state):
    state_type = str(state.get("state_type") or "").lower()
    payload = {"state_type": state_type}
    if state_type == "map":
        payload["options"] = [
            (o.get("index"), o.get("col"), o.get("row"), o.get("type"))
            for o in ((state.get("map") or {}).get("next_options") or [])
        ]
    elif state_type == "rewards":
        payload["items"] = [
            (i.get("index"), i.get("type"), i.get("description"))
            for i in ((state.get("rewards") or {}).get("items") or [])
        ]
        payload["can_proceed"] = (state.get("rewards") or {}).get("can_proceed")
    elif state_type == "card_reward":
        payload["cards"] = [
            (c.get("index"), c.get("id"), c.get("name"))
            for c in ((state.get("card_reward") or {}).get("cards") or [])
        ]
    elif state_type == "event":
        payload["options"] = [
            (o.get("index"), o.get("title"), o.get("is_locked"), o.get("is_proceed"), o.get("was_chosen"))
            for o in ((state.get("event") or {}).get("options") or [])
        ]
        payload["in_dialogue"] = (state.get("event") or {}).get("in_dialogue")
    elif state_type == "rest_site":
        payload["options"] = [
            (o.get("index"), o.get("id"), o.get("is_enabled"))
            for o in ((state.get("rest_site") or {}).get("options") or [])
        ]
    elif state_type in ("shop", "fake_merchant"):
        payload["items"] = [
            (i.get("index"), i.get("category"), i.get("price") or i.get("cost"), i.get("can_afford"), i.get("is_stocked"))
            for i in get_items_for_state(state, state_type)
        ]
        payload["can_proceed"] = get_can_proceed(state, state_type)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def choose_macro_action(macro_agent, state, allow_shop=False):
    if str(state.get("state_type") or "").lower() == "map":
        return choose_map_route_action(state)

    if not macro_agent:
        return None, {
            "top_actions": [],
            "chosen_action": None,
            "payload": None,
            "reason": "macro_model_not_loaded",
        }

    record = build_macro_record_from_state(state)
    state_vec = encode_macro_record(macro_agent["vocab"], record)
    expected_dim = macro_agent.get("input_dim") or len(state_vec)
    if len(state_vec) != expected_dim:
        return None, {
            "top_actions": [],
            "chosen_action": None,
            "payload": None,
            "reason": f"macro_feature_dim_mismatch actual={len(state_vec)} expected={expected_dim}",
        }

    state_tensor = torch.tensor([state_vec], dtype=torch.float32).to(macro_agent["device"])
    with torch.no_grad():
        outputs = macro_agent["model"](state_tensor)
        probs = torch.softmax(outputs[0], dim=0)
        sorted_indices = torch.argsort(outputs[0], descending=True)

    top_actions = []
    chosen_payload = None
    chosen_label = None
    chosen_reason = "no_legal_macro_action"
    for idx in sorted_indices:
        label = macro_agent["id_to_action"].get(idx.item(), "UNKNOWN")
        if label in ("UNKNOWN", "PAD"):
            continue
        payload, status = macro_label_to_payload(label, state, allow_shop=allow_shop)
        conf = probs[idx].item() * 100
        top_actions.append({"action": label, "confidence": round(conf, 2), "marker": status})
        if chosen_payload is None and payload:
            chosen_payload = payload
            chosen_label = label
            chosen_reason = status
        if len(top_actions) >= 6 and chosen_payload is not None:
            break

    if chosen_payload is None:
        chosen_payload, fallback_reason = macro_fallback_payload(state)
        if chosen_payload:
            chosen_label = chosen_payload.get("action")
            chosen_reason = fallback_reason

    return chosen_payload, {
        "top_actions": top_actions[:6],
        "chosen_action": chosen_label,
        "payload": chosen_payload,
        "reason": chosen_reason,
    }


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
    macro_processed_dir = os.path.join(os.path.dirname(__file__), "ProcessedMacroParams")
    encoder, id_to_action, model, device = load_agent(processed_dir)
    macro_agent = load_macro_agent(macro_processed_dir)
    session_id = f"ai_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    
    print(Fore.GREEN + Style.BRIGHT + "\n====== STS2 AI Control System v2 ======")
    print(Fore.WHITE + "  Enter any combat encounter, AI will auto-play when it's your turn.")
    print(Fore.WHITE + "  Press Ctrl+C to stop.\n")
    
    last_status_print = 0
    last_data_source = None
    last_macro_decision_key = None

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
            if state_type in ("shop", "fake_merchant") and not control.get("macro_shop_enabled", False):
                write_ai_logic_snapshot({
                    "timestamp": int(time.time() * 1000),
                    "session_id": session_id,
                    "state_type": state_type,
                    "mode": "macro",
                    "top_actions": [],
                    "chosen_action": None,
                    "payload": None,
                    "reason": "shop_protected",
                })
                if last_data_source != "human":
                    set_data_source("human")
                    last_data_source = "human"
                time.sleep(4.0)
                continue

            if control.get("macro_enabled", False) and state_type in ("map", "rewards", "card_reward", "event", "rest_site", "shop", "fake_merchant", "treasure"):
                allow_shop = bool(control.get("macro_shop_enabled", False))
                payload, macro_info = choose_macro_action(macro_agent, state, allow_shop=allow_shop)
                decision_key = json.dumps({
                    "signature": macro_state_signature(state),
                    "payload": payload,
                }, sort_keys=True, ensure_ascii=False)
                write_ai_logic_snapshot({
                    "timestamp": int(time.time() * 1000),
                    "session_id": session_id,
                    "state_type": state_type,
                    "mode": "macro",
                    "top_actions": macro_info.get("top_actions", []),
                    "chosen_action": macro_info.get("chosen_action"),
                    "payload": payload,
                    "reason": macro_info.get("reason"),
                })
                if payload and decision_key != last_macro_decision_key:
                    print(Fore.GREEN + Style.BRIGHT + f"  >>> MACRO EXECUTE: {macro_info.get('chosen_action')}")
                    print(Fore.WHITE + f"  Sending: {json.dumps(payload, ensure_ascii=False)}")
                    if last_data_source != "ai":
                        set_data_source("ai")
                        last_data_source = "ai"
                    success = send_action(payload)
                    last_macro_decision_key = decision_key
                    if not success and last_data_source != "human":
                        set_data_source("human")
                        last_data_source = "human"
                    time.sleep(1.5)
                    continue
                if state_type in ("shop", "fake_merchant") and not allow_shop:
                    time.sleep(4.0)
                    continue

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
        legal_candidates = enumerate_combat_actions(state)
        legal_candidate_catalog = public_candidate_catalog(legal_candidates, limit=24)
            
        # 特征编码 + 模型推理
        state_vec = encoder.encode(state)
        state_vec = align_feature_vector(state_vec, model.net[0].in_features)
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
                chosen_candidate = choose_candidate_for_card(legal_candidates, chosen_card)
                chosen_action = f"play_card_{chosen_card.get('id')}"
                chosen_candidate_label = chosen_candidate.label if chosen_candidate else chosen_action
                print(Fore.GREEN + Style.BRIGHT + f"  >>> EXECUTE: {chosen_candidate_label}")
                payload = chosen_candidate.payload if chosen_candidate else build_play_card_payload_for_card(chosen_card, alive_enemies)
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
                    "candidate_actions": legal_candidate_catalog,
                    "chosen_action": chosen_action,
                    "chosen_candidate": chosen_candidate_label,
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
                end_turn_candidate = next((c for c in legal_candidates if c.kind == "end_turn"), None)
                payload = end_turn_candidate.payload if end_turn_candidate else {"action": "end_turn"}
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
                    "candidate_actions": legal_candidate_catalog,
                    "chosen_action": "end_turn",
                    "chosen_candidate": end_turn_candidate.label if end_turn_candidate else "end_turn",
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
