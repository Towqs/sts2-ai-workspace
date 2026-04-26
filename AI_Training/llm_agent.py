import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

WORKSPACE = Path(__file__).resolve().parents[1]
AI_DIR = WORKSPACE / "AI_Training"
DATA_DIR = WORKSPACE / "RL_Datasets"
CONFIG_PATH = AI_DIR / "model_config.json"
LOGIC_PATH = AI_DIR / "llm_logic_state.json"
LLM_LOG_DIR = DATA_DIR / "LLM_Actions"
GAME_API = "http://localhost:15526/api/v1/singleplayer"

DEFAULT_CONFIG = {
    "enabled": False,
    "mode": "advisor",
    "provider": "openai_compatible",
    "base_url": "https://api.openai.com/v1",
    "api_key": "",
    "model": "",
    "temperature": 0.2,
    "max_tokens": 700,
    "decision_interval_sec": 3.0,
    "execute_combat": False,
    "confirm_shop": True,
    "max_actions_per_turn": 12,
}

COMBAT_TYPES = {"monster", "elite", "boss"}


def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_config():
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(read_json(CONFIG_PATH, {}))
    cfg["mode"] = cfg.get("mode") if cfg.get("mode") in ("advisor", "combat_auto") else "advisor"
    cfg["provider"] = "openai_compatible"
    cfg["temperature"] = float(cfg.get("temperature", 0.2) or 0.2)
    cfg["max_tokens"] = int(cfg.get("max_tokens", 700) or 700)
    cfg["decision_interval_sec"] = max(1.0, float(cfg.get("decision_interval_sec", 3.0) or 3.0))
    cfg["max_actions_per_turn"] = max(1, int(cfg.get("max_actions_per_turn", 12) or 12))
    return cfg


def request_json(url, method="GET", body=None, headers=None, timeout=20):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method=method, headers=headers or {})
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {raw[:500]}")
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw[:500].replace("\n", " ")
        raise RuntimeError(f"Non-JSON response from {url}: {preview}") from exc


def get_game_state():
    return request_json(GAME_API + "?format=json", timeout=3)


def post_game_action(payload):
    return request_json(GAME_API, method="POST", body=payload, timeout=3)


def card_cost(card, energy):
    cost = card.get("cost", 0)
    if cost == "X":
        return energy
    try:
        return int(cost)
    except (TypeError, ValueError):
        return 99


def enemy_hp(enemy):
    try:
        return int(enemy.get("hp", enemy.get("current_hp", 0)) or 0)
    except (TypeError, ValueError):
        return 0


def enemy_id(enemy):
    return enemy.get("entity_id") or enemy.get("id") or enemy.get("name") or ""


def compact_state(state):
    player = state.get("player", {})
    battle = state.get("battle", {})
    hand = []
    for i, card in enumerate(player.get("hand", [])):
        hand.append({
            "index": i,
            "id": card.get("id"),
            "name": card.get("name") or card.get("title") or card.get("id"),
            "cost": card.get("cost"),
            "can_play": bool(card.get("can_play", False)),
            "target_type": card.get("target_type"),
        })
    enemies = []
    for enemy in battle.get("enemies", []):
        if enemy_hp(enemy) <= 0:
            continue
        enemies.append({
            "entity_id": enemy_id(enemy),
            "name": enemy.get("name"),
            "hp": enemy_hp(enemy),
            "block": enemy.get("block", 0),
            "intents": enemy.get("intents", []),
        })
    return {
        "state_type": state.get("state_type"),
        "player": {
            "character": player.get("character"),
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "block": player.get("block"),
            "energy": player.get("energy"),
            "gold": player.get("gold"),
            "hand": hand,
            "relics": player.get("relics", [])[:12],
            "potions": player.get("potions", []),
        },
        "battle": {
            "turn": battle.get("turn"),
            "is_play_phase": battle.get("is_play_phase"),
            "player_actions_disabled": battle.get("player_actions_disabled"),
            "enemies": enemies,
        },
        "map": state.get("map", {}),
        "rewards": state.get("rewards", state.get("reward_options", [])),
        "screen": state.get("screen", {}),
        "next_options": state.get("next_options", []),
    }


def action_catalog(compact):
    state_type = str(compact.get("state_type") or "").lower()
    battle = compact.get("battle", {})
    player = compact.get("player", {})
    energy = player.get("energy") or 0
    actions = []

    if state_type in COMBAT_TYPES and battle.get("is_play_phase") and battle.get("turn") == "player":
        for card in player.get("hand", []):
            if card.get("can_play") and card_cost(card, energy) <= energy:
                args = {"card_index": card["index"]}
                if card.get("target_type") == "AnyEnemy":
                    args["target"] = "REQUIRED enemy entity_id"
                actions.append({"action": "combat_play_card", "args": args, "card": card.get("id")})
        actions.append({"action": "combat_end_turn", "args": {}})
    else:
        actions.extend([
            {"action": "wait", "args": {}},
            {"action": "proceed_to_map", "args": {}},
            {"action": "map_choose_node", "args": {"index": "0-based next option index"}},
            {"action": "rewards_claim", "args": {"index": "0-based reward index"}},
            {"action": "rewards_pick_card", "args": {"card_index": "0-based card reward index"}},
            {"action": "rewards_skip_card", "args": {}},
            {"action": "shop_purchase", "args": {"index": "0-based shop item index"}},
            {"action": "event_choose_option", "args": {"index": "0-based event option index"}},
            {"action": "rest_choose_option", "args": {"index": "0-based rest option index"}},
        ])
    return actions


def chat_url(base_url):
    base = (base_url or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_openai_compatible(cfg, compact, actions):
    if not cfg.get("api_key"):
        raise RuntimeError("Missing API key")
    if not cfg.get("model"):
        raise RuntimeError("Missing model name")

    system = (
        "You are a Slay the Spire 2 decision agent. "
        "Choose exactly one legal action from the provided action catalog. "
        "Return only JSON with keys: action, args, reason, confidence. "
        "Never invent card indexes or enemy ids. If no useful action is safe, choose wait."
    )
    user = {
        "mode": cfg.get("mode"),
        "rules": [
            "For enemy-targeted cards, target must be an entity_id from battle.enemies.",
            "Prefer playing useful zero-cost cards before spending energy.",
            "Do not end turn while affordable playable cards remain unless clearly justified.",
            "Shop purchases are suggestions unless execution is explicitly enabled.",
        ],
        "state": compact,
        "available_actions": actions,
    }
    body = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": cfg.get("temperature", 0.2),
        "max_tokens": cfg.get("max_tokens", 700),
    }
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
    result = request_json(chat_url(cfg.get("base_url")), method="POST", body=body, headers=headers, timeout=45)
    content = result["choices"][0]["message"]["content"]
    decision = extract_json(content)
    decision["_raw_content"] = content
    return decision


def normalize_decision(decision):
    aliases = {
        "play_card": "combat_play_card",
        "end_turn": "combat_end_turn",
        "claim_reward": "rewards_claim",
        "select_card_reward": "rewards_pick_card",
        "skip_card_reward": "rewards_skip_card",
        "choose_map_node": "map_choose_node",
        "choose_event_option": "event_choose_option",
        "choose_rest_option": "rest_choose_option",
    }
    action = aliases.get(decision.get("action"), decision.get("action"))
    args = decision.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    return {
        "action": action,
        "args": args,
        "reason": str(decision.get("reason", ""))[:1200],
        "confidence": decision.get("confidence"),
    }


def validate_and_convert(decision, state):
    decision = normalize_decision(decision)
    action = decision["action"]
    args = decision["args"]
    compact = compact_state(state)
    state_type = str(compact.get("state_type") or "").lower()
    battle = compact.get("battle", {})
    player = compact.get("player", {})

    if action == "wait":
        return None, "advisor_wait"

    if action == "combat_end_turn":
        if state_type not in COMBAT_TYPES or battle.get("turn") != "player":
            return None, "not_player_combat_turn"
        return {"action": "end_turn"}, "ok"

    if action == "combat_play_card":
        if state_type not in COMBAT_TYPES or battle.get("turn") != "player" or not battle.get("is_play_phase"):
            return None, "not_combat_play_phase"
        try:
            idx = int(args.get("card_index"))
        except (TypeError, ValueError):
            return None, "missing_card_index"
        hand = player.get("hand", [])
        if idx < 0 or idx >= len(hand):
            return None, "card_index_out_of_range"
        card = hand[idx]
        energy = player.get("energy") or 0
        if not card.get("can_play") or card_cost(card, energy) > energy:
            return None, "card_not_playable"
        payload = {"action": "play_card", "card_index": idx}
        if card.get("target_type") == "AnyEnemy":
            target = args.get("target")
            valid_targets = {e.get("entity_id") for e in battle.get("enemies", []) if e.get("entity_id")}
            if target not in valid_targets:
                return None, "missing_or_invalid_target"
            payload["target"] = target
        return payload, "ok"

    noncombat = {
        "proceed_to_map": {"action": "proceed"},
        "rewards_skip_card": {"action": "skip_card_reward"},
    }
    indexed = {
        "map_choose_node": ("choose_map_node", "index"),
        "rewards_claim": ("claim_reward", "index"),
        "rewards_pick_card": ("select_card_reward", "card_index"),
        "shop_purchase": ("shop_purchase", "index"),
        "event_choose_option": ("choose_event_option", "index"),
        "rest_choose_option": ("choose_rest_option", "index"),
    }
    if action in noncombat:
        return noncombat[action], "ok"
    if action in indexed:
        raw_action, key = indexed[action]
        try:
            value = int(args.get(key))
        except (TypeError, ValueError):
            return None, f"missing_{key}"
        return {"action": raw_action, key: value}, "ok"

    return None, f"unknown_action_{action}"


def append_llm_log(session_id, snapshot):
    LLM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LLM_LOG_DIR / f"llm_actions_{datetime.now():%Y-%m-%d}.jsonl"
    record = {
        "type": "llm_decision",
        "run_id": session_id,
        "timestamp": int(time.time() * 1000),
        "source": "llm",
        **snapshot,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def should_execute(cfg, payload, state):
    if not payload:
        return False
    if cfg.get("mode") != "combat_auto" or not cfg.get("execute_combat"):
        return False
    state_type = str(state.get("state_type") or "").lower()
    return state_type in COMBAT_TYPES and payload.get("action") in ("play_card", "end_turn")


def run_agent():
    session_id = f"llm_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    last_decision_key = None
    actions_this_turn = 0
    last_turn_id = None

    print("STS2 LLM agent started.")
    while True:
        cfg = load_config()
        if not cfg.get("enabled"):
            write_json(LOGIC_PATH, {
                "timestamp": int(time.time() * 1000),
                "status": "disabled",
                "message": "LLM agent process is running, but model access is disabled.",
            })
            time.sleep(2.0)
            continue

        try:
            state = get_game_state()
            compact = compact_state(state)
            turn_id = f"{compact.get('state_type')}:{compact.get('battle', {}).get('turn')}:{compact.get('player', {}).get('energy')}:{len(compact.get('player', {}).get('hand', []))}"
            if turn_id != last_turn_id:
                actions_this_turn = 0
                last_turn_id = turn_id
            if actions_this_turn >= cfg.get("max_actions_per_turn", 12):
                raise RuntimeError("max_actions_per_turn reached")

            actions = action_catalog(compact)
            decision = call_openai_compatible(cfg, compact, actions)
            normalized = normalize_decision(decision)
            payload, validation = validate_and_convert(normalized, state)
            execute = should_execute(cfg, payload, state)
            result = None
            ok = None

            decision_key = json.dumps({"payload": payload, "state": turn_id}, sort_keys=True, ensure_ascii=False)
            if execute and payload and decision_key != last_decision_key:
                result = post_game_action(payload)
                ok = "error" not in result
                last_decision_key = decision_key
                actions_this_turn += 1

            snapshot = {
                "timestamp": int(time.time() * 1000),
                "status": "ok",
                "session_id": session_id,
                "mode": cfg.get("mode"),
                "provider": cfg.get("provider"),
                "model": cfg.get("model"),
                "state_type": state.get("state_type"),
                "decision": normalized,
                "payload": payload,
                "validation": validation,
                "executed": bool(execute and payload),
                "ok": ok,
                "result": result,
                "compact_state": compact,
            }
            write_json(LOGIC_PATH, snapshot)
            append_llm_log(session_id, snapshot)
        except (HTTPError, URLError) as exc:
            write_json(LOGIC_PATH, {
                "timestamp": int(time.time() * 1000),
                "status": "error",
                "error": f"connection_error: {exc}",
            })
        except Exception as exc:
            write_json(LOGIC_PATH, {
                "timestamp": int(time.time() * 1000),
                "status": "error",
                "error": str(exc),
            })

        time.sleep(load_config().get("decision_interval_sec", 3.0))


if __name__ == "__main__":
    run_agent()
