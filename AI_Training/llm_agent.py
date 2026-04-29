import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from combat_actions import enumerate_combat_actions

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
    "action_selection_mode": "catalog_args",
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
    if cfg.get("action_selection_mode") not in ("catalog_args", "candidate_id"):
        cfg["action_selection_mode"] = "catalog_args"
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


def intent_damage(intent):
    text = str(intent.get("label") or intent.get("description") or "")
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if ("x" in text.lower() or "×" in text) and len(nums) >= 2:
        return nums[0] * nums[1]
    return nums[0] if nums else 0


def compact_state(state):
    player = state.get("player", {})
    battle = state.get("battle", {})
    run = state.get("run", {})
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
    incoming_damage = sum(intent_damage(intent) for enemy in enemies for intent in enemy.get("intents", []))
    block = player.get("block") or 0
    hp = player.get("hp") or 0
    net_incoming = max(0, incoming_damage - block)
    affordable_cards = [
        card for card in hand
        if card.get("can_play") and card_cost(card, player.get("energy") or 0) <= (player.get("energy") or 0)
    ]
    return {
        "state_type": state.get("state_type"),
        "run": {
            "act": run.get("act"),
            "floor": run.get("floor"),
            "ascension": run.get("ascension"),
        },
        "player": {
            "character": player.get("character"),
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "block": player.get("block"),
            "energy": player.get("energy"),
            "max_energy": player.get("max_energy"),
            "gold": player.get("gold"),
            "draw_pile_count": player.get("draw_pile_count"),
            "discard_pile_count": player.get("discard_pile_count"),
            "exhaust_pile_count": player.get("exhaust_pile_count"),
            "hand": hand,
            "affordable_cards": [{"index": c["index"], "id": c["id"], "cost": c["cost"]} for c in affordable_cards],
            "relics": player.get("relics", [])[:12],
            "potions": player.get("potions", []),
        },
        "battle": {
            "round": battle.get("round"),
            "turn": battle.get("turn"),
            "is_play_phase": battle.get("is_play_phase"),
            "player_actions_disabled": battle.get("player_actions_disabled"),
            "incoming_damage": incoming_damage,
            "net_incoming_after_block": net_incoming,
            "hp_after_incoming": hp - net_incoming,
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
                    args["target"] = "one battle.enemies entity_id"
                action = {"action": "combat_play_card", "args": args, "card": card.get("id")}
                if card.get("target_type") == "AnyEnemy":
                    action["target_options"] = [e.get("entity_id") for e in compact.get("battle", {}).get("enemies", []) if e.get("entity_id")]
                actions.append(action)
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


def add_candidate(candidates, candidate_id, label, payload, detail=None):
    item = {
        "id": candidate_id,
        "label": label,
        "payload": payload,
    }
    if detail:
        item["detail"] = detail
    candidates.append(item)


def items_for_state(state, state_type):
    if state_type == "rewards":
        return (state.get("rewards") or {}).get("items", []) or []
    if state_type == "shop":
        return (state.get("shop") or {}).get("items", []) or []
    if state_type == "fake_merchant":
        return ((state.get("fake_merchant") or {}).get("shop") or {}).get("items", []) or []
    return []


def can_proceed(state, state_type):
    if state_type in ("rewards", "shop", "rest_site"):
        return bool((state.get(state_type) or {}).get("can_proceed"))
    if state_type == "fake_merchant":
        return bool(((state.get("fake_merchant") or {}).get("shop") or {}).get("can_proceed"))
    if state_type == "treasure":
        return bool((state.get("treasure") or {}).get("can_proceed"))
    return False


def candidate_action_catalog(state, compact):
    state_type = str(compact.get("state_type") or state.get("state_type") or "").lower()
    candidates = []
    add_candidate(candidates, "wait", "wait", None)

    if state_type in COMBAT_TYPES and compact.get("battle", {}).get("is_play_phase") and compact.get("battle", {}).get("turn") == "player":
        for candidate in enumerate_combat_actions(state):
            item = candidate.to_dict(include_features=False)
            add_candidate(
                candidates,
                candidate.label,
                candidate.label,
                candidate.payload,
                {
                    "kind": item.get("kind"),
                    "card_id": item.get("card_id"),
                    "potion_id": item.get("potion_id"),
                    "card_index": item.get("card_index"),
                    "potion_slot": item.get("potion_slot"),
                    "target_id": item.get("target_id"),
                },
            )
        return candidates

    if state_type == "map":
        for fallback_index, option in enumerate(((state.get("map") or {}).get("next_options") or [])):
            index = int(option.get("index", fallback_index))
            add_candidate(
                candidates,
                f"map:{index}",
                f"choose map node {index}",
                {"action": "choose_map_node", "index": index},
                {"type": option.get("type"), "col": option.get("col"), "row": option.get("row")},
            )

    if state_type == "rewards":
        for fallback_index, item in enumerate(items_for_state(state, state_type)):
            index = int(item.get("index", fallback_index))
            add_candidate(
                candidates,
                f"reward:{index}",
                f"claim reward {index}",
                {"action": "claim_reward", "index": index},
                {
                    "type": item.get("type") or item.get("category"),
                    "description": item.get("description"),
                    "potion": item.get("potion_id") or item.get("potion_name"),
                },
            )

    if state_type == "card_reward":
        card_state = state.get("card_reward") or {}
        for fallback_index, card in enumerate(card_state.get("cards") or []):
            index = int(card.get("index", fallback_index))
            add_candidate(
                candidates,
                f"card_reward:{index}",
                f"pick card reward {index}",
                {"action": "select_card_reward", "card_index": index},
                {
                    "id": card.get("id"),
                    "name": card.get("name"),
                    "type": card.get("type"),
                    "rarity": card.get("rarity"),
                    "cost": card.get("cost"),
                },
            )
        if card_state.get("can_skip"):
            add_candidate(candidates, "card_reward:skip", "skip card reward", {"action": "skip_card_reward"})

    if state_type == "event":
        event = state.get("event") or {}
        if event.get("in_dialogue"):
            add_candidate(candidates, "event:advance_dialogue", "advance event dialogue", {"action": "advance_dialogue"})
        for fallback_index, option in enumerate(event.get("options") or []):
            if option.get("is_locked") or option.get("was_chosen"):
                continue
            index = int(option.get("index", fallback_index))
            add_candidate(
                candidates,
                f"event:{index}",
                f"choose event option {index}",
                {"action": "choose_event_option", "index": index},
                {"title": option.get("title"), "description": option.get("description"), "is_proceed": option.get("is_proceed")},
            )

    if state_type == "rest_site":
        for fallback_index, option in enumerate((state.get("rest_site") or {}).get("options") or []):
            if not option.get("is_enabled", True):
                continue
            index = int(option.get("index", fallback_index))
            add_candidate(
                candidates,
                f"rest:{index}",
                f"choose rest option {index}",
                {"action": "choose_rest_option", "index": index},
                {"id": option.get("id"), "name": option.get("name"), "description": option.get("description")},
            )

    if state_type in ("shop", "fake_merchant"):
        for fallback_index, item in enumerate(items_for_state(state, state_type)):
            if item.get("is_stocked") is False or item.get("can_afford") is False:
                continue
            index = int(item.get("index", fallback_index))
            add_candidate(
                candidates,
                f"shop:{index}",
                f"buy shop item {index}",
                {"action": "shop_purchase", "index": index},
                {
                    "category": item.get("category"),
                    "price": item.get("price"),
                    "card": item.get("card_id") or item.get("card_name"),
                    "relic": item.get("relic_id") or item.get("relic_name"),
                    "potion": item.get("potion_id") or item.get("potion_name"),
                },
            )

    if can_proceed(state, state_type):
        add_candidate(candidates, "proceed", "proceed", {"action": "proceed"})

    return candidates


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


def chat_response_content(result):
    choices = result.get("choices") if isinstance(result, dict) else None
    if not choices:
        return None
    choice = choices[0] if choices else {}
    message = choice.get("message") if isinstance(choice, dict) else None
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(part))
            return "\n".join(p for p in parts if p)
        if content is not None:
            return str(content)
        if message.get("tool_calls"):
            return ""
    if isinstance(choice, dict) and choice.get("text") is not None:
        return str(choice.get("text"))
    return None


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
            "Use battle.incoming_damage, battle.net_incoming_after_block, and battle.hp_after_incoming to decide whether blocking is mandatory.",
            "Use run.act, run.floor, battle.round, relics, potions, and pile counts when explaining risk.",
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
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Accept": "application/json"}
    result = request_json(chat_url(cfg.get("base_url")), method="POST", body=body, headers=headers, timeout=45)
    content = chat_response_content(result)
    if content is None:
        preview = json.dumps(result, ensure_ascii=False)[:1500]
        raise RuntimeError(f"Model response has no chat content: {preview}")
    try:
        decision = extract_json(content)
    except Exception as exc:
        preview = content[:1500].replace("\n", " ")
        raise RuntimeError(f"Model response did not contain JSON decision: {preview}") from exc
    decision["_raw_content"] = content
    return decision


def call_openai_candidate_selector(cfg, compact, candidates):
    if not cfg.get("api_key"):
        raise RuntimeError("Missing API key")
    if not cfg.get("model"):
        raise RuntimeError("Missing model name")

    system = (
        "You are a Slay the Spire 2 decision agent. "
        "Choose exactly one candidate id from the provided candidate_actions list. "
        "Return only JSON with keys: candidate_id, reason, confidence. "
        "Do not create actions, arguments, card indexes, enemy ids, or payloads. "
        "If no useful action is safe, choose candidate_id \"wait\"."
    )
    user = {
        "mode": cfg.get("mode"),
        "rules": [
            "candidate_id must exactly match one id in candidate_actions.",
            "Candidate payloads are read-only; never modify or invent them.",
            "Prefer useful zero-cost cards before spending energy.",
            "Do not choose end_turn while affordable playable cards remain unless clearly justified.",
            "Use battle.incoming_damage, battle.net_incoming_after_block, and battle.hp_after_incoming to decide whether blocking is mandatory.",
            "Shop purchases are suggestions unless execution is explicitly enabled.",
        ],
        "state": compact,
        "candidate_actions": candidates,
    }
    body = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": cfg.get("temperature", 0.2),
        "max_tokens": cfg.get("max_tokens", 700),
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Accept": "application/json"}
    result = request_json(chat_url(cfg.get("base_url")), method="POST", body=body, headers=headers, timeout=45)
    content = chat_response_content(result)
    if content is None:
        preview = json.dumps(result, ensure_ascii=False)[:1500]
        raise RuntimeError(f"Model response has no chat content: {preview}")
    try:
        decision = extract_json(content)
    except Exception as exc:
        preview = content[:1500].replace("\n", " ")
        raise RuntimeError(f"Model response did not contain JSON decision: {preview}") from exc
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


def normalize_candidate_decision(decision):
    candidate_id = decision.get("candidate_id") or decision.get("id") or decision.get("action")
    return {
        "candidate_id": str(candidate_id or ""),
        "reason": str(decision.get("reason", ""))[:1200],
        "confidence": decision.get("confidence"),
    }


def validate_candidate_decision(decision, candidates):
    normalized = normalize_candidate_decision(decision)
    candidate_id = normalized["candidate_id"]
    by_id = {item.get("id"): item for item in candidates}
    selected = by_id.get(candidate_id)
    if selected is None:
        return normalized, None, "candidate_id_not_found", None
    return normalized, selected.get("payload"), "ok" if selected.get("payload") else "advisor_wait", selected


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
            living_enemies = [e for e in battle.get("enemies", []) if e.get("entity_id")]
            valid_targets = {e.get("entity_id") for e in living_enemies}
            if target not in valid_targets:
                placeholder = str(target or "").strip().lower()
                if living_enemies and placeholder in ("", "required enemy entity_id", "one battle.enemies entity_id", "enemy", "target"):
                    target = min(living_enemies, key=enemy_hp).get("entity_id")
                    payload["target"] = target
                    return payload, "auto_targeted_lowest_hp"
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


def error_backoff_seconds(error_text, consecutive_errors):
    text = str(error_text or "").lower()
    if "invalid token" in text or "recaptcha" in text or "http 401" in text:
        return 300
    return min(300, 15 * (2 ** min(max(consecutive_errors - 1, 0), 4)))


def state_signature(compact):
    battle = compact.get("battle", {})
    player = compact.get("player", {})
    hand = player.get("hand", [])
    enemies = compact.get("battle", {}).get("enemies", [])
    return json.dumps({
        "state_type": compact.get("state_type"),
        "round": battle.get("round"),
        "turn": battle.get("turn"),
        "play_phase": battle.get("is_play_phase"),
        "energy": player.get("energy"),
        "hand": [(c.get("index"), c.get("id"), c.get("cost"), c.get("can_play")) for c in hand],
        "enemies": [(e.get("entity_id"), e.get("hp"), e.get("block")) for e in enemies],
    }, sort_keys=True, ensure_ascii=False)


def is_combat_action_phase(compact):
    state_type = str(compact.get("state_type") or "").lower()
    battle = compact.get("battle", {})
    return state_type in COMBAT_TYPES and battle.get("turn") == "player" and bool(battle.get("is_play_phase"))


def run_agent():
    session_id = f"llm_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    last_decision_key = None
    actions_this_turn = 0
    last_turn_id = None
    consecutive_errors = 0
    next_request_at = 0.0
    last_error = ""

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
            now = time.time()
            if next_request_at > now:
                remaining = max(1, int(next_request_at - now))
                write_json(LOGIC_PATH, {
                    "timestamp": int(now * 1000),
                    "status": "cooldown",
                    "error": last_error,
                    "retry_after_sec": remaining,
                    "message": f"LLM request paused after API error. Retry in {remaining}s.",
                })
                time.sleep(min(5.0, remaining))
                continue

            state = get_game_state()
            compact = compact_state(state)
            if cfg.get("mode") == "combat_auto" and not is_combat_action_phase(compact):
                write_json(LOGIC_PATH, {
                    "timestamp": int(time.time() * 1000),
                    "status": "waiting",
                    "session_id": session_id,
                    "mode": cfg.get("mode"),
                    "provider": cfg.get("provider"),
                    "model": cfg.get("model"),
                    "state_type": state.get("state_type"),
                    "message": "Combat Auto waits without model calls outside the player's combat action phase.",
                    "compact_state": compact,
                })
                time.sleep(load_config().get("decision_interval_sec", 3.0))
                continue

            turn_id = state_signature(compact)
            if turn_id != last_turn_id:
                actions_this_turn = 0
                last_turn_id = turn_id
            if actions_this_turn >= cfg.get("max_actions_per_turn", 12):
                raise RuntimeError("max_actions_per_turn reached")

            selection_mode = cfg.get("action_selection_mode", "catalog_args")
            actions = action_catalog(compact)
            candidates = candidate_action_catalog(state, compact)
            selected_candidate = None
            if selection_mode == "candidate_id":
                decision = call_openai_candidate_selector(cfg, compact, candidates)
                normalized, payload, validation, selected_candidate = validate_candidate_decision(decision, candidates)
            else:
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
                "action_selection_mode": selection_mode,
                "provider": cfg.get("provider"),
                "model": cfg.get("model"),
                "state_type": state.get("state_type"),
                "decision": normalized,
                "raw_response": decision.get("_raw_content"),
                "candidate_actions": candidates,
                "selected_candidate": selected_candidate,
                "payload": payload,
                "validation": validation,
                "executed": bool(execute and payload),
                "ok": ok,
                "result": result,
                "compact_state": compact,
            }
            write_json(LOGIC_PATH, snapshot)
            append_llm_log(session_id, snapshot)
            consecutive_errors = 0
            next_request_at = 0.0
            last_error = ""
        except (HTTPError, URLError) as exc:
            last_error = f"connection_error: {exc}"
            consecutive_errors += 1
            next_request_at = time.time() + error_backoff_seconds(last_error, consecutive_errors)
            write_json(LOGIC_PATH, {
                "timestamp": int(time.time() * 1000),
                "status": "error",
                "error": last_error,
                "retry_after_sec": max(1, int(next_request_at - time.time())),
            })
        except Exception as exc:
            last_error = str(exc)
            consecutive_errors += 1
            next_request_at = time.time() + error_backoff_seconds(last_error, consecutive_errors)
            write_json(LOGIC_PATH, {
                "timestamp": int(time.time() * 1000),
                "status": "error",
                "error": last_error,
                "retry_after_sec": max(1, int(next_request_at - time.time())),
            })

        time.sleep(load_config().get("decision_interval_sec", 3.0))


if __name__ == "__main__":
    run_agent()
