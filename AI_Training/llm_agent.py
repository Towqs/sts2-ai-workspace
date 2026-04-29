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
PLACEHOLDER_TARGETS = {
    "",
    "enemy",
    "target",
    "required enemy entity_id",
    "one battle.enemies entity_id",
    "enemy entity_id",
}
RAW_TO_CATALOG_ACTION = {
    "play_card": "combat_play_card",
    "use_potion": "combat_use_potion",
    "end_turn": "combat_end_turn",
    "choose_map_node": "map_choose_node",
    "claim_reward": "rewards_claim",
    "select_card_reward": "rewards_pick_card",
    "skip_card_reward": "rewards_skip_card",
    "shop_purchase": "shop_purchase",
    "choose_event_option": "event_choose_option",
    "choose_rest_option": "rest_choose_option",
    "advance_dialogue": "event_advance_dialogue",
    "proceed": "proceed_to_map",
}
CATALOG_TO_RAW_ACTION = {v: k for k, v in RAW_TO_CATALOG_ACTION.items()}
CATALOG_TO_RAW_ACTION.update({
    "wait": "wait",
})


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


def payload_with_policy(payload, policy_name, model_version):
    if not payload:
        return payload
    tagged = dict(payload)
    tagged["policy_name"] = policy_name
    tagged["model_version"] = model_version
    return tagged


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


def safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def needs_enemy_target(item):
    target_type = str((item or {}).get("target_type") or "").lower()
    return target_type in ("anyenemy", "enemy", "singleenemy")


def compact_card_by_index(compact, index):
    for card in ((compact.get("player") or {}).get("hand") or []):
        if safe_int(card.get("index"), -1) == index:
            return card
    return None


def compact_potion_by_slot(compact, slot):
    for potion in ((compact.get("player") or {}).get("potions") or []):
        if safe_int(potion.get("slot"), -1) == slot:
            return potion
    return None


def intent_damage(intent):
    if isinstance(intent, str):
        text = intent
    else:
        intent = intent or {}
        text = str(intent.get("label") or intent.get("description") or intent.get("type") or "")
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


def action_catalog_from_candidates(candidates):
    """Expose exact legal payloads while keeping the action+args prompt shape."""
    actions = []
    seen = set()
    for candidate in candidates:
        payload = candidate.get("payload")
        if not payload:
            item = {
                "action": "wait",
                "args": {},
                "candidate_id": candidate.get("id"),
                "label": candidate.get("label"),
            }
        else:
            raw_action = payload.get("action")
            catalog_action = RAW_TO_CATALOG_ACTION.get(raw_action, raw_action)
            args = {
                key: value for key, value in payload.items()
                if key not in ("action", "policy_name", "model_version")
            }
            item = {
                "action": catalog_action,
                "args": args,
                "candidate_id": candidate.get("id"),
                "label": candidate.get("label"),
            }
            if candidate.get("detail"):
                item["detail"] = candidate.get("detail")
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        actions.append(item)
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
        "Copy action and args exactly from one available_actions item. "
        "Never invent card indexes, potion slots, reward indexes, or enemy ids. "
        "If no useful action is safe, choose wait."
    )
    user = {
        "mode": cfg.get("mode"),
        "rules": [
            "available_actions already contains exact legal args. Copy one item; do not reinterpret placeholder text.",
            "For enemy-targeted cards, target must be an entity_id from battle.enemies.",
            "For potions, slot must be one listed potion slot; target must be one listed enemy entity_id when required.",
            "Prefer playing useful zero-cost cards before spending energy.",
            "Do not end turn while affordable playable cards remain unless clearly justified.",
            "Use battle.incoming_damage, battle.net_incoming_after_block, and battle.hp_after_incoming to decide whether blocking is mandatory.",
            "In boss or elite fights, consider using potions before the player is in lethal danger.",
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
        "use_potion": "combat_use_potion",
        "claim_reward": "rewards_claim",
        "select_card_reward": "rewards_pick_card",
        "skip_card_reward": "rewards_skip_card",
        "choose_map_node": "map_choose_node",
        "choose_event_option": "event_choose_option",
        "choose_rest_option": "rest_choose_option",
        "advance_dialogue": "event_advance_dialogue",
        "proceed": "proceed_to_map",
    }
    action = aliases.get(decision.get("action"), decision.get("action"))
    args = decision.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    return {
        "action": action,
        "args": args,
        "candidate_id": str(decision.get("candidate_id") or decision.get("id") or args.get("candidate_id") or ""),
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

    if action == "combat_use_potion":
        if state_type not in COMBAT_TYPES or battle.get("turn") != "player" or not battle.get("is_play_phase"):
            return None, "not_combat_play_phase"
        try:
            slot = int(args.get("slot"))
        except (TypeError, ValueError):
            return None, "missing_potion_slot"
        potions = player.get("potions", [])
        potion = next((item for item in potions if safe_int(item.get("slot"), -1) == slot), None)
        if not potion:
            return None, "potion_slot_not_found"
        if potion.get("can_use_in_combat") is False:
            return None, "potion_not_usable"
        payload = {"action": "use_potion", "slot": slot}
        if needs_enemy_target(potion):
            target = args.get("target")
            living_enemies = [e for e in battle.get("enemies", []) if e.get("entity_id")]
            valid_targets = {e.get("entity_id") for e in living_enemies}
            if target not in valid_targets:
                placeholder = str(target or "").strip().lower()
                if living_enemies and placeholder in PLACEHOLDER_TARGETS:
                    payload["target"] = min(living_enemies, key=enemy_hp).get("entity_id")
                    return payload, "auto_targeted_lowest_hp"
                return None, "missing_or_invalid_target"
            payload["target"] = target
        return payload, "ok"

    noncombat = {
        "proceed_to_map": {"action": "proceed"},
        "rewards_skip_card": {"action": "skip_card_reward"},
        "event_advance_dialogue": {"action": "advance_dialogue"},
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


def candidate_payload(candidate):
    payload = candidate.get("payload") if isinstance(candidate, dict) else None
    return payload if isinstance(payload, dict) else None


def payload_field(payload, *names):
    for name in names:
        if name in payload:
            return payload.get(name)
    return None


def find_candidate_for_catalog_decision(normalized, candidates):
    candidate_id = normalized.get("candidate_id")
    if candidate_id:
        for candidate in candidates:
            if candidate.get("id") == candidate_id:
                return candidate

    action = normalized.get("action")
    args = normalized.get("args") or {}
    raw_action = CATALOG_TO_RAW_ACTION.get(action, action)
    if raw_action == "wait":
        return next((item for item in candidates if item.get("id") == "wait"), None)

    wanted_index = safe_int(payload_field(args, "index", "card_index"), None)
    wanted_slot = safe_int(args.get("slot"), None)
    wanted_target = str(args.get("target") or args.get("target_id") or "").strip()
    wanted_target_lower = wanted_target.lower()

    matches = []
    loose_matches = []
    for candidate in candidates:
        payload = candidate_payload(candidate)
        if not payload or payload.get("action") != raw_action:
            continue
        if raw_action == "play_card":
            card_index = safe_int(payload.get("card_index"), None)
            if wanted_index is not None and card_index != wanted_index:
                continue
            target = str(payload.get("target") or "")
            if wanted_target and wanted_target_lower not in PLACEHOLDER_TARGETS and target and target != wanted_target:
                continue
            (matches if target == wanted_target or not target else loose_matches).append(candidate)
            continue
        if raw_action == "use_potion":
            slot = safe_int(payload.get("slot"), None)
            if wanted_slot is not None and slot != wanted_slot:
                continue
            target = str(payload.get("target") or "")
            if wanted_target and wanted_target_lower not in PLACEHOLDER_TARGETS and target and target != wanted_target:
                continue
            (matches if target == wanted_target or not target else loose_matches).append(candidate)
            continue
        if raw_action in ("choose_map_node", "claim_reward", "choose_event_option", "choose_rest_option", "shop_purchase"):
            if wanted_index is None or safe_int(payload.get("index"), None) == wanted_index:
                matches.append(candidate)
            continue
        if raw_action == "select_card_reward":
            if wanted_index is None or safe_int(payload.get("card_index"), None) == wanted_index:
                matches.append(candidate)
            continue
        if raw_action in ("end_turn", "skip_card_reward", "advance_dialogue", "proceed"):
            matches.append(candidate)

    return (matches or loose_matches or [None])[0]


def card_for_candidate(compact, candidate):
    payload = candidate_payload(candidate) or {}
    return compact_card_by_index(compact, safe_int(payload.get("card_index"), -1)) or {}


def choose_demo_progress_candidate(candidates, compact, prefer_zero_cost=False):
    playable = []
    potion = []
    for candidate in candidates:
        payload = candidate_payload(candidate)
        if not payload:
            continue
        if payload.get("action") == "play_card":
            playable.append(candidate)
        elif payload.get("action") == "use_potion":
            potion.append(candidate)
    if not playable and not potion:
        return None

    def score(candidate):
        payload = candidate_payload(candidate) or {}
        card = card_for_candidate(compact, candidate)
        card_id = str(card.get("id") or candidate.get("detail", {}).get("card_id") or "")
        cost = card_cost(card, (compact.get("player") or {}).get("energy") or 0)
        card_type = str(card.get("type") or "").lower()
        score_value = 0
        if payload.get("action") == "use_potion":
            score_value += 30
        if cost == 0:
            score_value += 50
        if "attack" in card_type:
            score_value += 20
        if "skill" in card_type:
            score_value += 12
        if any(key in card_id for key in ("OFFERING", "ANGER", "BULLY", "BREAKTHROUGH")):
            score_value += 8
        if prefer_zero_cost and cost != 0:
            score_value -= 40
        return -score_value

    pool = playable or potion
    return sorted(pool, key=score)[0]


def catalog_guardrail_payload(normalized, payload, validation, candidates, compact):
    affordable = (compact.get("player") or {}).get("affordable_cards") or []
    selected = find_candidate_for_catalog_decision(normalized, candidates)
    if selected:
        selected_payload = candidate_payload(selected)
        if selected_payload:
            if selected_payload.get("action") == "end_turn" and affordable:
                fallback = choose_demo_progress_candidate(candidates, compact, prefer_zero_cost=True)
                if fallback and fallback.get("id") != selected.get("id"):
                    return candidate_payload(fallback), "guarded_premature_end_turn", fallback
            if payload != selected_payload or validation != "ok":
                return selected_payload, f"{validation}->candidate_guardrail", selected
            return payload, validation, selected
        return None, "advisor_wait", selected

    if payload is None:
        fallback = choose_demo_progress_candidate(candidates, compact)
        if fallback:
            return candidate_payload(fallback), f"{validation}->demo_progress_fallback", fallback
        return payload, validation, None

    if payload.get("action") == "end_turn" and affordable:
        fallback = choose_demo_progress_candidate(candidates, compact, prefer_zero_cost=True)
        if fallback:
            return candidate_payload(fallback), "guarded_premature_end_turn", fallback

    return payload, validation, None


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
            candidates = candidate_action_catalog(state, compact)
            actions = action_catalog_from_candidates(candidates)
            selected_candidate = None
            if selection_mode == "candidate_id":
                decision = call_openai_candidate_selector(cfg, compact, candidates)
                normalized, payload, validation, selected_candidate = validate_candidate_decision(decision, candidates)
            else:
                decision = call_openai_compatible(cfg, compact, actions)
                normalized = normalize_decision(decision)
                payload, validation = validate_and_convert(normalized, state)
                payload, validation, selected_candidate = catalog_guardrail_payload(normalized, payload, validation, candidates, compact)
            policy_name = "llm_candidate_id" if selection_mode == "candidate_id" else "llm_catalog_args"
            model_version = str(cfg.get("model") or "")
            payload = payload_with_policy(payload, policy_name, model_version)
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
                "time": datetime.now().strftime("%H:%M:%S"),
                "status": "ok",
                "session_id": session_id,
                "mode": cfg.get("mode"),
                "action_selection_mode": selection_mode,
                "policy_name": policy_name,
                "model_version": model_version,
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
