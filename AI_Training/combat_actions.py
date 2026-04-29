import re
from dataclasses import dataclass


ACTION_TYPES = ("play_card", "use_potion", "end_turn")
CANDIDATE_FEATURE_DIM = 26


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_cost(item, energy):
    cost = item.get("cost", 0)
    if cost == "X":
        return energy
    try:
        return int(cost)
    except (TypeError, ValueError):
        return 99


def parse_number_text(text):
    return [int(n) for n in re.findall(r"\d+", str(text or ""))]


def number_hint(item, mode):
    text = " ".join(str(item.get(k) or "") for k in ("description", "name", "id", "type"))
    lowered = text.lower()
    if mode == "attack" and not any(k in lowered for k in ("deal", "damage", "attack", "造成", "伤害")):
        return 0
    if mode == "block" and not any(k in lowered for k in ("block", "defend", "armor", "格挡", "护甲")):
        return 0
    nums = parse_number_text(text)
    return nums[0] if nums else 0


def intent_damage(intent):
    if isinstance(intent, str):
        text = intent
    else:
        text = str((intent or {}).get("label") or (intent or {}).get("description") or (intent or {}).get("type") or "")
    nums = parse_number_text(text)
    if ("x" in text.lower() or "×" in text) and len(nums) >= 2:
        return nums[0] * nums[1]
    return nums[0] if nums else 0


def normalize_combat_state(state):
    if not isinstance(state, dict):
        return {"player": {}, "battle": {}, "run": {}}
    if isinstance(state.get("player"), dict) and isinstance(state.get("battle"), dict):
        return state

    player = {
        "hp": state.get("hp", 0),
        "max_hp": state.get("max_hp", 1),
        "block": state.get("block", 0),
        "energy": state.get("energy", 0),
        "max_energy": state.get("max_energy", 1),
        "hand": state.get("hand", []),
        "potions": state.get("potions", []),
    }
    battle = {
        "round": state.get("round", 0),
        "turn": str(state.get("turn", "")).lower(),
        "is_play_phase": state.get("is_play_phase", True),
        "enemies": state.get("enemies", []),
    }
    run = state.get("run") if isinstance(state.get("run"), dict) else {
        "act": state.get("act", 0),
        "floor": state.get("floor", 0),
        "ascension": state.get("ascension", 0),
    }
    normalized = dict(state)
    normalized["player"] = player
    normalized["battle"] = battle
    normalized["run"] = run
    return normalized


def enemy_hp(enemy):
    return safe_int(enemy.get("hp", enemy.get("current_hp", 0)))


def enemy_max_hp(enemy):
    return max(safe_float(enemy.get("max_hp"), enemy_hp(enemy) or 1), 1.0)


def enemy_block(enemy):
    return safe_float(enemy.get("block"), 0.0)


def enemy_target_id(enemy):
    return enemy.get("entity_id") or enemy.get("id") or enemy.get("name") or ""


def enemy_intent_damage(enemy):
    return sum(intent_damage(intent) for intent in enemy.get("intents", []) or [])


def alive_enemies(state):
    normalized = normalize_combat_state(state)
    enemies = normalized.get("battle", {}).get("enemies", []) or []
    return [e for e in enemies if isinstance(e, dict) and enemy_hp(e) > 0]


def requires_enemy_target(item):
    target_type = str(item.get("target_type") or "").lower()
    return target_type in ("anyenemy", "enemy", "singleenemy")


def is_playable_card(card, energy):
    return bool(card.get("can_play", False)) and parse_cost(card, energy) <= energy


def is_usable_potion(potion):
    if potion.get("can_use_in_combat") is False:
        return False
    if "can_use_in_combat" not in potion and "target_type" not in potion:
        return False
    return True


@dataclass(frozen=True)
class CombatCandidate:
    kind: str
    label: str
    payload: dict
    features: tuple
    card_id: str = ""
    potion_id: str = ""
    card_index: int = -1
    potion_slot: int = -1
    target_id: str = ""
    target_index: int = -1
    target_effective_hp: float = 0.0

    def to_dict(self, include_features=False):
        item = {
            "kind": self.kind,
            "label": self.label,
            "payload": self.payload,
            "card_id": self.card_id,
            "potion_id": self.potion_id,
            "card_index": self.card_index,
            "potion_slot": self.potion_slot,
            "target_id": self.target_id,
            "target_index": self.target_index,
            "target_effective_hp": self.target_effective_hp,
        }
        if include_features:
            item["features"] = list(self.features)
        return item


def _action_type_one_hot(kind):
    return [1.0 if kind == item else 0.0 for item in ACTION_TYPES]


def _candidate_features(state, kind, card=None, potion=None, target=None, target_index=-1):
    normalized = normalize_combat_state(state)
    player = normalized.get("player", {})
    battle = normalized.get("battle", {})
    hand = player.get("hand", []) or []
    enemies = alive_enemies(normalized)
    hp = safe_float(player.get("hp"))
    max_hp = max(safe_float(player.get("max_hp"), 1.0), 1.0)
    block = safe_float(player.get("block"))
    energy = safe_float(player.get("energy"))
    max_energy = max(safe_float(player.get("max_energy"), 1.0), 1.0)
    incoming = sum(enemy_intent_damage(enemy) for enemy in enemies)
    net_incoming = max(0.0, incoming - block)
    affordable_count = sum(1 for item in hand if isinstance(item, dict) and is_playable_card(item, energy))

    item = card or potion or {}
    cost = parse_cost(item, energy) if card else 0
    damage_hint = number_hint(item, "attack")
    block_hint = number_hint(item, "block")
    item_type = str(item.get("type") or "").lower()
    target_required = requires_enemy_target(item)
    has_target = target is not None
    target_hp = safe_float(target.get("hp")) if target else 0.0
    target_block = enemy_block(target) if target else 0.0
    target_intent = enemy_intent_damage(target) if target else 0.0
    target_effective_hp = target_hp + target_block
    lowest_effective_hp = min((safe_float(e.get("hp")) + enemy_block(e) for e in enemies), default=0.0)

    features = []
    features.extend(_action_type_one_hot(kind))
    features.extend([
        min(cost / max_energy, 3.0),
        1.0 if card and is_playable_card(card, energy) else 0.0,
        min(damage_hint / 60.0, 2.0),
        min(block_hint / 60.0, 2.0),
        1.0 if "attack" in item_type else 0.0,
        1.0 if "skill" in item_type else 0.0,
        1.0 if "power" in item_type else 0.0,
        max((energy - cost) / max_energy, -1.0),
        1.0 if target_required else 0.0,
        1.0 if has_target else 0.0,
        min(target_hp / max(enemy_max_hp(target), 1.0), 2.0) if target else 0.0,
        min(target_block / 50.0, 2.0),
        min(target_intent / max_hp, 2.0),
        min(target_effective_hp / 100.0, 2.0),
        1.0 if target and damage_hint and damage_hint >= target_effective_hp else 0.0,
        1.0 if target and target_effective_hp == lowest_effective_hp else 0.0,
        hp / max_hp,
        min(block / 50.0, 2.0),
        energy / max_energy,
        min(incoming / max_hp, 2.0),
        min(net_incoming / max_hp, 2.0),
        min(affordable_count / 10.0, 1.0),
        min(len(enemies) / 5.0, 1.0),
    ])
    if len(features) != CANDIDATE_FEATURE_DIM:
        raise ValueError(f"candidate feature dim drifted: {len(features)} != {CANDIDATE_FEATURE_DIM}")
    return tuple(float(x) for x in features)


def enumerate_combat_actions(state, include_end_turn=True, include_potions=True):
    normalized = normalize_combat_state(state)
    player = normalized.get("player", {})
    battle = normalized.get("battle", {})
    hand = player.get("hand", []) or []
    potions = player.get("potions", normalized.get("potions", [])) or []
    energy = safe_float(player.get("energy"))
    enemies = alive_enemies(normalized)
    candidates = []

    for hand_pos, card in enumerate(hand):
        if not isinstance(card, dict) or not is_playable_card(card, energy):
            continue
        card_index = safe_int(card.get("index"), hand_pos)
        card_id = str(card.get("id") or card.get("name") or "UNKNOWN")
        if requires_enemy_target(card):
            for target_index, target in enumerate(enemies):
                target_id = enemy_target_id(target)
                if not target_id:
                    continue
                payload = {"action": "play_card", "card_index": card_index, "target": target_id}
                label = f"play_card:{card_id}:h{card_index}->t{target_index}:{target_id}"
                candidates.append(CombatCandidate(
                    kind="play_card",
                    label=label,
                    payload=payload,
                    features=_candidate_features(normalized, "play_card", card=card, target=target, target_index=target_index),
                    card_id=card_id,
                    card_index=card_index,
                    target_id=target_id,
                    target_index=target_index,
                    target_effective_hp=safe_float(target.get("hp")) + enemy_block(target),
                ))
        else:
            payload = {"action": "play_card", "card_index": card_index}
            label = f"play_card:{card_id}:h{card_index}"
            candidates.append(CombatCandidate(
                kind="play_card",
                label=label,
                payload=payload,
                features=_candidate_features(normalized, "play_card", card=card),
                card_id=card_id,
                card_index=card_index,
            ))

    if include_potions:
        for potion in potions:
            if not isinstance(potion, dict) or not is_usable_potion(potion):
                continue
            slot = safe_int(potion.get("slot"), -1)
            if slot < 0:
                continue
            potion_id = str(potion.get("id") or potion.get("name") or "UNKNOWN")
            if requires_enemy_target(potion):
                for target_index, target in enumerate(enemies):
                    target_id = enemy_target_id(target)
                    if not target_id:
                        continue
                    payload = {"action": "use_potion", "slot": slot, "target": target_id}
                    label = f"use_potion:{potion_id}:s{slot}->t{target_index}:{target_id}"
                    candidates.append(CombatCandidate(
                        kind="use_potion",
                        label=label,
                        payload=payload,
                        features=_candidate_features(normalized, "use_potion", potion=potion, target=target, target_index=target_index),
                        potion_id=potion_id,
                        potion_slot=slot,
                        target_id=target_id,
                        target_index=target_index,
                        target_effective_hp=safe_float(target.get("hp")) + enemy_block(target),
                    ))
            else:
                payload = {"action": "use_potion", "slot": slot}
                label = f"use_potion:{potion_id}:s{slot}"
                candidates.append(CombatCandidate(
                    kind="use_potion",
                    label=label,
                    payload=payload,
                    features=_candidate_features(normalized, "use_potion", potion=potion),
                    potion_id=potion_id,
                    potion_slot=slot,
                ))

    if include_end_turn:
        candidates.append(CombatCandidate(
            kind="end_turn",
            label="end_turn",
            payload={"action": "end_turn"},
            features=_candidate_features(normalized, "end_turn"),
        ))

    return candidates


def match_logged_action(candidates, action):
    if not isinstance(action, dict):
        return -1
    kind = action.get("action") or action.get("action_type")
    card_id = action.get("card_id") or action.get("card") or action.get("id")
    card_index = action.get("card_index")
    potion_id = action.get("potion_id")
    potion_slot = action.get("slot")
    target_id = action.get("target") or action.get("target_id")
    target_id = "" if target_id in (None, "none", "self", "player") else str(target_id)

    for idx, candidate in enumerate(candidates):
        if candidate.kind != kind:
            continue
        if kind == "end_turn":
            return idx
        if kind == "play_card":
            if card_id and candidate.card_id != str(card_id):
                continue
            if card_index is not None and candidate.card_index != safe_int(card_index, -999):
                continue
            if target_id and candidate.target_id != target_id:
                continue
            return idx
        if kind == "use_potion":
            if potion_id and candidate.potion_id != str(potion_id):
                continue
            if potion_slot is not None and candidate.potion_slot != safe_int(potion_slot, -999):
                continue
            if target_id and candidate.target_id != target_id:
                continue
            return idx
    return -1


def choose_candidate_for_card(candidates, card):
    card_id = str(card.get("id") or card.get("name") or "")
    card_index = safe_int(card.get("index"), -1)
    matches = [
        candidate for candidate in candidates
        if candidate.kind == "play_card"
        and (candidate.card_index == card_index or (card_index < 0 and candidate.card_id == card_id))
    ]
    if not matches:
        return None
    targeted = [candidate for candidate in matches if candidate.target_id]
    if targeted:
        return min(targeted, key=lambda item: item.target_effective_hp)
    return matches[0]


def candidate_feature_rows(candidates):
    return [list(candidate.features) for candidate in candidates]


def public_candidate_catalog(candidates, limit=None):
    selected = candidates if limit is None else candidates[:limit]
    return [candidate.to_dict(include_features=False) for candidate in selected]
