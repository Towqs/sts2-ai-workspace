import math
import re


DECK_SUMMARY_VERSION = "deck_summary_v1"


def _safe_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, list) else []


def _text_blob(card):
    if not isinstance(card, dict):
        return ""
    parts = []
    for key in ("id", "name", "card_id", "card_name", "title", "type", "rarity", "description", "text"):
        value = card.get(key)
        if value is not None:
            parts.append(str(value))
    keywords = card.get("keywords")
    if isinstance(keywords, list):
        parts.extend(str(item) for item in keywords)
    return " ".join(parts).lower()


def normalize_card_id(card):
    if not isinstance(card, dict):
        return ""
    value = (
        card.get("id")
        or card.get("card_id")
        or card.get("name")
        or card.get("card_name")
        or card.get("title")
        or ""
    )
    return re.sub(r"\+", "", str(value).strip().upper().replace(" ", "_"))


def card_cost(card):
    if not isinstance(card, dict):
        return 0.0
    cost = card.get("cost", card.get("energy_cost", 0))
    if str(cost).upper() == "X":
        return 3.0
    return _safe_float(cost, 0.0)


def normalized_card_type(card):
    value = card.get("type") if isinstance(card, dict) else card
    text = str(value or "").strip().lower()
    if "attack" in text or "\u653b\u51fb" in text:
        return "attack"
    if "skill" in text or "\u6280\u80fd" in text:
        return "skill"
    if "power" in text or "\u80fd\u529b" in text:
        return "power"
    if "curse" in text or "\u8bc5\u5492" in text:
        return "curse"
    if "status" in text or "\u72b6\u6001" in text:
        return "status"
    return "other"


def _has_any(text, terms):
    return any(term in text for term in terms)


def classify_card(card):
    """Return coarse Ironclad role tags for a card-like dict."""
    text = _text_blob(card)
    card_id = normalize_card_id(card)
    card_id_l = card_id.lower()
    ctype = normalized_card_type(card)
    tags = {
        "type": ctype,
        "card_id": card_id,
        "cost": card_cost(card),
        "damage": ctype == "attack",
        "block": ctype == "skill" and _has_any(text, ("block", "defend", "armor", "\u683c\u6321", "\u62a4\u7532")),
        "draw": _has_any(text, ("draw", "card draw", "\u62bd\u724c", "\u62bd")),
        "scaling": _has_any(text, ("strength", "dexterity", "whenever", "each turn", "\u529b\u91cf", "\u654f\u6377", "\u6bcf\u56de\u5408")),
        "strength": _has_any(text, ("strength", "\u529b\u91cf")),
        "multihit": _has_any(card_id_l, ("twin_strike", "sword_boomerang", "pummel", "whirlwind", "reaper")),
        "block_payoff": _has_any(card_id_l, ("body_slam", "barricade", "entrench", "juggernaut")),
        "exhaust": _has_any(text, ("exhaust", "\u6d88\u8017")),
        "exhaust_payoff": _has_any(card_id_l, ("feel_no_pain", "dark_embrace", "corruption", "charons_ashes")),
        "vulnerable": _has_any(text, ("vulnerable", "\u6613\u4f24")),
        "self_damage": _has_any(card_id_l, ("hemokinesis", "bloodletting", "offering", "rupture", "combust", "brutality")),
        "aoe": _has_any(text, ("all enemies", "aoe", "\u6240\u6709\u654c\u4eba", "\u5168\u4f53")),
        "premium": False,
    }

    premium_ids = {
        "BATTLE_TRANCE",
        "SHRUG_IT_OFF",
        "FEEL_NO_PAIN",
        "DARK_EMBRACE",
        "CORRUPTION",
        "BARRICADE",
        "BODY_SLAM",
        "INFLAME",
        "DEMON_FORM",
        "SPOT_WEAKNESS",
        "LIMIT_BREAK",
        "IMMOLATE",
        "FEED",
    }
    tags["premium"] = card_id in premium_ids or tags["block_payoff"] or tags["exhaust_payoff"]

    if card_id in {"BASH", "UPPERCUT", "SHOCKWAVE", "THUNDERCLAP"}:
        tags["vulnerable"] = True
    if card_id in {"INFLAME", "DEMON_FORM", "SPOT_WEAKNESS", "LIMIT_BREAK", "FLEX"}:
        tags["strength"] = True
        tags["scaling"] = True
    if card_id in {"POMMEL_STRIKE", "SHRUG_IT_OFF", "BATTLE_TRANCE", "BURNING_PACT", "OFFERING", "DARK_EMBRACE"}:
        tags["draw"] = True
    if card_id in {"DEFEND", "SHRUG_IT_OFF", "TRUE_GRIT", "IMPERVIOUS", "FLAME_BARRIER", "POWER_THROUGH", "SECOND_WIND"}:
        tags["block"] = True
    if card_id in {"TRUE_GRIT", "SECOND_WIND", "FIEND_FIRE", "SEVER_SOUL", "BURNING_PACT", "CORRUPTION"}:
        tags["exhaust"] = True

    return tags


def _deck_cards(state):
    state = _as_dict(state)
    player = _as_dict(state.get("player"))
    candidates = (
        player.get("deck"),
        player.get("cards"),
        player.get("master_deck"),
        state.get("deck"),
        state.get("cards"),
    )
    for value in candidates:
        cards = _as_list(value)
        if cards:
            return cards
    return []


def _overview_counts(state):
    state = _as_dict(state)
    player = _as_dict(state.get("player"))
    overview = player.get("deck_overview") or state.get("deck_overview")
    return overview if isinstance(overview, dict) else {}


def build_deck_summary(state):
    state = _as_dict(state)
    player = _as_dict(state.get("player"))
    cards = [card for card in _deck_cards(state) if isinstance(card, dict)]
    overview = _overview_counts(state)
    summary = {
        "deck_summary_version": DECK_SUMMARY_VERSION,
        "deck_size": _safe_int(player.get("deck_size", state.get("deck_size", len(cards))), len(cards)),
        "attack_count": 0,
        "skill_count": 0,
        "power_count": 0,
        "curse_count": 0,
        "status_count": 0,
        "other_count": 0,
        "damage_count": 0,
        "block_count": 0,
        "draw_count": 0,
        "scaling_count": 0,
        "strength_sources": 0,
        "multihit_count": 0,
        "block_payoffs": 0,
        "exhaust_cards": 0,
        "exhaust_payoffs": 0,
        "vulnerable_sources": 0,
        "self_damage_enablers": 0,
        "aoe_count": 0,
        "avg_cost": 0.0,
        "cost_0": 0,
        "cost_1": 0,
        "cost_2": 0,
        "cost_3_plus": 0,
        "bloat_score": 0.0,
        "archetype_scores": {},
    }

    costs = []
    if cards:
        for card in cards:
            tags = classify_card(card)
            ctype = tags["type"]
            if ctype == "attack":
                summary["attack_count"] += 1
            elif ctype == "skill":
                summary["skill_count"] += 1
            elif ctype == "power":
                summary["power_count"] += 1
            elif ctype == "curse":
                summary["curse_count"] += 1
            elif ctype == "status":
                summary["status_count"] += 1
            else:
                summary["other_count"] += 1
            for key in (
                ("damage", "damage_count"),
                ("block", "block_count"),
                ("draw", "draw_count"),
                ("scaling", "scaling_count"),
                ("strength", "strength_sources"),
                ("multihit", "multihit_count"),
                ("block_payoff", "block_payoffs"),
                ("exhaust", "exhaust_cards"),
                ("exhaust_payoff", "exhaust_payoffs"),
                ("vulnerable", "vulnerable_sources"),
                ("self_damage", "self_damage_enablers"),
                ("aoe", "aoe_count"),
            ):
                if tags[key[0]]:
                    summary[key[1]] += 1
            cost = tags["cost"]
            costs.append(cost)
            if cost <= 0:
                summary["cost_0"] += 1
            elif cost <= 1:
                summary["cost_1"] += 1
            elif cost <= 2:
                summary["cost_2"] += 1
            else:
                summary["cost_3_plus"] += 1
    elif overview:
        for key, count in overview.items():
            bucket = normalized_card_type(key)
            if bucket == "attack":
                summary["attack_count"] += _safe_int(count)
            elif bucket == "skill":
                summary["skill_count"] += _safe_int(count)
            elif bucket == "power":
                summary["power_count"] += _safe_int(count)
            else:
                summary["other_count"] += _safe_int(count)

    counted = sum(summary[key] for key in ("attack_count", "skill_count", "power_count", "curse_count", "status_count", "other_count"))
    summary["deck_size"] = max(summary["deck_size"], counted)
    deck_size = max(summary["deck_size"], 1)
    summary["avg_cost"] = round(sum(costs) / max(len(costs), 1), 3) if costs else 0.0
    summary["attack_ratio"] = round(summary["attack_count"] / deck_size, 4)
    summary["skill_ratio"] = round(summary["skill_count"] / deck_size, 4)
    summary["power_ratio"] = round(summary["power_count"] / deck_size, 4)
    summary["draw_density"] = round(summary["draw_count"] / deck_size, 4)
    summary["block_density"] = round(summary["block_count"] / deck_size, 4)
    summary["damage_density"] = round(summary["damage_count"] / deck_size, 4)
    summary["bloat_score"] = round(max(0, summary["deck_size"] - 22) / 18.0, 4)
    summary["archetype_scores"] = {
        "strength_multihit": round(
            summary["strength_sources"] * 1.4
            + summary["multihit_count"] * 1.2
            + summary["vulnerable_sources"] * 0.5
            + summary["attack_count"] * 0.08,
            4,
        ),
        "barricade_block": round(
            summary["block_payoffs"] * 1.7
            + summary["block_count"] * 0.45
            + summary["skill_count"] * 0.08,
            4,
        ),
        "exhaust_engine": round(
            summary["exhaust_payoffs"] * 1.8
            + summary["exhaust_cards"] * 0.8
            + summary["draw_count"] * 0.35,
            4,
        ),
        "self_damage_rupture": round(summary["self_damage_enablers"] * 1.4 + summary["strength_sources"] * 0.25, 4),
    }
    return summary


def public_deck_summary(summary):
    keys = (
        "deck_summary_version",
        "deck_size",
        "attack_count",
        "skill_count",
        "power_count",
        "draw_count",
        "block_count",
        "scaling_count",
        "strength_sources",
        "multihit_count",
        "block_payoffs",
        "exhaust_cards",
        "exhaust_payoffs",
        "vulnerable_sources",
        "avg_cost",
        "bloat_score",
        "archetype_scores",
    )
    return {key: summary.get(key) for key in keys}
