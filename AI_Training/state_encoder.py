import math


STATE_FEATURES_VERSION = "state_features_v1"


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


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


def build_state_summary(state):
    """Return a small, versioned state summary for option-scoring logs."""
    state = _dict(state)
    player = _dict(state.get("player"))
    run = _dict(state.get("run"))
    battle = _dict(state.get("battle"))
    hp = _safe_float(player.get("hp", state.get("hp")), 0.0)
    max_hp = max(_safe_float(player.get("max_hp", state.get("max_hp")), 1.0), 1.0)
    deck = _list(player.get("deck") or player.get("cards") or state.get("deck"))
    relics = _list(player.get("relics") or state.get("relics"))
    potions = _list(player.get("potions") or state.get("potions"))
    enemies = _list(battle.get("enemies") or state.get("enemies"))
    return {
        "state_features_version": STATE_FEATURES_VERSION,
        "state_type": str(state.get("state_type") or state.get("room_type") or ""),
        "character": str(player.get("character") or state.get("character") or ""),
        "act": _safe_int(run.get("act", state.get("act")), 0),
        "floor": _safe_int(run.get("floor", state.get("floor")), 0),
        "hp_ratio": round(hp / max_hp, 4),
        "gold": _safe_int(player.get("gold", state.get("gold")), 0),
        "deck_size": _safe_int(
            player.get("deck_size", state.get("deck_size", len(deck))),
            len(deck),
        ),
        "relic_count": _safe_int(player.get("relic_count", len(relics)), len(relics)),
        "potion_count": len([p for p in potions if isinstance(p, dict)]),
        "enemy_count": len([e for e in enemies if isinstance(e, dict)]),
    }


def lightweight_state_features(state):
    """Stable tiny feature vector for first-phase shadow diagnostics."""
    summary = build_state_summary(state)
    return [
        summary["act"] / 4.0,
        summary["floor"] / 60.0,
        summary["hp_ratio"],
        summary["gold"] / 500.0,
        summary["deck_size"] / 80.0,
        summary["relic_count"] / 40.0,
        summary["potion_count"] / 5.0,
        summary["enemy_count"] / 5.0,
    ]
