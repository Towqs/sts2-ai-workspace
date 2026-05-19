from __future__ import annotations

import copy
import math
from pathlib import Path

from deck_summary import build_deck_summary, classify_card, public_deck_summary
from options.base import OPTION_FEATURES_VERSION, OPTION_SCHEMA_VERSION, Option, OptionResult, ranked_options
from state_encoder import STATE_FEATURES_VERSION


CARD_OPTION_FEATURES_VERSION = "card_option_features_v1"
CARD_SCORER_VERSION = "ironclad_card_scorer_v1"
SCORER_LOGIC_VERSION = "ironclad_card_scorer_logic_v1_5b"
TEMPLATE_LOGIC_VERSION = "ironclad_template_lock_v1"
SKIP_LOGIC_VERSION = "ironclad_skip_logic_v1_5b"


DEFAULT_TEMPLATE_CONFIG = {
    "default_template": "strength_multihit",
    "option_card_scorer": {"mode": "shadow"},
    "active_canary": {
        "only_when_confidence_gap_gte": 1.0,
        "fallback_to_old_when_gap_lt": 0.3,
        "allow_skip_when_deck_size_gte": 22,
        "allow_skip_when_best_card_score_lte": 0.5,
        "max_active_ratio_per_run": 0.35,
        "max_card_index": 2,
        "early_act_guard_enabled": True,
        "early_act_guard_act": 1,
        "early_act_guard_max_floor": 5,
        "early_act_guard_max_deck_size": 16,
        "early_act_guard_barricade_gap": 1.5,
        "early_act_guard_min_damage_density": 0.22,
    },
    "template_selection": {
        "mode": "locked_after_warmup",
        "warmup_card_rewards": 3,
        "switch_margin": 1.0,
        "switch_patience": 2,
        "min_consistency_target": 0.65,
    },
    "templates": {
        "strength_multihit": {
            "enabled": True,
            "description": "Strength plus multi-hit attacks.",
        },
        "barricade_block": {
            "enabled": True,
            "description": "Block scaling plus Body Slam style payoffs.",
        },
        "exhaust_engine": {
            "enabled": True,
            "description": "Exhaust, draw, and defensive engine payoffs.",
        },
        "self_damage_rupture": {
            "enabled": False,
            "description": "Self-damage growth package; disabled in phase 1.",
        },
    },
    "skip": {
        "soft_deck_size": 20,
        "hard_deck_size": 24,
        "very_large_deck_size": 28,
        "huge_deck_size": 32,
        "low_best_score": 0.8,
        "target_best_score": 1.0,
        "low_archetype_fit": 0.3,
        "low_confidence_gap": 0.2,
    },
}


def default_config_path():
    return Path(__file__).resolve().parents[1] / "configs" / "archetype_templates.yaml"


def _parse_scalar(value):
    text = str(value).strip()
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", "~"):
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text.strip("\"'")


def _deep_merge(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _simple_yaml(path):
    data = {}
    section = None
    current_template = None
    current_nested = None
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return data

    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0:
            current_template = None
            current_nested = None
            if value:
                data[key] = _parse_scalar(value)
                section = None
            else:
                section = key
                data.setdefault(section, {})
            continue
        if section == "option_card_scorer" and indent == 2:
            data.setdefault(section, {})[key] = _parse_scalar(value)
            continue
        if section == "active_canary" and indent == 2:
            data.setdefault(section, {})[key] = _parse_scalar(value)
            continue
        if section == "skip" and indent == 2:
            data.setdefault(section, {})[key] = _parse_scalar(value)
            continue
        if section == "template_selection" and indent == 2:
            data.setdefault(section, {})[key] = _parse_scalar(value)
            continue
        if section == "templates":
            templates = data.setdefault("templates", {})
            if indent == 2 and not value:
                current_template = key
                current_nested = None
                templates.setdefault(current_template, {})
                continue
            if indent == 4 and current_template:
                if value:
                    templates.setdefault(current_template, {})[key] = _parse_scalar(value)
                    current_nested = None
                else:
                    current_nested = key
                    templates.setdefault(current_template, {}).setdefault(current_nested, {})
                continue
            if indent == 6 and current_template and current_nested:
                templates[current_template][current_nested][key] = _parse_scalar(value)
    return data


def load_template_config(path=None):
    config = copy.deepcopy(DEFAULT_TEMPLATE_CONFIG)
    path = Path(path) if path else default_config_path()
    if not path.exists():
        return config
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        loaded = _simple_yaml(path)
    if isinstance(loaded, dict):
        _deep_merge(config, loaded)
    return config


def normalize_card_scorer_mode(value):
    if isinstance(value, dict):
        value = value.get("mode")
    text = str(value or "shadow").strip().lower()
    return text if text in ("off", "shadow", "active", "active_canary", "active_canary_noop") else "shadow"


def enabled_templates(config=None):
    config = config or load_template_config()
    templates = config.get("templates") or {}
    return [
        template_id
        for template_id, item in templates.items()
        if isinstance(item, dict) and bool(item.get("enabled", False))
    ]


def archetype_consistency(deck_summary, config=None):
    config = config or load_template_config()
    allowed = set(enabled_templates(config))
    raw_scores = (deck_summary or {}).get("archetype_scores") or {}
    scores = {
        key: round(float(max(raw_scores.get(key, 0.0), 0.0)), 4)
        for key in allowed
    }
    if not scores:
        return {"selected": "", "score": 0.0, "consistency": 0.0, "scores": {}}
    selected, score = max(scores.items(), key=lambda item: item[1])
    total = sum(scores.values())
    consistency = float(score / total) if total > 1e-9 else 0.0
    return {
        "selected": selected,
        "score": round(score, 4),
        "consistency": round(consistency, 4),
        "scores": scores,
    }


def select_template(deck_summary, config=None, preferred=None):
    config = config or load_template_config()
    allowed = enabled_templates(config)
    if preferred in allowed:
        return preferred
    consistency = archetype_consistency(deck_summary, config)
    selected = consistency.get("selected")
    if selected and consistency.get("score", 0.0) >= 1.0:
        return selected
    default_template = str(config.get("default_template") or "strength_multihit")
    if default_template in allowed:
        return default_template
    return allowed[0] if allowed else default_template


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


def _run_context(state):
    state = state if isinstance(state, dict) else {}
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    hp = _safe_float(player.get("hp"), 0.0)
    max_hp = max(_safe_float(player.get("max_hp"), 1.0), 1.0)
    return {
        "act": _safe_int(run.get("act", state.get("act")), 1),
        "floor": _safe_int(run.get("floor", state.get("floor")), 0),
        "hp_ratio": hp / max_hp,
    }


def _add(score, reasons, amount, reason):
    if abs(amount) > 1e-9:
        score += amount
        reasons.append(reason)
    return score


def _add_component(breakdown, key, amount):
    if abs(amount) > 1e-9:
        breakdown[key] = round(float(breakdown.get(key, 0.0) + amount), 4)


def _add_score(score, reasons, breakdown, key, amount, reason):
    if abs(amount) > 1e-9:
        score += amount
        reasons.append(reason)
        _add_component(breakdown, key, amount)
    return score


def score_card(state, deck_summary, card, template_id=None, config=None):
    """Score one candidate card for the current Ironclad deck context."""
    config = config or load_template_config()
    deck_summary = deck_summary or build_deck_summary(state)
    template_id = template_id or select_template(deck_summary, config)
    tags = classify_card(card)
    ctx = _run_context(state)
    deck_size = max(_safe_int(deck_summary.get("deck_size"), 0), 0)
    block_density = _safe_float(deck_summary.get("block_density"), 0.0)
    block_payoffs = _safe_int(deck_summary.get("block_payoffs"), 0)
    card_id = str(tags.get("card_id") or "")
    self_damage_enabled = bool(((config.get("templates") or {}).get("self_damage_rupture") or {}).get("enabled", False))
    score = 0.0
    reasons = []
    breakdown = {
        "base_power": 0.0,
        "archetype_fit": 0.0,
        "role_need": 0.0,
        "synergy": 0.0,
        "rarity_bonus": 0.0,
        "cost_penalty": 0.0,
        "duplicate_penalty": 0.0,
        "deck_bloat_penalty": 0.0,
    }

    if tags["type"] in ("curse", "status"):
        score = _add_score(score, reasons, breakdown, "base_power", -4.0, "curse/status card")

    if ctx["act"] <= 1 and ctx["floor"] <= 8 and tags["damage"]:
        score = _add_score(score, reasons, breakdown, "role_need", 1.4, "early damage")
    if deck_summary.get("damage_density", 0.0) < 0.32 and tags["damage"]:
        score = _add_score(score, reasons, breakdown, "role_need", 1.0, "fills damage gap")
    if deck_summary.get("block_density", 0.0) < 0.22 and tags["block"]:
        score = _add_score(score, reasons, breakdown, "role_need", 1.1, "fills block gap")
    if deck_size >= 18 and deck_summary.get("draw_count", 0) < 2 and tags["draw"]:
        score = _add_score(score, reasons, breakdown, "role_need", 1.1, "adds draw")
    if (ctx["act"] >= 2 or ctx["floor"] >= 9) and tags["scaling"]:
        score = _add_score(score, reasons, breakdown, "role_need", 0.9, "adds scaling")
    if deck_summary.get("vulnerable_sources", 0) < 1 and tags["vulnerable"]:
        score = _add_score(score, reasons, breakdown, "role_need", 0.7, "adds vulnerable")
    if tags["aoe"] and deck_summary.get("aoe_count", 0) < 1:
        score = _add_score(score, reasons, breakdown, "role_need", 0.7, "adds aoe")

    if template_id == "strength_multihit":
        if tags["strength"]:
            score = _add_score(score, reasons, breakdown, "archetype_fit", 1.9, "strength template payoff")
        if tags["multihit"]:
            score = _add_score(score, reasons, breakdown, "archetype_fit", 1.7, "multi-hit strength payoff")
        if tags["vulnerable"]:
            score = _add_score(score, reasons, breakdown, "synergy", 0.7, "vulnerable burst setup")
    elif template_id == "barricade_block":
        if tags["block_payoff"]:
            score = _add_score(score, reasons, breakdown, "archetype_fit", 2.1, "block payoff")
        if tags["block"]:
            score = _add_score(score, reasons, breakdown, "archetype_fit", 1.2, "block package")
        if tags["damage"] and not tags["block_payoff"] and deck_size >= 16:
            score = _add_score(score, reasons, breakdown, "archetype_fit", -0.5, "off-template attack")
    elif template_id == "exhaust_engine":
        if tags["exhaust_payoff"]:
            score = _add_score(score, reasons, breakdown, "archetype_fit", 2.1, "exhaust payoff")
        if tags["exhaust"]:
            score = _add_score(score, reasons, breakdown, "archetype_fit", 1.3, "exhaust enabler")
        if tags["draw"]:
            score = _add_score(score, reasons, breakdown, "synergy", 0.8, "engine draw")
        if tags["damage"] and not (tags["exhaust"] or tags["premium"]) and deck_size >= 16:
            score = _add_score(score, reasons, breakdown, "archetype_fit", -0.4, "low engine fit")
    elif template_id == "self_damage_rupture":
        if tags["self_damage"]:
            score = _add_score(score, reasons, breakdown, "archetype_fit", 1.0, "self-damage package disabled by default")

    if card_id == "BODY_SLAM" and template_id == "barricade_block":
        has_body_slam_support = block_payoffs > 0 or block_density >= 0.35 or (
            block_density >= 0.22 and deck_summary.get("damage_density", 0.0) < 0.32
        )
        if not has_body_slam_support:
            score = _add_score(score, reasons, breakdown, "archetype_fit", -1.8, "body slam lacks block support")
        elif block_payoffs <= 0 and block_density < 0.35:
            score = _add_score(score, reasons, breakdown, "archetype_fit", -0.35, "body slam speculative")

    if tags["self_damage"] and template_id != "self_damage_rupture" and not self_damage_enabled:
        score = _add_score(score, reasons, breakdown, "archetype_fit", -0.7, "self-damage template disabled")
        if ctx["hp_ratio"] < 0.55:
            score = _add_score(score, reasons, breakdown, "archetype_fit", -0.5, "low hp self-damage risk")

    rarity = str((card or {}).get("rarity") or "").lower()
    if "rare" in rarity:
        score = _add_score(score, reasons, breakdown, "rarity_bonus", 0.25, "rare card")
    elif "uncommon" in rarity:
        score = _add_score(score, reasons, breakdown, "rarity_bonus", 0.1, "uncommon card")

    if tags["cost"] >= 3 and ctx["act"] <= 1 and ctx["floor"] <= 8:
        score = _add_score(score, reasons, breakdown, "cost_penalty", -0.7, "early high cost")
    elif tags["cost"] >= 3:
        score = _add_score(score, reasons, breakdown, "cost_penalty", -0.25, "high cost")
    if deck_size >= 24 and not tags["premium"] and score < 1.2:
        score = _add_score(score, reasons, breakdown, "deck_bloat_penalty", -0.7, "deck bloat pressure")
    if deck_size >= 28 and not tags["premium"] and score < 2.2:
        score = _add_score(score, reasons, breakdown, "deck_bloat_penalty", -0.4, "deck 28+ bloat pressure")
    if deck_size >= 32 and not tags["premium"] and score < 3.0:
        score = _add_score(score, reasons, breakdown, "deck_bloat_penalty", -0.8, "deck 32+ severe bloat")

    return {
        "score": round(float(score), 4),
        "reasons": reasons[:5] or ["neutral fit"],
        "score_breakdown": breakdown,
        "template_id": template_id,
        "tags": {key: value for key, value in tags.items() if isinstance(value, (bool, int, float, str))},
    }


def _candidate_diagnostics(scored_cards):
    scores = []
    archetype_fits = []
    duplicate_like = 0
    low_fit = 0
    for item in scored_cards or []:
        if isinstance(item, dict):
            score = _safe_float(item.get("score"), 0.0)
            breakdown = item.get("score_breakdown") if isinstance(item.get("score_breakdown"), dict) else {}
            tags = item.get("tags") if isinstance(item.get("tags"), dict) else {}
        else:
            score = _safe_float(item, 0.0)
            breakdown = {}
            tags = {}
        scores.append(score)
        fit = _safe_float(breakdown.get("archetype_fit"), 0.0)
        archetype_fits.append(fit)
        if _safe_float(breakdown.get("duplicate_penalty"), 0.0) < 0.0:
            duplicate_like += 1
        if fit < 0.3 and not tags.get("premium"):
            low_fit += 1

    sorted_scores = sorted(scores, reverse=True)
    best = sorted_scores[0] if sorted_scores else 0.0
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    return {
        "best_card_score": round(float(best), 4),
        "second_best_card_score": round(float(second), 4),
        "confidence_gap": round(float(best - second), 4),
        "max_archetype_fit": round(float(max(archetype_fits) if archetype_fits else 0.0), 4),
        "all_candidates_low_fit": bool(scored_cards) and low_fit >= len(scored_cards),
        "all_candidates_duplicates_or_low_fit": bool(scored_cards) and (duplicate_like + low_fit) >= len(scored_cards),
    }


def score_skip(state, deck_summary, card_scores, template_id=None, config=None):
    config = config or load_template_config()
    skip_cfg = config.get("skip") or {}
    ctx = _run_context(state)
    deck_size = _safe_int((deck_summary or {}).get("deck_size"), 0)
    diagnostics = _candidate_diagnostics(card_scores)
    best = diagnostics["best_card_score"]
    confidence_gap = diagnostics["confidence_gap"]
    max_archetype_fit = diagnostics["max_archetype_fit"]
    score = -1.0
    reasons = ["default take bias"]
    breakdown = {
        "base_take_bias": -1.0,
        "deck_bloat_bonus": 0.0,
        "low_value_bonus": 0.0,
        "low_fit_bonus": 0.0,
        "low_confidence_bonus": 0.0,
        "template_coherence_bonus": 0.0,
        "early_deck_penalty": 0.0,
    }
    soft_size = _safe_int(skip_cfg.get("soft_deck_size"), 20)
    hard_size = _safe_int(skip_cfg.get("hard_deck_size"), 24)
    very_large_size = _safe_int(skip_cfg.get("very_large_deck_size"), 28)
    huge_size = _safe_int(skip_cfg.get("huge_deck_size"), 32)
    low_best = _safe_float(skip_cfg.get("low_best_score"), 0.8)
    target_best = _safe_float(skip_cfg.get("target_best_score"), 1.0)
    low_fit_threshold = _safe_float(skip_cfg.get("low_archetype_fit"), 0.3)
    low_gap = _safe_float(skip_cfg.get("low_confidence_gap"), 0.2)

    if deck_size >= soft_size and best < low_best:
        score += 0.5
        _add_component(breakdown, "deck_bloat_bonus", 0.5)
        reasons.append("deck 20+ with weak best card")
    if deck_size >= 22 and best < target_best:
        score += 0.8
        _add_component(breakdown, "deck_bloat_bonus", 0.8)
        reasons.append("deck 22+ with low value offer")
    if deck_size >= hard_size:
        score += 0.8
        _add_component(breakdown, "deck_bloat_bonus", 0.8)
        reasons.append("deck 24+ bloat pressure")
    if deck_size >= very_large_size:
        score += 0.8
        _add_component(breakdown, "deck_bloat_bonus", 0.8)
        reasons.append("deck 28+ strong bloat pressure")
    if deck_size >= huge_size:
        score += 1.2
        _add_component(breakdown, "deck_bloat_bonus", 1.2)
        reasons.append("deck 32+ severe bloat pressure")
    elif deck_size >= soft_size:
        score += 0.25
        _add_component(breakdown, "deck_bloat_bonus", 0.25)
        reasons.append("deck size caution")

    if best < 0.0:
        score += 1.1
        _add_component(breakdown, "low_value_bonus", 1.1)
        reasons.append("all candidates harmful")
    elif best < low_best:
        score += 0.5
        _add_component(breakdown, "low_value_bonus", 0.5)
        reasons.append("low best card score")

    if max_archetype_fit < low_fit_threshold:
        score += 0.6
        _add_component(breakdown, "low_fit_bonus", 0.6)
        reasons.append("low archetype fit")
    if confidence_gap < low_gap and deck_size >= soft_size:
        score += 0.4
        _add_component(breakdown, "low_confidence_bonus", 0.4)
        reasons.append("low confidence in large deck")
    if diagnostics["all_candidates_duplicates_or_low_fit"]:
        score += 0.6
        _add_component(breakdown, "low_fit_bonus", 0.6)
        reasons.append("all candidates duplicate or low-fit")

    consistency = archetype_consistency(deck_summary, config)
    if deck_size >= soft_size and consistency.get("consistency", 0.0) >= 0.65 and best < target_best:
        score += 0.4
        _add_component(breakdown, "template_coherence_bonus", 0.4)
        reasons.append("template already coherent")

    if ctx["act"] <= 1 and ctx["floor"] <= 8 and deck_size < 18:
        score -= 1.3
        _add_component(breakdown, "early_deck_penalty", -1.3)
        reasons.append("early deck still needs cards")

    return {
        "score": round(float(score), 4),
        "reasons": reasons[:5],
        "score_breakdown": breakdown,
        "diagnostics": diagnostics,
        "template_id": template_id or select_template(deck_summary, config),
    }


def confidence_gap_from_options(options):
    scores = sorted([float(option.score) for option in options], reverse=True)
    if len(scores) < 2:
        return 0.0
    return round(scores[0] - scores[1], 4)


def build_card_reward_options(state, mode=None, template_id=None, config=None, deck_summary=None):
    config = config or load_template_config()
    mode = normalize_card_scorer_mode(mode if mode is not None else (config.get("option_card_scorer") or {}).get("mode"))
    deck_summary = deck_summary or build_deck_summary(state)
    template_id = select_template(deck_summary, config, preferred=template_id)
    card_reward = (state or {}).get("card_reward") if isinstance(state, dict) else {}
    cards = (card_reward or {}).get("cards") or []
    options = []
    scored_cards = []
    for fallback_index, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        index = _safe_int(card.get("index"), fallback_index)
        scored = score_card(state, deck_summary, card, template_id=template_id, config=config)
        scored_cards.append(scored)
        options.append(Option(
            label=f"choose_card:index_{index}",
            payload={"action": "select_card_reward", "card_index": index},
            kind="card_reward",
            score=scored["score"],
            reasons=scored["reasons"],
            features=[],
            metadata={
                "card": {
                    "id": card.get("id") or card.get("card_id") or "",
                    "name": card.get("name") or card.get("card_name") or "",
                    "type": card.get("type") or "",
                    "rarity": card.get("rarity") or "",
                    "cost": card.get("cost", 0),
                    "index": index,
                },
                "tags": scored["tags"],
                "score_breakdown": scored["score_breakdown"],
                "template_id": template_id,
                "scorer_version": CARD_SCORER_VERSION,
            },
            index=index,
        ))

    if (card_reward or {}).get("can_skip"):
        skipped = score_skip(state, deck_summary, scored_cards, template_id=template_id, config=config)
        options.append(Option(
            label="skip_reward",
            payload={"action": "skip_card_reward"},
            kind="card_reward",
            score=skipped["score"],
            reasons=skipped["reasons"],
            features=[],
            metadata={
                "template_id": template_id,
                "scorer_version": CARD_SCORER_VERSION,
                "skip": True,
                "score_breakdown": skipped["score_breakdown"],
                "skip_diagnostics": skipped["diagnostics"],
            },
            index=len(options),
        ))

    ranked = ranked_options(options)
    selected = ranked[0] if ranked else None
    return OptionResult(
        options=ranked,
        selected=selected,
        mode=mode,
        template_id=template_id,
        option_schema=OPTION_SCHEMA_VERSION,
        option_features_version=CARD_OPTION_FEATURES_VERSION,
        state_features_version=STATE_FEATURES_VERSION,
        deck_summary=public_deck_summary(deck_summary),
        archetype_consistency=archetype_consistency(deck_summary, config),
        confidence_gap=confidence_gap_from_options(ranked),
        template_lock=dict((state or {}).get("_card_template_lock") or {}),
    )


def card_result_top_actions(result, limit=6):
    if not result:
        return []
    actions = []
    for option in result.options[:limit]:
        actions.append({
            "action": option.label,
            "confidence": round(max(float(option.score), 0.0) * 20.0, 2),
            "marker": f"shadow_score={option.score:+.2f}; {' / '.join(option.reasons) or '-'}",
        })
    return actions


def card_result_log_payload(result, include_options=True):
    if not result:
        return {}
    data = result.to_dict(include_features=False)
    if not include_options:
        data.pop("options", None)
    return data
