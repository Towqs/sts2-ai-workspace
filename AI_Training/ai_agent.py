import os
import json
import math
import random
import time
import uuid
import re
import traceback
from datetime import datetime
import requests
import torch
import torch.nn as nn
import numpy as np
from colorama import init, Fore, Style

from combat_actions import (
    CANDIDATE_FEATURE_DIM,
    choose_candidate_for_card,
    enumerate_combat_actions,
    public_candidate_catalog,
)
from data_pipeline import StateEncoder
from deck_summary import build_deck_summary
from train_candidate_bc import CandidateBCScorer
from macro_data_pipeline import Vocab as MacroVocab, encode_record as encode_macro_record
from train_macro_bc import MacroBCModel
from ppo_policy import PPO_INPUT_DIM, load_ppo_policy
from state_encoder import STATE_FEATURES_VERSION
from options.base import OPTION_FEATURES_VERSION, OPTION_SCHEMA_VERSION
from options.cards import (
    CARD_OPTION_FEATURES_VERSION,
    CARD_SCORER_VERSION,
    archetype_consistency,
    build_card_reward_options,
    card_result_log_payload,
    card_result_top_actions,
    load_template_config,
    normalize_card_scorer_mode,
    select_template,
)

init(autoreset=True)

PORT = 15526
API_URL = f"http://127.0.0.1:{PORT}/api/v1/singleplayer"
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL_PATH = os.path.join(os.path.dirname(__file__), "control_state.json")
AI_LOGIC_PATH = os.path.join(os.path.dirname(__file__), "ai_logic_state.json")
AI_LOG_DIR = os.path.join(WORKSPACE_DIR, "RL_Datasets", "AI_Combat")
PPO_LOG_DIR = os.path.join(WORKSPACE_DIR, "RL_Datasets", "PPO")
OPTION_SHADOW_LOG_DIR = os.path.join(WORKSPACE_DIR, "RL_Datasets", "OptionShadow")
CARD_TEMPLATE_LOCKS = {}

DEFAULT_CONTROL = {
    "ai_enabled": True,
    "macro_enabled": False,
    "macro_shop_enabled": False,
    "macro_card_reward_weight": 0.35,
    "record_ai_actions": True,
    "include_ai_in_training": False,
    "game_speed_enabled": False,
    "game_speed_multiplier": 2.0,
    "ai_min_training_quality": "partial_act1",
    "ai_accept_failed_after_act1": True,
    "ai_require_no_invalid_actions": True,
    "exploration_enabled": False,
    "exploration_mode": "aggressive",
    "self_play_constraint_mode": "explore",
    "combat_exploration_epsilon": 0.35,
    "macro_exploration_epsilon": 0.25,
    "exploration_top_k": 5,
    "exploration_temperature": 1.35,
    "policy_mode": "current_rl",
    "ppo_seed_mode": "fixed",
    "ppo_fixed_seed": "101",
    "option_card_scorer": {"mode": "shadow"},
}

from train_bc import CombatBCModel

def build_model_version(prefix, metadata_path, metadata):
    samples = metadata.get("samples", 0)
    features = metadata.get("features", 0)
    try:
        stamp = datetime.fromtimestamp(os.path.getmtime(metadata_path)).strftime("%Y%m%d%H%M%S")
    except Exception:
        stamp = "unknown"
    return f"{prefix}:s{samples}:f{features}:{stamp}"


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
    metadata["model_version"] = metadata.get("model_version") or build_model_version("bc_combat", metadata_path, metadata)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    weight_input_dim = int(state_dict.get("net.0.weight").shape[1])
    input_dim = int(metadata.get("features") or weight_input_dim)
    if input_dim != weight_input_dim:
        input_dim = weight_input_dim
    vocab_output_dim = len(vocab['actions'])
    weight_output_dim = int(state_dict.get("net.5.weight").shape[0])
    output_dim = vocab_output_dim
    if vocab_output_dim != weight_output_dim:
        output_dim = weight_output_dim
        id_to_action = {idx: action for idx, action in id_to_action.items() if idx < output_dim}
        metadata["action_space_mismatch"] = {
            "vocab_actions": int(vocab_output_dim),
            "checkpoint_actions": int(weight_output_dim),
        }
        print(
            Fore.YELLOW
            + f"[CombatBC] Vocab has {vocab_output_dim} actions but checkpoint has {weight_output_dim}. "
            + "Using checkpoint action space; retrain combat BC to include newer actions."
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CombatBCModel(input_dim, output_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(Fore.CYAN + f"[*] AI Brain Loaded! Action space: {output_dim} | Features: {input_dim} | Device: {device}")
    return encoded, id_to_action, model, device, metadata


def load_candidate_agent(processed_dir):
    rl_model_path = os.path.join(processed_dir, "candidate_rl_model_best.pth")
    rl_metadata_path = os.path.join(processed_dir, "candidate_rl_metadata.json")
    if os.path.exists(rl_model_path):
        model_path = rl_model_path
        metadata_path = rl_metadata_path
        model_prefix = "candidate_rl"
    else:
        model_path = os.path.join(processed_dir, "candidate_bc_model_best.pth")
        metadata_path = os.path.join(processed_dir, "candidate_metadata.json")
        model_prefix = "candidate_bc"
    if not os.path.exists(model_path):
        print(Fore.YELLOW + "[Candidate] Candidate scorer not found. Falling back to legacy combat BC.")
        return None

    metadata = {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        pass

    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        weight_input_dim = int(state_dict.get("net.0.weight").shape[1])
        input_dim = int(metadata.get("features") or weight_input_dim)
        if input_dim != weight_input_dim:
            input_dim = weight_input_dim
        if input_dim <= CANDIDATE_FEATURE_DIM:
            raise ValueError(f"candidate input dim too small: {input_dim}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CandidateBCScorer(input_dim).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        metadata["model_version"] = metadata.get("model_version") or build_model_version(model_prefix, metadata_path, metadata)
        print(Fore.CYAN + f"[*] Candidate Scorer Loaded! Source: {model_prefix} | Features: {input_dim} | Device: {device}")
        return {
            "model": model,
            "device": device,
            "input_dim": input_dim,
            "state_dim": input_dim - CANDIDATE_FEATURE_DIM,
            "metadata": metadata,
            "model_version": metadata["model_version"],
            "model_path": os.path.basename(model_path),
        }
    except Exception as exc:
        print(Fore.YELLOW + f"[Candidate] Candidate scorer failed to load: {exc}. Falling back to legacy combat BC.")
        return None


def align_feature_vector(vector, expected_dim):
    if len(vector) == expected_dim:
        return vector
    if len(vector) > expected_dim:
        return vector[:expected_dim]
    return np.pad(vector, (0, expected_dim - len(vector)), mode="constant")


def combat_text(item):
    parts = []
    for key in ("id", "name", "type", "description", "target_type"):
        value = item.get(key)
        if value is not None:
            parts.append(str(value))
    keywords = item.get("keywords")
    if keywords:
        try:
            parts.append(json.dumps(keywords, ensure_ascii=False))
        except Exception:
            parts.append(str(keywords))
    return " ".join(parts).lower()


def combat_core_text(item):
    parts = []
    for key in ("id", "name", "type", "description", "target_type"):
        value = item.get(key)
        if value is not None:
            parts.append(str(value))
    return " ".join(parts).lower()


def combat_numbers(text):
    import re
    return [int(n) for n in re.findall(r"\d+", str(text or ""))]


def combat_number_hint(item, mode):
    text = combat_core_text(item)
    if mode == "attack" and not any(k in text for k in ("deal", "attack", "造成", "攻击")):
        return 0
    if mode == "block" and not any(k in text for k in ("block", "defend", "armor", "格挡", "护甲", "防御")):
        return 0
    nums = combat_numbers(text)
    return nums[0] if nums else 0


def card_id_key(card):
    return str(card.get("id") or card.get("card_id") or "").upper()


def card_play_cost(card, energy=0.0):
    cost = card.get("cost_for_turn", card.get("cost", 0))
    if str(cost).upper() == "X":
        return max(float(energy or 0.0), 0.0)
    try:
        return max(float(cost), 0.0)
    except (TypeError, ValueError):
        return 99.0


def known_card_effect_hints(card):
    card_id = card_id_key(card)
    text = combat_text(card)
    damage = 0.0
    block = 0.0

    known_damage = {
        "STRIKE": 6.0,
        "STRIKE_IRONCLAD": 6.0,
        "BASH": 8.0,
        "ANGER": 6.0,
        "HEADBUTT": 9.0,
        "THUNDERCLAP": 4.0,
        "SWORD_BOOMERANG": 9.0,
        "FORGOTTEN_RITUAL": 10.0,
        "FIGHT_ME": 10.0,
        "CINDER": 18.0,
        "SETUP_STRIKE": 6.0,
    }
    known_block = {
        "DEFEND": 5.0,
        "DEFEND_IRONCLAD": 5.0,
    }
    if card_id in known_damage:
        damage = known_damage[card_id]
    elif "attack" in text or "攻击" in text:
        damage = 6.0
    if card_id in known_block:
        block = known_block[card_id]
    elif "defend" in text or "防御" in text or "block" in text or "格挡" in text:
        block = 5.0
    return damage, block


def known_card_utility_hints(card):
    card_id = card_id_key(card)
    text = combat_text(card)
    scaling_ids = {"FIGHT_ME", "SETUP_STRIKE", "INFLAME", "DEMON_FORM"}
    setup_attack_ids = {"BASH", "THUNDERCLAP", "FIGHT_ME", "SETUP_STRIKE", "FORGOTTEN_RITUAL"}
    energy_ids = {"BLOODLETTING", "FORGOTTEN_RITUAL"}
    power_ids = {"DRUM_OF_BATTLE"}
    setup_attack_terms = (
        "vulnerable", "weak", "strength", "dexterity", "draw", "energy", "apply", "gain",
        "易伤", "虚弱", "力量", "敏捷", "抽", "能量", "施加", "获得",
    )
    return {
        "scaling": card_id in scaling_ids or has_any(text, ("strength", "dexterity", "力量", "敏捷")),
        "energy": card_id in energy_ids or has_any(text, ("energy", "能量", "ironclad_energy")),
        "exhaust_synergy": card_id == "FORGOTTEN_RITUAL" or has_any(text, ("本回合消耗", "if you exhausted", "if you exhaust")),
        "power": card_id in power_ids or "power" in str(card.get("type") or "").lower() or "能力" in str(card.get("type") or ""),
        "setup_attack": card_id in setup_attack_ids or has_any(text, setup_attack_terms),
    }


def card_end_turn_hand_damage(card):
    text = combat_text(card)
    card_id = card_id_key(card)
    if card_id == "BECKON":
        return 6.0

    has_end_turn = "end of your turn" in text or "\u56de\u5408\u7ed3\u675f" in text
    has_hand_clause = "in your hand" in text or "\u624b\u724c" in text
    has_life_loss = (
        "lose" in text
        or "damage" in text
        or "\u5931\u53bb" in text
        or "\u751f\u547d" in text
    )
    if has_end_turn and has_hand_clause and has_life_loss:
        nums = combat_numbers(text)
        return max(nums) if nums else 0.0
    return 0.0


def hand_end_turn_damage(hand):
    return sum(
        card_end_turn_hand_damage(card)
        for card in hand or []
        if isinstance(card, dict)
    )


def card_self_damage(card):
    if card_end_turn_hand_damage(card) > 0:
        return 0.0
    text = combat_text(card)
    card_key = f"{card.get('id') or ''} {card.get('name') or ''} {text}".lower()
    known_self_damage = {
        "bloodletting": 3.0,
        "放血": 3.0,
        "offering": 6.0,
        "祭品": 6.0,
        "hemokinesis": 2.0,
        "御血术": 2.0,
        "combust": 1.0,
        "燃烧": 1.0,
        "brutality": 1.0,
        "残暴": 1.0,
        "残虐": 1.0,
        "jax": 3.0,
        "j.a.x": 3.0,
        "neows_fury": 7.0,
        "涅奥之怒": 7.0,
    }
    for key, value in known_self_damage.items():
        if key in card_key:
            return value
    import re
    patterns = (
        r"失去\s*(\d+)\s*点?生命",
        r"lose\s*(\d+)\s*(?:hp|health|life)",
        r"take\s*(\d+)\s*damage",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return safe_num(match.group(1), 0.0)
    return 0.0


def enemy_incoming_damage(enemies):
    total = 0.0
    for enemy in enemies or []:
        for intent in enemy.get("intents", []) or []:
            if isinstance(intent, dict):
                text = " ".join(str(intent.get(k) or "") for k in ("label", "description", "type", "title"))
            else:
                text = str(intent or "")
            lowered = text.lower()
            if not any(k in lowered for k in ("attack", "damage", "攻击", "伤害", "攻势")):
                continue
            nums = combat_numbers(text)
            if ("x" in text.lower() or "×" in text or "脳" in text) and len(nums) >= 2:
                total += nums[0] * nums[1]
            elif nums:
                total += nums[0]
            else:
                total += 6.0
    return total


def enemy_blob(enemy):
    parts = []
    for key in ("id", "entity_id", "name", "description"):
        value = enemy.get(key)
        if value is not None:
            parts.append(str(value))
    for key in ("status", "statuses", "powers", "buffs", "debuffs"):
        values = enemy.get(key) or []
        if isinstance(values, dict):
            values = [values]
        for value in values:
            if isinstance(value, dict):
                for subkey in ("id", "name", "type", "label", "title", "description"):
                    subvalue = value.get(subkey)
                    if subvalue is not None:
                        parts.append(str(subvalue))
            else:
                parts.append(str(value))
    try:
        parts.append(json.dumps(enemy.get("intents") or [], ensure_ascii=False))
    except Exception:
        parts.append(str(enemy.get("intents") or ""))
    return " ".join(parts).lower()


def enemies_damage_ineffective(enemies):
    immune_terms = (
        "intangible", "无形", "虚无", "免疫伤害", "不受伤害", "不会受到伤害",
        "不会受到任何", "本回合不会受到", "任何伤害",
        "take no damage", "cannot take damage", "no damage",
    )
    return any(has_any(enemy_blob(enemy), immune_terms) for enemy in enemies or [])


def card_for_candidate(candidate, state):
    player = (state or {}).get("player") or {}
    hand = player.get("hand") or []
    for card in hand:
        if isinstance(card, dict) and safe_int(card.get("index"), -9999) == candidate.card_index:
            return card
    if 0 <= candidate.card_index < len(hand) and isinstance(hand[candidate.card_index], dict):
        card = hand[candidate.card_index]
        if str(card.get("id") or card.get("name") or "") == candidate.card_id:
            return card
    for card in hand:
        if isinstance(card, dict) and str(card.get("id") or card.get("name") or "") == candidate.card_id:
            return card
    return {}


def potion_for_candidate(candidate, state):
    player = (state or {}).get("player") or {}
    potions = player.get("potions") or []
    for potion in potions:
        if isinstance(potion, dict) and safe_int(potion.get("slot"), -1) == candidate.potion_slot:
            return potion
    return {}


def potion_id_key(potion):
    return str(potion.get("id") or potion.get("potion_id") or "").upper()


def card_effect_profile(card):
    text = combat_text(card)
    card_id = card_id_key(card)
    card_type = str(card.get("type") or "").lower()
    damage = combat_number_hint(card, "attack")
    block_gain = combat_number_hint(card, "block")
    known_damage, known_block = known_card_effect_hints(card)
    if damage <= 0:
        damage = known_damage
    if block_gain <= 0:
        block_gain = known_block
    utility_hints = known_card_utility_hints(card)
    self_damage = card_self_damage(card)
    end_turn_hand_damage = card_end_turn_hand_damage(card)
    status_like_ids = {
        "SLIMED",
        "WOUND",
        "DAZED",
        "BURN",
        "VOID",
        "PARASITE",
    }
    status_like = (
        card_id in status_like_ids
        or "status" in card_type
        or "状态" in card_type
        or "黏液" in text
    )
    utility_terms = (
        "draw", "抽", "vulnerable", "易伤", "weak", "虚弱", "strength", "力量",
        "dexterity", "敏捷", "energy", "能量", "exhaust", "消耗", "upgrade", "升级",
        "apply", "给予",
    )
    is_attack = "attack" in card_type or "攻击" in card_type or damage > 0
    return {
        "text": text,
        "card_id": card_id,
        "card_type": card_type,
        "cost": card_play_cost(card),
        "is_attack": is_attack,
        "damage": damage,
        "block_gain": block_gain,
        "self_damage": self_damage,
        "end_turn_hand_damage": end_turn_hand_damage,
        "status_like": status_like,
        "has_utility": any(term in text for term in utility_terms) or any(utility_hints.values()),
        "scaling": utility_hints["scaling"],
        "energy": utility_hints["energy"],
        "exhaust_synergy": utility_hints["exhaust_synergy"],
        "power_like": utility_hints["power"] or "power" in card_type,
        "setup_attack": bool(
            is_attack
            and (
                utility_hints["setup_attack"]
                or utility_hints["scaling"]
                or utility_hints["energy"]
                or utility_hints["exhaust_synergy"]
            )
        ),
    }


def status_amount(entries, names):
    wanted = {str(name).lower() for name in names}
    total = 0.0
    for item in entries or []:
        if not isinstance(item, dict):
            continue
        fields = (
            item.get("id"),
            item.get("name"),
            item.get("type"),
            item.get("label"),
            item.get("title"),
        )
        text = " ".join(str(value or "").lower() for value in fields)
        if any(name in text for name in wanted):
            total += safe_num(item.get("amount", item.get("stack", item.get("stacks", 1))), 1.0)
    return total


def card_gain_strength(card):
    text = combat_text(card)
    card_id = card_id_key(card)
    known = {
        "INFLAME": 2.0,
        "DEMON_FORM": 2.0,
        "FLEX_POTION": 5.0,
    }
    if card_id in known:
        return known[card_id]
    if not has_any(text, ("strength", "\u529b\u91cf")):
        return 0.0
    nums = combat_numbers(text)
    return max(nums) if nums else 1.0


def card_gain_energy(card):
    text = combat_text(card)
    card_id = card_id_key(card)
    known = {
        "BLOODLETTING": 2.0,
        "FORGOTTEN_RITUAL": 1.0,
    }
    if card_id in known:
        return known[card_id]
    if not has_any(text, ("energy", "\u80fd\u91cf")):
        return 0.0
    nums = combat_numbers(text)
    if has_any(text, ("gain", "\u83b7\u5f97", "\u56de\u590d")):
        return max(nums) if nums else 1.0
    return 0.0


def card_applies_vulnerable(card):
    text = combat_text(card)
    return has_any(text, ("vulnerable", "\u6613\u4f24"))


def card_discount_sensitive(card):
    text = combat_text(card)
    card_id = card_id_key(card)
    if card_id in {"DROPKICK", "BODY_SLAM"}:
        return True
    has_cost = has_any(text, ("cost", "free", "\u8d39\u7528", "\u82b1\u8d39", "\u514d\u8d39"))
    has_play_clause = has_any(text, (
        "card played", "cards played", "play a card", "played this turn",
        "\u6253\u51fa", "\u6253\u51fa\u724c", "\u672c\u56de\u5408", "\u964d\u4f4e", "\u51cf\u5c11",
    ))
    return bool(has_cost and has_play_clause)


def card_trigger_on_block(card):
    text = combat_text(card)
    return has_any(text, (
        "when you gain block", "whenever you gain block", "gain block",
        "\u83b7\u5f97\u683c\u6321\u65f6", "\u5f53\u4f60\u83b7\u5f97\u683c\u6321", "\u83b7\u5f97\u683c\u6321",
    )) and card_effect_profile(card)["damage"] > 0


def card_sequence_cost(card, energy, played_count):
    cost = card_play_cost(card, energy)
    if card_discount_sensitive(card):
        cost = max(0.0, cost - played_count)
    return cost


def enemy_vulnerable_multiplier(enemies):
    for enemy in enemies or []:
        statuses = []
        for key in ("status", "statuses", "powers", "buffs", "debuffs"):
            value = enemy.get(key) if isinstance(enemy, dict) else None
            if isinstance(value, list):
                statuses.extend(value)
            elif isinstance(value, dict):
                statuses.append(value)
        if status_amount(statuses, ("vulnerable", "\u6613\u4f24")) > 0:
            return 1.5
    return 1.0


def estimate_card_damage(card, strength=0.0, vulnerable_mult=1.0):
    profile = card_effect_profile(card)
    if not profile["is_attack"] or profile["damage"] <= 0:
        return 0.0
    base = max(0.0, profile["damage"] + strength)
    return math.floor(base * max(vulnerable_mult, 1.0))


def hand_burst_plan(state, max_depth=6):
    player = (state or {}).get("player") or {}
    battle = (state or {}).get("battle") or {}
    hand = [
        dict(card)
        for card in (player.get("hand") or [])
        if isinstance(card, dict) and card.get("can_play", False)
    ]
    if not hand:
        return {
            "max_damage": 0.0,
            "sequence": [],
            "first_card_index": None,
            "first_card_id": "",
            "lethal": False,
            "target_effective_hp": 0.0,
            "reason": "no_playable_cards",
        }
    enemies = [enemy for enemy in (battle.get("enemies") or []) if isinstance(enemy, dict) and safe_num(enemy.get("hp"), 0.0) > 0]
    target_effective_hp = min((safe_num(e.get("hp"), 0.0) + safe_num(e.get("block"), 0.0) for e in enemies), default=0.0)
    player_status = []
    for key in ("status", "statuses", "powers", "buffs"):
        value = player.get(key)
        if isinstance(value, list):
            player_status.extend(value)
        elif isinstance(value, dict):
            player_status.append(value)
    base_strength = status_amount(player_status, ("strength", "\u529b\u91cf"))
    base_vulnerable = enemy_vulnerable_multiplier(enemies)
    start_energy = safe_num(player.get("energy"), 0.0)
    best = {
        "damage": 0.0,
        "sequence": [],
        "first_card_index": None,
        "first_card_id": "",
        "lethal": False,
        "target_effective_hp": target_effective_hp,
        "reason": "search",
    }

    def search(cards, energy, strength, vulnerable_mult, played_count, damage, sequence):
        nonlocal best
        if damage > best["damage"]:
            first = sequence[0] if sequence else {}
            best = {
                "damage": damage,
                "sequence": list(sequence),
                "first_card_index": first.get("index"),
                "first_card_id": first.get("id") or first.get("name") or "",
                "lethal": bool(target_effective_hp > 0 and damage >= target_effective_hp),
                "target_effective_hp": target_effective_hp,
                "reason": "burst_search",
            }
        if len(sequence) >= max_depth or not cards:
            return

        playable = []
        for pos, card in enumerate(cards):
            if card_sequence_cost(card, energy, played_count) <= energy:
                playable.append((pos, card))
        playable.sort(
            key=lambda row: (
                estimate_card_damage(row[1], strength, vulnerable_mult),
                card_effect_profile(row[1])["setup_attack"],
                -card_sequence_cost(row[1], energy, played_count),
            ),
            reverse=True,
        )
        for pos, card in playable[:8]:
            cost = card_sequence_cost(card, energy, played_count)
            profile = card_effect_profile(card)
            next_cards = cards[:pos] + cards[pos + 1:]
            next_energy = max(0.0, energy - cost) + card_gain_energy(card)
            next_strength = strength + card_gain_strength(card)
            next_vulnerable = 1.5 if card_applies_vulnerable(card) else vulnerable_mult
            dealt = estimate_card_damage(card, strength, vulnerable_mult)
            if profile["block_gain"] > 0:
                for other in next_cards:
                    if card_trigger_on_block(other):
                        dealt += estimate_card_damage(other, strength, vulnerable_mult)
            entry = {
                "index": safe_int(card.get("index"), pos),
                "id": card.get("id") or card.get("name") or "",
                "cost": cost,
                "damage": round(dealt, 2),
            }
            search(
                next_cards,
                next_energy,
                next_strength,
                next_vulnerable,
                played_count + 1,
                damage + dealt,
                sequence + [entry],
            )

    search(hand, start_energy, base_strength, base_vulnerable, 0, 0.0, [])
    return {
        "max_damage": round(best["damage"], 2),
        "sequence": best["sequence"],
        "first_card_index": best["first_card_index"],
        "first_card_id": best["first_card_id"],
        "lethal": best["lethal"],
        "target_effective_hp": round(best["target_effective_hp"], 2),
        "reason": best["reason"],
    }


def burst_sequence_label(plan, limit=3):
    sequence = (plan or {}).get("sequence") or []
    names = [str(item.get("id") or item.get("index") or "?") for item in sequence[:limit] if isinstance(item, dict)]
    if not names:
        return ""
    suffix = "..." if len(sequence) > limit else ""
    return ">".join(names) + suffix


def burst_candidate_adjustment(candidate, burst_plan):
    if not burst_plan or burst_plan.get("max_damage", 0.0) <= 0:
        return 0.0, ""
    max_damage = safe_num(burst_plan.get("max_damage"), 0.0)
    first_index = burst_plan.get("first_card_index")
    target_hp = safe_num(burst_plan.get("target_effective_hp"), 0.0)
    lethal = bool(burst_plan.get("lethal"))
    sequence = burst_sequence_label(burst_plan)
    marker_parts = []
    adjustment = 0.0

    if candidate.kind == "play_card" and first_index is not None:
        if safe_int(candidate.card_index, -1) == safe_int(first_index, -2):
            bonus = min(26.0, 6.0 + max_damage * 0.35)
            if lethal:
                bonus += 24.0
                marker_parts.append("burst_lethal")
            else:
                marker_parts.append("burst_first")
            adjustment += bonus
            marker_parts.append(f"dmg={max_damage:g}")
            if target_hp > 0:
                marker_parts.append(f"hp={target_hp:g}")
            if sequence:
                marker_parts.append(f"seq={sequence}")
        elif lethal:
            adjustment -= 10.0
            marker_parts.append("not_burst_lethal_first")
    elif candidate.kind == "end_turn":
        if lethal:
            adjustment -= 80.0
            marker_parts.append("end_turn_skips_lethal_burst")
        elif max_damage >= 10.0:
            adjustment -= min(28.0, 6.0 + max_damage * 0.6)
            marker_parts.append(f"end_turn_skips_burst={max_damage:g}")

    return adjustment, ",".join(marker_parts)


def playable_setup_attack_exists(player, energy, exclude_index=None):
    for hand_index, hand_card in enumerate(player.get("hand", []) or []):
        if not isinstance(hand_card, dict) or not hand_card.get("can_play", False):
            continue
        card_index = safe_int(hand_card.get("index"), hand_index)
        if exclude_index is not None and card_index == exclude_index:
            continue
        if card_play_cost(hand_card, energy) > energy:
            continue
        if card_effect_profile(hand_card)["setup_attack"]:
            return True
    return False


def potion_damage_hint(potion):
    known_damage = {
        "FIRE_POTION": 20.0,
        "EXPLOSIVE_AMPOULE": 10.0,
        "POWDERED_DEMISE": 10.0,
    }
    potion_id = potion_id_key(potion)
    if potion_id in known_damage:
        return known_damage[potion_id]
    text = combat_core_text(potion)
    if not any(k in text for k in ("deal", "attack", "fire", "造成", "攻击", "火焰")):
        return 0
    nums = combat_numbers(text)
    if not nums:
        return 0
    return max(nums)


def potion_block_hint(potion, current_block=0.0):
    potion_id = potion_id_key(potion)
    if potion_id == "BLOCK_POTION":
        return 12.0
    if potion_id == "FORTIFIER":
        return max(0.0, safe_num(current_block, 0.0) * 2.0)
    text = combat_core_text(potion)
    if not any(k in text for k in ("block", "defend", "armor", "\u683c\u6321", "\u9632\u5fa1", "\u62a4\u7532")):
        return 0.0
    nums = combat_numbers(text)
    return max(nums) if nums else 0.0


def potion_heal_hint(potion, max_hp=1.0):
    potion_id = potion_id_key(potion)
    if potion_id == "BLOOD_POTION":
        return max(1.0, max_hp * 0.20)
    text = combat_core_text(potion)
    if not any(k in text for k in ("heal", "recover", "\u56de\u590d", "\u6cbb\u7597")):
        return 0.0
    nums = combat_numbers(text)
    return max(nums) if nums else 0.0


def potion_tactical_profile(candidate, state):
    potion = potion_for_candidate(candidate, state)
    text = combat_text(potion)
    potion_id = potion_id_key(potion)
    player = (state or {}).get("player") or {}
    battle = (state or {}).get("battle") or {}
    state_type = str((state or {}).get("state_type") or "").lower()
    enemies = battle.get("enemies") or []
    hp = safe_num(player.get("hp"), 0.0)
    max_hp = max(safe_num(player.get("max_hp"), 1.0), 1.0)
    hp_ratio = hp / max_hp
    incoming = enemy_incoming_damage(enemies)
    block = safe_num(player.get("block"), 0.0)
    net_incoming = max(0.0, incoming - block)
    round_no = safe_int(battle.get("round"), 0)
    damage = potion_damage_hint(potion)
    block_gain = potion_block_hint(potion, block)
    heal_gain = potion_heal_hint(potion, max_hp)
    defensive = any(k in text for k in ("block", "格挡", "weak", "虚弱"))
    scaling = any(k in text for k in ("dexterity", "敏捷", "strength", "力量"))
    defensive_ids = {"BLOCK_POTION", "FORTIFIER", "WEAK_POTION"}
    scaling_ids = {"DEXTERITY_POTION", "STRENGTH_POTION", "FLEX_POTION", "POWER_POTION"}
    resource_ids = {"ENERGY_POTION", "ATTACK_POTION", "SKILL_POTION", "COLORLESS_POTION", "CLARITY"}
    healing_ids = {"BLOOD_POTION"}
    defensive = potion_id in defensive_ids or any(k in text for k in ("block", "weak", "\u683c\u6321", "\u865a\u5f31"))
    scaling = potion_id in scaling_ids or any(k in text for k in ("dexterity", "strength", "\u654f\u6377", "\u529b\u91cf"))
    resource = potion_id in resource_ids or any(k in text for k in ("energy", "free", "card", "\u80fd\u91cf", "\u514d\u8d39", "\u624b\u724c"))
    healing = potion_id in healing_ids or heal_gain > 0
    lethal = bool(damage > 0 and candidate.target_effective_hp and damage >= candidate.target_effective_hp)
    prevents_lethal = bool(
        hp > 0
        and net_incoming >= hp
        and (
            (block_gain > 0 and max(0.0, net_incoming - block_gain) < hp)
            or (heal_gain > 0 and net_incoming < hp + heal_gain)
        )
    )
    severe_threat = net_incoming >= max(10.0, hp * 0.45)
    boss_or_elite = state_type in ("boss", "elite")
    boss_fight = state_type == "boss"
    defensive_useful = bool(defensive and incoming > 0 and (prevents_lethal or severe_threat))
    healing_useful = bool(healing and (prevents_lethal or (hp_ratio <= (0.55 if boss_fight else 0.30) and (boss_or_elite or incoming > 0))))
    resource_useful = bool(resource and (boss_fight or prevents_lethal or (boss_or_elite and hp_ratio <= 0.30 and round_no >= 3 and severe_threat)))
    scaling_useful = bool(scaling and boss_or_elite and round_no <= (4 if boss_fight else 2) and hp_ratio >= 0.35)
    damage_useful = bool(damage > 0 and boss_fight)
    urgent = prevents_lethal or severe_threat
    useful = lethal or damage_useful or defensive_useful or healing_useful or resource_useful or scaling_useful
    return {
        "potion_id": potion_id,
        "text": text,
        "damage": damage,
        "block_gain": block_gain,
        "heal_gain": heal_gain,
        "defensive": defensive,
        "scaling": scaling,
        "resource": resource,
        "healing": healing,
        "lethal": lethal,
        "damage_useful": damage_useful,
        "boss_fight": boss_fight,
        "prevents_lethal": prevents_lethal,
        "severe_threat": severe_threat,
        "defensive_useful": defensive_useful,
        "healing_useful": healing_useful,
        "resource_useful": resource_useful,
        "scaling_useful": scaling_useful,
        "urgent": urgent,
        "useful": useful,
        "incoming": incoming,
        "net_incoming": net_incoming,
        "hp_ratio": hp_ratio,
        "round_no": round_no,
    }


def candidate_should_play_before_end_turn(candidate, state):
    if candidate.kind == "play_card":
        card = card_for_candidate(candidate, state)
        profile = card_effect_profile(card)
        player = (state or {}).get("player") or {}
        battle = (state or {}).get("battle") or {}
        hp = safe_num(((state or {}).get("player") or {}).get("hp"), 0.0)
        max_hp = max(safe_num(((state or {}).get("player") or {}).get("max_hp"), 1.0), 1.0)
        hp_ratio = hp / max_hp
        incoming = enemy_incoming_damage(battle.get("enemies") or [])
        block = safe_num(player.get("block"), 0.0)
        net_incoming = max(0.0, incoming - block)
        if profile["end_turn_hand_damage"] > 0:
            return True
        if profile["self_damage"] > 0 and (hp <= profile["self_damage"] + 1 or hp_ratio <= 0.35):
            return False
        if profile["status_like"] and profile["damage"] <= 0 and profile["block_gain"] <= 0:
            return False
        if profile["damage"] > 0:
            return True
        if profile["block_gain"] > 0 and net_incoming > 0:
            return True
        if "power" in profile["card_type"]:
            return True
        return bool(profile["has_utility"] and not profile["status_like"])
    if candidate.kind == "use_potion":
        return potion_tactical_profile(candidate, state)["useful"]
    return False


def should_force_delay_end_turn(candidates, state):
    if not state:
        return False, ""
    useful = [candidate for candidate in candidates if candidate_should_play_before_end_turn(candidate, state)]
    if not useful:
        return False, ""
    player = state.get("player") or {}
    energy = safe_num(player.get("energy"), 0.0)
    battle = state.get("battle") or {}
    incoming = enemy_incoming_damage(battle.get("enemies") or [])
    if incoming > 0:
        return True, "useful_action_available_under_threat"
    if any(c.kind == "play_card" for c in useful) and energy > 0:
        return True, "useful_card_available"
    if any(c.kind == "use_potion" for c in useful):
        return True, "useful_potion_available"
    return False, ""


def tactical_candidate_adjustment(candidate, state):
    if not state:
        return 0.0, ""
    player = state.get("player") or {}
    battle = state.get("battle") or {}
    enemies = battle.get("enemies") or []
    hp = safe_num(player.get("hp"), 0.0)
    max_hp = max(safe_num(player.get("max_hp"), 1.0), 1.0)
    hp_ratio = hp / max_hp
    block = safe_num(player.get("block"), 0.0)
    energy = safe_num(player.get("energy"), 0.0)
    incoming = enemy_incoming_damage(enemies)
    net_incoming = max(0.0, incoming - block)
    damage_ineffective = enemies_damage_ineffective(enemies)
    adjustment = 0.0
    reasons = []

    if candidate.kind == "play_card":
        card = card_for_candidate(candidate, state)
        profile = card_effect_profile(card)
        damage = profile["damage"]
        block_gain = profile["block_gain"]
        self_damage = profile["self_damage"]
        end_turn_hand_damage = profile["end_turn_hand_damage"]
        card_type = profile["card_type"]
        exhaust_count = safe_num(player.get("exhaust_pile_count") or player.get("exhaust_count"), 0.0)
        playable_attack_count = sum(
            1
            for hand_card in player.get("hand", []) or []
            if isinstance(hand_card, dict)
            and hand_card.get("can_play", False)
            and card_effect_profile(hand_card)["is_attack"]
        )
        card_cost = card_play_cost(card, energy)
        is_attack = bool(profile["is_attack"])
        setup_attack = bool(profile["setup_attack"])
        has_setup_attack_waiting = playable_setup_attack_exists(player, energy, exclude_index=candidate.card_index)
        stable_or_free = net_incoming < hp or card_cost <= 0 or block_gain > 0
        candidate_lethal = bool(damage > 0 and candidate.target_effective_hp and damage >= candidate.target_effective_hp)

        if end_turn_hand_damage > 0:
            adjustment += 12.0
            reasons.append(f"clear_end_turn_hand_damage={end_turn_hand_damage:g}")
            if net_incoming + hand_end_turn_damage(player.get("hand")) >= hp and hp > 0:
                adjustment += 18.0
                reasons.append("prevents_end_turn_lethal")

        if profile["status_like"] and damage <= 0 and block_gain <= 0 and end_turn_hand_damage <= 0:
            adjustment -= 8.0
            reasons.append("status_card_penalty")

        if self_damage > 0:
            if hp <= self_damage + 1 or hp_ratio <= 0.35:
                return -100.0, f"blocked_self_damage hp={hp:g} cost={self_damage:g}"
            adjustment -= 2.0
            reasons.append(f"self_damage={self_damage:g}")

        if is_attack and damage > 0 and card_cost <= 0 and not profile["status_like"]:
            adjustment += 10.0
            reasons.append("zero_cost_attack")
            if profile["has_utility"]:
                adjustment += 3.0
                reasons.append("zero_cost_utility_attack")

        if setup_attack and playable_attack_count >= 2 and stable_or_free:
            adjustment += 12.0
            reasons.append("setup_attack_before_plain")
        elif (
            is_attack
            and damage > 0
            and not setup_attack
            and has_setup_attack_waiting
            and stable_or_free
            and not candidate_lethal
        ):
            adjustment -= 6.0
            reasons.append("setup_attack_available")

        if incoming <= 0:
            if damage > 0 or "attack" in card_type:
                adjustment += 6.0
                reasons.append("no_incoming_attack")
            if block_gain > 0 and damage <= 0:
                adjustment -= 10.0
                reasons.append("no_incoming_block_penalty")
            if profile["power_like"] or profile["scaling"]:
                adjustment += 5.0
                reasons.append("safe_turn_setup")
        else:
            if block_gain > 0 and damage <= 0 and net_incoming <= 0:
                adjustment -= 8.0
                reasons.append("already_blocked")
            elif block_gain > 0 and damage <= 0 and block + block_gain > incoming + 6 and not profile["has_utility"]:
                adjustment -= 4.0
                reasons.append("overblock_penalty")
            if net_incoming >= hp and hp > 0:
                if block_gain > 0:
                    adjustment += 12.0
                    reasons.append("lethal_block")
                elif damage > 0 and not (candidate.target_effective_hp and damage >= candidate.target_effective_hp):
                    adjustment -= 6.0
                    reasons.append("nonlethal_attack_under_lethal")
            if net_incoming >= max(8.0, hp * 0.35) and block_gain > 0:
                adjustment += 5.0
                reasons.append("high_threat_block")
            if hp_ratio <= 0.35 and block_gain > 0:
                adjustment += 3.0
                reasons.append("low_hp_block")

        if damage_ineffective:
            if profile["power_like"] or profile["scaling"] or profile["energy"]:
                adjustment += 16.0
                reasons.append("damage_immune_setup")
            elif damage > 0 and block_gain <= 0 and not profile["has_utility"]:
                adjustment -= 16.0
                reasons.append("damage_immune_attack")

        if profile["power_like"] and (safe_int(battle.get("round"), 0) in (0, 1, 2) or damage_ineffective or incoming <= 0):
            adjustment += 4.0
            reasons.append("early_power")
        if profile["scaling"] and (playable_attack_count >= 2 or damage_ineffective or incoming <= 0):
            adjustment += 4.0
            reasons.append("scaling_combo")
        if profile["exhaust_synergy"]:
            if exhaust_count > 0:
                adjustment += 7.0
                reasons.append("exhaust_synergy_ready")
            else:
                adjustment -= 4.0
                reasons.append("exhaust_synergy_not_ready")

        if damage > 0 and candidate.target_effective_hp and damage >= candidate.target_effective_hp:
            adjustment += 8.0
            reasons.append("lethal")

    elif candidate.kind == "use_potion":
        profile = potion_tactical_profile(candidate, state)
        already_used_potion = bool(state.get("_potion_used_this_turn"))
        boss_fight = bool(profile.get("boss_fight"))
        if already_used_potion and not boss_fight and not (profile["lethal"] or profile["prevents_lethal"]):
            return -1000.0, "one_potion_per_turn_hard"
        if profile["lethal"]:
            adjustment += 14.0
            reasons.append("lethal_potion")
        elif boss_fight and profile["damage"] > 0:
            adjustment += 8.0
            reasons.append("boss_damage_potion")
        elif profile["prevents_lethal"]:
            adjustment += 12.0
            reasons.append("prevents_lethal_potion")
        elif profile["defensive_useful"]:
            adjustment += 5.0
            reasons.append("severe_threat_defensive_potion")
        elif profile["healing_useful"]:
            adjustment += 4.0
            reasons.append("critical_heal_potion")
        elif profile["resource_useful"]:
            adjustment += 3.0
            reasons.append("desperate_resource_potion")
        elif profile["scaling_useful"]:
            adjustment += 5.0 if boss_fight else 2.0
            reasons.append("boss_setup_potion" if boss_fight else "safe_setup_potion")
        elif boss_fight and (profile["resource"] or profile["defensive"] or profile["healing"]):
            adjustment += 3.0
            reasons.append("boss_unrestricted_potion")
        if not profile["useful"] and not boss_fight:
            adjustment -= 20.0
            reasons.append("save_potion")
        if incoming <= 0 and (profile["defensive"] or profile["scaling"]) and not boss_fight:
            adjustment -= 6.0
            reasons.append("no_incoming_save_potion")

    elif candidate.kind == "end_turn":
        end_turn_hand_loss = hand_end_turn_damage(player.get("hand"))
        if incoming <= 0:
            adjustment += 0.5
        if end_turn_hand_loss > 0:
            adjustment -= min(24.0, end_turn_hand_loss * 2.0)
            reasons.append(f"hand_end_turn_damage={end_turn_hand_loss:g}")
        if net_incoming + end_turn_hand_loss >= hp and hp > 0:
            adjustment -= 100.0
            reasons.append("never_end_turn_into_hand_lethal")
        if net_incoming >= hp and hp > 0:
            adjustment -= 4.0
            reasons.append("lethal_incoming")

    return adjustment, ",".join(reasons)


def score_combat_candidates(candidate_agent, state_vec, candidates, state=None, exploration=None):
    if not candidate_agent or not candidates:
        return None, [], "candidate_model_unavailable"
    model = candidate_agent["model"]
    device = candidate_agent["device"]
    constraint_mode = str((exploration or {}).get("constraint_mode") or "guarded").lower()
    if constraint_mode not in ("guarded", "explore", "free"):
        constraint_mode = "guarded"
    use_tactical_guards = constraint_mode != "free"
    tactical_scale = 1.0 if constraint_mode == "guarded" else 0.35 if constraint_mode == "explore" else 0.0
    force_delay_end_turn, delay_reason = should_force_delay_end_turn(candidates, state) if use_tactical_guards else (False, "")
    burst_plan = hand_burst_plan(state) if use_tactical_guards and state else None
    lethal_end_turn = False
    if state:
        player = state.get("player") or {}
        battle = state.get("battle") or {}
        hp = safe_num(player.get("hp"), 0.0)
        block = safe_num(player.get("block"), 0.0)
        incoming = enemy_incoming_damage(battle.get("enemies") or [])
        end_turn_hand_loss = hand_end_turn_damage(player.get("hand"))
        lethal_end_turn = bool(
            hp > 0
            and max(0.0, incoming - block) + end_turn_hand_loss >= hp
            and any(c.kind != "end_turn" for c in candidates)
        )
    state_part = align_feature_vector(state_vec, int(candidate_agent["state_dim"]))
    rows = []
    for candidate in candidates:
        rows.append(np.concatenate([
            state_part,
            np.array(candidate.features, dtype=np.float32),
        ]))
    if not rows:
        return None, [], "no_candidates"
    x = torch.tensor(np.asarray(rows, dtype=np.float32), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    ranked = []
    for candidate, logit, prob in zip(candidates, logits.detach().cpu().numpy(), probs):
        adjustment, marker = tactical_candidate_adjustment(candidate, state) if use_tactical_guards else (0.0, "")
        burst_adjustment, burst_marker = burst_candidate_adjustment(candidate, burst_plan) if use_tactical_guards else (0.0, "")
        adjustment += burst_adjustment
        marker = ",".join(x for x in (marker, burst_marker) if x)
        adjustment *= tactical_scale
        if lethal_end_turn and use_tactical_guards:
            if candidate.kind == "end_turn":
                adjustment -= 100.0 * tactical_scale
                marker = ",".join(x for x in (marker, "never_end_turn_into_lethal") if x)
            elif candidate.kind == "use_potion":
                profile = potion_tactical_profile(candidate, state)
                if profile["lethal"] or profile["prevents_lethal"] or profile["resource_useful"]:
                    adjustment += 8.0 * tactical_scale
                    marker = ",".join(x for x in (marker, "desperate_potion") if x)
                else:
                    adjustment -= 8.0 * tactical_scale
                    marker = ",".join(x for x in (marker, "not_a_rescue_potion") if x)
        if candidate.kind == "end_turn" and force_delay_end_turn:
            adjustment -= 100.0 * tactical_scale
            marker = ",".join(x for x in (marker, delay_reason) if x)
        base_score = float(logit)
        ranked.append({
            "candidate": candidate,
            "base_score": base_score,
            "score": base_score + adjustment,
            "adjustment": adjustment,
            "confidence": float(prob),
            "marker": marker,
        })
    ranked.sort(key=lambda item: item["score"], reverse=True)
    chosen = ranked[0] if ranked else None
    decision_source = f"candidate_scorer_{constraint_mode}"
    if chosen and exploration and exploration.get("enabled"):
        selected, explored, selected_rank = sample_ranked_entry(
            ranked,
            exploration.get("combat_epsilon", 0.0),
            exploration.get("top_k", 1),
            exploration.get("temperature", 1.0),
            score_key="score",
        )
        if selected:
            chosen = selected
            chosen["explored"] = explored
            chosen["exploration_rank"] = selected_rank
            if explored:
                decision_source = "candidate_scorer_tactical_explore"
    return chosen, ranked, decision_source


def candidate_top_actions(ranked, limit=5):
    top_actions = []
    for item in ranked[:limit]:
        candidate = item["candidate"]
        confidence = item.get("confidence")
        if confidence is None:
            confidence = item.get("prob")
        if confidence is None:
            confidence = max(float(item.get("score", 0.0)), 0.0)
        top_actions.append({
            "action": candidate.label,
            "confidence": round(float(confidence) * 100.0, 2),
            "marker": " / ".join(x for x in [candidate.kind, item.get("marker")] if x),
        })
    return top_actions


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
    metadata["model_version"] = metadata.get("model_version") or build_model_version("macro_bc", metadata_path, metadata)
    input_dim = int(metadata.get("features") or 115)
    output_dim = len(vocab.tables["actions"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MacroBCModel(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(Fore.CYAN + f"[*] Macro Brain Loaded! Action space: {output_dim} | Features: {input_dim} | Device: {device}")
    return {"vocab": vocab, "id_to_action": id_to_action, "model": model, "device": device, "input_dim": input_dim, "metadata": metadata, "model_version": metadata["model_version"]}


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
    if state_type == "treasure":
        treasure = state.get("treasure") or {}
        for key in ("relics", "items", "rewards", "reward_items"):
            items = treasure.get(key)
            if items:
                rows = []
                for fallback_index, item in enumerate(items or []):
                    if not isinstance(item, dict):
                        continue
                    row = dict(item)
                    if key == "relics":
                        row.setdefault("type", "relic")
                    row.setdefault("index", fallback_index)
                    rows.append(row)
                return rows
        relic = treasure.get("relic") or treasure.get("reward") or {}
        if isinstance(relic, dict) and relic:
            item = dict(relic)
            item.setdefault("type", "relic")
            item.setdefault("index", treasure.get("index", 0))
            return [item]
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


def reward_item_type(item):
    return str(item.get("type") or item.get("category") or "").lower()


def reward_matches_type(item, reward_type):
    item_type = reward_item_type(item)
    return item_type == reward_type or (reward_type == "card" and item_type == "special_card")


def potion_is_empty_slot(potion):
    potion_id = potion_id_key(potion)
    name = str(potion.get("name") or potion.get("potion_name") or "").strip().upper()
    return (not potion_id and not name) or potion_id in ("EMPTY", "EMPTY_POTION_SLOT", "NONE")


def filled_potions(player):
    return [
        potion
        for potion in (player.get("potions") or [])
        if isinstance(potion, dict) and not potion_is_empty_slot(potion)
    ]


def potion_slot_capacity(player):
    filled = len(filled_potions(player))
    for key in ("potion_slots_total", "max_potions", "potion_capacity", "potion_slots"):
        value = player.get(key)
        if isinstance(value, (list, tuple)):
            return max(len(value), 0)
        if isinstance(value, dict):
            return max(len(value), 0)
        count = safe_int(value, 0)
        if count > 0:
            return count
    return max(3, filled)


def potion_slots_full(player):
    potions = filled_potions(player)
    filled = max(safe_int(player.get("potion_slots_filled"), len(potions)), len(potions))
    capacity = potion_slot_capacity(player)
    return bool(capacity > 0 and filled >= capacity)


def normalize_potion_like(item):
    if not isinstance(item, dict):
        return {}
    return {
        "id": item.get("potion_id") or item.get("id") or item.get("item_id"),
        "name": item.get("potion_name") or item.get("name") or item.get("item_name"),
        "description": (
            item.get("potion_description")
            or item.get("description")
            or item.get("item_description")
        ),
        "type": item.get("type") or item.get("category"),
        "slot": item.get("slot"),
    }


def potion_keep_value(potion):
    potion = normalize_potion_like(potion)
    potion_id = potion_id_key(potion)
    text = combat_text(potion)
    known = {
        "FIRE_POTION": 4.5,
        "EXPLOSIVE_AMPOULE": 4.0,
        "POWDERED_DEMISE": 4.0,
        "BLOOD_POTION": 3.8,
        "BLOCK_POTION": 3.5,
        "ENERGY_POTION": 3.4,
        "STRENGTH_POTION": 3.2,
        "FORTIFIER": 3.2,
        "ANCIENT_POTION": 3.0,
        "WEAK_POTION": 2.8,
        "VULNERABILITY_POTION": 2.8,
        "DEXTERITY_POTION": 2.8,
        "SPEED_POTION": 2.7,
        "DROPLET_OF_PRECOGNITION": 2.6,
        "MAZALETHS_GIFT": 2.6,
        "CLARITY": 2.4,
        "FLEX_POTION": 2.2,
        "COLORLESS_POTION": 2.1,
        "ATTACK_POTION": 2.0,
        "SKILL_POTION": 2.0,
    }
    score = known.get(potion_id, 1.5)
    if has_any(text, ("deal", "damage", "attack", "strength", "vulnerable", "fire")):
        score += 0.5
    if has_any(text, ("block", "heal", "recover", "weak")):
        score += 0.4
    if has_any(text, ("energy", "draw", "card", "free")):
        score += 0.3
    return round(score, 3)


def potion_slot_number(potion, fallback_index):
    slot = safe_int(potion.get("slot"), -1)
    return slot if slot >= 0 else fallback_index


def reward_potion_payload(state, item, fallback_index):
    player = state.get("player") or {}
    reward_potion = normalize_potion_like(item)
    reward_index = int(item.get("index", fallback_index))
    if not potion_slots_full(player):
        return {"action": "claim_reward", "index": reward_index}, "available"

    inventory = filled_potions(player)
    if not inventory:
        return {"action": "claim_reward", "index": reward_index}, "available_no_inventory"

    valued = [
        (potion_keep_value(potion), potion_slot_number(potion, idx), potion)
        for idx, potion in enumerate(inventory)
    ]
    valued.sort(key=lambda row: row[0])
    worst_value, worst_slot, worst_potion = valued[0]
    reward_value = potion_keep_value(reward_potion)
    reward_name = reward_potion.get("name") or reward_potion.get("id") or "reward_potion"
    worst_name = worst_potion.get("name") or worst_potion.get("id") or f"slot_{worst_slot}"
    if reward_value >= worst_value + 0.75:
        return (
            {"action": "discard_potion", "slot": int(worst_slot)},
            f"full_potion_slots_discard slot={worst_slot} replace={worst_name}->{reward_name} {worst_value:.2f}->{reward_value:.2f}",
        )
    return (
        None,
        f"skip_full_potion_reward keep={worst_name}:{worst_value:.2f} reward={reward_name}:{reward_value:.2f}",
    )


def reward_item_claim_payload(state, item, fallback_index, state_type=None):
    if str(state_type or (state or {}).get("state_type") or "").lower() == "treasure":
        return {"action": "claim_treasure_relic", "index": int(item.get("index", fallback_index))}, "treasure_relic_available"
    if reward_item_type(item) == "potion":
        return reward_potion_payload(state, item, fallback_index)
    return {"action": "claim_reward", "index": int(item.get("index", fallback_index))}, "available"


def choose_reward_rule_action(state, state_type):
    items = list(get_items_for_state(state, state_type))
    priority = {"relic": 0, "gold": 1, "potion": 2, "card": 3, "special_card": 3}
    ranked = sorted(
        enumerate(items),
        key=lambda row: (priority.get(reward_item_type(row[1]), 4), safe_int(row[1].get("index"), row[0])),
    )
    top_actions = []
    for fallback_index, item in ranked:
        payload, status = reward_item_claim_payload(state, item, fallback_index, state_type)
        item_type = reward_item_type(item) or "item"
        label_index = safe_int(item.get("index"), fallback_index)
        marker = status
        top_actions.append({
            "action": f"claim_reward:index_{label_index}:{item_type}",
            "confidence": 100.0 - min(len(top_actions), 8),
            "marker": marker,
        })
        if payload:
            return payload, {
                "top_actions": top_actions,
                "chosen_action": f"claim_reward:index_{label_index}:{item_type}",
                "payload": payload,
                "reason": f"{state_type}_claim_before_proceed:{status}",
            }
    if get_can_proceed(state, state_type):
        payload = {"action": "proceed"}
        return payload, {
            "top_actions": top_actions or [{"action": "proceed", "confidence": 100.0, "marker": "no_claimable_rewards"}],
            "chosen_action": "proceed",
            "payload": payload,
            "reason": f"{state_type}_no_claimable_proceed",
        }
    return None, {
        "top_actions": top_actions,
        "chosen_action": None,
        "payload": None,
        "reason": f"{state_type}_no_claimable_no_proceed",
    }


def rest_option_blob(option):
    parts = []
    for key in ("id", "option_id", "title", "name", "description", "option_title"):
        value = option.get(key)
        if value is not None:
            parts.append(str(value))
    return " ".join(parts).lower()


def rest_option_kind(option):
    text = rest_option_blob(option)
    if has_any(text, ("smith", "upgrade", "强化", "升级", "锻造")):
        return "smith"
    if has_any(text, ("heal", "rest", "sleep", "休息", "治疗", "回复")):
        return "heal"
    return "other"


def is_pre_boss_rest_site(state):
    run = (state or {}).get("run") or {}
    floor = safe_int(run.get("floor"), 0)
    act = safe_int(run.get("act"), 0)
    map_state = (state or {}).get("map") or {}
    next_options = map_state.get("next_options") or []
    if any(str(option.get("type") or "").lower() == "boss" for option in next_options if isinstance(option, dict)):
        return True
    # STS2 act maps report the final rest site immediately before the boss around floor 15.
    return bool(act >= 1 and floor >= 15)


def choose_rest_site_rule_action(state, exploration=None):
    rest_site = state.get("rest_site") or {}
    options = rest_site.get("options") or []
    enabled = [o for o in options if o.get("is_enabled", True)]
    if not enabled:
        return None, {
            "top_actions": [],
            "chosen_action": None,
            "payload": None,
            "reason": "rest_no_enabled_option",
        }

    player = state.get("player") or {}
    hp = safe_num(player.get("hp"), 0.0)
    max_hp = max(safe_num(player.get("max_hp"), 1.0), 1.0)
    hp_ratio = hp / max_hp
    smith = next((o for o in enabled if rest_option_kind(o) == "smith"), None)
    heal = next((o for o in enabled if rest_option_kind(o) == "heal"), None)
    pre_boss = is_pre_boss_rest_site(state)

    if pre_boss and heal and hp < max_hp:
        chosen = heal
        reason = f"rest_rule_pre_boss_heal hp={hp_ratio:.2f}"
    elif hp_ratio >= 0.62 and smith:
        chosen = smith
        reason = f"rest_rule_smith_high_hp hp={hp_ratio:.2f}"
    elif hp_ratio <= 0.45 and heal:
        chosen = heal
        reason = f"rest_rule_heal_low_hp hp={hp_ratio:.2f}"
    elif smith:
        chosen = smith
        reason = f"rest_rule_smith_stable_hp hp={hp_ratio:.2f}"
    elif heal:
        chosen = heal
        reason = f"rest_rule_heal_only hp={hp_ratio:.2f}"
    else:
        chosen = enabled[0]
        reason = "rest_rule_first_enabled"

    scored_options = []
    for option in enabled:
        kind = rest_option_kind(option)
        option_index = int(option.get("index", enabled.index(option)))
        score = 100.0 if option is chosen else (60.0 if kind == "smith" else 40.0)
        if pre_boss and kind == "heal" and hp < max_hp:
            score = max(score, 120.0)
        elif pre_boss and kind == "smith" and hp < max_hp:
            score = min(score, 20.0)
        scored_options.append({
            "option": option,
            "score": score,
            "index": option_index,
            "kind": kind,
        })
    if exploration and exploration.get("enabled"):
        selected, explored, selected_rank = sample_ranked_entry(
            sorted(scored_options, key=lambda item: item["score"], reverse=True),
            exploration.get("macro_epsilon", 0.0),
            exploration.get("top_k", 1),
            exploration.get("temperature", 1.0),
            score_key="score",
        )
        if selected:
            chosen = selected["option"]
            if explored:
                reason = f"{reason}; explore_rank={selected_rank}"
    index = int(chosen.get("index", enabled.index(chosen)))
    top_actions = []
    for item in sorted(scored_options, key=lambda row: row["score"], reverse=True):
        option = item["option"]
        top_actions.append({
            "action": f"choose_rest_option:index_{item['index']}:{item['kind']}",
            "confidence": item["score"],
            "marker": f"hp={hp_ratio:.2f}; id={option.get('id') or option.get('option_id') or '-'}",
        })
    return {"action": "choose_rest_option", "index": index}, {
        "top_actions": top_actions,
        "chosen_action": f"choose_rest_option:index_{index}:{rest_option_kind(chosen)}",
        "payload": {"action": "choose_rest_option", "index": index},
        "reason": reason,
    }


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
        wanted_category = wanted if wanted in ("card", "relic", "potion", "card_removal") else None
        if wanted_category:
            ranked = rank_shop_items(state, state_type, category_filter=wanted_category)
            if ranked:
                best = ranked[0]
                return best["item"], best["fallback_index"], f"shop_rule_{wanted_category}:{best['score']:+.2f}"
            return None, None, f"shop_{wanted_category}_unavailable"
        for fallback_index, item in available:
            if shop_item_matches(item, wanted):
                return item, fallback_index, "available"
        return None, None, "shop_item_unavailable"

    ranked = rank_shop_items(state, state_type)
    if not ranked:
        return None, None, "no_safe_shop_purchase"
    best = ranked[0]
    return best["item"], best["fallback_index"], f"shop_rule_purchase:{best['score']:+.2f}"


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
        for fallback_index, option in enumerate(options):
            option_index = safe_int(option.get("index"), fallback_index)
            if option_index == index:
                return {"action": "choose_map_node", "index": option_index}, "available"
        if 0 <= index < len(options):
            option = options[index] or {}
            return {"action": "choose_map_node", "index": safe_int(option.get("index"), index)}, "available"
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
        if state_type not in ("rewards", "treasure"):
            return None, "not_rewards_or_treasure"
        reward_type = label.split(":", 1)[1].lower().split(" ", 1)[0]
        last_status = f"reward_{reward_type}_unavailable"
        for fallback_index, item in enumerate(get_items_for_state(state, state_type)):
            if reward_matches_type(item, reward_type):
                payload, status = reward_item_claim_payload(state, item, fallback_index, state_type)
                if payload:
                    return payload, status
                last_status = status
        return None, last_status

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

    if label.startswith("choose_card:"):
        if state_type != "card_reward":
            return None, "not_card_reward"
        wanted = label.split(":", 1)[1].lower()
        cards = ((state.get("card_reward") or {}).get("cards") or [])
        for fallback_index, card in enumerate(cards):
            fields = (card.get("id"), card.get("name"), card.get("card_id"), card.get("card_name"))
            if any(str(value or "").lower() == wanted for value in fields):
                return {"action": "select_card_reward", "card_index": int(card.get("index", fallback_index))}, "available"
        return None, "card_unavailable"

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
        payload, info = choose_rest_site_rule_action(state)
        if payload:
            return payload, info.get("reason", "rest_rule")
        return None, "no_rest_option_enabled"

    if label.startswith("buy_item:"):
        if state_type not in ("shop", "fake_merchant"):
            return None, "not_shop"
        if not allow_shop:
            return None, "shop_protected"
        wanted = label.split(":", 1)[1].lower()
        item, fallback_index, status = choose_shop_item(state, state_type, wanted)
        if item:
            return {"action": "shop_purchase", "index": int(item.get("index", fallback_index))}, status
        return None, status

    if label == "proceed":
        if state_type in ("shop", "fake_merchant") and not allow_shop:
            return None, "shop_protected"
        if state_type in ("rewards", "treasure"):
            payload, info = choose_reward_rule_action(state, state_type)
            if payload and payload.get("action") != "proceed":
                return None, "claim_reward_before_proceed"
        if get_can_proceed(state, state_type):
            return {"action": "proceed"}, "available"
        return None, "proceed_unavailable"

    return None, "unsupported_macro_label"


def macro_fallback_payload(state):
    state_type = str(state.get("state_type") or "").lower()
    if state_type == "map":
        payload, info = choose_map_route_action(state)
        return payload, info.get("reason", "map_route_fallback")
    if state_type in ("rewards", "treasure"):
        payload, info = choose_reward_rule_action(state, state_type)
        return payload, info.get("reason", f"{state_type}_reward_rule")
    if state_type == "rest_site" and get_can_proceed(state, state_type):
        return {"action": "proceed"}, f"{state_type}_proceed"
    return None, ""


def safe_num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def card_blob(card):
    parts = []
    for key in ("id", "name", "type", "description", "target_type", "rarity"):
        value = card.get(key)
        if value is not None:
            parts.append(str(value))
    keywords = card.get("keywords")
    if keywords:
        try:
            parts.append(json.dumps(keywords, ensure_ascii=False))
        except Exception:
            parts.append(str(keywords))
    return " ".join(parts).lower()


def normalized_card_type(card_or_type):
    value = card_or_type.get("type") if isinstance(card_or_type, dict) else card_or_type
    text = str(value or "").strip().lower()
    if "attack" in text or "攻击" in text:
        return "attack"
    if "skill" in text or "技能" in text:
        return "skill"
    if "power" in text or "能力" in text:
        return "power"
    if "curse" in text or "诅咒" in text:
        return "curse"
    if "status" in text or "状态" in text:
        return "status"
    return "other"


def has_any(text, keywords):
    return any(keyword in text for keyword in keywords)


def card_traits(card):
    text = card_blob(card)
    ctype = normalized_card_type(card)
    cost = safe_num(card.get("cost"), 0.0)
    rarity = str(card.get("rarity") or "").lower()
    known_damage, known_block = known_card_effect_hints(card)
    utility_hints = known_card_utility_hints(card)
    return {
        "type": ctype,
        "cost": cost,
        "rarity": rarity,
        "is_attack": ctype == "attack",
        "is_skill": ctype == "skill",
        "is_power": ctype == "power" or utility_hints["power"],
        "damage": ctype == "attack" or known_damage > 0 or has_any(text, ("damage", "伤害")),
        "block": known_block > 0 or (ctype == "skill" and has_any(text, ("block", "格挡", "防御", "护甲"))),
        "draw": has_any(text, ("draw", "抽", "card draw", "抽牌")),
        "aoe": has_any(text, ("all enemies", "aoe", "所有敌人", "全部敌人", "全体", "每个敌人")),
        "scaling": utility_hints["scaling"] or has_any(text, ("strength", "dexterity", "focus", "力量", "敏捷", "集中", "每回合", "whenever", "永久")),
    }


def deck_profile_from_state(state):
    player = state.get("player") or {}
    deck = player.get("deck") or player.get("cards") or []
    overview = player.get("deck_overview") or {}
    profile = {
        "total": safe_int(player.get("deck_size") or player.get("deck_total") or len(deck)),
        "attack": 0,
        "skill": 0,
        "power": 0,
        "other": 0,
        "draw": 0,
        "aoe": 0,
        "scaling": 0,
    }

    if isinstance(deck, list) and deck:
        for card in deck:
            if not isinstance(card, dict):
                continue
            traits = card_traits(card)
            ctype = traits["type"]
            profile[ctype if ctype in ("attack", "skill", "power") else "other"] += 1
            profile["draw"] += 1 if traits["draw"] else 0
            profile["aoe"] += 1 if traits["aoe"] else 0
            profile["scaling"] += 1 if traits["scaling"] or traits["is_power"] else 0
        profile["total"] = max(profile["total"], sum(profile[k] for k in ("attack", "skill", "power", "other")))
    elif isinstance(overview, dict):
        for key, count in overview.items():
            ctype = normalized_card_type(key)
            bucket = ctype if ctype in ("attack", "skill", "power") else "other"
            profile[bucket] += safe_int(count)
        profile["total"] = max(profile["total"], sum(profile[k] for k in ("attack", "skill", "power", "other")))

    total = max(profile["total"], 1)
    profile["attack_ratio"] = round(profile["attack"] / total, 3)
    profile["skill_ratio"] = round(profile["skill"] / total, 3)
    profile["power_ratio"] = round(profile["power"] / total, 3)
    return profile


def public_deck_profile(profile):
    return {
        "total": profile.get("total", 0),
        "attack": profile.get("attack", 0),
        "skill": profile.get("skill", 0),
        "power": profile.get("power", 0),
        "draw": profile.get("draw", 0),
        "aoe": profile.get("aoe", 0),
        "scaling": profile.get("scaling", 0),
        "attack_ratio": profile.get("attack_ratio", 0),
        "skill_ratio": profile.get("skill_ratio", 0),
    }


def option_card_scorer_mode(control=None):
    if control is None:
        try:
            control = load_control()
        except Exception:
            control = DEFAULT_CONTROL
    setting = (control or {}).get("option_card_scorer", DEFAULT_CONTROL.get("option_card_scorer"))
    return normalize_card_scorer_mode(setting)


def card_reward_signature_for_template_lock(state):
    state = state if isinstance(state, dict) else {}
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    cards = ((state.get("card_reward") or {}).get("cards") or [])
    card_keys = []
    for fallback_index, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        card_keys.append((
            safe_int(card.get("index"), fallback_index),
            str(card.get("id") or card.get("card_id") or ""),
            str(card.get("name") or card.get("card_name") or ""),
        ))
    return json.dumps({
        "act": safe_int(run.get("act"), 0),
        "floor": safe_int(run.get("floor"), 0),
        "cards": card_keys,
    }, sort_keys=True, ensure_ascii=False)


def locked_template_for_card_reward(state, session_id=""):
    config = load_template_config()
    deck_summary = build_deck_summary(state)
    candidate = select_template(deck_summary, config)
    consistency = archetype_consistency(deck_summary, config)
    scores = consistency.get("scores") or {}
    lock_cfg = config.get("template_selection") or {}
    if str(lock_cfg.get("mode") or "locked_after_warmup") != "locked_after_warmup":
        return candidate, deck_summary, {
            "mode": str(lock_cfg.get("mode") or "free"),
            "candidate_template": candidate,
            "template_scores": scores,
            "locked": False,
        }

    key = session_id or "global"
    state_key = card_reward_signature_for_template_lock(state)
    lock = CARD_TEMPLATE_LOCKS.setdefault(key, {
        "seen": set(),
        "card_reward_count": 0,
        "template_counts": {},
        "locked_template": "",
        "challenger_template": "",
        "challenger_count": 0,
    })
    lock.setdefault("template_counts", {})
    warmup = max(safe_int(lock_cfg.get("warmup_card_rewards"), 3), 0)
    margin = safe_num(lock_cfg.get("switch_margin"), 1.0)
    patience = max(safe_int(lock_cfg.get("switch_patience"), 2), 1)
    switched = False

    if state_key not in lock["seen"]:
        lock["seen"].add(state_key)
        lock["card_reward_count"] += 1
        lock["template_counts"][candidate] = safe_int(lock["template_counts"].get(candidate), 0) + 1
        if not lock["locked_template"] and lock["card_reward_count"] >= warmup:
            lock["locked_template"] = max(
                lock["template_counts"].items(),
                key=lambda item: (safe_int(item[1], 0), safe_num(scores.get(item[0]), 0.0), item[0]),
            )[0]
        elif lock["locked_template"]:
            locked = lock["locked_template"]
            candidate_score = safe_num(scores.get(candidate), 0.0)
            locked_score = safe_num(scores.get(locked), 0.0)
            if candidate != locked and candidate_score - locked_score >= margin:
                if lock["challenger_template"] == candidate:
                    lock["challenger_count"] += 1
                else:
                    lock["challenger_template"] = candidate
                    lock["challenger_count"] = 1
                if lock["challenger_count"] >= patience:
                    lock["locked_template"] = candidate
                    lock["challenger_template"] = ""
                    lock["challenger_count"] = 0
                    switched = True
            else:
                lock["challenger_template"] = ""
                lock["challenger_count"] = 0

    selected = lock["locked_template"] or candidate
    lock_info = {
        "mode": "locked_after_warmup",
        "warmup_card_rewards": warmup,
        "switch_margin": margin,
        "switch_patience": patience,
        "card_reward_count": lock["card_reward_count"],
        "candidate_template": candidate,
        "template_counts": dict(lock["template_counts"]),
        "locked_template": lock["locked_template"],
        "selected_template": selected,
        "challenger_template": lock["challenger_template"],
        "challenger_count": lock["challenger_count"],
        "template_scores": scores,
        "locked": bool(lock["locked_template"]),
        "switched": switched,
    }
    return selected, deck_summary, lock_info


def build_shadow_card_result(state, mode=None):
    mode = normalize_card_scorer_mode(mode or option_card_scorer_mode())
    if mode == "off":
        return mode, None, ""
    try:
        session_id = str((state or {}).get("_ai_session_id") or "")
        template_id, deck_summary, lock_info = locked_template_for_card_reward(state, session_id=session_id)
        state_with_lock = dict(state or {})
        state_with_lock["_card_template_lock"] = lock_info
        return mode, build_card_reward_options(
            state_with_lock,
            mode=mode,
            template_id=template_id,
            deck_summary=deck_summary,
        ), ""
    except Exception as exc:
        return mode, None, str(exc)


def card_scorer_info_payload(mode, result, error=""):
    if mode == "off":
        return {"mode": "off"}
    payload = {
        "mode": mode,
        "scorer_version": CARD_SCORER_VERSION,
        "option_schema": OPTION_SCHEMA_VERSION,
        "state_features_version": STATE_FEATURES_VERSION,
        "option_features_version": CARD_OPTION_FEATURES_VERSION,
    }
    if error:
        payload["error"] = error
    if result:
        payload.update(card_result_log_payload(result, include_options=True))
        payload["top_actions"] = card_result_top_actions(result)
    return payload


def active_card_scorer_choice(state, card_baseline_weight=1.0):
    requested_mode = option_card_scorer_mode()
    mode = "active_canary" if requested_mode == "active_canary" else "active"
    mode, result, error = build_shadow_card_result(state, mode=mode)
    if not result or not result.selected:
        return None, {
            "top_actions": [],
            "chosen_action": None,
            "payload": None,
            "reason": f"card_scorer_active_unavailable: {error or 'no_options'}",
            "deck_profile": {},
            "reward_baseline": {
                "mode": "card_scorer_active",
                "weight": card_baseline_weight,
                "error": error,
            },
            "card_scorer": card_scorer_info_payload(mode, result, error),
        }
    selected = result.selected
    fallback_reason = ""
    if mode == "active_canary":
        entries, profile = card_reward_baseline_entries(state)
        entries.sort(key=lambda item: item["score"], reverse=True)
        old = entries[0] if entries else None
        canary_cfg = (load_template_config().get("active_canary") or {})
        min_gap = safe_num(canary_cfg.get("only_when_confidence_gap_gte"), 1.0)
        fallback_gap = safe_num(canary_cfg.get("fallback_to_old_when_gap_lt"), 0.3)
        allow_skip_deck = safe_int(canary_cfg.get("allow_skip_when_deck_size_gte"), 22)
        max_card_index = safe_int(canary_cfg.get("max_card_index"), 2)
        deck_size = safe_int((result.deck_summary or {}).get("deck_size"), 0)
        selected_index = safe_int(getattr(selected, "index", -1), -1)
        if old:
            if result.confidence_gap < fallback_gap:
                fallback_reason = f"canary_low_gap<{fallback_gap:g}"
            elif result.confidence_gap < min_gap:
                fallback_reason = f"canary_gap_below_takeover<{min_gap:g}"
            elif selected.label == "skip_reward" and deck_size < allow_skip_deck:
                fallback_reason = f"canary_skip_deck<{allow_skip_deck}"
            elif selected.label.startswith("choose_card:index_") and selected_index > max_card_index:
                fallback_reason = f"canary_extra_card_index>{max_card_index}"
            if fallback_reason:
                selected = type(result.selected)(
                    label=old["label"],
                    payload=old["payload"],
                    kind="card_reward",
                    score=old["score"],
                    reasons=list(old.get("reasons") or []) + [fallback_reason],
                    metadata={"card": old.get("card") or {}, "score_breakdown": {}, "canary_fallback": fallback_reason},
                    index=safe_int((old.get("payload") or {}).get("card_index"), -1),
                )
    top_actions = card_result_top_actions(result)
    return selected.payload, {
        "top_actions": top_actions,
        "chosen_action": selected.label,
        "payload": selected.payload,
        "reason": f"card_scorer_{mode}: " + (" / ".join(selected.reasons) or "highest score"),
        "deck_profile": result.deck_summary,
        "reward_baseline": {
            "mode": f"card_scorer_{mode}",
            "weight": card_baseline_weight,
            "chosen_score": round(float(selected.score), 3),
            "chosen_reason": selected.reasons,
            "template_id": result.template_id,
            "archetype_consistency": result.archetype_consistency,
            "canary_fallback_reason": fallback_reason,
        },
        "card_scorer": card_scorer_info_payload(mode, result, error),
        "archetype_consistency": result.archetype_consistency,
    }


def score_reward_card(card, profile, state):
    traits = card_traits(card)
    run = state.get("run") or {}
    player = state.get("player") or {}
    act = safe_int(run.get("act"), 1)
    floor = safe_int(run.get("floor"), 0)
    hp = safe_num(player.get("hp"), 0.0)
    max_hp = max(safe_num(player.get("max_hp"), 1.0), 1.0)
    hp_ratio = hp / max_hp
    deck_size = max(safe_int(profile.get("total")), 1)
    attack = safe_int(profile.get("attack"))
    skill = safe_int(profile.get("skill"))
    power = safe_int(profile.get("power"))
    attack_ratio = safe_num(profile.get("attack_ratio"))
    skill_ratio = safe_num(profile.get("skill_ratio"))

    score = 0.0
    reasons = []

    if traits["is_attack"]:
        if act <= 1 and floor <= 8:
            score += 2.0
            reasons.append("Act1前期需要输出")
        if attack < 5:
            score += 1.8
            reasons.append("攻击牌偏少")
        if attack_ratio < 0.32:
            score += 1.2
            reasons.append("攻击比例低")
        if attack_ratio > 0.45 and deck_size >= 16:
            score -= 1.0
            reasons.append("攻击比例已高")

    if traits["is_skill"]:
        if skill_ratio < 0.30 and deck_size >= 12:
            score += 1.2
            reasons.append("技能牌偏少")
        if act >= 2:
            score += 0.8
            reasons.append("中后期需要防御/功能")
        if hp_ratio < 0.55 and traits["block"]:
            score += 1.4
            reasons.append("低血量优先格挡")

    if traits["is_power"]:
        if floor >= 8 or act >= 2:
            score += 1.2
            reasons.append("需要长期成长")
        if power < 2:
            score += 0.8
            reasons.append("能力牌偏少")
        if act <= 1 and floor <= 5:
            score -= 0.5
            reasons.append("前期能力牌过慢")

    if traits["draw"]:
        if deck_size >= 18 and safe_int(profile.get("draw")) < 2:
            score += 1.6
            reasons.append("牌组变厚需要过牌")
        elif safe_int(profile.get("draw")) < 1:
            score += 0.8
            reasons.append("缺少过牌")

    if traits["aoe"] and safe_int(profile.get("aoe")) < 1:
        score += 1.3
        reasons.append("缺少AOE")

    if traits["scaling"] and (act >= 2 or floor >= 10):
        score += 1.1
        reasons.append("中后期补成长")

    if "rare" in traits["rarity"]:
        score += 0.35
        reasons.append("稀有牌")
    elif "uncommon" in traits["rarity"]:
        score += 0.15
        reasons.append("罕见牌")

    if traits["cost"] >= 3:
        penalty = 0.8 if act <= 1 and floor <= 8 else 0.25
        score -= penalty
        reasons.append("高费用")

    if deck_size >= 28 and score < 1.0:
        score -= 0.7
        reasons.append("牌组已厚")

    return round(score, 3), reasons[:3]


def score_skip_card_reward(card_scores, profile, state):
    run = state.get("run") or {}
    act = safe_int(run.get("act"), 1)
    floor = safe_int(run.get("floor"), 0)
    deck_size = safe_int(profile.get("total"))
    best = max(card_scores) if card_scores else 0.0
    if deck_size >= 30 and best < 1.25:
        return 2.0, ["牌组过厚且候选不强"]
    if deck_size >= 24 and best < 0.5:
        return 1.0, ["候选牌收益低"]
    if act <= 1 and floor <= 8 and deck_size < 18:
        return -2.0, ["前期不建议跳过"]
    return -1.0, ["默认优先拿有效牌"]


def card_reward_baseline_entries(state):
    cards = ((state.get("card_reward") or {}).get("cards") or [])
    profile = deck_profile_from_state(state)
    entries = []
    card_scores = []
    for fallback_index, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        index = safe_int(card.get("index"), fallback_index)
        score, reasons = score_reward_card(card, profile, state)
        card_scores.append(score)
        entries.append({
            "label": f"choose_card:index_{index}",
            "payload": {"action": "select_card_reward", "card_index": index},
            "score": score,
            "reasons": reasons,
            "card": card,
        })
    if (state.get("card_reward") or {}).get("can_skip"):
        skip_score, skip_reasons = score_skip_card_reward(card_scores, profile, state)
        entries.append({
            "label": "skip_reward",
            "payload": {"action": "skip_card_reward"},
            "score": skip_score,
            "reasons": skip_reasons,
            "card": {},
        })
    return entries, profile


def choose_card_reward_baseline_action(state, card_baseline_weight=1.0):
    scorer_mode = option_card_scorer_mode()
    if scorer_mode in ("active", "active_canary"):
        return active_card_scorer_choice(state, card_baseline_weight=card_baseline_weight)
    shadow_mode, shadow_result, shadow_error = build_shadow_card_result(state, mode=scorer_mode)
    shadow_info = card_scorer_info_payload(shadow_mode, shadow_result, shadow_error)
    entries, profile = card_reward_baseline_entries(state)
    if not entries:
        return None, {
            "top_actions": [],
            "chosen_action": None,
            "payload": None,
            "reason": "no_card_reward_options",
            "deck_profile": public_deck_profile(profile),
            "reward_baseline": {
                "mode": "baseline_only",
                "weight": card_baseline_weight,
                "card_scorer_mode": scorer_mode,
            },
            "card_scorer": shadow_info,
            "archetype_consistency": (shadow_info or {}).get("archetype_consistency"),
        }
    entries.sort(key=lambda item: item["score"], reverse=True)
    best = entries[0]
    top_actions = [
        {
            "action": item["label"],
            "confidence": round(max(item["score"], 0.0) * 20, 2),
            "marker": f"baseline={item['score']:+.2f}; {' / '.join(item['reasons']) or '-'}",
        }
        for item in entries[:6]
    ]
    return best["payload"], {
        "top_actions": top_actions,
        "chosen_action": best["label"],
        "payload": best["payload"],
        "reason": "card_reward_baseline: " + (" / ".join(best["reasons"]) or "highest score"),
        "deck_profile": public_deck_profile(profile),
        "reward_baseline": {
            "mode": "baseline_only",
            "weight": card_baseline_weight,
            "chosen_score": best["score"],
            "chosen_reason": best["reasons"],
            "card_scorer_mode": scorer_mode,
        },
        "card_scorer": shadow_info,
        "archetype_consistency": (shadow_info or {}).get("archetype_consistency"),
    }


def card_attack_hint(card):
    traits = card_traits(card)
    text = combat_core_text(card)
    known_damage, _known_block = known_card_effect_hints(card)
    if known_damage > 0:
        return known_damage
    if not (traits["is_attack"] or has_any(text, ("deal", "attack", "造成", "攻击"))):
        return 0
    nums = combat_numbers(text)
    if not nums:
        return 0
    if len(nums) >= 2 and nums[1] <= 8 and has_any(text, ("times", "×", "次", "多次")):
        return nums[0] * nums[1]
    return nums[0]


def card_block_hint(card):
    text = combat_core_text(card)
    _known_damage, known_block = known_card_effect_hints(card)
    if known_block > 0:
        return known_block
    if not has_any(text, ("block", "defend", "armor", "格挡", "护甲", "防御")):
        return 0
    nums = combat_numbers(text)
    return nums[0] if nums else 0


def is_starter_strike_or_defend(card):
    card_id = str(card.get("id") or card.get("card_id") or "").upper()
    name = str(card.get("name") or card.get("card_name") or "")
    return card_id in ("STRIKE_IRONCLAD", "DEFEND_IRONCLAD") or name in ("打击", "防御", "Strike", "Defend") or name.lower() in ("strike", "defend")


def is_status_or_curse_card(card):
    traits = card_traits(card)
    text = card_blob(card)
    if traits["type"] in ("curse", "status"):
        return True
    return has_any(text, (
        "curse", "status", "wound", "burn", "dazed", "slimed", "void",
        "诅咒", "状态", "伤口", "灼伤", "晕眩", "黏液", "虚空",
    ))


def is_safe_sacrifice_card(card):
    return is_status_or_curse_card(card) or is_starter_strike_or_defend(card)


def card_long_term_value(card):
    traits = card_traits(card)
    text = card_blob(card)
    attack = card_attack_hint(card)
    block = card_block_hint(card)
    score = 0.0
    reasons = []

    if is_status_or_curse_card(card):
        score -= 6.0
        reasons.append("负面牌")

    if is_starter_strike_or_defend(card):
        score -= 1.8
        reasons.append("基础打防")
    elif "basic" in traits["rarity"]:
        score -= 0.5
        reasons.append("基础牌")

    if traits["is_attack"]:
        score += 1.8 + min(attack / 10.0, 2.4)
        reasons.append("攻击牌")
        if attack >= 10:
            score += 0.7
            reasons.append("高伤害")
    if "vulnerable" in text or "易伤" in text:
        score += 1.2
        reasons.append("易伤")
    if traits["block"] or block:
        score += 0.8 + min(block / 12.0, 1.1)
        reasons.append("格挡")
    if traits["draw"]:
        score += 1.0
        reasons.append("过牌")
    if traits["scaling"] or traits["is_power"]:
        score += 1.3
        reasons.append("成长")
    if traits["aoe"]:
        score += 1.0
        reasons.append("AOE")
    if "energy" in text or "能量" in text:
        score += 0.6
        reasons.append("能量")
    if "exhaust" in text or "消耗" in text:
        score -= 0.4
        reasons.append("消耗")
    if card.get("is_upgraded"):
        score += 0.8
        reasons.append("已升级")
    if "rare" in traits["rarity"]:
        score += 0.4
        reasons.append("稀有牌")
    elif "uncommon" in traits["rarity"]:
        score += 0.2
        reasons.append("罕见牌")
    score -= max(traits["cost"] - 2.0, 0.0) * 0.35
    return round(score, 3), reasons[:3]


def card_sacrifice_priority(card, state=None):
    value, value_reasons = card_long_term_value(card)
    traits = card_traits(card)
    text = card_blob(card)
    score = -value
    reasons = ["低长期价值"] + value_reasons[:1]

    if is_status_or_curse_card(card):
        score += 8.0
        reasons = ["优先清理负面牌"]
    elif is_starter_strike_or_defend(card):
        score += 5.0
        reasons = ["优先处理基础打防"]

    if state:
        profile = deck_profile_from_state(state)
        if is_starter_strike_or_defend(card):
            attack_ratio = safe_num(profile.get("attack_ratio"))
            skill_ratio = safe_num(profile.get("skill_ratio"))
            if traits["is_attack"] and attack_ratio >= skill_ratio:
                score += 0.6
                reasons.append("攻击牌偏多")
            if traits["is_skill"] and skill_ratio > attack_ratio + 0.08:
                score += 0.6
                reasons.append("技能牌偏多")

    if card.get("is_upgraded"):
        score -= 2.0
        reasons.append("保护已升级牌")
    if traits["is_power"] or traits["draw"] or traits["scaling"] or traits["aoe"]:
        score -= 2.0
        reasons.append("保护功能牌")
    if traits["is_attack"] and card_attack_hint(card) >= 10:
        score -= 2.5
        reasons.append("保护核心输出")
    if "rare" in traits["rarity"] or "uncommon" in traits["rarity"]:
        score -= 0.8
        reasons.append("保护高稀有度")
    if has_any(text, ("vulnerable", "易伤", "weak", "虚弱")):
        score -= 1.2
        reasons.append("保护关键状态牌")
    if value >= 4.0:
        score -= 3.0
        reasons.append("禁止牺牲强牌")

    return round(score, 3), reasons[:3]


def detect_card_select_operation(screen):
    screen_type = str(screen.get("screen_type") or "").lower()
    prompt = " ".join(
        str(screen.get(key) or "").lower()
        for key in ("prompt", "instructions_title", "instructions_description")
    )
    text = f"{screen_type} {prompt}"

    if has_any(text, ("draw pile", "top of", "抽牌堆顶", "抽牌堆顶部", "牌堆顶", "放到抽牌堆")):
        return "topdeck"
    if "upgrade" in screen_type or has_any(text, ("upgrade", "升级", "强化")):
        return "upgrade"
    if "transform" in screen_type or has_any(text, ("transform", "变换", "变化", "变形", "转化")):
        return "transform"
    if has_any(text, (
        "remove", "delete", "sell", "sold", "sacrifice", "lose a card", "exhaust",
        "移除", "删除", "出售", "卖", "献祭", "失去一张", "失去1张", "消耗", "抛弃", "弃掉",
    )):
        return "sacrifice"
    if "discard" in text and "discard pile" not in text and "弃牌堆" not in text:
        return "sacrifice"
    if "丢弃" in text and "弃牌堆" not in text:
        return "sacrifice"
    return "choose"


def score_card_select_for_operation(card, operation, state=None):
    value, value_reasons = card_long_term_value(card)
    traits = card_traits(card)
    text = card_blob(card)

    if operation == "upgrade":
        if card.get("is_upgraded"):
            return -5.0, ["已升级，跳过"]
        score = value
        reasons = ["升级高价值牌"] + value_reasons[:2]
        if has_any(text, ("vulnerable", "易伤", "strength", "力量")):
            score += 1.0
            reasons.append("升级关键状态/成长")
        if card_attack_hint(card) >= 8:
            score += 0.8
            reasons.append("升级输出")
        if card_block_hint(card) >= 10:
            score += 0.5
            reasons.append("升级防御")
        if is_starter_strike_or_defend(card):
            score -= 0.6
            reasons.append("基础牌优先级低")
        return round(score, 3), reasons[:3]

    if operation == "topdeck":
        score = value
        reasons = ["放到抽牌堆顶"] + value_reasons[:2]
        if traits["is_attack"]:
            score += min(card_attack_hint(card) / 8.0, 1.8)
            reasons.append("下回合输出")
        if traits["draw"] or "energy" in text or "能量" in text:
            score += 0.8
            reasons.append("下回合启动")
        if traits["is_power"]:
            score -= 0.5
            reasons.append("能力牌偏慢")
        return round(score, 3), reasons[:3]

    if operation in ("sacrifice", "transform"):
        score, reasons = card_sacrifice_priority(card, state)
        if operation == "transform":
            reasons = ["变换低价值牌"] + reasons[:2]
            if is_starter_strike_or_defend(card):
                score += 0.8
            if is_status_or_curse_card(card):
                score += 0.4
        else:
            reasons = ["移除/卖掉低价值牌"] + reasons[:2]
        return round(score, 3), reasons[:3]

    return value, ["选择高价值牌"] + value_reasons[:2]


def is_selection_state_type(state_type):
    return str(state_type or "").lower() in ("card_select", "hand_select")


def selection_screen_for_state(state):
    state_type = str((state or {}).get("state_type") or "").lower()
    if state_type == "hand_select":
        return (state or {}).get("hand_select") or {}
    return (state or {}).get("card_select") or {}


def selection_memory_signature(state):
    state_type = str((state or {}).get("state_type") or "").lower()
    screen = selection_screen_for_state(state)
    run = (state or {}).get("run") or {}
    battle = (state or {}).get("battle") or {}
    return json.dumps({
        "state_type": state_type,
        "screen_type": screen.get("screen_type") or screen.get("mode"),
        "prompt": screen.get("prompt") or "",
        "act": run.get("act"),
        "floor": run.get("floor"),
        "round": battle.get("round") if state_type == "hand_select" else None,
    }, sort_keys=True, ensure_ascii=False)


def selection_selected_count(state, screen):
    remembered = {
        safe_int(index, -1)
        for index in (state or {}).get("_selection_selected_indices", [])
    }
    legacy_remembered = {
        safe_int(index, -1)
        for index in (state or {}).get("_card_select_selected_indices", [])
    }
    observed = screen.get("selected_cards") or []
    return max(
        len({index for index in remembered | legacy_remembered if index >= 0}),
        len(observed) if isinstance(observed, list) else 0,
    )


def selection_target_count(screen, state_type="card_select"):
    for key in ("required_count", "target_count", "max_select", "max_cards", "card_count"):
        if key in (screen or {}):
            value = safe_int(screen.get(key), 0)
            if value > 0:
                return min(value, 6)

    text = " ".join(
        str((screen or {}).get(key) or "").lower()
        for key in ("prompt", "instructions_title", "instructions_description", "screen_type", "mode")
    )
    patterns = (
        r"(?:select|choose|pick|discard|exhaust|upgrade|transform|remove)\s+(?:up to\s+)?(\d+)",
        r"(\d+)\s+(?:cards|card)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = safe_int(match.group(1), 0)
            if value > 0:
                return min(value, 6)

    cn_numbers = {"一": 1, "二": 2, "两": 2, "兩": 2, "三": 3, "四": 4, "五": 5}
    for token, value in cn_numbers.items():
        if f"{token}张" in text or f"{token}張" in text or f"{token}card" in text:
            return value

    return 1


def choose_card_select_action(state):
    screen = state.get("card_select") or {}
    operation = detect_card_select_operation(screen)
    remembered_indices = {
        safe_int(index, -1)
        for index in (
            state.get("_selection_selected_indices", [])
            or state.get("_card_select_selected_indices", [])
        )
    }
    observed_indices = {
        safe_int(card.get("index"), -1)
        for card in (screen.get("selected_cards") or [])
        if isinstance(card, dict)
    }
    excluded_indices = {index for index in remembered_indices | observed_indices if index >= 0}
    selected_count = selection_selected_count(state, screen)
    target_count = selection_target_count(screen, "card_select")
    preview_showing = bool(screen.get("preview_showing"))

    if screen.get("can_confirm") and (preview_showing or selected_count >= target_count):
        return {"action": "confirm_selection"}, {
            "top_actions": [{"action": "confirm_selection", "confidence": 100.0, "marker": "selection_ready"}],
            "chosen_action": "confirm_selection",
            "payload": {"action": "confirm_selection"},
            "reason": f"card_select_confirm_ready selected={selected_count}/{target_count}",
        }

    cards = screen.get("cards") or []
    if not cards:
        if screen.get("can_cancel"):
            return {"action": "cancel_selection"}, {
                "top_actions": [{"action": "cancel_selection", "confidence": 100.0, "marker": "no_cards"}],
                "chosen_action": "cancel_selection",
                "payload": {"action": "cancel_selection"},
                "reason": "card_select_no_cards",
            }
        return None, {
            "top_actions": [],
            "chosen_action": None,
            "payload": None,
            "reason": "card_select_no_cards",
        }

    scored = []
    for fallback_index, card in enumerate(cards):
        index = safe_int(card.get("index"), fallback_index)
        if index in excluded_indices:
            continue
        score, reasons = score_card_select_for_operation(card, operation, state)
        scored.append({
            "index": index,
            "id": card.get("id"),
            "name": card.get("name"),
            "score": round(score, 3),
            "reasons": reasons,
            "safe": is_safe_sacrifice_card(card) if operation in ("sacrifice", "transform") else True,
        })

    if not scored:
        if screen.get("can_confirm"):
            return {"action": "confirm_selection"}, {
                "top_actions": [{"action": "confirm_selection", "confidence": 100.0, "marker": "all_remembered_selected"}],
                "chosen_action": "confirm_selection",
                "payload": {"action": "confirm_selection"},
                "reason": "card_select_confirm_after_memory",
            }
        return None, {
            "top_actions": [],
            "chosen_action": None,
            "payload": None,
            "reason": "card_select_waiting_after_memory",
        }

    if operation in ("sacrifice", "transform"):
        safe_scored = [item for item in scored if item.get("safe")]
        if safe_scored:
            scored = safe_scored
        elif screen.get("can_confirm") and selected_count > 0:
            return {"action": "confirm_selection"}, {
                "top_actions": [{"action": "confirm_selection", "confidence": 100.0, "marker": "no_more_safe_cards"}],
                "chosen_action": "confirm_selection",
                "payload": {"action": "confirm_selection"},
                "reason": f"card_select_{operation}_confirm_no_more_safe_targets",
            }
        elif screen.get("can_cancel"):
            return {"action": "cancel_selection"}, {
                "top_actions": [{"action": "cancel_selection", "confidence": 100.0, "marker": "no_safe_card_to_remove"}],
                "chosen_action": "cancel_selection",
                "payload": {"action": "cancel_selection"},
                "reason": f"card_select_{operation}_cancel_no_safe_target",
            }

    best = max(scored, key=lambda item: item["score"])
    payload = {"action": "select_card", "index": best["index"]}
    top_actions = [
        {
            "action": f"select_card:index_{item['index']}:{item.get('id') or item.get('name')}",
            "confidence": round(max(item["score"], 0.0) * 20, 2),
            "marker": f"operation={operation}; target={selected_count}/{target_count}; score={item['score']:+.2f}; {'safe' if item.get('safe') else 'unsafe'}; {' / '.join(item['reasons']) or '-'}",
        }
        for item in sorted(scored, key=lambda item: item["score"], reverse=True)[:6]
    ]
    return payload, {
        "top_actions": top_actions,
        "chosen_action": f"select_card:index_{best['index']}:{best.get('id') or best.get('name')}",
        "payload": payload,
        "reason": f"card_select_{operation}: " + (" / ".join(best["reasons"]) or "highest score"),
    }


def choose_hand_select_action(state):
    screen = state.get("hand_select") or {}
    operation = detect_card_select_operation({
        "screen_type": screen.get("mode"),
        "prompt": screen.get("prompt"),
    })
    remembered_indices = {
        safe_int(index, -1)
        for index in (
            state.get("_selection_selected_indices", [])
            or state.get("_card_select_selected_indices", [])
        )
    }
    observed_indices = {
        safe_int(card.get("index"), -1)
        for card in (screen.get("selected_cards") or [])
        if isinstance(card, dict)
    }
    excluded_indices = {index for index in remembered_indices | observed_indices if index >= 0}
    selected_count = selection_selected_count(state, screen)
    target_count = selection_target_count(screen, "hand_select")

    if screen.get("can_confirm") and selected_count >= target_count:
        return {"action": "combat_confirm_selection"}, {
            "top_actions": [{"action": "combat_confirm_selection", "confidence": 100.0, "marker": "selection_ready"}],
            "chosen_action": "combat_confirm_selection",
            "payload": {"action": "combat_confirm_selection"},
            "reason": f"hand_select_confirm_ready selected={selected_count}/{target_count}",
        }

    cards = screen.get("cards") or []
    if not cards:
        if screen.get("can_confirm"):
            return {"action": "combat_confirm_selection"}, {
                "top_actions": [{"action": "combat_confirm_selection", "confidence": 100.0, "marker": "no_cards_confirm"}],
                "chosen_action": "combat_confirm_selection",
                "payload": {"action": "combat_confirm_selection"},
                "reason": "hand_select_no_cards_confirm",
            }
        return None, {
            "top_actions": [],
            "chosen_action": None,
            "payload": None,
            "reason": "hand_select_no_cards",
        }

    scored = []
    for fallback_index, card in enumerate(cards):
        index = safe_int(card.get("index"), fallback_index)
        if index in excluded_indices:
            continue
        score, reasons = score_card_select_for_operation(card, operation, state)
        scored.append({
            "index": index,
            "id": card.get("id"),
            "name": card.get("name"),
            "score": round(score, 3),
            "reasons": reasons,
            "safe": is_safe_sacrifice_card(card) if operation in ("sacrifice", "transform") else True,
        })

    if not scored:
        if screen.get("can_confirm"):
            return {"action": "combat_confirm_selection"}, {
                "top_actions": [{"action": "combat_confirm_selection", "confidence": 100.0, "marker": "all_remembered_selected"}],
                "chosen_action": "combat_confirm_selection",
                "payload": {"action": "combat_confirm_selection"},
                "reason": "hand_select_confirm_after_memory",
            }
        return None, {
            "top_actions": [],
            "chosen_action": None,
            "payload": None,
            "reason": "hand_select_waiting_after_memory",
        }

    if operation in ("sacrifice", "transform"):
        safe_scored = [item for item in scored if item.get("safe")]
        if safe_scored:
            scored = safe_scored
        elif screen.get("can_confirm") and selected_count > 0:
            return {"action": "combat_confirm_selection"}, {
                "top_actions": [{"action": "combat_confirm_selection", "confidence": 100.0, "marker": "no_more_safe_cards"}],
                "chosen_action": "combat_confirm_selection",
                "payload": {"action": "combat_confirm_selection"},
                "reason": f"hand_select_{operation}_confirm_no_more_safe_targets",
            }

    best = max(scored, key=lambda item: item["score"])
    payload = {"action": "combat_select_card", "card_index": best["index"]}
    top_actions = [
        {
            "action": f"combat_select_card:index_{item['index']}:{item.get('id') or item.get('name')}",
            "confidence": round(max(item["score"], 0.0) * 20, 2),
            "marker": f"operation={operation}; target={selected_count}/{target_count}; score={item['score']:+.2f}; {'safe' if item.get('safe') else 'unsafe'}; {' / '.join(item['reasons']) or '-'}",
        }
        for item in sorted(scored, key=lambda item: item["score"], reverse=True)[:6]
    ]
    return payload, {
        "top_actions": top_actions,
        "chosen_action": f"combat_select_card:index_{best['index']}:{best.get('id') or best.get('name')}",
        "payload": payload,
        "reason": f"hand_select_{operation}: " + (" / ".join(best["reasons"]) or "highest score"),
    }


def choose_bundle_select_action(state):
    screen = state.get("bundle_select") or {}
    if screen.get("preview_showing") and screen.get("can_confirm"):
        return {"action": "confirm_bundle_selection"}, {
            "top_actions": [{"action": "confirm_bundle_selection", "confidence": 100.0, "marker": "bundle_preview_ready"}],
            "chosen_action": "confirm_bundle_selection",
            "payload": {"action": "confirm_bundle_selection"},
            "reason": "bundle_select_confirm_preview",
        }

    scored = []
    for fallback_index, bundle in enumerate(screen.get("bundles") or []):
        if not isinstance(bundle, dict):
            continue
        index = safe_int(bundle.get("index"), fallback_index)
        cards = [card for card in (bundle.get("cards") or []) if isinstance(card, dict)]
        card_scores = []
        reasons = []
        for card in cards:
            score, card_reasons = score_card_select_for_operation(card, "choose", state)
            card_scores.append(score)
            if card_reasons:
                reasons.append(f"{card.get('id') or card.get('name')}: {card_reasons[0]}")
        if card_scores:
            score = sum(card_scores) / len(card_scores) + max(card_scores) * 0.25
        else:
            score = -1.0
            reasons.append("empty bundle")
        scored.append({
            "index": index,
            "score": round(score, 3),
            "card_count": len(cards),
            "reasons": reasons[:3],
        })

    if not scored:
        if screen.get("can_cancel"):
            return {"action": "cancel_bundle_selection"}, {
                "top_actions": [{"action": "cancel_bundle_selection", "confidence": 100.0, "marker": "no_bundles"}],
                "chosen_action": "cancel_bundle_selection",
                "payload": {"action": "cancel_bundle_selection"},
                "reason": "bundle_select_no_bundles",
            }
        return None, {
            "top_actions": [],
            "chosen_action": None,
            "payload": None,
            "reason": "bundle_select_no_bundles",
        }

    best = max(scored, key=lambda item: item["score"])
    payload = {"action": "select_bundle", "index": best["index"]}
    top_actions = [
        {
            "action": f"select_bundle:index_{item['index']}",
            "confidence": round(max(item["score"], 0.0) * 20, 2),
            "marker": f"cards={item['card_count']}; score={item['score']:+.2f}; {' / '.join(item['reasons']) or '-'}",
        }
        for item in sorted(scored, key=lambda item: item["score"], reverse=True)[:6]
    ]
    return payload, {
        "top_actions": top_actions,
        "chosen_action": f"select_bundle:index_{best['index']}",
        "payload": payload,
        "reason": "bundle_select_choose: " + (" / ".join(best["reasons"]) or "highest bundle value"),
    }


def shop_price(item):
    return safe_num(item.get("price") or item.get("cost"), 9999.0)


def shop_item_category(item):
    return str(item.get("category") or item.get("type") or "").lower()


def card_from_shop_item(item):
    return {
        "id": item.get("card_id") or item.get("id"),
        "name": item.get("card_name") or item.get("name"),
        "type": item.get("card_type") or item.get("type"),
        "cost": item.get("card_cost") or item.get("cost"),
        "star_cost": item.get("card_star_cost"),
        "description": item.get("card_description") or item.get("description"),
        "rarity": item.get("card_rarity") or item.get("rarity"),
        "keywords": item.get("keywords") or [],
        "is_upgraded": item.get("is_upgraded", False),
    }


def shop_item_blob(item):
    parts = []
    for key in (
        "category", "type", "item_id", "item_name",
        "card_id", "card_name", "card_description", "card_rarity",
        "relic_id", "relic_name", "relic_description",
        "potion_id", "potion_name", "potion_description",
    ):
        value = item.get(key)
        if value is not None:
            parts.append(str(value))
    keywords = item.get("keywords")
    if keywords:
        try:
            parts.append(json.dumps(keywords, ensure_ascii=False))
        except Exception:
            parts.append(str(keywords))
    return " ".join(parts).lower()


def best_deck_sacrifice_candidate(state):
    deck = ((state.get("player") or {}).get("deck") or [])
    best = None
    for card in deck:
        if not isinstance(card, dict):
            continue
        score, reasons = card_sacrifice_priority(card, state)
        item = {"card": card, "score": score, "reasons": reasons}
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def best_safe_deck_sacrifice_candidate(state):
    deck = ((state.get("player") or {}).get("deck") or [])
    best = None
    for card in deck:
        if not isinstance(card, dict) or not is_safe_sacrifice_card(card):
            continue
        score, reasons = card_sacrifice_priority(card, state)
        item = {"card": card, "score": score, "reasons": reasons}
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def best_shop_removal_candidate(state):
    deck = ((state.get("player") or {}).get("deck") or [])
    best = None
    for card in deck:
        if not isinstance(card, dict) or not is_status_or_curse_card(card):
            continue
        score, reasons = card_sacrifice_priority(card, state)
        item = {"card": card, "score": score, "reasons": reasons}
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def score_shop_card(item, state):
    card = card_from_shop_item(item)
    profile = deck_profile_from_state(state)
    reward_score, reward_reasons = score_reward_card(card, profile, state)
    value, value_reasons = card_long_term_value(card)
    price = shop_price(item)
    score = reward_score + value * 0.35 - price / 75.0
    reasons = reward_reasons[:2] + value_reasons[:1]
    if item.get("on_sale"):
        score += 0.7
        reasons.append("打折")
    if score < 0.4 and safe_int(profile.get("total")) >= 20:
        score -= 0.8
        reasons.append("避免牌组变厚")
    return round(score, 3), reasons[:4]


def score_shop_relic(item, state):
    text = shop_item_blob(item)
    player = state.get("player") or {}
    gold = safe_num(player.get("gold"), 0.0)
    price = shop_price(item)
    score = 3.2 - price / 115.0
    reasons = ["遗物收益"]

    if has_any(text, ("energy", "能量")):
        score += 2.4
        reasons.append("能量")
    if has_any(text, ("draw", "card reward", "抽", "抽牌", "卡牌奖励")):
        score += 1.6
        reasons.append("过牌/奖励")
    if has_any(text, ("heal", "recover", "回复", "治疗", "生命")):
        score += 1.2
        reasons.append("续航")
    if has_any(text, ("elite", "boss", "精英", "boss")):
        score += 1.0
        reasons.append("关键战收益")
    if has_any(text, ("strength", "damage", "力量", "伤害")):
        score += 0.9
        reasons.append("输出收益")
    if price <= gold * 0.45:
        score += 0.5
        reasons.append("价格可接受")
    if price > gold * 0.85:
        score -= 0.8
        reasons.append("消耗金币过多")
    return round(score, 3), reasons[:4]


def score_shop_potion(item, state):
    text = shop_item_blob(item)
    player = state.get("player") or {}
    potions = player.get("potions") or []
    hp = safe_num(player.get("hp"), 0.0)
    max_hp = max(safe_num(player.get("max_hp"), 1.0), 1.0)
    hp_ratio = hp / max_hp
    price = shop_price(item)
    score = 1.0 - price / 55.0
    reasons = ["药水"]

    if len(potions) >= 3:
        score -= 3.0
        reasons.append("药水位可能已满")
    if has_any(text, ("damage", "attack", "strength", "vulnerable", "伤害", "攻击", "力量", "易伤")):
        score += 1.8
        reasons.append("战斗爆发")
    if has_any(text, ("block", "armor", "格挡", "护甲", "防御")):
        score += 1.2
        reasons.append("保命")
    if has_any(text, ("draw", "energy", "抽牌", "能量")):
        score += 1.5
        reasons.append("启动")
    if hp_ratio < 0.45:
        score += 0.8
        reasons.append("低血量备药")
    return round(score, 3), reasons[:4]


def score_shop_removal(item, state):
    profile = deck_profile_from_state(state)
    deck_size = safe_int(profile.get("total"))
    best = best_shop_removal_candidate(state)
    price = shop_price(item)
    score = -price / 90.0
    reasons = ["删牌"]

    if best:
        score += max(best["score"], 0.0) * 0.9
        name = (best["card"].get("name") or best["card"].get("id") or "低价值牌")
        reasons.append(f"可删{name}")
    if deck_size >= 18:
        score += 1.0
        reasons.append("牌组偏厚")
    if deck_size >= 24:
        score += 0.8
        reasons.append("压缩牌组")
    if not best:
        score -= 99.0
        reasons.append("无诅咒不花钱删牌")
    elif best["score"] < 2.0:
        score -= 1.5
        reasons.append("缺少明显垃圾牌")
    return round(score, 3), reasons[:4]


def score_shop_item(item, state):
    category = shop_item_category(item)
    if category == "card":
        return score_shop_card(item, state)
    if category == "relic":
        return score_shop_relic(item, state)
    if category == "potion":
        return score_shop_potion(item, state)
    if category == "card_removal":
        return score_shop_removal(item, state)
    return -99.0, ["未知商品类型"]


def rank_shop_items(state, state_type, category_filter=None):
    ranked = []
    for fallback_index, item in enumerate(get_items_for_state(state, state_type)):
        if not item.get("is_stocked", True) or not item.get("can_afford", True):
            continue
        if category_filter and shop_item_category(item) != category_filter:
            continue
        score, reasons = score_shop_item(item, state)
        if (category_filter is None or category_filter == "card_removal") and score < 0.75:
            continue
        ranked.append({
            "fallback_index": fallback_index,
            "item": item,
            "score": score,
            "reasons": reasons,
            "category": shop_item_category(item),
        })
    ranked.sort(key=lambda row: (
        row["score"],
        1 if row["item"].get("on_sale") else 0,
        -shop_price(row["item"]),
    ), reverse=True)
    return ranked


def choose_shop_rule_action(state, state_type, exploration=None):
    ranked = [row for row in rank_shop_items(state, state_type) if row["category"] != "card_removal"]
    if ranked:
        best = ranked[0]
        if exploration and exploration.get("enabled"):
            selected, _, _ = sample_ranked_entry(
                ranked,
                exploration.get("macro_epsilon", 0.0),
                exploration.get("top_k", 1),
                exploration.get("temperature", 1.0),
                score_key="score",
            )
            if selected:
                best = selected
        payload = {"action": "shop_purchase", "index": int(best["item"].get("index", best["fallback_index"]))}
        top_actions = [
            {
                "action": f"shop_purchase:index_{int(row['item'].get('index', row['fallback_index']))}:{row['category']}",
                "confidence": round(max(row["score"], 0.0) * 20, 2),
                "marker": f"shop_score={row['score']:+.2f}; price={shop_price(row['item']):.0f}; {' / '.join(row['reasons']) or '-'}",
            }
            for row in ranked[:6]
        ]
        return payload, {
            "top_actions": top_actions,
            "chosen_action": f"shop_purchase:index_{payload['index']}:{best['category']}",
            "payload": payload,
            "reason": "shop_rule: " + (" / ".join(best["reasons"]) or "highest score"),
        }

    if get_can_proceed(state, state_type):
        payload = {"action": "proceed"}
        return payload, {
            "top_actions": [{"action": "proceed", "confidence": 100.0, "marker": "no_positive_shop_purchase"}],
            "chosen_action": "proceed",
            "payload": payload,
            "reason": "shop_rule_no_positive_purchase",
        }
    if state_type in ("shop", "fake_merchant"):
        payload = {"action": "proceed"}
        return payload, {
            "top_actions": [{"action": "proceed", "confidence": 80.0, "marker": "close_shop_inventory"}],
            "chosen_action": "proceed",
            "payload": payload,
            "reason": "shop_rule_close_inventory",
        }
    return None, {
        "top_actions": [],
        "chosen_action": None,
        "payload": None,
        "reason": "shop_rule_no_action",
    }


def choose_card_reward_mixed_action(macro_agent, state, outputs, probs, card_baseline_weight, exploration=None):
    scorer_mode = option_card_scorer_mode()
    if scorer_mode in ("active", "active_canary"):
        return choose_card_reward_baseline_action(state, card_baseline_weight)
    shadow_mode, shadow_result, shadow_error = build_shadow_card_result(state, mode=scorer_mode)
    shadow_info = card_scorer_info_payload(shadow_mode, shadow_result, shadow_error)
    baseline_entries, profile = card_reward_baseline_entries(state)
    baseline_by_label = {entry["label"]: entry for entry in baseline_entries}
    baseline_by_payload = {json.dumps(entry["payload"], sort_keys=True): entry for entry in baseline_entries}
    ranked = []

    for idx, logit in enumerate(outputs):
        label = macro_agent["id_to_action"].get(idx, "UNKNOWN")
        if label in ("UNKNOWN", "PAD"):
            continue
        if not (label.startswith("choose_card:") or label == "skip_reward"):
            continue
        payload, status = macro_label_to_payload(label, state)
        if not payload:
            continue
        entry = baseline_by_label.get(label) or baseline_by_payload.get(json.dumps(payload, sort_keys=True))
        baseline_score = safe_num((entry or {}).get("score"))
        reasons = (entry or {}).get("reasons", [])
        adjusted = float(logit.item()) + card_baseline_weight * baseline_score
        ranked.append({
            "label": label,
            "payload": payload,
            "status": status,
            "prob": float(probs[idx].item()) * 100.0,
            "baseline_score": baseline_score,
            "adjusted": adjusted,
            "reasons": reasons,
        })

    if not ranked:
        return choose_card_reward_baseline_action(state, card_baseline_weight)

    ranked.sort(key=lambda item: item["adjusted"], reverse=True)
    best = ranked[0]
    if exploration and exploration.get("enabled"):
        selected, _, selected_rank = sample_ranked_entry(
            ranked,
            exploration.get("macro_epsilon", 0.0),
            exploration.get("top_k", 1),
            exploration.get("temperature", 1.0),
            score_key="adjusted",
        )
        if selected:
            best = selected
            if selected_rank > 0:
                best["reasons"] = list(best["reasons"]) + [f"explore_rank={selected_rank}"]
    top_actions = [
        {
            "action": item["label"],
            "confidence": round(item["prob"], 2),
            "marker": (
                f"{item['status']}; baseline={item['baseline_score']:+.2f}; "
                f"mixed={item['adjusted']:+.2f}; {' / '.join(item['reasons']) or '-'}"
            ),
        }
        for item in ranked[:6]
    ]
    return best["payload"], {
        "top_actions": top_actions,
        "chosen_action": best["label"],
        "payload": best["payload"],
        "reason": "card_reward_mixed: " + (" / ".join(best["reasons"]) or best["status"]),
        "deck_profile": public_deck_profile(profile),
        "reward_baseline": {
            "mode": "model_plus_baseline",
            "weight": card_baseline_weight,
            "chosen_score": round(best["baseline_score"], 3),
            "chosen_mixed_score": round(best["adjusted"], 3),
            "chosen_reason": best["reasons"],
            "card_scorer_mode": scorer_mode,
        },
        "card_scorer": shadow_info,
        "archetype_consistency": (shadow_info or {}).get("archetype_consistency"),
    }


def event_option_blob(option):
    parts = []
    for key in ("title", "description", "relic_name", "relic_description"):
        value = option.get(key)
        if value is not None:
            parts.append(str(value))
    keywords = option.get("keywords")
    if keywords:
        try:
            parts.append(json.dumps(keywords, ensure_ascii=False))
        except Exception:
            parts.append(str(keywords))
    return " ".join(parts).lower()


def estimate_event_hp_cost(text):
    if not has_any(text, ("hp", "health", "生命", "血")):
        return 0
    if not has_any(text, ("lose", "loss", "失去", "扣", "支付", "花费", "付出", "take", "受到")):
        return 0
    nums = combat_numbers(text)
    return nums[0] if nums else 0


def score_event_option_rule(option, state):
    text = event_option_blob(option)
    player = state.get("player") or {}
    hp = safe_num(player.get("hp"), 0.0)
    max_hp = max(safe_num(player.get("max_hp"), 1.0), 1.0)
    hp_ratio = hp / max_hp
    gold = safe_num(player.get("gold"), 0.0)
    score = 0.0
    reasons = []

    if option.get("is_proceed"):
        score += 0.4
        reasons.append("继续")

    if has_any(text, ("relic", "遗物")):
        score += 3.0
        reasons.append("遗物收益")
    if has_any(text, ("potion", "药水")):
        score += 1.2
        reasons.append("药水收益")
    if has_any(text, ("gold", "金币", "金钱")) and has_any(text, ("gain", "获得", "拾起")):
        score += 1.5
        reasons.append("金币收益")
    if has_any(text, ("max hp", "最大生命")) and has_any(text, ("gain", "获得", "提升", "增加")):
        score += 2.0
        reasons.append("最大生命收益")
    if has_any(text, ("upgrade", "升级", "强化", "附魔")):
        score += 1.6
        reasons.append("卡牌强化")
    if has_any(text, ("choose", "选择", "add", "加入", "card pack", "卡牌包")) and has_any(text, ("card", "牌")):
        score += 1.0
        reasons.append("卡牌收益")

    hp_cost = estimate_event_hp_cost(text)
    if hp_cost:
        penalty = hp_cost / max_hp * (3.0 if hp_ratio < 0.45 else 1.4)
        if hp - hp_cost <= max(6.0, max_hp * 0.12):
            penalty += 8.0
        elif hp_ratio < 0.35:
            penalty += 3.0
        score -= penalty
        reasons.append(f"扣血成本{hp_cost}")

    if has_any(text, ("max hp", "最大生命")) and has_any(text, ("lose", "失去", "降低", "减少")):
        score -= 4.0
        reasons.append("最大生命损失")
    if has_any(text, ("lose all gold", "失去所有金币", "失去全部金币")):
        score -= min(4.0, gold / 80.0)
        reasons.append("失去金币")

    card_loss = has_any(text, (
        "remove", "lose a card", "lose 1 card", "sell", "sacrifice",
        "移除", "失去一张牌", "失去1张牌", "出售", "卖掉", "卖出", "献祭", "discard", "丢弃", "抛弃", "弃掉",
    ))
    transform = has_any(text, ("transform", "变换", "变化为", "变形", "转化"))
    if card_loss or transform:
        if has_any(text, ("starter", "初始牌", "打击", "防御", "strike", "defend")):
            score -= 0.6
            reasons.append("只动初始牌")
        else:
            best_sacrifice = best_deck_sacrifice_candidate(state)
            has_obvious_junk = bool(best_sacrifice and best_sacrifice["score"] >= 3.0)
            penalty = 2.0 if has_obvious_junk else (3.5 if hp_ratio >= 0.45 else 1.6)
            score -= penalty
            reasons.append("可控删牌" if has_obvious_junk else "可能损失核心牌")

    if has_any(text, ("next", "again", "reroll", "下一", "再选", "重抽", "继续选择")):
        score += 0.5
        reasons.append("可继续选择")
    if has_any(text, ("leave", "ignore", "skip", "离开", "无视", "跳过")):
        score += 0.2
        reasons.append("安全离开")

    return round(score, 3), reasons[:4]


def event_option_has_cost_risk(option):
    text = event_option_blob(option)
    if has_any(text, (
        "lose", "loss", "失去", "remove", "移除", "sell", "出售", "卖", "sacrifice", "献祭",
        "hp", "生命", "血", "支付", "花费", "付出", "discard", "丢弃", "抛弃", "弃掉",
    )):
        return True
    if has_any(text, ("transform", "变换", "变化为", "变形", "转化")) and not has_any(text, ("starter", "初始牌", "打击", "防御", "strike", "defend")):
        return True
    return False


def choose_event_option_rule_action(state, exploration=None):
    event = state.get("event") or {}
    if event.get("in_dialogue"):
        return {"action": "advance_dialogue"}, {
            "top_actions": [{"action": "advance_dialogue", "confidence": 100.0, "marker": "event_dialogue"}],
            "chosen_action": "advance_dialogue",
            "payload": {"action": "advance_dialogue"},
            "reason": "event_dialogue",
        }

    options = event.get("options") or []
    available = [o for o in options if not o.get("is_locked") and not o.get("was_chosen")]
    if len(available) == 1:
        index = safe_int(available[0].get("index"), 0)
        payload = {"action": "choose_event_option", "index": index}
        return payload, {
            "top_actions": [{"action": f"choose_event_option:index_{index}", "confidence": 100.0, "marker": "single_option"}],
            "chosen_action": f"choose_event_option:index_{index}",
            "payload": payload,
            "reason": "event_single_option",
        }
    if not available:
        return None, None

    scored = []
    for fallback_index, option in enumerate(available):
        index = safe_int(option.get("index"), fallback_index)
        score, reasons = score_event_option_rule(option, state)
        scored.append({
            "index": index,
            "title": option.get("title"),
            "score": score,
            "reasons": reasons,
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0]
    if exploration and exploration.get("enabled"):
        selected, _, selected_rank = sample_ranked_entry(
            scored,
            exploration.get("macro_epsilon", 0.0),
            exploration.get("top_k", 1),
            exploration.get("temperature", 1.0),
            score_key="score",
        )
        if selected:
            best = selected
            if selected_rank > 0:
                best["reasons"] = list(best["reasons"]) + [f"explore_rank={selected_rank}"]
    payload = {"action": "choose_event_option", "index": best["index"]}
    top_actions = [
        {
            "action": f"choose_event_option:index_{item['index']}:{item.get('title')}",
            "confidence": round(max(item["score"], 0.0) * 20, 2),
            "marker": f"rule_score={item['score']:+.2f}; {' / '.join(item['reasons']) or '-'}",
        }
        for item in sorted(scored, key=lambda item: item["score"], reverse=True)[:6]
    ]
    return payload, {
        "top_actions": top_actions,
        "chosen_action": f"choose_event_option:index_{best['index']}:{best.get('title')}",
        "payload": payload,
        "reason": "event_general_rule: " + (" / ".join(best["reasons"]) or "highest score"),
    }


def choose_crystal_sphere_rule_action(state, exploration=None):
    crystal = state.get("crystal_sphere") or {}
    tool = str(crystal.get("tool") or "").lower()
    can_proceed = bool(crystal.get("can_proceed"))
    divinations_left_text = str(crystal.get("divinations_left_text") or "")
    cells = crystal.get("cells") or []
    clickable_cells = crystal.get("clickable_cells") or [
        cell for cell in cells
        if isinstance(cell, dict) and cell.get("is_clickable")
    ]
    revealed_items = crystal.get("revealed_items") or [
        cell for cell in cells
        if isinstance(cell, dict) and cell.get("item_type")
    ]
    grid_width = max(safe_int(crystal.get("grid_width"), 11), 1)
    grid_height = max(safe_int(crystal.get("grid_height"), 11), 1)
    center_x = (grid_width - 1) / 2.0
    center_y = (grid_height - 1) / 2.0
    divinations_match = re.search(r"-?\d+", divinations_left_text)
    divinations_left = safe_int(divinations_match.group(0), 999) if divinations_match else 999

    if can_proceed or divinations_left <= 0:
        payload = {"action": "crystal_sphere_proceed"}
        reason = "crystal_sphere_can_proceed" if can_proceed else "crystal_sphere_no_divinations_left"
        return payload, {
            "top_actions": [{"action": "crystal_sphere_proceed", "confidence": 100.0, "marker": reason}],
            "chosen_action": "crystal_sphere_proceed",
            "payload": payload,
            "reason": reason,
        }

    if tool != "big":
        payload = {"action": "crystal_sphere_set_tool", "tool": "big"}
        return payload, {
            "top_actions": [{"action": "crystal_sphere_set_tool:big", "confidence": 100.0, "marker": "prefer_big_tool"}],
            "chosen_action": "crystal_sphere_set_tool:big",
            "payload": payload,
            "reason": "crystal_sphere_set_big_tool",
        }

    if not clickable_cells:
        return None, None

    def cell_priority(cell):
        x = safe_num(cell.get("x"))
        y = safe_num(cell.get("y"))
        return ((x - center_x) ** 2 + (y - center_y) ** 2, y, x)

    ranked_cells = sorted(clickable_cells, key=cell_priority)
    best_cell = ranked_cells[0]
    payload = {
        "action": "crystal_sphere_click_cell",
        "x": int(best_cell.get("x", 0)),
        "y": int(best_cell.get("y", 0)),
    }
    top_actions = [
        {
            "action": f"crystal_sphere_click_cell:{int(cell.get('x', 0))},{int(cell.get('y', 0))}",
            "confidence": round(max(1.0, 100.0 - cell_priority(cell)[0] * 4.0), 2),
            "marker": "clickable",
        }
        for cell in ranked_cells[:6]
    ]
    if revealed_items:
        top_actions[0]["marker"] = f"revealed={len(revealed_items)}"
    return payload, {
        "top_actions": top_actions,
        "chosen_action": f"crystal_sphere_click_cell:{int(best_cell.get('x', 0))},{int(best_cell.get('y', 0))}",
        "payload": payload,
        "reason": f"crystal_sphere_click_center revealed={len(revealed_items)} clickable={len(clickable_cells)}",
    }


def route_type_score(node_type, hp_ratio, gold, floor=0):
    node_type = str(node_type or "").lower()
    if "boss" in node_type:
        return 0.0
    if "elite" in node_type:
        if hp_ratio < 0.55:
            return -14.0
        if floor and floor <= 7:
            return 0.5 if hp_ratio >= 0.85 else -5.0
        if hp_ratio >= 0.80:
            return 4.0
        if hp_ratio >= 0.65:
            return -1.5
        return -8.0
    if "rest" in node_type or "camp" in node_type:
        if hp_ratio < 0.35:
            return 10.0
        if hp_ratio < 0.55:
            return 7.0
        if hp_ratio < 0.70:
            return 4.0
        return 1.0
    if "shop" in node_type or "merchant" in node_type:
        if hp_ratio < 0.35:
            return 4.0 if gold >= 75 else 1.5
        if gold >= 180:
            return 3.5
        if gold >= 110:
            return 2.0
        if gold >= 75:
            return 1.0
        return -1.5
    if "treasure" in node_type:
        return 4.0 if hp_ratio < 0.45 else 2.5
    if "event" in node_type or "unknown" in node_type or "ancient" in node_type or "?" in node_type:
        if hp_ratio < 0.35:
            return 8.0
        if hp_ratio < 0.55:
            return 5.5
        if floor and floor <= 7:
            return 4.0
        return 3.0
    if "monster" in node_type:
        if hp_ratio < 0.25:
            return -8.0
        if hp_ratio < 0.35:
            return -5.0
        if hp_ratio < 0.55:
            return -3.0
        if hp_ratio < 0.70:
            return -1.0
        return -0.5 if floor and floor <= 7 else 1.0
    return 0.5


def map_node_key(node):
    return (safe_int(node.get("col"), -999), safe_int(node.get("row"), -999))


def map_node_type(node):
    return str((node or {}).get("type") or "").lower()


def is_route_buffer_node(node_type, gold):
    return (
        "rest" in node_type
        or "camp" in node_type
        or (("shop" in node_type or "merchant" in node_type) and gold >= 75)
    )


def route_elite_risk(option, map_state, gold, max_depth=7):
    nodes = {
        map_node_key(node): node
        for node in (map_state.get("nodes") or [])
        if isinstance(node, dict)
    }
    start_key = map_node_key(option)

    def scan(key, depth, buffered, path):
        if depth > max_depth or key in path:
            return None, None
        node = nodes.get(key)
        if not node:
            return None, None
        node_type = map_node_type(node)
        if buffered:
            return None, None
        if "elite" in node_type:
            return depth, depth
        if is_route_buffer_node(node_type, gold):
            return None, None

        child_keys = []
        for child in node.get("children", []) or []:
            if isinstance(child, (list, tuple)) and len(child) >= 2:
                child_keys.append((safe_int(child[0], -999), safe_int(child[1], -999)))
        if not child_keys:
            return None, None

        nearest = []
        forced = []
        for child_key in child_keys:
            child_nearest, child_forced = scan(child_key, depth + 1, False, path | {key})
            if child_nearest is not None:
                nearest.append(child_nearest)
            if child_forced is not None:
                forced.append(child_forced)

        nearest_depth = min(nearest) if nearest else None
        forced_depth = min(forced) if forced and len(forced) == len(child_keys) else None
        return nearest_depth, forced_depth

    return scan(start_key, 0, False, set())


def route_lookahead_adjustment(option, map_state, hp_ratio, gold, floor, max_depth=4):
    nodes = {
        map_node_key(node): node
        for node in (map_state.get("nodes") or [])
        if isinstance(node, dict)
    }
    start_key = map_node_key(option)
    start_node = nodes.get(start_key, option)
    stack = []
    for child in start_node.get("children", []) or []:
        if isinstance(child, (list, tuple)) and len(child) >= 2:
            stack.append(((safe_int(child[0], -999), safe_int(child[1], -999)), 1, False))

    score = 0.0
    markers = []
    nearest_elite_depth, forced_elite_depth = route_elite_risk(option, map_state, gold, max_depth=7)
    if forced_elite_depth is not None and floor + forced_elite_depth <= 8:
        penalty = 32.0 / max(forced_elite_depth, 1)
        if hp_ratio < 0.95:
            penalty += 8.0 / max(forced_elite_depth, 1)
        score -= penalty
        markers.append(f"forced_elite_d{forced_elite_depth}")
    elif nearest_elite_depth is not None and floor + nearest_elite_depth <= 8:
        score -= 8.0 / max(nearest_elite_depth, 1)
        markers.append(f"early_elite_d{nearest_elite_depth}")
    seen = set()
    while stack:
        key, depth, has_rest_buffer = stack.pop()
        if depth > max_depth or (key, depth, has_rest_buffer) in seen:
            continue
        seen.add((key, depth, has_rest_buffer))
        node = nodes.get(key)
        if not node:
            continue
        node_type = map_node_type(node)
        weight = 0.16 / max(depth, 1)
        score += route_type_score(node_type, hp_ratio, gold, floor + depth) * weight

        is_rest = "rest" in node_type or "camp" in node_type
        is_shop = "shop" in node_type or "merchant" in node_type
        is_buffer = is_rest or (is_shop and gold >= 75)
        buffered = has_rest_buffer or is_buffer
        if "elite" in node_type and not has_rest_buffer:
            if hp_ratio < 0.55:
                penalty = 14.0 / max(depth, 1)
            elif hp_ratio < 0.70:
                penalty = 8.0 / max(depth, 1)
            elif floor <= 7 and hp_ratio < 0.95:
                penalty = 12.0 / max(depth, 1)
            else:
                penalty = 2.5 / max(depth, 1)
            score -= penalty
            markers.append(f"elite_d{depth}")
        if is_rest and hp_ratio < 0.65:
            score += 2.0 / max(depth, 1)
            markers.append(f"rest_d{depth}")
        if is_shop and gold >= 75 and hp_ratio < 0.90:
            score += 1.5 / max(depth, 1)
            markers.append(f"shop_d{depth}")

        for child in node.get("children", []) or []:
            if isinstance(child, (list, tuple)) and len(child) >= 2:
                stack.append(((safe_int(child[0], -999), safe_int(child[1], -999)), depth + 1, buffered))

    return score, ",".join(markers[:4])


def choose_map_route_action(state, exploration=None):
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
    map_state = state.get("map") or {}

    scored = []
    for fallback_index, option in enumerate(options):
        node_type = option.get("type")
        score = route_type_score(node_type, hp_ratio, gold, floor)
        leads = option.get("leads_to") or []
        score += min(len(leads), 3) * 0.25
        for lead in leads[:4]:
            score += route_type_score(lead.get("type"), hp_ratio, gold, floor + 1) * 0.35
        lookahead_score, lookahead_marker = route_lookahead_adjustment(option, map_state, hp_ratio, gold, floor)
        score += lookahead_score
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
            "lookahead": lookahead_marker,
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    best_score = scored[0]["score"]
    tied = [item for item in scored if best_score - item["score"] <= 0.25]
    if len(tied) > 1:
        center = sum(safe_num(o.get("col")) for o in options) / len(options)
        if floor % 2:
            best = max(tied, key=lambda item: (item["leads"], safe_num(item["col"]) - center))
        else:
            best = max(tied, key=lambda item: (item["leads"], center - safe_num(item["col"])))
    else:
        best = scored[0]
    if exploration and exploration.get("enabled"):
        selected, _, selected_rank = sample_ranked_entry(
            scored,
            exploration.get("macro_epsilon", 0.0),
            exploration.get("top_k", 1),
            exploration.get("temperature", 1.0),
            score_key="score",
        )
        if selected:
            best = selected
            if selected_rank > 0:
                best["lookahead"] = ",".join(x for x in [best.get("lookahead"), f"explore_rank={selected_rank}"] if x)

    payload = {"action": "choose_map_node", "index": best["index"]}
    top_actions = [
        {
            "action": f"route:index_{item['index']}:{item['type']}",
            "confidence": item["score"],
            "marker": f"col={item['col']} leads={item['leads']} {item['lookahead']}",
        }
        for item in scored[:6]
    ]
    return payload, {
        "top_actions": top_actions,
        "chosen_action": f"route:index_{best['index']}:{best['type']}",
        "payload": payload,
        "reason": f"route_score hp={hp_ratio:.2f} gold={int(gold)}",
    }


def macro_state_signature(state):
    state_type = str(state.get("state_type") or "").lower()
    run = state.get("run") or {}
    player = state.get("player") or {}
    payload = {
        "state_type": state_type,
        "act": run.get("act"),
        "floor": run.get("floor"),
    }
    if state_type == "map":
        payload["options"] = [
            (o.get("index"), o.get("col"), o.get("row"), o.get("type"))
            for o in ((state.get("map") or {}).get("next_options") or [])
        ]
    elif state_type == "rewards":
        player = state.get("player") or {}
        potions = filled_potions(player)
        payload["items"] = [
            (i.get("index"), i.get("type"), i.get("description"))
            for i in ((state.get("rewards") or {}).get("items") or [])
        ]
        payload["can_proceed"] = (state.get("rewards") or {}).get("can_proceed")
        payload["potion_slots_filled"] = max(
            safe_int(player.get("potion_slots_filled"), len(potions)),
            len(potions),
        )
        payload["potion_slots_capacity"] = potion_slot_capacity(player)
        payload["potions"] = [
            (p.get("slot"), p.get("id") or p.get("potion_id"), p.get("name") or p.get("potion_name"))
            for p in potions
        ]
    elif state_type == "card_reward":
        payload["cards"] = [
            (c.get("index"), c.get("id"), c.get("name"))
            for c in ((state.get("card_reward") or {}).get("cards") or [])
        ]
        payload["can_skip"] = (state.get("card_reward") or {}).get("can_skip")
    elif state_type == "card_select":
        card_select = state.get("card_select") or {}
        payload["screen_type"] = card_select.get("screen_type")
        payload["prompt"] = card_select.get("prompt")
        payload["can_confirm"] = card_select.get("can_confirm")
        payload["can_cancel"] = card_select.get("can_cancel")
        payload["cards"] = [
            (c.get("index"), c.get("id"), c.get("name"))
            for c in (card_select.get("cards") or [])
        ]
    elif state_type == "hand_select":
        hand_select = state.get("hand_select") or {}
        payload["mode"] = hand_select.get("mode")
        payload["prompt"] = hand_select.get("prompt")
        payload["can_confirm"] = hand_select.get("can_confirm")
        payload["selected_count"] = selection_selected_count(state, hand_select)
        payload["target_count"] = selection_target_count(hand_select, "hand_select")
        payload["cards"] = [
            (c.get("index"), c.get("id"), c.get("name"))
            for c in (hand_select.get("cards") or [])
        ]
    elif state_type == "bundle_select":
        bundle_select = state.get("bundle_select") or {}
        payload["screen_type"] = bundle_select.get("screen_type")
        payload["prompt"] = bundle_select.get("prompt")
        payload["can_confirm"] = bundle_select.get("can_confirm")
        payload["can_cancel"] = bundle_select.get("can_cancel")
        payload["preview_showing"] = bundle_select.get("preview_showing")
        payload["bundles"] = [
            (
                b.get("index"),
                [
                    (c.get("index"), c.get("id"), c.get("name"))
                    for c in (b.get("cards") or [])
                    if isinstance(c, dict)
                ],
            )
            for b in (bundle_select.get("bundles") or [])
            if isinstance(b, dict)
        ]
    elif state_type == "event":
        payload["options"] = [
            (o.get("index"), o.get("title"), o.get("is_locked"), o.get("is_proceed"), o.get("was_chosen"))
            for o in ((state.get("event") or {}).get("options") or [])
        ]
        payload["in_dialogue"] = (state.get("event") or {}).get("in_dialogue")
    elif state_type == "crystal_sphere":
        crystal = state.get("crystal_sphere") or {}
        payload["tool"] = crystal.get("tool")
        payload["can_proceed"] = crystal.get("can_proceed")
        payload["clickable_count"] = len(crystal.get("clickable_cells") or [])
        payload["revealed_count"] = len(crystal.get("revealed_items") or [])
        payload["divinations_left_text"] = crystal.get("divinations_left_text")
    elif state_type == "rest_site":
        rest_site = state.get("rest_site") or {}
        payload["options"] = [
            (o.get("index"), o.get("id"), o.get("is_enabled"))
            for o in (rest_site.get("options") or [])
        ]
        payload["can_proceed"] = rest_site.get("can_proceed")
        payload["hp"] = player.get("hp")
        payload["max_hp"] = player.get("max_hp")
    elif state_type in ("shop", "fake_merchant"):
        payload["items"] = [
            (i.get("index"), i.get("category"), i.get("price") or i.get("cost"), i.get("can_afford"), i.get("is_stocked"))
            for i in get_items_for_state(state, state_type)
        ]
        payload["can_proceed"] = get_can_proceed(state, state_type)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def choose_macro_action(macro_agent, state, allow_shop=False, card_baseline_weight=0.35, exploration=None):
    state_type = str(state.get("state_type") or "").lower()
    constraint_mode = str((exploration or {}).get("constraint_mode") or "guarded").lower()
    if constraint_mode not in ("guarded", "explore", "free"):
        constraint_mode = "guarded"
    prefer_model = constraint_mode in ("explore", "free") and macro_agent
    effective_allow_shop = bool(allow_shop or constraint_mode in ("explore", "free"))
    effective_card_baseline_weight = 0.0 if constraint_mode == "free" else (min(card_baseline_weight, 0.10) if constraint_mode == "explore" else card_baseline_weight)

    if state_type == "map":
        return choose_map_route_action(state, exploration=exploration)
    if state_type in ("rewards", "treasure"):
        return choose_reward_rule_action(state, state_type)
    if state_type == "card_select":
        return choose_card_select_action(state)
    if state_type == "hand_select":
        return choose_hand_select_action(state)
    if state_type == "bundle_select":
        return choose_bundle_select_action(state)
    if state_type == "event":
        event_payload, event_info = choose_event_option_rule_action(state, exploration=exploration)
        if event_payload:
            return event_payload, event_info
    if state_type == "crystal_sphere":
        crystal_payload, crystal_info = choose_crystal_sphere_rule_action(state, exploration=exploration)
        if crystal_payload:
            return crystal_payload, crystal_info
    if state_type == "rest_site" and not prefer_model:
        rest_payload, rest_info = choose_rest_site_rule_action(state, exploration=exploration)
        if rest_payload:
            return rest_payload, rest_info
    if state_type in ("shop", "fake_merchant") and effective_allow_shop:
        shop_payload, shop_info = choose_shop_rule_action(state, state_type, exploration=exploration)
        if shop_payload:
            return shop_payload, shop_info

    if not macro_agent:
        if state_type == "card_reward":
            return choose_card_reward_baseline_action(state, effective_card_baseline_weight)
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

    if state_type == "card_reward":
        return choose_card_reward_mixed_action(macro_agent, state, outputs[0], probs, effective_card_baseline_weight, exploration=exploration)

    top_actions = []
    ranked_actions = []
    for idx in sorted_indices:
        label = macro_agent["id_to_action"].get(idx.item(), "UNKNOWN")
        if label in ("UNKNOWN", "PAD"):
            continue
        payload, status = macro_label_to_payload(label, state, allow_shop=effective_allow_shop)
        conf = probs[idx].item() * 100
        top_actions.append({"action": label, "confidence": round(conf, 2), "marker": status})
        if payload:
            ranked_actions.append({
                "label": label,
                "payload": payload,
                "status": status,
                "score": float(outputs[0][idx].item()),
            })
        if len(top_actions) >= 6:
            break

    chosen_payload = None
    chosen_label = None
    chosen_reason = "no_legal_macro_action"
    if ranked_actions:
        chosen = ranked_actions[0]
        if exploration and exploration.get("enabled"):
            selected, _, selected_rank = sample_ranked_entry(
                ranked_actions,
                exploration.get("macro_epsilon", 0.0),
                exploration.get("top_k", 1),
                exploration.get("temperature", 1.0),
                score_key="score",
            )
            if selected:
                chosen = selected
                if selected_rank > 0:
                    chosen["status"] = f"{chosen['status']}; explore_rank={selected_rank}"
        chosen_payload = chosen["payload"]
        chosen_label = chosen["label"]
        chosen_reason = chosen["status"]

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


def normalize_policy_mode(value):
    mode = str(value or "current_rl").strip().lower()
    return mode if mode in ("current_rl", "ppo_experiment", "ppo_best") else "current_rl"


def ppo_active(control):
    return normalize_policy_mode((control or {}).get("policy_mode")) in ("ppo_experiment", "ppo_best")


def load_active_ppo_agent(control):
    mode = normalize_policy_mode((control or {}).get("policy_mode"))
    if mode not in ("ppo_experiment", "ppo_best"):
        return None
    processed_dir = os.path.join(os.path.dirname(__file__), "ProcessedPPOParams")
    return load_ppo_policy(processed_dir, mode=mode, allow_untrained=(mode == "ppo_experiment"))


def action_payload_key(payload):
    return json.dumps(payload or {}, sort_keys=True, ensure_ascii=False)


def ppo_align_row(row):
    return align_feature_vector(np.asarray(row, dtype=np.float32), PPO_INPUT_DIM).astype(np.float32)


def macro_action_features(state, label, payload, score=0.0, index=0, total=1):
    state_type = str((state or {}).get("state_type") or "").lower()
    player = (state or {}).get("player") or {}
    run = (state or {}).get("run") or {}
    action = str((payload or {}).get("action") or "").lower()
    state_types = ("map", "rewards", "card_reward", "card_select", "hand_select", "event", "crystal_sphere", "rest_site", "shop", "fake_merchant", "treasure")
    action_types = ("choose_map_node", "claim_reward", "select_card_reward", "skip_card_reward", "choose_event_option", "choose_rest_option", "shop_purchase", "proceed", "select_card", "confirm_selection", "combat_select_card", "combat_confirm_selection")
    hp = safe_num(player.get("hp"), 0.0)
    max_hp = max(safe_num(player.get("max_hp"), 1.0), 1.0)
    features = []
    features.extend([1.0 if state_type == item else 0.0 for item in state_types])
    features.extend([1.0 if action == item else 0.0 for item in action_types])
    features.extend([
        hp / max_hp,
        min(safe_num(player.get("gold"), 0.0) / 300.0, 2.0),
        min(safe_num(run.get("floor"), 0.0) / 17.0, 2.0),
        min(safe_num(run.get("act"), 0.0) / 4.0, 2.0),
        min(safe_num(player.get("deck_size"), len(player.get("deck") or [])) / 40.0, 2.0),
        min(float(index) / 12.0, 1.0),
        min(float(total) / 12.0, 1.0),
        max(min(float(score) / 10.0, 5.0), -5.0),
        1.0 if get_can_proceed(state, state_type) else 0.0,
        1.0 if is_pre_boss_rest_site(state) else 0.0,
    ])
    return np.asarray(features, dtype=np.float32)


def macro_row(macro_agent, state, candidate, index=0, total=1):
    state_vec = np.zeros(0, dtype=np.float32)
    if macro_agent:
        try:
            record = build_macro_record_from_state(state)
            state_vec = np.asarray(encode_macro_record(macro_agent["vocab"], record), dtype=np.float32)
        except Exception:
            state_vec = np.zeros(0, dtype=np.float32)
    action_vec = macro_action_features(
        state,
        candidate.get("label", ""),
        candidate.get("payload"),
        candidate.get("score", 0.0),
        index,
        total,
    )
    return ppo_align_row(np.concatenate([state_vec, action_vec]))


def enumerate_macro_candidates(state, macro_agent=None, allow_shop=False, card_baseline_weight=0.35, exploration=None):
    state_type = str((state or {}).get("state_type") or "").lower()
    candidates = []

    def add(label, payload, score=0.0, marker=""):
        if payload:
            candidates.append({"label": label, "payload": payload, "score": float(score), "marker": marker})

    if state_type == "map":
        for fallback_index, option in enumerate(((state.get("map") or {}).get("next_options") or [])):
            if not isinstance(option, dict):
                continue
            idx = safe_int(option.get("index"), fallback_index)
            node_type = str(option.get("type") or "")
            add(f"select_map_node:index_{idx}:{node_type}", {"action": "choose_map_node", "index": idx}, 1.0, node_type)
    elif state_type in ("rewards", "treasure"):
        for fallback_index, item in enumerate(get_items_for_state(state, state_type)):
            payload, status = reward_item_claim_payload(state, item, fallback_index, state_type)
            item_type = reward_item_type(item) or "item"
            idx = safe_int(item.get("index"), fallback_index)
            add(f"claim_reward:index_{idx}:{item_type}", payload, 2.0, status)
        if get_can_proceed(state, state_type):
            add("proceed", {"action": "proceed"}, 0.25, "available")
    elif state_type == "card_reward":
        entries, _profile = card_reward_baseline_entries(state)
        for entry in entries:
            add(entry["label"], entry["payload"], entry.get("score", 0.0), "card_reward")
    elif state_type == "event":
        for fallback_index, option in enumerate(((state.get("event") or {}).get("options") or [])):
            if not isinstance(option, dict) or option.get("is_locked"):
                continue
            idx = safe_int(option.get("index"), fallback_index)
            score, reasons = score_event_option_rule(option, state)
            add(f"choose_event_option:index_{idx}:{option.get('title')}", {"action": "choose_event_option", "index": idx}, score, " / ".join(reasons))
    elif state_type == "rest_site":
        for fallback_index, option in enumerate(((state.get("rest_site") or {}).get("options") or [])):
            if not isinstance(option, dict) or not option.get("is_enabled", True):
                continue
            idx = safe_int(option.get("index"), fallback_index)
            kind = rest_option_kind(option)
            score = 3.0 if kind == "heal" and is_pre_boss_rest_site(state) else 2.0 if kind == "smith" else 1.0
            add(f"choose_rest_option:index_{idx}:{kind}", {"action": "choose_rest_option", "index": idx}, score, kind)
    elif state_type in ("shop", "fake_merchant") and allow_shop:
        for row in rank_shop_items(state, state_type):
            idx = safe_int(row["item"].get("index"), row["fallback_index"])
            add(f"shop_purchase:index_{idx}:{row['category']}", {"action": "shop_purchase", "index": idx}, row["score"], " / ".join(row["reasons"]))
        if get_can_proceed(state, state_type):
            add("proceed", {"action": "proceed"}, 0.25, "available")

    if not candidates:
        payload, info = choose_macro_action(macro_agent, state, allow_shop=allow_shop, card_baseline_weight=card_baseline_weight, exploration=exploration)
        add(info.get("chosen_action") or ((payload or {}).get("action") or "macro_action"), payload, 1.0, info.get("reason", "fallback"))

    total = max(len(candidates), 1)
    for idx, candidate in enumerate(candidates):
        candidate["features"] = macro_row(macro_agent, state, candidate, idx, total).tolist()
    return candidates


def ppo_select(ppo_agent, feature_rows, fallback_index=0, deterministic=False):
    if not ppo_agent or not feature_rows:
        return None
    feature_count = len(feature_rows)
    chosen_index = max(0, min(int(fallback_index or 0), feature_count - 1))
    try:
        device = ppo_agent["device"]
        x = torch.tensor(np.asarray(feature_rows, dtype=np.float32), dtype=torch.float32).to(device)
        with torch.no_grad():
            logits, values = ppo_agent["model"](x)
            logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            values = torch.nan_to_num(values.float(), nan=0.0, posinf=1000000.0, neginf=-1000000.0)
            probs = torch.softmax(logits, dim=0)
            if (not torch.isfinite(probs).all()) or float(probs.sum().detach().cpu()) <= 0.0:
                probs = torch.ones_like(logits) / max(int(logits.numel()), 1)
            probs = probs.clamp_min(1e-8)
            probs = probs / probs.sum()
            log_probs = torch.log(probs)
        if ppo_agent.get("trained"):
            if deterministic:
                chosen_index = int(torch.argmax(logits).item())
            else:
                chosen_index = int(torch.multinomial(probs, 1).item())
    except Exception as exc:
        print(Fore.YELLOW + f"  [PPO] Selection failed, using fallback candidate. {exc}")
        logits = torch.zeros(feature_count, dtype=torch.float32)
        values = torch.zeros(feature_count, dtype=torch.float32)
        probs = torch.ones(feature_count, dtype=torch.float32) / max(feature_count, 1)
        log_probs = torch.log(probs.clamp_min(1e-8))
    ranked = sorted(
        [
            {
                "index": i,
                "score": float(logits[i].detach().cpu()),
                "confidence": float(probs[i].detach().cpu()),
            }
            for i in range(len(feature_rows))
        ],
        key=lambda item: item["score"],
        reverse=True,
    )
    return {
        "chosen_index": chosen_index,
        "logprob": float(log_probs[chosen_index].detach().cpu()),
        "value": float(values[chosen_index].detach().cpu()),
        "ranked": ranked,
    }


def ppo_state_summary(state):
    state = state or {}
    player = state.get("player") or {}
    run = state.get("run") or {}
    battle = state.get("battle") or {}
    return {
        "state_type": state.get("state_type"),
        "act": safe_int(run.get("act"), 0),
        "floor": safe_int(run.get("floor"), 0),
        "hp": safe_int(player.get("hp"), 0),
        "max_hp": safe_int(player.get("max_hp"), 0),
        "gold": safe_int(player.get("gold"), 0),
        "energy": safe_int(player.get("energy"), 0),
        "round": safe_int(battle.get("round"), 0),
    }


def ppo_enemy_hp_by_id(state):
    battle = (state or {}).get("battle") or {}
    rows = {}
    for idx, enemy in enumerate(battle.get("enemies") or []):
        if not isinstance(enemy, dict):
            continue
        key = str(enemy.get("entity_id") or enemy.get("combat_id") or enemy.get("id") or enemy.get("name") or idx)
        hp = max(0.0, safe_num(enemy.get("hp", enemy.get("current_hp", 0.0)), 0.0))
        block = max(0.0, safe_num(enemy.get("block"), 0.0))
        rows[key] = hp + block
    return rows


def ppo_status_entries(entity):
    entries = []
    for key in ("status", "statuses", "powers", "buffs", "debuffs"):
        value = (entity or {}).get(key)
        if isinstance(value, list):
            entries.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            entries.append(value)
    return entries


def ppo_player_strength(state):
    player = (state or {}).get("player") or {}
    return status_amount(ppo_status_entries(player), ("strength", "\u529b\u91cf"))


def ppo_payload_card(state, payload):
    if (payload or {}).get("action") != "play_card":
        return {}
    player = (state or {}).get("player") or {}
    hand = player.get("hand") or []
    card_index = safe_int((payload or {}).get("card_index", (payload or {}).get("index")), -1)
    for idx, card in enumerate(hand):
        if not isinstance(card, dict):
            continue
        if safe_int(card.get("index"), idx) == card_index or idx == card_index:
            return card
    return {}


def ppo_card_is_body_slam(card):
    text = combat_text(card)
    card_id = card_id_key(card)
    return card_id == "BODY_SLAM" or has_any(text, ("body slam", "\u5168\u8eab\u649e\u51fb", "block as damage", "\u683c\u6321\u9020\u6210\u4f24\u5bb3"))


def ppo_card_is_multi_hit(card):
    text = combat_text(card)
    card_id = card_id_key(card)
    if card_id in {"SWORD_BOOMERANG", "TWIN_STRIKE", "PUMMEL", "WHIRLWIND", "CLEAVE"}:
        return True
    return has_any(text, ("times", "all enemies", "\u6b21", "\u6240\u6709\u654c\u4eba", "\u591a\u6bb5"))


def ppo_ironclad_reward_terms(before, after, payload, damage, hp_loss):
    terms = {}
    before_type = str((before or {}).get("state_type") or "").lower()
    if before_type not in ("monster", "elite", "boss"):
        return terms
    player_before = (before or {}).get("player") or {}
    player_after = (after or {}).get("player") or {}
    battle = (before or {}).get("battle") or {}
    enemies = battle.get("enemies") or []
    incoming = enemy_incoming_damage(enemies)
    before_block = safe_num(player_before.get("block"), 0.0)
    after_block = safe_num(player_after.get("block"), before_block)
    block_delta = max(0.0, after_block - before_block)
    if block_delta > 0 and incoming > before_block:
        useful_block = min(block_delta, max(0.0, incoming - before_block))
        terms["ic_effective_block"] = min(0.35, useful_block * 0.025)
        excess_block = max(0.0, block_delta - useful_block)
        if excess_block > 8:
            terms["ic_excess_block"] = -min(0.12, excess_block * 0.01)

    card = ppo_payload_card(before, payload)
    if not card:
        return terms
    profile = card_effect_profile(card)
    strength_before = ppo_player_strength(before)
    strength_after = ppo_player_strength(after)
    strength_gain = max(0.0, strength_after - strength_before)
    round_no = max(1, safe_int(battle.get("round"), 1))

    if strength_gain > 0:
        terms["ic_strength_gain"] = min(0.35, 0.12 + strength_gain * 0.06)
        if round_no <= 3:
            terms["ic_early_scaling"] = 0.12
    if profile["is_attack"] and damage > 0 and strength_before > 0:
        terms["ic_strength_attack"] = min(0.35, strength_before * 0.025 + damage * 0.004)
        if ppo_card_is_multi_hit(card):
            terms["ic_multi_hit_strength"] = 0.12
    if card_applies_vulnerable(card):
        terms["ic_vulnerable_setup"] = 0.12 if damage <= 0 else 0.18
    if ppo_card_is_body_slam(card):
        if damage > 0:
            terms["ic_block_to_damage"] = min(0.4, 0.16 + damage * 0.01)
        elif before_block > 0:
            terms["ic_missed_block_slam"] = -0.15
    if profile["self_damage"] > 0:
        converted = damage > 0 or strength_gain > 0 or card_gain_energy(card) > 0
        terms["ic_self_damage_value"] = 0.12 if converted else -min(0.3, profile["self_damage"] * 0.05)
        if hp_loss > 0 and not converted:
            terms["ic_bad_self_damage"] = -min(0.3, hp_loss * 0.04)
    return terms


def ppo_combat_reward_terms(state_before, state_after, ok, payload):
    if not ok:
        return {"illegal_action": -1.0}
    before = state_before or {}
    after = state_after or {}
    before_run = before.get("run") or {}
    after_run = after.get("run") or {}
    before_type = str(before.get("state_type") or "").lower()
    after_type = str(after.get("state_type") or "").lower()
    terms = {}
    floor_delta = safe_int(after_run.get("floor"), 0) - safe_int(before_run.get("floor"), 0)
    if floor_delta > 0:
        terms["floor_delta"] = 0.05 * floor_delta
    before_is_combat = before_type in ("monster", "elite", "boss")
    after_is_combat = after_type in ("monster", "elite", "boss")
    if before_is_combat:
        before_hp = ppo_enemy_hp_by_id(before)
        after_hp = ppo_enemy_hp_by_id(after) if after_is_combat else {}
        damage = 0.0
        kills = 0
        for key, hp_before in before_hp.items():
            hp_after = after_hp.get(key, 0.0)
            damage += max(0.0, hp_before - hp_after)
            if hp_before > 0 and hp_after <= 0:
                kills += 1
        player_before = before.get("player") or {}
        player_after = after.get("player") or {}
        hp_loss = max(0.0, safe_num(player_before.get("hp"), 0.0) - safe_num(player_after.get("hp"), 0.0))
        if damage > 0:
            total_before_hp = max(1.0, sum(before_hp.values()))
            terms["combat_damage"] = min(0.35, 0.35 * damage / total_before_hp)
        if kills > 0:
            kill_value = 0.35 if before_type == "monster" else 0.65 if before_type == "elite" else 1.0
            terms["enemy_kill"] = min(1.0, kills * kill_value)
        if hp_loss > 0:
            max_hp = max(1.0, safe_num(player_before.get("max_hp"), 1.0))
            terms["hp_loss"] = -min(0.65, 0.65 * hp_loss / max_hp)
        if (payload or {}).get("action") == "end_turn" and damage <= 0:
            terms["idle_end_turn"] = -0.08
        terms.update(ppo_ironclad_reward_terms(before, after, payload, damage, hp_loss))
        if before_type in ("monster", "elite", "boss") and not after_is_combat:
            round_no = max(1, safe_int((before.get("battle") or {}).get("round"), 1))
            terms["combat_win"] = 0.7 if before_type == "monster" else 1.0 if before_type == "elite" else 1.5
            terms["combat_speed"] = max(0.0, 0.35 - max(0, round_no - 3) * 0.07)
            if hp_loss <= 0:
                terms["clean_finish"] = 0.15
    if after_type == "boss" and before_type != "boss":
        terms["enter_boss"] = 1.0
    if safe_int(before_run.get("act"), 0) < 2 and safe_int(after_run.get("act"), 0) >= 2:
        terms["act2_reached"] = 3.0
    progress = boss_hp_progress(before, after)
    if progress:
        terms["boss_damage"] = min(0.5, safe_num(progress.get("damage"), 0.0) * 0.02)
    if before_type == "rest_site" and is_pre_boss_rest_site(before):
        player = before.get("player") or {}
        hp = safe_num(player.get("hp"), 0.0)
        max_hp = safe_num(player.get("max_hp"), 0.0)
        if hp < max_hp and (payload or {}).get("action") == "choose_rest_option":
            options = (before.get("rest_site") or {}).get("options") or []
            chosen_idx = safe_int((payload or {}).get("index"), -1)
            chosen = next((o for o in options if safe_int(o.get("index"), -2) == chosen_idx), {})
            kind = rest_option_kind(chosen)
            terms["pre_boss_rest"] = 0.8 if kind == "heal" else -0.8 if kind == "smith" else 0.0
    return {key: round(float(value), 4) for key, value in terms.items() if abs(float(value)) > 1e-9}


def ppo_step_reward(state_before, state_after, ok, payload):
    reward = sum(ppo_combat_reward_terms(state_before, state_after, ok, payload).values())
    return round(float(max(-1.0, min(1.0, reward))), 4)


def append_ppo_rollout(session_id, mode, state_before, state_after, payload, ok, candidates, features, chosen_index, ppo_decision, behavior_policy):
    if mode not in ("ppo_experiment", "ppo_best") or not candidates or chosen_index is None or not ppo_decision:
        return
    os.makedirs(PPO_LOG_DIR, exist_ok=True)
    path = os.path.join(PPO_LOG_DIR, f"ppo_rollouts_{datetime.now():%Y-%m-%d}.jsonl")
    run = (state_before or {}).get("run") or {}
    state_type = (state_before or {}).get("state_type")
    reward_terms = ppo_combat_reward_terms(state_before, state_after, ok, payload)
    reward = ppo_step_reward(state_before, state_after, ok, payload)
    old_logprob = float(ppo_decision.get("logprob", 0.0))
    old_value = float(ppo_decision.get("value", 0.0))
    record = {
        "type": "ppo_step",
        "timestamp": int(time.time() * 1000),
        "run_id": session_id,
        "episode_id": session_id,
        "seed": str((state_before or {}).get("seed") or run.get("seed") or ""),
        "act": safe_int(run.get("act"), 0),
        "floor": safe_int(run.get("floor"), 0),
        "state_type": state_type,
        "screen_type": state_type,
        "policy_mode": mode,
        "policy_version": str(ppo_decision.get("policy_version") or behavior_policy or mode),
        "behavior_policy": behavior_policy,
        "reward_schema": "ironclad_v1",
        "option_schema": OPTION_SCHEMA_VERSION,
        "state_features_version": STATE_FEATURES_VERSION,
        "option_features_version": OPTION_FEATURES_VERSION,
        "ok": bool(ok),
        "action": payload,
        "chosen_index": int(chosen_index),
        "action_index": int(chosen_index),
        "logprob": old_logprob,
        "value": old_value,
        "old_logprob": old_logprob,
        "old_value": old_value,
        "reward": reward,
        "reward_terms": reward_terms,
        "done": False,
        "truncated": False,
        "legal_option_count": int(len(candidates)),
        "features": features,
        "candidates": [
            {
                "label": c.get("label") if isinstance(c, dict) else getattr(c, "label", ""),
                "kind": c.get("kind", "macro") if isinstance(c, dict) else getattr(c, "kind", ""),
                "payload": c.get("payload") if isinstance(c, dict) else getattr(c, "payload", {}),
            }
            for c in candidates
        ],
        "state_before": ppo_state_summary(state_before),
        "state_after": ppo_state_summary(state_after),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception as exc:
        print(Fore.YELLOW + f"[PPO] rollout write failed: {exc}")


def score_distribution(values):
    finite = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    if not finite:
        return {"count": 0, "mean": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    return {
        "count": len(finite),
        "mean": round(mean, 4),
        "variance": round(variance, 4),
        "min": round(min(finite), 4),
        "max": round(max(finite), 4),
    }


def card_label_index(label):
    text = str(label or "")
    if not text.startswith("choose_card:index_"):
        return -1
    return safe_int(text.rsplit("_", 1)[-1], -1)


def append_card_scorer_shadow_log(session_id, state, actual_payload, macro_info, behavior_policy, model_version=""):
    scorer = (macro_info or {}).get("card_scorer")
    if not isinstance(scorer, dict) or scorer.get("mode") == "off":
        return
    options = scorer.get("options") or []
    selected = scorer.get("selected") or {}
    old_label = (macro_info or {}).get("chosen_action")
    if not old_label:
        if (actual_payload or {}).get("action") == "skip_card_reward":
            old_label = "skip_reward"
        elif (actual_payload or {}).get("action") == "select_card_reward":
            old_label = f"choose_card:index_{safe_int((actual_payload or {}).get('card_index', (actual_payload or {}).get('index')), -1)}"
    raw_scorer_label = selected.get("label")
    scorer_label = raw_scorer_label
    options_by_label = {
        option.get("label"): option
        for option in options
        if isinstance(option, dict)
    }
    old_option = options_by_label.get(old_label) or {}
    effective_selected = selected
    effective_fallback_reason = ""
    raw_scorer_card = {
        "index": selected.get("index", -1),
        "card_id": selected.get("card_id") or "",
        "name": selected.get("name") or "",
        "type": selected.get("card_type") or "",
        "cost": selected.get("cost", 0),
    }
    confidence_gap_value = safe_num(scorer.get("confidence_gap"), 0.0)
    if old_label and raw_scorer_label and old_label != raw_scorer_label and old_option:
        raw_index = card_label_index(raw_scorer_label)
        if confidence_gap_value < 0.15:
            effective_fallback_reason = "shadow_low_confidence_fallback"
        elif raw_index > 2:
            effective_fallback_reason = "shadow_extra_card_index_fallback"
        if effective_fallback_reason:
            scorer_label = old_label
            effective_selected = dict(old_option)
            effective_selected.setdefault("label", old_label)
            effective_selected.setdefault("payload", old_option.get("payload"))
            effective_selected["raw_scorer_action"] = raw_scorer_label
            effective_selected["fallback_reason"] = effective_fallback_reason
    skip_option = options_by_label.get("skip_reward") or {}
    skip_breakdown = skip_option.get("score_breakdown") if isinstance(skip_option.get("score_breakdown"), dict) else {}
    skip_diagnostics = (
        ((skip_option.get("metadata") or {}).get("skip_diagnostics"))
        if isinstance(skip_option.get("metadata"), dict)
        else None
    )
    if not isinstance(skip_diagnostics, dict):
        skip_diagnostics = {}
    old_card = {
        "index": old_option.get("index", safe_int((actual_payload or {}).get("card_index", (actual_payload or {}).get("index")), -1)),
        "card_id": old_option.get("card_id") or "",
        "name": old_option.get("name") or "",
        "type": old_option.get("card_type") or "",
        "cost": old_option.get("cost", 0),
    }
    scorer_card = {
        "index": effective_selected.get("index", -1),
        "card_id": effective_selected.get("card_id") or "",
        "name": effective_selected.get("name") or "",
        "type": effective_selected.get("card_type") or "",
        "cost": effective_selected.get("cost", 0),
    }
    run = (state or {}).get("run") or {}
    scores = [
        safe_num(option.get("score"), 0.0)
        for option in options
        if isinstance(option, dict)
    ]
    template_scores = (
        ((scorer.get("template_lock") or {}).get("template_scores"))
        or ((scorer.get("archetype_consistency") or {}).get("scores"))
        or {}
    )
    deck_summary = scorer.get("deck_summary") if isinstance(scorer.get("deck_summary"), dict) else {}
    record = {
        "type": "card_scorer_shadow",
        "timestamp": int(time.time() * 1000),
        "run_id": session_id,
        "episode_id": session_id,
        "seed": str((state or {}).get("seed") or run.get("seed") or ""),
        "act": safe_int(run.get("act"), 0),
        "floor": safe_int(run.get("floor"), 0),
        "screen_type": "card_reward",
        "policy_version": str(model_version or behavior_policy or ""),
        "behavior_policy": behavior_policy,
        "reward_schema": "ironclad_shadow_v1",
        "actual_payload": actual_payload,
        "actual_action": (actual_payload or {}).get("action"),
        "action_index": safe_int((actual_payload or {}).get("card_index", (actual_payload or {}).get("index")), -1),
        "old_logprob": None,
        "old_value": None,
        "done": False,
        "truncated": False,
        "actual_skip": (actual_payload or {}).get("action") == "skip_card_reward",
        "legacy_chosen_action": old_label,
        "old_policy_action": old_label,
        "old_policy_card": old_card,
        "raw_scorer_action": raw_scorer_label,
        "raw_scorer_card": raw_scorer_card,
        "effective_fallback_reason": effective_fallback_reason,
        "low_confidence_fallback": effective_fallback_reason == "shadow_low_confidence_fallback",
        "extra_card_index_fallback": effective_fallback_reason == "shadow_extra_card_index_fallback",
        "recommended_action": scorer_label,
        "scorer_action": scorer_label,
        "scorer_card": scorer_card,
        "recommended_payload": effective_selected.get("payload"),
        "scorer_disagreed_with_old_policy": bool(old_label and scorer_label and old_label != scorer_label),
        "confidence_gap": confidence_gap_value,
        "skip_score": skip_option.get("score"),
        "skip_score_breakdown": skip_breakdown,
        "skip_reasons": skip_option.get("reasons") or [],
        "skip_diagnostics": skip_diagnostics,
        "best_card_score": skip_diagnostics.get("best_card_score"),
        "second_best_card_score": skip_diagnostics.get("second_best_card_score"),
        "max_archetype_fit": skip_diagnostics.get("max_archetype_fit"),
        "skip_recommended": scorer_label == "skip_reward",
        "skip_available": any(
            (option or {}).get("label") == "skip_reward"
            for option in options
            if isinstance(option, dict)
        ),
        "legal_option_count": scorer.get("legal_option_count", len(options)),
        "option_schema": scorer.get("option_schema", OPTION_SCHEMA_VERSION),
        "state_features_version": scorer.get("state_features_version", STATE_FEATURES_VERSION),
        "option_features_version": scorer.get("option_features_version", CARD_OPTION_FEATURES_VERSION),
        "template_id": scorer.get("template_id"),
        "template_scores": template_scores,
        "template_lock": scorer.get("template_lock"),
        "template_locked": bool((scorer.get("template_lock") or {}).get("locked")),
        "candidate_template": (scorer.get("template_lock") or {}).get("candidate_template"),
        "locked_template": (scorer.get("template_lock") or {}).get("locked_template"),
        "archetype_consistency": scorer.get("archetype_consistency"),
        "deck_size": deck_summary.get("deck_size"),
        "deck_summary": deck_summary,
        "score_distribution": score_distribution(scores),
        "reward_terms": {},
        "reward_term_distribution": {},
        "raw_selected": selected,
        "selected": effective_selected,
        "options": options,
    }
    try:
        os.makedirs(OPTION_SHADOW_LOG_DIR, exist_ok=True)
        path = os.path.join(OPTION_SHADOW_LOG_DIR, f"card_scorer_{datetime.now():%Y-%m-%d}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception as exc:
        print(Fore.YELLOW + f"[CardScorer] shadow log write failed: {exc}")


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


def choose_card_to_play(sorted_indices, id_to_action, hand_cards, energy, exploration=None):
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
            if exploration and exploration.get("enabled"):
                ranked_cards = []
                seen_ids = set()
                for inner_idx in sorted_indices:
                    inner_action = id_to_action.get(inner_idx.item(), "UNKNOWN")
                    if not inner_action.startswith("play_card_"):
                        continue
                    inner_card_id = inner_action.replace("play_card_", "")
                    if inner_card_id in playable_by_id and inner_card_id not in seen_ids:
                        seen_ids.add(inner_card_id)
                        ranked_cards.append({
                            "card": playable_by_id[inner_card_id],
                            "score": float(len(sorted_indices) - len(ranked_cards)),
                        })
                selected, _, _ = sample_ranked_entry(
                    ranked_cards,
                    exploration.get("combat_epsilon", 0.0),
                    exploration.get("top_k", 1),
                    exploration.get("temperature", 1.0),
                    score_key="score",
                )
                if selected:
                    return selected["card"]
            return playable_by_id[card_id]

    # 模型词表里没有的新牌，兜底打第一张可支付的牌。
    return affordable[0]


def load_exploration_config(control):
    enabled = bool(control.get("exploration_enabled", False))
    mode = str(control.get("exploration_mode") or "aggressive").lower()
    if mode not in ("aggressive", "off"):
        mode = "aggressive"
    constraint_mode = str(control.get("self_play_constraint_mode") or "explore").lower()
    if constraint_mode not in ("guarded", "explore", "free"):
        constraint_mode = "explore"
    try:
        combat_epsilon = max(0.0, min(float(control.get("combat_exploration_epsilon", 0.35)), 1.0))
    except (TypeError, ValueError):
        combat_epsilon = 0.35
    try:
        macro_epsilon = max(0.0, min(float(control.get("macro_exploration_epsilon", 0.25)), 1.0))
    except (TypeError, ValueError):
        macro_epsilon = 0.25
    try:
        top_k = max(1, min(int(control.get("exploration_top_k", 5)), 12))
    except (TypeError, ValueError):
        top_k = 5
    try:
        temperature = max(0.1, min(float(control.get("exploration_temperature", 1.35)), 5.0))
    except (TypeError, ValueError):
        temperature = 1.35
    return {
        "enabled": enabled and mode != "off",
        "mode": mode,
        "constraint_mode": constraint_mode,
        "combat_epsilon": combat_epsilon,
        "macro_epsilon": macro_epsilon,
        "top_k": top_k,
        "temperature": temperature,
    }


def sample_ranked_entry(ranked, epsilon, top_k, temperature, score_key="score"):
    if not ranked:
        return None, False, -1
    pool = ranked[:max(1, min(len(ranked), int(top_k or 1)))]
    if len(pool) == 1 or epsilon <= 0.0 or random.random() >= epsilon:
        return pool[0], False, 0

    scores = [float(item.get(score_key, 0.0)) for item in pool]
    peak = max(scores)
    safe_temp = max(float(temperature or 1.0), 0.1)
    weights = [math.exp((score - peak) / safe_temp) for score in scores]
    total = sum(weights)
    if total <= 0:
        return pool[0], False, 0

    pick = random.random() * total
    running = 0.0
    for index, (item, weight) in enumerate(zip(pool, weights)):
        running += weight
        if pick <= running:
            return item, index > 0, index
    return pool[-1], True, len(pool) - 1


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


def control_speed_multiplier(control):
    if not control.get("game_speed_enabled", False):
        return 1.0
    try:
        return max(1.0, min(float(control.get("game_speed_multiplier", 2.0)), 6.0))
    except (TypeError, ValueError):
        return 2.0


def agent_sleep(seconds, control=None):
    multiplier = control_speed_multiplier(control or load_control())
    time.sleep(max(0.05, float(seconds) / multiplier))


def payload_with_policy(payload, policy_name, model_version):
    if not payload:
        return payload
    tagged = dict(payload)
    tagged["policy_name"] = policy_name
    tagged["model_version"] = model_version
    return tagged


def combat_enemies_from_state(state):
    if not isinstance(state, dict):
        return []
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    return [e for e in (battle.get("enemies") or []) if isinstance(e, dict)]


def enemy_identity(enemy):
    return str(enemy.get("entity_id") or enemy.get("combat_id") or enemy.get("id") or enemy.get("name") or "")


def boss_hp_snapshot(state):
    if not isinstance(state, dict) or str(state.get("state_type") or "").lower() != "boss":
        return None
    enemies = combat_enemies_from_state(state)
    rows = []
    total_hp = 0
    total_max_hp = 0
    for index, enemy in enumerate(enemies):
        max_hp = safe_int(enemy.get("max_hp"), 0)
        hp = max(0, safe_int(enemy.get("hp", enemy.get("current_hp")), 0))
        if max_hp <= 0 and hp <= 0:
            continue
        identity = enemy_identity(enemy) or f"enemy_{index}"
        total_hp += hp
        total_max_hp += max(max_hp, hp)
        rows.append({
            "id": identity,
            "name": enemy.get("name") or enemy.get("id") or identity,
            "hp": hp,
            "max_hp": max(max_hp, hp),
            "block": safe_int(enemy.get("block"), 0),
        })
    if not rows:
        return None
    return {
        "total_hp": total_hp,
        "total_max_hp": total_max_hp,
        "hp_ratio": round(total_hp / max(total_max_hp, 1), 4),
        "enemies": rows,
    }


def boss_hp_progress(before, after):
    before_snapshot = boss_hp_snapshot(before)
    after_snapshot = boss_hp_snapshot(after)
    if not before_snapshot and not after_snapshot:
        return None
    progress = {
        "before": before_snapshot,
        "after": after_snapshot,
        "damage": 0,
        "hp_ratio_drop": 0.0,
    }
    if before_snapshot and after_snapshot:
        progress["damage"] = max(0, before_snapshot["total_hp"] - after_snapshot["total_hp"])
        progress["hp_ratio_drop"] = round(before_snapshot["hp_ratio"] - after_snapshot["hp_ratio"], 4)
    return progress


def append_ai_action_log(session_id, action_payload, state_before, state_after, ok, policy_name="", model_version=""):
    control = load_control()
    if not control.get("record_ai_actions", True):
        return

    os.makedirs(AI_LOG_DIR, exist_ok=True)
    path = os.path.join(AI_LOG_DIR, f"ai_combat_run_{datetime.now():%Y-%m-%d}.jsonl")
    boss_progress = boss_hp_progress(state_before, state_after)
    record = {
        "type": "action",
        "run_id": session_id,
        "timestamp": int(time.time() * 1000),
        "source": "ai",
        "policy_name": policy_name,
        "model_version": model_version,
        "action_type": action_payload.get("action"),
        "action_data": action_payload,
        "ok": bool(ok),
        "state_before": state_before,
        "state_after": state_after,
    }
    if boss_progress:
        record["boss_hp_progress"] = boss_progress
        record["boss_damage_delta"] = boss_progress.get("damage", 0)
        record["boss_hp_ratio_drop"] = boss_progress.get("hp_ratio_drop", 0.0)
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


def combat_turn_key(state):
    run = (state or {}).get("run") or {}
    battle = (state or {}).get("battle") or {}
    return json.dumps({
        "act": run.get("act"),
        "floor": run.get("floor"),
        "round": battle.get("round"),
        "turn": battle.get("turn"),
        "state_type": (state or {}).get("state_type"),
    }, sort_keys=True, ensure_ascii=False)


def run_agent():
    processed_dir = os.path.join(os.path.dirname(__file__), "ProcessedParams")
    macro_processed_dir = os.path.join(os.path.dirname(__file__), "ProcessedMacroParams")
    encoder, id_to_action, model, device, combat_metadata = load_agent(processed_dir)
    candidate_agent = load_candidate_agent(processed_dir)
    macro_agent = load_macro_agent(macro_processed_dir)
    startup_control = load_control()
    ppo_agent = load_active_ppo_agent(startup_control)
    ppo_policy_mode = normalize_policy_mode(startup_control.get("policy_mode"))
    combat_policy_name = "bc_combat"
    combat_model_version = combat_metadata.get("model_version", "")
    candidate_policy_name = "candidate_bc_combat"
    candidate_model_version = (candidate_agent or {}).get("model_version", "")
    if candidate_agent and str(candidate_agent.get("model_path") or "").startswith("candidate_rl_"):
        candidate_policy_name = "candidate_rl_combat"
    macro_policy_name = "macro_mixed"
    macro_model_version = (macro_agent or {}).get("model_version", "rules")
    ppo_model_version = (ppo_agent or {}).get("model_version", "")
    session_id = f"ai_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    
    print(Fore.GREEN + Style.BRIGHT + "\n====== STS2 AI Control System v2 ======")
    print(Fore.WHITE + "  Enter any combat encounter, AI will auto-play when it's your turn.")
    print(Fore.WHITE + "  Press Ctrl+C to stop.\n")
    
    last_status_print = 0
    last_data_source = None
    last_macro_decision_key = None
    last_macro_decision_ts = 0.0
    card_select_memory = {}
    potion_used_turn_key = None

    while True:
        agent_sleep(0.8)
        
        state = fetch_game_state()
        if not state:
            now = time.time()
            if now - last_status_print > 5:
                print(Fore.RED + "[Waiting] Game API not reachable... Is the game running?")
                last_status_print = now
            continue

        control = load_control()
        exploration = load_exploration_config(control)
        current_policy_mode = normalize_policy_mode(control.get("policy_mode"))
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
        if not is_selection_state_type(state_type) and card_select_memory:
            card_select_memory.clear()
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
            constraint_mode = exploration.get("constraint_mode", "guarded")
            if state_type in ("shop", "fake_merchant") and not control.get("macro_shop_enabled", False) and constraint_mode == "guarded":
                write_ai_logic_snapshot({
                    "timestamp": int(time.time() * 1000),
                    "session_id": session_id,
                    "state_type": state_type,
                    "mode": "macro",
                    "policy_name": macro_policy_name,
                    "model_version": macro_model_version,
                    "top_actions": [],
                    "chosen_action": None,
                    "payload": None,
                    "reason": "shop_protected",
                })
                if last_data_source != "human":
                    set_data_source("human")
                    last_data_source = "human"
                agent_sleep(4.0, control)
                continue

            if control.get("macro_enabled", False) and state_type in ("map", "rewards", "card_reward", "card_select", "hand_select", "bundle_select", "event", "crystal_sphere", "rest_site", "shop", "fake_merchant", "treasure"):
                allow_shop = bool(control.get("macro_shop_enabled", False) or constraint_mode in ("explore", "free"))
                card_baseline_weight = safe_num(control.get("macro_card_reward_weight"), 0.35)
                state["_ai_session_id"] = session_id
                card_select_key = None
                if is_selection_state_type(state_type):
                    card_select_key = selection_memory_signature(state)
                    remembered_selection = sorted(card_select_memory.get(card_select_key, set()))
                    state["_selection_selected_indices"] = remembered_selection
                    state["_card_select_selected_indices"] = remembered_selection
                payload, macro_info = choose_macro_action(
                    macro_agent,
                    state,
                    allow_shop=allow_shop,
                    card_baseline_weight=card_baseline_weight,
                    exploration=exploration,
                )
                macro_ppo_candidates = []
                macro_ppo_features = []
                macro_ppo_decision = None
                macro_ppo_chosen_index = None
                if current_policy_mode in ("ppo_experiment", "ppo_best") and ppo_agent:
                    macro_ppo_candidates = enumerate_macro_candidates(
                        state,
                        macro_agent=macro_agent,
                        allow_shop=allow_shop,
                        card_baseline_weight=card_baseline_weight,
                        exploration=exploration,
                    )
                    payload_key = action_payload_key(payload)
                    fallback_index = next(
                        (i for i, c in enumerate(macro_ppo_candidates) if action_payload_key(c.get("payload")) == payload_key),
                        -1,
                    )
                    if fallback_index < 0 and payload:
                        macro_ppo_candidates.append({
                            "label": macro_info.get("chosen_action") or payload.get("action") or "macro_fallback",
                            "payload": payload,
                            "score": 1.0,
                            "marker": macro_info.get("reason", "fallback"),
                            "features": macro_row(macro_agent, state, {
                                "label": macro_info.get("chosen_action") or payload.get("action") or "macro_fallback",
                                "payload": payload,
                                "score": 1.0,
                            }, len(macro_ppo_candidates), len(macro_ppo_candidates) + 1).tolist(),
                        })
                        fallback_index = len(macro_ppo_candidates) - 1
                    macro_ppo_features = [c.get("features", []) for c in macro_ppo_candidates]
                    macro_ppo_decision = ppo_select(
                        ppo_agent,
                        macro_ppo_features,
                        fallback_index=max(0, fallback_index),
                        deterministic=(current_policy_mode == "ppo_best"),
                    )
                    if macro_ppo_decision:
                        macro_ppo_chosen_index = int(macro_ppo_decision["chosen_index"])
                        if ppo_agent.get("trained") and 0 <= macro_ppo_chosen_index < len(macro_ppo_candidates):
                            chosen_ppo_macro = macro_ppo_candidates[macro_ppo_chosen_index]
                            payload = chosen_ppo_macro.get("payload")
                            macro_info["chosen_action"] = chosen_ppo_macro.get("label")
                            macro_info["payload"] = payload
                            macro_info["reason"] = f"ppo_macro:{ppo_agent.get('source')} {chosen_ppo_macro.get('marker', '')}"
                        macro_info["top_actions"] = [
                            {
                                "action": macro_ppo_candidates[item["index"]].get("label"),
                                "confidence": round(item["confidence"] * 100.0, 2),
                                "marker": "ppo_macro",
                            }
                            for item in (macro_ppo_decision.get("ranked") or [])[:6]
                            if 0 <= item["index"] < len(macro_ppo_candidates)
                        ] or macro_info.get("top_actions", [])
                decision_key = json.dumps({
                    "signature": macro_state_signature(state),
                    "payload": payload,
                }, sort_keys=True, ensure_ascii=False)
                snapshot_policy_name = current_policy_mode if current_policy_mode in ("ppo_experiment", "ppo_best") and ppo_agent and ppo_agent.get("trained") else macro_policy_name
                snapshot_model_version = ppo_model_version if current_policy_mode in ("ppo_experiment", "ppo_best") and ppo_agent and ppo_agent.get("trained") else macro_model_version
                if state_type == "card_reward":
                    append_card_scorer_shadow_log(
                        session_id,
                        state,
                        payload,
                        macro_info,
                        snapshot_policy_name,
                        snapshot_model_version,
                    )
                write_ai_logic_snapshot({
                    "timestamp": int(time.time() * 1000),
                    "session_id": session_id,
                    "state_type": state_type,
                    "mode": "macro",
                    "policy_name": snapshot_policy_name,
                    "model_version": snapshot_model_version,
                    "top_actions": macro_info.get("top_actions", []),
                    "chosen_action": macro_info.get("chosen_action"),
                    "payload": payload,
                    "reason": macro_info.get("reason"),
                    "constraint_mode": constraint_mode,
                    "deck_profile": macro_info.get("deck_profile"),
                    "reward_baseline": macro_info.get("reward_baseline"),
                    "card_scorer": macro_info.get("card_scorer"),
                    "archetype_consistency": macro_info.get("archetype_consistency"),
                    "exploration": exploration,
                })
                now_ts = time.time()
                macro_retry_after = 1.0 if state_type in ("rewards", "treasure", "card_reward") else 2.0 if state_type in ("rest_site", "event", "card_select", "hand_select", "bundle_select") else 6.0
                should_retry_macro = bool(
                    payload
                    and decision_key == last_macro_decision_key
                    and now_ts - last_macro_decision_ts >= macro_retry_after
                )
                if payload and (decision_key != last_macro_decision_key or should_retry_macro):
                    print(Fore.GREEN + Style.BRIGHT + f"  >>> MACRO EXECUTE: {macro_info.get('chosen_action')}")
                    print(Fore.WHITE + f"  Sending: {json.dumps(payload, ensure_ascii=False)}")
                    if last_data_source != "ai":
                        set_data_source("ai")
                        last_data_source = "ai"
                    raw_payload = payload
                    active_macro_policy = current_policy_mode if current_policy_mode in ("ppo_experiment", "ppo_best") and ppo_agent and ppo_agent.get("trained") else macro_policy_name
                    active_macro_version = ppo_model_version if current_policy_mode in ("ppo_experiment", "ppo_best") and ppo_agent and ppo_agent.get("trained") else macro_model_version
                    payload = payload_with_policy(payload, active_macro_policy, active_macro_version)
                    success = send_action(payload)
                    if is_selection_state_type(state_type) and card_select_key and success:
                        action_name = raw_payload.get("action")
                        if action_name in ("select_card", "combat_select_card"):
                            selected = card_select_memory.setdefault(card_select_key, set())
                            selected.add(safe_int(raw_payload.get("index", raw_payload.get("card_index")), -1))
                        elif action_name in ("confirm_selection", "cancel_selection", "combat_confirm_selection"):
                            card_select_memory.pop(card_select_key, None)
                    last_macro_decision_key = decision_key
                    last_macro_decision_ts = now_ts
                    if not success and last_data_source != "human":
                        set_data_source("human")
                        last_data_source = "human"
                    state_after = fetch_game_state()
                    append_ppo_rollout(
                        session_id,
                        current_policy_mode,
                        state,
                        state_after,
                        raw_payload,
                        success,
                        macro_ppo_candidates,
                        macro_ppo_features,
                        macro_ppo_chosen_index,
                        macro_ppo_decision,
                        active_macro_policy,
                    )
                    macro_post_delay = 0.6 if state_type in ("rewards", "treasure", "card_reward") else 1.0
                    agent_sleep(macro_post_delay, control)
                    continue
                if state_type in ("shop", "fake_merchant") and not allow_shop:
                    agent_sleep(4.0, control)
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
        current_turn_key = combat_turn_key(state)
        state["_potion_used_this_turn"] = (current_turn_key == potion_used_turn_key)
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

            legacy_top_actions = list(top_actions)
            candidate_decision, ranked_candidates, decision_source = score_combat_candidates(
                candidate_agent,
                state_vec,
                legal_candidates,
                state,
                exploration=exploration,
            )
            combat_ppo_features = []
            combat_ppo_decision = None
            combat_ppo_chosen_index = None
            if current_policy_mode in ("ppo_experiment", "ppo_best") and ppo_agent and legal_candidates:
                try:
                    ppo_state_part = align_feature_vector(state_vec, max(0, PPO_INPUT_DIM - CANDIDATE_FEATURE_DIM))
                    combat_ppo_features = [
                        ppo_align_row(np.concatenate([
                            np.asarray(ppo_state_part, dtype=np.float32),
                            np.asarray(candidate.features, dtype=np.float32),
                        ])).tolist()
                        for candidate in legal_candidates
                    ]
                    fallback_index = 0
                    if candidate_decision:
                        fallback_candidate = candidate_decision.get("candidate")
                        fallback_index = next(
                            (idx for idx, candidate in enumerate(legal_candidates) if candidate is fallback_candidate),
                            0,
                        )
                    combat_ppo_decision = ppo_select(
                        ppo_agent,
                        combat_ppo_features,
                        fallback_index=fallback_index,
                        deterministic=(current_policy_mode == "ppo_best"),
                    )
                    if combat_ppo_decision:
                        combat_ppo_chosen_index = int(combat_ppo_decision.get("chosen_index", 0))
                        if ppo_agent.get("trained") and 0 <= combat_ppo_chosen_index < len(legal_candidates):
                            chosen_ppo_candidate = legal_candidates[combat_ppo_chosen_index]
                            chosen_score = next(
                                (
                                    float(item.get("score", 0.0))
                                    for item in (combat_ppo_decision.get("ranked") or [])
                                    if int(item.get("index", -1)) == combat_ppo_chosen_index
                                ),
                                0.0,
                            )
                            candidate_decision = {
                                "candidate": chosen_ppo_candidate,
                                "score": chosen_score,
                                "reason": f"ppo_combat:{ppo_agent.get('source')}",
                            }
                            decision_source = f"ppo_{current_policy_mode}"
                            ranked_candidates = [
                                {
                                    "candidate": legal_candidates[item["index"]],
                                    "score": float(item.get("score", item.get("confidence", 0.0))),
                                    "confidence": float(item.get("confidence", 0.0)),
                                    "reason": "ppo_combat",
                                }
                                for item in (combat_ppo_decision.get("ranked") or [])
                                if 0 <= int(item.get("index", -1)) < len(legal_candidates)
                            ]
                except Exception as exc:
                    print(Fore.YELLOW + f"  [PPO] Combat policy unavailable, using current RL. {exc}")
            
            # === 选择最佳合法动作 ===
            if candidate_decision:
                chosen_candidate = candidate_decision["candidate"]
                chosen_action = chosen_candidate.kind
                chosen_candidate_label = chosen_candidate.label
                top_actions = candidate_top_actions(ranked_candidates)
                ppo_is_active = bool(
                    current_policy_mode in ("ppo_experiment", "ppo_best")
                    and ppo_agent
                    and ppo_agent.get("trained")
                    and combat_ppo_decision
                )
                active_policy_name = current_policy_mode if ppo_is_active else candidate_policy_name
                active_model_version = ppo_model_version if ppo_is_active else candidate_model_version
                print(Fore.CYAN + "  [Candidate] Top 5 candidate scores:")
                for item in top_actions:
                    print(f"    {item['action'][:58]:58s}  {item['confidence']:5.1f}% [{item['marker']}]")
                print(Fore.GREEN + Style.BRIGHT + f"  >>> EXECUTE: {chosen_candidate_label}")
                raw_payload = chosen_candidate.payload
                payload = payload_with_policy(raw_payload, active_policy_name, active_model_version)
                write_ai_logic_snapshot({
                    "timestamp": int(time.time() * 1000),
                    "session_id": session_id,
                    "state_type": state_type,
                    "policy_name": active_policy_name,
                    "model_version": active_model_version,
                    "fallback_policy_name": combat_policy_name,
                    "fallback_model_version": combat_model_version,
                    "decision_source": decision_source,
                    "hp": hp,
                    "block": block,
                    "energy": energy,
                    "hand": available_card_ids,
                    "playable": playable_card_ids,
                    "enemies": [{"id": get_enemy_target_id(e), "name": e.get("name"), "hp": get_enemy_hp(e)} for e in alive_enemies],
                    "top_actions": top_actions,
                    "legacy_top_actions": legacy_top_actions,
                    "candidate_actions": legal_candidate_catalog,
                    "chosen_action": chosen_action,
                    "chosen_candidate": chosen_candidate_label,
                    "payload": payload,
                    "reason": "candidate scorer plus tactical guards chose the highest scoring legal action",
                    "constraint_mode": exploration.get("constraint_mode", "guarded"),
                    "exploration": exploration,
                })
                if payload:
                    print(Fore.WHITE + f"  Sending: {json.dumps(payload)}")
                    if last_data_source != "ai":
                        set_data_source("ai")
                        last_data_source = "ai"
                    success = send_action(payload)
                    state_after = fetch_game_state()
                    append_ai_action_log(session_id, payload, state, state_after, success, active_policy_name, active_model_version)
                    append_ppo_rollout(
                        session_id,
                        current_policy_mode,
                        state,
                        state_after,
                        raw_payload,
                        success,
                        legal_candidates,
                        combat_ppo_features,
                        combat_ppo_chosen_index,
                        combat_ppo_decision,
                        active_policy_name,
                    )
                    if success and chosen_candidate.kind == "use_potion":
                        potion_used_turn_key = current_turn_key
                    if success:
                        agent_sleep(1.5, control)
                    else:
                        agent_sleep(0.5, control)
                else:
                    print(Fore.RED + "  [Bug] Candidate scorer returned empty payload")
                continue

            chosen_card = choose_card_to_play(sorted_indices, id_to_action, hand_cards, energy, exploration=exploration)

            if chosen_card:
                chosen_candidate = choose_candidate_for_card(legal_candidates, chosen_card)
                chosen_action = f"play_card_{chosen_card.get('id')}"
                chosen_candidate_label = chosen_candidate.label if chosen_candidate else chosen_action
                print(Fore.GREEN + Style.BRIGHT + f"  >>> EXECUTE: {chosen_candidate_label}")
                payload = chosen_candidate.payload if chosen_candidate else build_play_card_payload_for_card(chosen_card, alive_enemies)
                payload = payload_with_policy(payload, combat_policy_name, combat_model_version)
                write_ai_logic_snapshot({
                    "timestamp": int(time.time() * 1000),
                    "session_id": session_id,
                    "state_type": state_type,
                    "policy_name": combat_policy_name,
                    "model_version": combat_model_version,
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
                    "constraint_mode": exploration.get("constraint_mode", "guarded"),
                    "exploration": exploration,
                })
                
                if payload:
                    print(Fore.WHITE + f"  Sending: {json.dumps(payload)}")
                    if last_data_source != "ai":
                        set_data_source("ai")
                        last_data_source = "ai"
                    success = send_action(payload)
                    state_after = fetch_game_state()
                    append_ai_action_log(session_id, payload, state, state_after, success, combat_policy_name, combat_model_version)
                    if success:
                        agent_sleep(1.5, control)  # 等动画播完
                    else:
                        agent_sleep(0.5, control)  # 失败了也稍微等一下再重试
                else:
                    print(Fore.RED + "  [Bug] Could not build payload")
            else:
                print(Fore.RED + "  [No affordable playable card found, ending turn]")
                end_turn_candidate = next((c for c in legal_candidates if c.kind == "end_turn"), None)
                payload = end_turn_candidate.payload if end_turn_candidate else {"action": "end_turn"}
                payload = payload_with_policy(payload, combat_policy_name, combat_model_version)
                write_ai_logic_snapshot({
                    "timestamp": int(time.time() * 1000),
                    "session_id": session_id,
                    "state_type": state_type,
                    "policy_name": combat_policy_name,
                    "model_version": combat_model_version,
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
                    "constraint_mode": exploration.get("constraint_mode", "guarded"),
                    "exploration": exploration,
                })
                if last_data_source != "ai":
                    set_data_source("ai")
                    last_data_source = "ai"
                success = send_action(payload)
                state_after = fetch_game_state()
                append_ai_action_log(session_id, payload, state, state_after, success, combat_policy_name, combat_model_version)
                agent_sleep(1.5, control)
                        
if __name__ == "__main__":
    try:
        run_agent()
    except Exception:
        error_path = os.path.join(os.path.dirname(__file__), "ai_agent_error.log")
        with open(error_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] ai_agent crashed\n")
            traceback.print_exc(file=f)
        raise
