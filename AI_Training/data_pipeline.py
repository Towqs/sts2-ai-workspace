import os
import json
import glob
import hashlib
import numpy as np
import time
import re
from datetime import datetime

from combat_actions import (
    CANDIDATE_FEATURE_DIM,
    candidate_feature_rows,
    enumerate_combat_actions,
    match_logged_action,
)

# 超参数/常量定义
MAX_HAND = 10
MAX_ENEMIES = 5
MAX_RELICS = 20
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL_PATH = os.path.join(os.path.dirname(__file__), "control_state.json")
DISCARDED_RUNS_PATH = os.path.join(WORKSPACE_DIR, "RL_Datasets", "discarded_runs.json")
RUN_LABELS_PATH = os.path.join(WORKSPACE_DIR, "RL_Datasets", "run_labels.json")
QUALITY_ORDER = {
    "failed_run": -1,
    "unknown": 0,
    "before_act1_boss": 0,
    "partial_act1": 1,
    "partial_act2": 2,
    "perfect_run": 3,
}


def _path_signature(path):
    rel = os.path.relpath(path, WORKSPACE_DIR)
    try:
        stat = os.stat(path)
        return {
            "path": rel,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except OSError:
        return {
            "path": rel,
            "missing": True,
        }


def _vocab_build_signature(filepaths, ctx):
    payload = {
        "files": [_path_signature(path) for path in sorted(filepaths)],
        "discarded": _path_signature(DISCARDED_RUNS_PATH),
        "labels": _path_signature(RUN_LABELS_PATH),
        "filter": {
            "include_ai": bool(ctx.include_ai),
            "min_quality": ctx.min_quality,
            "ai_min_quality": ctx.ai_min_quality,
            "ai_accept_failed_after_act1": bool(ctx.ai_accept_failed_after_act1),
            "ai_require_no_invalid_actions": bool(ctx.ai_require_no_invalid_actions),
            "disabled_since": ctx.disabled_since,
            "disabled_ranges": ctx.disabled_ranges,
        },
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def iter_json_objects(filepath):
    """Yield JSON objects from JSONL or concatenated JSON logs."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    decoder = json.JSONDecoder()
    idx = 0
    length = len(content)
    while idx < length:
        while idx < length and content[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            obj, new_idx = decoder.raw_decode(content, idx)
            if isinstance(obj, dict):
                yield obj
            idx = new_idx
        except json.JSONDecodeError:
            idx += 1


def _read_json_file(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _record_state(record):
    state = record.get("state_before") or record.get("state") or {}
    return state if isinstance(state, dict) else {}


def _run_values_from_record(record):
    state = _record_state(record)
    run = state.get("run") if isinstance(state.get("run"), dict) else state
    return _safe_int(run.get("act")), _safe_int(run.get("floor"))


def _record_is_invalid(record):
    if record.get("ok") is False:
        return True
    action_type = str(record.get("action_type") or record.get("type") or "").lower()
    return action_type in {"invalid_action", "error"}


def _build_run_progress(filepaths):
    progress = {}
    for filepath in filepaths:
        for record in iter_json_objects(filepath):
            run_id = record.get("run_id")
            if not run_id:
                continue
            item = progress.setdefault(run_id, {
                "records": 0,
                "max_act": 0,
                "max_floor": 0,
                "invalid_actions": 0,
                "sources": set(),
            })
            act, floor = _run_values_from_record(record)
            item["records"] += 1
            item["max_act"] = max(item["max_act"], act)
            item["max_floor"] = max(item["max_floor"], floor)
            item["sources"].add(record.get("source") or "unknown")
            if _record_is_invalid(record):
                item["invalid_actions"] += 1
    return progress


class FilterContext:
    """Pre-load all filter config once so we don't re-read JSON files per record."""

    def __init__(self):
        control = _read_json_file(CONTROL_PATH, {})
        self.include_ai = bool(control.get("include_ai_in_training", False))
        ai_min_q = control.get("ai_min_training_quality", "partial_act1")
        self.ai_min_quality = ai_min_q if ai_min_q in QUALITY_ORDER else "partial_act1"
        self.ai_accept_failed_after_act1 = bool(control.get("ai_accept_failed_after_act1", True))
        self.ai_require_no_invalid_actions = bool(control.get("ai_require_no_invalid_actions", True))
        self.disabled_since = control.get("collection_disabled_since")
        self.disabled_ranges = control.get("collection_disabled_ranges", [])
        min_q = control.get("min_training_quality", "unknown")
        self.min_quality = min_q if min_q in QUALITY_ORDER else "unknown"
        self.discarded = set(
            _read_json_file(DISCARDED_RUNS_PATH, {"discarded": []}).get("discarded", [])
        )
        self.labels = _read_json_file(RUN_LABELS_PATH, {"labels": {}}).get("labels", {})
        self.run_progress = {}

    def prepare(self, filepaths):
        self.run_progress = _build_run_progress(filepaths)

    def record_is_collectable(self, timestamp):
        ts = int(timestamp or 0)
        if self.disabled_since and ts >= int(self.disabled_since):
            return False
        for start, end in self.disabled_ranges:
            if int(start) <= ts <= int(end):
                return False
        return True

    def run_quality(self, run_id):
        return self.labels.get(run_id, {}).get("quality", "unknown")

    def run_reached_act2(self, run_id):
        progress = self.run_progress.get(run_id, {})
        return progress.get("max_act", 0) >= 2 or progress.get("max_floor", 0) > 17

    def run_has_invalid_actions(self, run_id):
        return self.run_progress.get(run_id, {}).get("invalid_actions", 0) > 0

    def source_allowed(self, data):
        run_id = data.get("run_id")
        source = data.get("source")
        if source != "ai":
            rq = self.run_quality(run_id)
            return QUALITY_ORDER.get(rq, 0) >= QUALITY_ORDER.get(self.min_quality, 0)
        if not self.include_ai:
            return False
        if self.ai_require_no_invalid_actions and self.run_has_invalid_actions(run_id):
            return False
        rq = self.run_quality(run_id)
        if QUALITY_ORDER.get(rq, 0) >= QUALITY_ORDER.get(self.ai_min_quality, 0):
            return True
        if self.ai_accept_failed_after_act1 and rq in {"failed_run", "unknown"}:
            return self.run_reached_act2(run_id)
        return False


# Keep module-level helpers for backward compatibility (used outside build_dataset)
def _include_ai_in_training():
    control = _read_json_file(CONTROL_PATH, {})
    return bool(control.get("include_ai_in_training", False))


def _record_is_collectable(timestamp):
    control = _read_json_file(CONTROL_PATH, {})
    ts = int(timestamp or 0)
    disabled_since = control.get("collection_disabled_since")
    if disabled_since and ts >= int(disabled_since):
        return False
    for start, end in control.get("collection_disabled_ranges", []):
        if int(start) <= ts <= int(end):
            return False
    return True


def _discarded_run_ids():
    data = _read_json_file(DISCARDED_RUNS_PATH, {"discarded": []})
    return set(data.get("discarded", []))


def _run_quality(run_id):
    labels = _read_json_file(RUN_LABELS_PATH, {"labels": {}}).get("labels", {})
    return labels.get(run_id, {}).get("quality", "unknown")


def _min_training_quality():
    control = _read_json_file(CONTROL_PATH, {})
    quality = control.get("min_training_quality", "unknown")
    return quality if quality in QUALITY_ORDER else "unknown"


def _ai_min_training_quality():
    control = _read_json_file(CONTROL_PATH, {})
    quality = control.get("ai_min_training_quality", "partial_act1")
    return quality if quality in QUALITY_ORDER else "partial_act1"


def _parse_card_cost(card, energy):
    cost = card.get("cost", 0)
    if cost == "X":
        return energy
    try:
        return int(cost)
    except (TypeError, ValueError):
        return 99


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_number_text(text):
    return [int(n) for n in re.findall(r"\d+", str(text or ""))]


def _intent_damage(intent):
    label = str(intent.get("label") or "")
    nums = _parse_number_text(label)
    if "x" in label.lower() or "×" in label:
        return nums[0] * nums[1] if len(nums) >= 2 else (nums[0] if nums else 0)
    if nums:
        return nums[0]
    nums = _parse_number_text(intent.get("description"))
    return nums[0] if nums else 0


def _card_number_hint(card, mode):
    text = " ".join(str(card.get(k) or "") for k in ("description", "name", "id", "type"))
    lowered = text.lower()
    if mode == "attack" and not any(k in lowered for k in ("deal", "damage", "attack", "造成", "伤害")):
        return 0
    if mode == "block" and not any(k in lowered for k in ("block", "defend", "armor", "格挡", "护甲")):
        return 0
    nums = _parse_number_text(text)
    return nums[0] if nums else 0


def _normalize_combat_state(state):
    """Accept both API-shaped and collector-minimal combat states."""
    if not isinstance(state, dict):
        return {}
    if isinstance(state.get("player"), dict) and isinstance(state.get("battle"), dict):
        return state

    player = {
        "character": state.get("character"),
        "hp": state.get("hp", 0),
        "max_hp": state.get("max_hp", 1),
        "block": state.get("block", 0),
        "energy": state.get("energy", 0),
        "max_energy": state.get("max_energy", 1),
        "hand": state.get("hand", []),
        "draw_pile_count": state.get("draw_pile_count", 0),
        "discard_pile_count": state.get("discard_pile_count", 0),
        "exhaust_pile_count": state.get("exhaust_pile_count", 0),
        "relics": state.get("relics", []),
        "potions": state.get("potions", []),
        "gold": state.get("gold", 0),
        "status": state.get("status", []),
    }
    enemies = []
    for enemy in state.get("enemies", []) or []:
        if not isinstance(enemy, dict):
            continue
        intents = enemy.get("intents", [])
        if intents and isinstance(intents[0], str):
            intents = [{"type": value, "label": "", "description": value} for value in intents]
        enemies.append({
            "entity_id": enemy.get("entity_id") or enemy.get("id") or enemy.get("name"),
            "name": enemy.get("name") or enemy.get("id") or enemy.get("entity_id") or "UNKNOWN",
            "hp": enemy.get("hp", 0),
            "max_hp": enemy.get("max_hp", 1),
            "block": enemy.get("block", 0),
            "status": enemy.get("status", []),
            "intents": intents or [],
        })
    battle = {
        "round": state.get("round", 0),
        "turn": str(state.get("turn", "")).lower(),
        "is_play_phase": state.get("is_play_phase", False),
        "enemies": enemies,
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


def _has_affordable_playable_card(state):
    state = _normalize_combat_state(state)
    player = state.get("player", {})
    energy = player.get("energy", state.get("energy", 0))
    hand = player.get("hand", state.get("hand", []))
    for card in hand:
        if card.get("can_play", False) and _parse_card_cost(card, energy) <= energy:
            return True
    return False


def should_use_record(data, state, action, ctx=None):
    """过滤掉 AI 自举失败样本，以及明显浪费费用的 end_turn 标签。
    ctx: 可选 FilterContext，传入时避免每条记录重复读磁盘。
    """
    if ctx:
        if not ctx.record_is_collectable(data.get("timestamp")):
            return False
        if data.get("run_id") in ctx.discarded:
            return False
        if not ctx.source_allowed(data):
            return False
    else:
        if not _record_is_collectable(data.get("timestamp")):
            return False
        if data.get("run_id") in _discarded_run_ids():
            return False
        run_quality = _run_quality(data.get("run_id"))
        if data.get("source") == "ai":
            if not _include_ai_in_training() or _record_is_invalid(data):
                return False
            min_quality = _ai_min_training_quality()
            if QUALITY_ORDER.get(run_quality, 0) < QUALITY_ORDER.get(min_quality, 0):
                return False
        else:
            min_quality = _min_training_quality()
            if QUALITY_ORDER.get(run_quality, 0) < QUALITY_ORDER.get(min_quality, 0):
                return False

    act_type = action.get("action", action.get("action_type", data.get("action_type", "UNKNOWN")))
    if act_type == "end_turn" and _has_affordable_playable_card(state):
        return False

    return True

class VocabBuilder:
    """第一遍扫描数据文件，构建全局词表（实体ID映射到整型索引）"""
    def __init__(self):
        self.cards = {"PAD": 0, "UNKNOWN": 1}
        self.relics = {"PAD": 0, "UNKNOWN": 1}
        self.enemies = {"PAD": 0, "UNKNOWN": 1}
        self.status = {"PAD": 0, "UNKNOWN": 1}
        self.actions = {"PAD": 0, "UNKNOWN": 1, "end_turn": 2}
        self.intent_types = {"PAD": 0, "UNKNOWN": 1}

    def _add_to_vocab(self, vocab, item):
        if item is not None and str(item).strip() != "":
            if item not in vocab:
                vocab[item] = len(vocab)
            
    def _iter_json_objects(self, filepath):
        """Helper to yield JSON objects from a file that might have newlines or multiple root objects."""
        yield from iter_json_objects(filepath)

    def scan_files(self, filepaths, ctx=None):
        print(f"Scanning {len(filepaths)} files to build vocab...")
        count = 0
        for filepath in filepaths:
            for data in self._iter_json_objects(filepath):
                try:
                    state = data.get("state", data.get("state_before"))
                    action = data.get("action", data.get("action_data"))
                    
                    # 兼容 v3 格式里的顶层类型
                    act_type = data.get("action_type") or data.get("type")
                    if action is None and act_type in ["action", "play_card", "end_turn"]: 
                        action = data

                    if state and action and should_use_record(data, state, action, ctx=ctx):
                        self._process_state(state)
                        self._process_action(action)
                        count += 1
                except Exception as e:
                    pass
        print(f"Done. Scanned {count} valid state-action pairs.")

    def _process_state(self, state):
        state = _normalize_combat_state(state)
        player = state.get("player", {})
        battle = state.get("battle", {})
        
        for c in player.get("hand", []):
            self._add_to_vocab(self.cards, c.get("id"))
            
        for r in player.get("relics", []):
            self._add_to_vocab(self.relics, r.get("id"))
            
        for s in player.get("status", []):
            self._add_to_vocab(self.status, s.get("id"))
            
        for e in battle.get("enemies", []):
            # entity_id contains unique suffix like NIBBIT_0, we want the base enemy type: name or prefix
            e_name = e.get("name", e.get("entity_id", "UNKNOWN"))
            self._add_to_vocab(self.enemies, e_name)
            for s in e.get("status", []):
                self._add_to_vocab(self.status, s.get("id"))
            for intent in e.get("intents", []):
                self._add_to_vocab(self.intent_types, intent.get("type"))
                
    def _process_action(self, action):
        act_type = action.get("action", action.get("action_type", "UNKNOWN"))
        if act_type == "play_card":
            card_id = action.get("card_id")
            act_name = f"play_card_{card_id}"
            self._add_to_vocab(self.actions, act_name)
            
    def save(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "cards": self.cards,
                "relics": self.relics,
                "enemies": self.enemies,
                "status": self.status,
                "actions": self.actions,
                "intent_types": self.intent_types
            }, f, indent=2, ensure_ascii=False)
        print(f"Vocab saved to {output_path}")
        print(f"Vocab sizes: Cards={len(self.cards)}, Enemies={len(self.enemies)}, Actions={len(self.actions)}")


class StateEncoder:
    """第二遍编码数据：将状态JSON转为数值形式的特征向量"""
    def __init__(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocabs = json.load(f)
            self.cards = vocabs["cards"]
            self.relics = vocabs["relics"]
            self.enemies = vocabs["enemies"]
            self.status = vocabs["status"]
            self.actions = vocabs["actions"]
            self.intent_types = vocabs["intent_types"]
            
        self.action_size = len(self.actions)

    def _get_id(self, vocab, key):
        return vocab.get(key, 1) # 1 is UNKNOWN
        
    def encode(self, state):
        state = _normalize_combat_state(state)
        player = state.get("player", {})
        battle = state.get("battle", {})
        run = state.get("run", {})
        
        features = []
        
        # 1. 玩家全局状态 (连续特征，尽量归一化)
        cur_hp = player.get("hp", 0)
        max_hp = max(player.get("max_hp", 1), 1)
        block = player.get("block", 0)
        energy = player.get("energy", 0)
        max_energy = max(player.get("max_energy", 1), 1)
        hand = player.get("hand", [])
        enemies = [e for e in battle.get("enemies", []) if _safe_float(e.get("hp")) > 0]
        incoming_damage = sum(_intent_damage(intent) for e in enemies for intent in e.get("intents", []))
        net_incoming = max(0.0, incoming_damage - _safe_float(block))
        playable = [c for c in hand if c.get("can_play", False)]
        affordable = [c for c in playable if _parse_card_cost(c, energy) <= energy]
        zero_cost = [c for c in affordable if _parse_card_cost(c, energy) == 0]
        affordable_attack = sum(_card_number_hint(c, "attack") for c in affordable)
        affordable_block = sum(_card_number_hint(c, "block") for c in affordable)
        enemy_effective_hp = [_safe_float(e.get("hp")) + _safe_float(e.get("block")) for e in enemies]
        can_kill_any = bool(enemy_effective_hp and affordable_attack >= min(enemy_effective_hp))
        
        features.extend([
            cur_hp / max_hp,
            min(block / 50.0, 1.0), # 软裁剪
            energy / max_energy,
            player.get("draw_pile_count", 0) / 30.0,
            player.get("discard_pile_count", 0) / 30.0,
            player.get("exhaust_pile_count", 0) / 30.0,
            _safe_float(run.get("act")) / 4.0,
            _safe_float(run.get("floor")) / 60.0,
            _safe_float(battle.get("round")) / 20.0,
            min(incoming_damage / max_hp, 2.0),
            min(net_incoming / max_hp, 2.0),
            max((cur_hp - net_incoming) / max_hp, -1.0),
            len(enemies) / MAX_ENEMIES,
            len(playable) / MAX_HAND,
            len(affordable) / MAX_HAND,
            len(zero_cost) / MAX_HAND,
            min(affordable_attack / 100.0, 2.0),
            min(affordable_block / 100.0, 2.0),
            1.0 if can_kill_any else 0.0,
            1.0 if net_incoming >= cur_hp and net_incoming > 0 else 0.0,
            1.0 if net_incoming >= max_hp * 0.3 else 0.0
        ])
        
        # 2. 手牌特征 (MAX_HAND 个槽位)
        # 每个牌槽位: [card_vocab_id, normalized_cost, is_upgraded]
        for i in range(MAX_HAND):
            if i < len(hand):
                c = hand[i]
                c_id = self._get_id(self.cards, c.get("id"))
                cost = 0
                try: 
                    cost_str = c.get("cost", "0")
                    if cost_str not in ["X", "Unplayable"]:
                        cost = int(cost_str)
                except:
                    cost = 0
                is_upg = 1.0 if c.get("is_upgraded", False) else 0.0
                features.extend([float(c_id), float(cost) / max_energy, is_upg])
            else:
                features.extend([0.0, 0.0, 0.0]) # PAD
                
        # 3. 敌人特征 (MAX_ENEMIES 个槽位)
        # 每个敌人槽位: [enemy_vocab_id, hp_norm, block_norm, primary_intent_vocab_id, intent_dmg_norm]
        for i in range(MAX_ENEMIES):
            if i < len(enemies):
                e = enemies[i]
                e_name = e.get("name", e.get("entity_id", "UNKNOWN"))
                e_id = self._get_id(self.enemies, e_name)
                e_hp = e.get("hp", 0)
                e_max_hp = max(e.get("max_hp", 1), 1)
                e_block = e.get("block", 0)
                
                intents = e.get("intents", [])
                primary_intent = intents[0] if intents else {}
                i_type = self._get_id(self.intent_types, primary_intent.get("type"))
                
                dmg = 0
                try:
                    if primary_intent.get("label"): dmg = int(primary_intent.get("label"))
                except: pass
                
                features.extend([float(e_id), e_hp / e_max_hp, min(e_block / 50.0, 1.0), float(i_type), min(dmg / max_hp, 1.0)])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0]) # PAD

        # TODO 还可以增加遗物特征模型
        return np.array(features, dtype=np.float32)

    def encode_action(self, action):
        act_type = action.get("action", action.get("action_type", "UNKNOWN"))
        if act_type == "play_card":
            card_id = action.get("card_id")
            act_name = f"play_card_{card_id}"
            return self._get_id(self.actions, act_name)
        elif act_type == "end_turn":
            return self._get_id(self.actions, "end_turn")
        else:
            return 1 # UNKNOWN


def build_dataset(data_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.json")
    
    # 一次性加载过滤配置，避免每条记录都读磁盘
    ctx = FilterContext()

    # 获取所有日志文件
    filepaths = []
    for d in data_dirs:
        filepaths.extend(glob.glob(os.path.join(d, "*.jsonl")))
        
    print(f"Found {len(filepaths)} data files.")
    ctx.prepare(filepaths)
    vocab_signature = _vocab_build_signature(filepaths, ctx)
    
    # 1. 扫描词表（当文件或过滤签名变化时强制重建）
    rebuild_vocab = True
    if os.path.exists(vocab_path):
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                old_vocab = json.load(f)
            old_signature = old_vocab.get("_build_signature")
            if old_signature == vocab_signature:
                rebuild_vocab = False
                print("Using existing vocab (dataset signature unchanged)...")
            else:
                print("Vocab outdated (dataset signature changed). Rebuilding...")
        except Exception:
            pass
    if rebuild_vocab:
        builder = VocabBuilder()
        builder.scan_files(filepaths, ctx=ctx)
        builder.save(vocab_path)
        # 追加构建元信息到 vocab.json
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            vocab_data["_build_file_count"] = len(filepaths)
            vocab_data["_build_include_ai"] = ctx.include_ai
            vocab_data["_build_signature"] = vocab_signature
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # 2. 编码数据
    encoder = StateEncoder(vocab_path)
    
    X_data = [] # 状态向量
    Y_data = [] # 动作ID (标签)
    candidate_X_data = [] # 状态向量 + 候选动作特征
    candidate_Y_data = [] # 1 表示该候选动作匹配人类记录
    candidate_group_data = [] # 同一个原始样本下的候选动作分组
    candidate_groups = 0
    candidate_match_misses = 0
    accepted_sources = {}
    accepted_run_qualities = {}
    accepted_runs = {}  # run_id -> {source, quality, samples}
    skipped_runs = {}   # run_id -> reason
    
    print("Encoding data into features...")
    start_time = time.time()
    
    for filepath in filepaths:
        for data in VocabBuilder()._iter_json_objects(filepath):
            try:
                state = data.get("state", data.get("state_before"))
                action = data.get("action", data.get("action_data"))
                
                # 兼容新格式
                act_type = data.get("action_type") or data.get("type")
                if action is None and act_type in ["action", "play_card", "end_turn"]: 
                    action = data

                # 只采集有效的战斗行为
                if (
                    state
                    and action
                    and action.get("action") != "battle_start"
                    and should_use_record(data, state, action, ctx=ctx)
                ):
                    state_vec = encoder.encode(state)
                    action_id = encoder.encode_action(action)
                    
                    if action_id != 1: # 排除 UNKNOWN
                        group_id = len(Y_data)
                        X_data.append(state_vec)
                        Y_data.append(action_id)
                        source = data.get("source") or "unknown"
                        accepted_sources[source] = accepted_sources.get(source, 0) + 1
                        run_id = data.get("run_id") or "unknown"
                        quality = ctx.run_quality(run_id)
                        accepted_run_qualities[quality] = accepted_run_qualities.get(quality, 0) + 1
                        # per-run tracking
                        if run_id not in accepted_runs:
                            accepted_runs[run_id] = {"source": source, "quality": quality, "samples": 0}
                        accepted_runs[run_id]["samples"] += 1

                        candidates = enumerate_combat_actions(state)
                        matched_idx = match_logged_action(candidates, action)
                        if matched_idx >= 0:
                            for idx, candidate_features in enumerate(candidate_feature_rows(candidates)):
                                candidate_X_data.append(np.concatenate([
                                    state_vec,
                                    np.array(candidate_features, dtype=np.float32),
                                ]))
                                candidate_Y_data.append(1 if idx == matched_idx else 0)
                                candidate_group_data.append(group_id)
                            candidate_groups += 1
                        else:
                            candidate_match_misses += 1
            except Exception as e:
                pass
                    
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.int64)
    candidate_feature_total = (X_data.shape[1] if X_data.ndim == 2 else 0) + CANDIDATE_FEATURE_DIM
    candidate_X_data = (
        np.array(candidate_X_data, dtype=np.float32)
        if candidate_X_data
        else np.zeros((0, candidate_feature_total), dtype=np.float32)
    )
    candidate_Y_data = np.array(candidate_Y_data, dtype=np.int64)
    candidate_group_data = np.array(candidate_group_data, dtype=np.int64)
    
    elapsed = time.time() - start_time
    human_samples = accepted_sources.get("human", 0)
    ai_samples = accepted_sources.get("ai", 0)
    total_samples = int(len(Y_data))
    print(f"Encoded {total_samples} samples in {elapsed:.2f} seconds.")
    print(f"  Human: {human_samples} ({human_samples*100/max(total_samples,1):.1f}%)  AI: {ai_samples} ({ai_samples*100/max(total_samples,1):.1f}%)  Runs: {len(accepted_runs)}")
    
    # 3. 保存 Numpy 数据集
    np.save(os.path.join(output_dir, 'X_train.npy'), X_data)
    np.save(os.path.join(output_dir, 'Y_train.npy'), Y_data)
    np.save(os.path.join(output_dir, 'candidate_X_train.npy'), candidate_X_data)
    np.save(os.path.join(output_dir, 'candidate_Y_train.npy'), candidate_Y_data)
    np.save(os.path.join(output_dir, 'candidate_group_train.npy'), candidate_group_data)

    # per-run 详细列表（按样本数降序）
    runs_detail = []
    for rid, info in sorted(accepted_runs.items(), key=lambda x: -x[1]["samples"]):
        runs_detail.append({
            "run_id": rid,
            "source": info["source"],
            "quality": info["quality"],
            "samples": info["samples"],
        })

    metadata = {
        "samples": total_samples,
        "human_samples": human_samples,
        "ai_samples": ai_samples,
        "human_ratio": round(human_samples / max(total_samples, 1), 4),
        "ai_ratio": round(ai_samples / max(total_samples, 1), 4),
        "accepted_run_count": len(accepted_runs),
        "accepted_runs": runs_detail,
        "features": int(X_data.shape[1]) if X_data.ndim == 2 else 0,
        "feature_version": 2,
        "candidate_rows": int(len(candidate_Y_data)),
        "candidate_groups": int(candidate_groups),
        "candidate_positive": int(candidate_Y_data.sum()) if len(candidate_Y_data) else 0,
        "candidate_match_misses": int(candidate_match_misses),
        "candidate_feature_dim": int(CANDIDATE_FEATURE_DIM),
        "candidate_total_features": int(candidate_X_data.shape[1]) if candidate_X_data.ndim == 2 else 0,
        "accepted_sources": accepted_sources,
        "accepted_run_qualities": accepted_run_qualities,
        "include_ai": ctx.include_ai,
        "min_training_quality": ctx.min_quality,
        "ai_min_training_quality": ctx.ai_min_quality,
        "ai_accept_failed_after_act1": ctx.ai_accept_failed_after_act1,
        "ai_require_no_invalid_actions": ctx.ai_require_no_invalid_actions,
        "ai_qualified_run_ids": sorted(
            run_id
            for run_id, progress in ctx.run_progress.items()
            if "ai" in progress.get("sources", set())
            and (progress.get("max_act", 0) >= 2 or progress.get("max_floor", 0) > 17)
        ),
        "build_timestamp": datetime.now().isoformat(timespec="seconds"),
        "build_elapsed_sec": round(elapsed, 2),
        "data_file_count": len(filepaths),
        "data_signature": vocab_signature,
        "feature_notes": [
            "act_floor_round",
            "incoming_damage",
            "net_incoming_after_block",
            "hp_after_incoming",
            "affordable_playable_counts",
            "affordable_attack_block_estimates",
            "lethal_and_threat_flags",
            "candidate_action_feature_rows_for_future_scorer",
        ],
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved matrices. X shape: {X_data.shape}, Y shape: {Y_data.shape}")
    print(f"Candidate matrices. X shape: {candidate_X_data.shape}, positives: {int(candidate_Y_data.sum()) if len(candidate_Y_data) else 0}, misses: {candidate_match_misses}")

if __name__ == "__main__":
    DATA_DIRS = [
        os.path.join(WORKSPACE_DIR, "RL_Datasets", "Combat"),
        os.path.join(WORKSPACE_DIR, "RL_Datasets", "Human", "Combat"),
        os.path.join(WORKSPACE_DIR, "RL_Datasets", "AI", "Combat"),
        os.path.join(WORKSPACE_DIR, "RL_Datasets", "AI_Combat"),
    ]
    OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "AI_Training", "ProcessedParams")
    
    build_dataset(DATA_DIRS, OUTPUT_DIR)
