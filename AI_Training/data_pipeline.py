import os
import json
import glob
import numpy as np
import time

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


def _read_json_file(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


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


def _parse_card_cost(card, energy):
    cost = card.get("cost", 0)
    if cost == "X":
        return energy
    try:
        return int(cost)
    except (TypeError, ValueError):
        return 99


def _has_affordable_playable_card(state):
    player = state.get("player", {})
    energy = player.get("energy", state.get("energy", 0))
    hand = player.get("hand", state.get("hand", []))
    for card in hand:
        if card.get("can_play", False) and _parse_card_cost(card, energy) <= energy:
            return True
    return False


def should_use_record(data, state, action):
    """过滤掉 AI 自举失败样本，以及明显浪费费用的 end_turn 标签。"""
    if not _record_is_collectable(data.get("timestamp")):
        return False

    if data.get("run_id") in _discarded_run_ids():
        return False

    run_quality = _run_quality(data.get("run_id"))
    min_quality = _min_training_quality()
    if QUALITY_ORDER.get(run_quality, 0) < QUALITY_ORDER.get(min_quality, 0):
        return False

    if data.get("source") == "ai" and not _include_ai_in_training():
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
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        decoder = json.JSONDecoder()
        idx = 0
        length = len(content)
        while idx < length:
            # Skip whitespace
            while idx < length and content[idx].isspace():
                idx += 1
            if idx >= length: break
            
            try:
                obj, new_idx = decoder.raw_decode(content, idx)
                yield obj
                idx = new_idx
            except json.JSONDecodeError:
                idx += 1 # Not ideal but prevents infinite loops if corrupted

    def scan_files(self, filepaths):
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

                    if state and action and should_use_record(data, state, action):
                        self._process_state(state)
                        self._process_action(action)
                        count += 1
                except Exception as e:
                    pass
        print(f"Done. Scanned {count} valid state-action pairs.")

    def _process_state(self, state):
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
        player = state.get("player", {})
        battle = state.get("battle", {})
        
        features = []
        
        # 1. 玩家全局状态 (连续特征，尽量归一化)
        cur_hp = player.get("hp", 0)
        max_hp = max(player.get("max_hp", 1), 1)
        block = player.get("block", 0)
        energy = player.get("energy", 0)
        max_energy = max(player.get("max_energy", 1), 1)
        
        features.extend([
            cur_hp / max_hp,
            min(block / 50.0, 1.0), # 软裁剪
            energy / max_energy,
            player.get("draw_pile_count", 0) / 30.0,
            player.get("discard_pile_count", 0) / 30.0,
            player.get("exhaust_pile_count", 0) / 30.0
        ])
        
        # 2. 手牌特征 (MAX_HAND 个槽位)
        # 每个牌槽位: [card_vocab_id, normalized_cost, is_upgraded]
        hand = player.get("hand", [])
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
        enemies = battle.get("enemies", [])
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
    
    # 获取所有日志文件
    filepaths = []
    for d in data_dirs:
        filepaths.extend(glob.glob(os.path.join(d, "*.jsonl")))
        
    print(f"Found {len(filepaths)} data files.")
    
    # 1. 扫描词表
    if not os.path.exists(vocab_path):
        builder = VocabBuilder()
        builder.scan_files(filepaths)
        builder.save(vocab_path)
    else:
        print("Using existing vocab...")

    # 2. 编码数据
    encoder = StateEncoder(vocab_path)
    
    X_data = [] # 状态向量
    Y_data = [] # 动作ID (标签)
    
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
                    and should_use_record(data, state, action)
                ):
                    state_vec = encoder.encode(state)
                    action_id = encoder.encode_action(action)
                    
                    if action_id != 1: # 排除 UNKNOWN
                        X_data.append(state_vec)
                        Y_data.append(action_id)
            except Exception as e:
                pass
                    
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.int64)
    
    print(f"Encoded {len(X_data)} samples in {time.time() - start_time:.2f} seconds.")
    
    # 3. 保存 Numpy 数据集
    np.save(os.path.join(output_dir, 'X_train.npy'), X_data)
    np.save(os.path.join(output_dir, 'Y_train.npy'), Y_data)
    print(f"Saved matrices. X shape: {X_data.shape}, Y shape: {Y_data.shape}")

if __name__ == "__main__":
    DATA_DIRS = [
        os.path.join(WORKSPACE_DIR, "RL_Datasets", "Combat"),
        os.path.join(WORKSPACE_DIR, "RL_Datasets", "Human", "Combat"),
        os.path.join(WORKSPACE_DIR, "RL_Datasets", "AI_Combat"),
    ]
    OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "AI_Training", "ProcessedParams")
    
    build_dataset(DATA_DIRS, OUTPUT_DIR)
