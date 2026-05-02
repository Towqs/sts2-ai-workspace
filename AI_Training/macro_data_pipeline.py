import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE_DIR / "RL_Datasets"
CONTROL_PATH = WORKSPACE_DIR / "AI_Training" / "control_state.json"
DISCARDED_RUNS_PATH = DATA_DIR / "discarded_runs.json"
RUN_LABELS_PATH = DATA_DIR / "run_labels.json"
OUTPUT_DIR = WORKSPACE_DIR / "AI_Training" / "ProcessedMacroParams"

QUALITY_ORDER = {
    "failed_run": -1,
    "unknown": 0,
    "before_act1_boss": 0,
    "partial_act1": 1,
    "partial_act2": 2,
    "perfect_run": 3,
}

MAX_RELICS = 12
MAX_POTIONS = 3
MAX_MAP_OPTIONS = 7
MAX_REWARD_ITEMS = 8
MAX_CARD_OPTIONS = 5
MAX_EVENT_OPTIONS = 6

ALLOWED_MACRO_ACTIONS = {
    "select_map_node",
    "claim_reward",
    "choose_card",
    "skip_reward",
    "choose_event_option",
    "choose_rest_option",
    "buy_item",
    "proceed",
    "cancel",
}


def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def run_values_from_record(record):
    state = record.get("state") if isinstance(record.get("state"), dict) else {}
    screen = screen_state(record) if isinstance(record, dict) else {}
    run = ((screen.get("run") or {}) if isinstance(screen.get("run"), dict) else {})
    return (
        safe_int(run.get("act", state.get("act"))),
        safe_int(run.get("floor", state.get("floor"))),
    )


def record_is_invalid(record):
    if record.get("ok") is False:
        return True
    action_type = str(record.get("action_type") or record.get("type") or "").lower()
    return action_type in {"invalid_action", "error"}


def build_run_progress(files):
    progress = {}
    for path in files:
        for record in iter_jsonl(path):
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
            act, floor = run_values_from_record(record)
            item["records"] += 1
            item["max_act"] = max(item["max_act"], act)
            item["max_floor"] = max(item["max_floor"], floor)
            item["sources"].add(record.get("source") or "unknown")
            if record_is_invalid(record):
                item["invalid_actions"] += 1
    return progress


def macro_files():
    roots = [
        DATA_DIR / "Macro",
        DATA_DIR / "Human" / "Macro",
        DATA_DIR / "AI" / "Macro",
        DATA_DIR / "Processed" / "Macro",
    ]
    files = []
    for root in roots:
        if root.exists():
            files.extend(root.glob("*.jsonl"))
    return sorted(files)


def control():
    return read_json(CONTROL_PATH, {})


def include_ai():
    return bool(control().get("include_ai_in_training", False))


def record_is_collectable(record):
    ctl = control()
    ts = int(record.get("timestamp") or 0)
    disabled_since = ctl.get("collection_disabled_since")
    if disabled_since and ts >= int(disabled_since):
        return False
    for span in ctl.get("collection_disabled_ranges", []):
        if len(span) == 2 and int(span[0]) <= ts <= int(span[1]):
            return False
    return True


def discarded_run_ids():
    return set(read_json(DISCARDED_RUNS_PATH, {"discarded": []}).get("discarded", []))


def run_quality(run_id):
    labels = read_json(RUN_LABELS_PATH, {"labels": {}}).get("labels", {})
    return labels.get(run_id, {}).get("quality", "unknown")


def min_training_quality():
    quality = control().get("min_training_quality", "unknown")
    return quality if quality in QUALITY_ORDER else "unknown"


def ai_min_training_quality():
    quality = control().get("ai_min_training_quality", "partial_act1")
    return quality if quality in QUALITY_ORDER else "partial_act1"


class FilterContext:
    def __init__(self, files):
        self.control = control()
        self.include_ai = bool(self.control.get("include_ai_in_training", False))
        self.min_quality = self.control.get("min_training_quality", "unknown")
        if self.min_quality not in QUALITY_ORDER:
            self.min_quality = "unknown"
        self.ai_min_quality = self.control.get("ai_min_training_quality", "partial_act1")
        if self.ai_min_quality not in QUALITY_ORDER:
            self.ai_min_quality = "partial_act1"
        self.ai_accept_failed_after_act1 = bool(self.control.get("ai_accept_failed_after_act1", True))
        self.ai_require_no_invalid_actions = bool(self.control.get("ai_require_no_invalid_actions", True))
        self.discarded = discarded_run_ids()
        self.labels = read_json(RUN_LABELS_PATH, {"labels": {}}).get("labels", {})
        self.run_progress = build_run_progress(files)

    def record_is_collectable(self, record):
        ts = int(record.get("timestamp") or 0)
        disabled_since = self.control.get("collection_disabled_since")
        if disabled_since and ts >= int(disabled_since):
            return False
        for span in self.control.get("collection_disabled_ranges", []):
            if len(span) == 2 and int(span[0]) <= ts <= int(span[1]):
                return False
        return True

    def run_quality(self, run_id):
        return self.labels.get(run_id, {}).get("quality", "unknown")

    def run_reached_act2(self, run_id):
        progress = self.run_progress.get(run_id, {})
        return progress.get("max_act", 0) >= 2 or progress.get("max_floor", 0) > 17

    def run_has_invalid_actions(self, run_id):
        return self.run_progress.get(run_id, {}).get("invalid_actions", 0) > 0

    def source_allowed(self, record):
        run_id = record.get("run_id")
        source = record.get("source")
        if source != "ai":
            quality = self.run_quality(run_id)
            return QUALITY_ORDER.get(quality, 0) >= QUALITY_ORDER.get(self.min_quality, 0)
        if not self.include_ai:
            return False
        if self.ai_require_no_invalid_actions and self.run_has_invalid_actions(run_id):
            return False
        quality = self.run_quality(run_id)
        if QUALITY_ORDER.get(quality, 0) >= QUALITY_ORDER.get(self.ai_min_quality, 0):
            return True
        if self.ai_accept_failed_after_act1 and quality in {"failed_run", "unknown"}:
            return self.run_reached_act2(run_id)
        return False


def should_use_record(record, ctx=None):
    if record.get("type") != "macro_action":
        return False
    if record.get("action_type") not in ALLOWED_MACRO_ACTIONS:
        return False
    if not record.get("state"):
        return False
    if not record.get("action_type"):
        return False
    if ctx:
        if not ctx.record_is_collectable(record):
            return False
        if record.get("run_id") in ctx.discarded:
            return False
        if not ctx.source_allowed(record):
            return False
    else:
        if not record_is_collectable(record):
            return False
        if record.get("run_id") in discarded_run_ids():
            return False
        if record.get("source") == "ai":
            if not include_ai() or record_is_invalid(record):
                return False
            if QUALITY_ORDER.get(run_quality(record.get("run_id")), 0) < QUALITY_ORDER.get(ai_min_training_quality(), 0):
                return False
    if record.get("action_type") == "select_map_node":
        screen = record.get("screen_state") or record.get("screen") or {}
        options = ((screen.get("map") or {}).get("next_options") or [])
        if len(options) <= 1:
            return False
    if not ctx and record.get("source") != "ai":
        min_quality = min_training_quality()
        if QUALITY_ORDER.get(run_quality(record.get("run_id")), 0) < QUALITY_ORDER.get(min_quality, 0):
            return False
    return True


class Vocab:
    def __init__(self):
        self.tables = {
            "actions": {"PAD": 0, "UNKNOWN": 1},
            "characters": {"PAD": 0, "UNKNOWN": 1},
            "rooms": {"PAD": 0, "UNKNOWN": 1},
            "screens": {"PAD": 0, "UNKNOWN": 1},
            "relics": {"PAD": 0, "UNKNOWN": 1},
            "potions": {"PAD": 0, "UNKNOWN": 1},
            "node_types": {"PAD": 0, "UNKNOWN": 1},
            "reward_types": {"PAD": 0, "UNKNOWN": 1},
            "cards": {"PAD": 0, "UNKNOWN": 1},
            "card_types": {"PAD": 0, "UNKNOWN": 1},
            "rarities": {"PAD": 0, "UNKNOWN": 1},
        }

    def add(self, table, value):
        key = str(value or "").strip()
        if not key:
            return
        items = self.tables[table]
        if key not in items:
            items[key] = len(items)

    def get(self, table, value):
        return self.tables[table].get(str(value or "").strip(), 1)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.tables, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.tables = read_json(path, obj.tables)
        return obj


def screen_state(record):
    screen = record.get("screen_state")
    return screen if isinstance(screen, dict) else {}


def state_type(record):
    return screen_state(record).get("state_type") or record.get("state", {}).get("room_type") or "unknown"


def norm_action_label(record):
    action_type = record.get("action_type") or "UNKNOWN"
    action_data = record.get("action_data") or {}
    screen = screen_state(record)

    if action_type == "select_map_node":
        col = action_data.get("col")
        row = action_data.get("row")
        for option in (screen.get("map") or {}).get("next_options", []):
            if option.get("col") == col and option.get("row") == row:
                return f"select_map_node:index_{option.get('index')}"
        node_type = action_data.get("node_type") or "unknown"
        return f"select_map_node:type_{node_type}"

    if action_type == "choose_card":
        card_id = action_data.get("card_id") or action_data.get("card_name") or action_data.get("card_title")
        cards = (screen.get("card_reward") or {}).get("cards", [])
        for card in cards:
            if card.get("id") == card_id or card.get("name") == card_id:
                return f"choose_card:index_{card.get('index')}"
        return f"choose_card:{card_id or 'unknown'}"

    if action_type == "claim_reward":
        reward_type = str(action_data.get("reward_type") or "unknown").replace("Reward", "").lower()
        return f"claim_reward:{reward_type}"

    if action_type == "choose_event_option":
        options = (screen.get("event") or {}).get("options", [])
        if len(options) == 1:
            return "choose_event_option:index_0"
        return "choose_event_option:unknown"

    if action_type == "choose_rest_option":
        return f"choose_rest_option:{action_data.get('option') or action_data.get('action') or 'unknown'}"

    if action_type == "buy_item":
        item_id = action_data.get("item_id") or action_data.get("item_name") or action_data.get("action") or "unknown"
        return f"buy_item:{item_id}"

    if action_type in ("skip_reward", "proceed", "cancel"):
        return action_type

    return action_type


def add_record_to_vocab(vocab, record):
    state = record.get("state") or {}
    screen = screen_state(record)
    action_data = record.get("action_data") or {}

    vocab.add("actions", norm_action_label(record))
    vocab.add("characters", state.get("character"))
    vocab.add("rooms", state.get("room_type"))
    vocab.add("screens", state_type(record))

    for relic in state.get("relics", []):
        vocab.add("relics", relic.get("id") or relic.get("name"))
    for potion in state.get("potions", []):
        vocab.add("potions", potion.get("id") or potion.get("name"))

    if record.get("action_type") == "select_map_node":
        vocab.add("node_types", action_data.get("node_type"))
    for option in (screen.get("map") or {}).get("next_options", []):
        vocab.add("node_types", option.get("type"))
    for item in (screen.get("rewards") or {}).get("items", []):
        vocab.add("reward_types", item.get("type"))
    for card in (screen.get("card_reward") or {}).get("cards", []):
        vocab.add("cards", card.get("id") or card.get("name"))
        vocab.add("card_types", card.get("type"))
        vocab.add("rarities", card.get("rarity"))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def card_cost(card):
    cost = card.get("cost")
    if cost == "X":
        return 3.0
    return safe_float(cost, 0.0)


def encode_record(vocab, record):
    state = record.get("state") or {}
    screen = screen_state(record)
    features = []

    hp = safe_float(state.get("hp"))
    max_hp = max(safe_float(state.get("max_hp"), 1.0), 1.0)
    features.extend([
        safe_float(state.get("act")) / 4.0,
        safe_float(state.get("floor")) / 60.0,
        safe_float(state.get("ascension")) / 20.0,
        hp / max_hp,
        max_hp / 120.0,
        safe_float(state.get("gold")) / 500.0,
        safe_float(state.get("deck_size")) / 80.0,
        safe_float(state.get("relic_count")) / 40.0,
        safe_float(state.get("potion_slots_filled")) / 5.0,
        vocab.get("characters", state.get("character")) / max(len(vocab.tables["characters"]), 1),
        vocab.get("rooms", state.get("room_type")) / max(len(vocab.tables["rooms"]), 1),
        vocab.get("screens", state_type(record)) / max(len(vocab.tables["screens"]), 1),
    ])

    relics = state.get("relics", [])
    for i in range(MAX_RELICS):
        if i < len(relics):
            features.append(vocab.get("relics", relics[i].get("id") or relics[i].get("name")) / max(len(vocab.tables["relics"]), 1))
        else:
            features.append(0.0)

    potions = state.get("potions", [])
    for i in range(MAX_POTIONS):
        if i < len(potions):
            features.append(vocab.get("potions", potions[i].get("id") or potions[i].get("name")) / max(len(vocab.tables["potions"]), 1))
        else:
            features.append(0.0)

    map_options = (screen.get("map") or {}).get("next_options", [])
    for i in range(MAX_MAP_OPTIONS):
        if i < len(map_options):
            option = map_options[i]
            features.extend([
                vocab.get("node_types", option.get("type")) / max(len(vocab.tables["node_types"]), 1),
                safe_float(option.get("col")) / 8.0,
                safe_float(option.get("row")) / 20.0,
                len(option.get("leads_to", [])) / 4.0,
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

    reward_items = (screen.get("rewards") or {}).get("items", [])
    for i in range(MAX_REWARD_ITEMS):
        if i < len(reward_items):
            item = reward_items[i]
            features.extend([
                vocab.get("reward_types", item.get("type")) / max(len(vocab.tables["reward_types"]), 1),
                safe_float(item.get("gold_amount")) / 200.0,
            ])
        else:
            features.extend([0.0, 0.0])

    cards = (screen.get("card_reward") or {}).get("cards", [])
    for i in range(MAX_CARD_OPTIONS):
        if i < len(cards):
            card = cards[i]
            features.extend([
                vocab.get("cards", card.get("id") or card.get("name")) / max(len(vocab.tables["cards"]), 1),
                vocab.get("card_types", card.get("type")) / max(len(vocab.tables["card_types"]), 1),
                vocab.get("rarities", card.get("rarity")) / max(len(vocab.tables["rarities"]), 1),
                card_cost(card) / 5.0,
                1.0 if card.get("is_upgraded") else 0.0,
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    event_options = (screen.get("event") or {}).get("options", [])
    features.append(len(event_options) / MAX_EVENT_OPTIONS)
    for i in range(MAX_EVENT_OPTIONS):
        if i < len(event_options):
            option = event_options[i]
            features.extend([
                1.0 if option.get("is_locked") else 0.0,
                1.0 if option.get("is_proceed") else 0.0,
                1.0 if option.get("was_chosen") else 0.0,
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def build_dataset():
    start_time = time.time()
    files = macro_files()
    print(f"Found {len(files)} macro files.")
    ctx = FilterContext(files)
    records = []
    skipped = Counter()
    for path in files:
        for record in iter_jsonl(path):
            if should_use_record(record, ctx=ctx):
                records.append(record)
            else:
                skipped["filtered"] += 1

    print(f"Collected {len(records)} macro samples; filtered {skipped['filtered']} records.")

    vocab = Vocab()
    for record in records:
        add_record_to_vocab(vocab, record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vocab_path = OUTPUT_DIR / "vocab.json"
    vocab.save(vocab_path)

    x_rows = []
    y_rows = []
    class_counts = Counter()
    source_counts = Counter()
    quality_counts = Counter()
    accepted_runs = {}  # run_id -> {source, quality, samples}
    for record in records:
        label = norm_action_label(record)
        action_id = vocab.get("actions", label)
        if action_id <= 1:
            continue
        x_rows.append(encode_record(vocab, record))
        y_rows.append(action_id)
        class_counts[label] += 1
        source = record.get("source") or "unknown"
        source_counts[source] += 1
        run_id = record.get("run_id") or "unknown"
        quality = ctx.run_quality(run_id)
        quality_counts[quality] += 1
        if run_id not in accepted_runs:
            accepted_runs[run_id] = {"source": source, "quality": quality, "samples": 0}
        accepted_runs[run_id]["samples"] += 1

    if x_rows:
        x_data = np.vstack(x_rows).astype(np.float32)
        y_data = np.array(y_rows, dtype=np.int64)
    else:
        x_data = np.zeros((0, 1), dtype=np.float32)
        y_data = np.zeros((0,), dtype=np.int64)

    human_samples = source_counts.get("human", 0)
    ai_samples = source_counts.get("ai", 0)
    total = int(len(y_data))

    runs_detail = []
    for rid, info in sorted(accepted_runs.items(), key=lambda x: -x[1]["samples"]):
        runs_detail.append({
            "run_id": rid,
            "source": info["source"],
            "quality": info["quality"],
            "samples": info["samples"],
        })

    np.save(OUTPUT_DIR / "X_train.npy", x_data)
    np.save(OUTPUT_DIR / "Y_train.npy", y_data)
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "samples": total,
            "human_samples": human_samples,
            "ai_samples": ai_samples,
            "human_ratio": round(human_samples / max(total, 1), 4),
            "ai_ratio": round(ai_samples / max(total, 1), 4),
            "accepted_run_count": len(accepted_runs),
            "accepted_runs": runs_detail,
            "features": int(x_data.shape[1]) if x_data.ndim == 2 else 0,
            "classes": class_counts,
            "accepted_sources": source_counts,
            "accepted_run_qualities": quality_counts,
            "files": [str(p.relative_to(WORKSPACE_DIR)) for p in files],
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
            "build_timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "build_elapsed_sec": round(time.time() - start_time, 2),
            "data_file_count": len(files),
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved macro dataset: X={x_data.shape}, Y={y_data.shape}")
    print(f"  Human: {human_samples} ({human_samples*100/max(total,1):.1f}%)  AI: {ai_samples} ({ai_samples*100/max(total,1):.1f}%)  Runs: {len(accepted_runs)}")
    print("Top macro labels:")
    for label, count in class_counts.most_common(20):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    build_dataset()
