import json
from collections import Counter
from datetime import datetime
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "RL_Datasets"
DISCARDED_PATH = DATA_DIR / "discarded_runs.json"
RUN_LABELS_PATH = DATA_DIR / "run_labels.json"
SELF_PLAY_SCORES_PATH = DATA_DIR / "self_play_scores.json"

QUALITY_LABELS = {
    "failed_run": "失败",
    "unknown": "未知",
    "before_act1_boss": "一关Boss前",
    "partial_act1": "一关Boss",
    "partial_act2": "二关Boss",
    "perfect_run": "通关完美",
}
QUALITY_ORDER = {
    "failed_run": -1,
    "unknown": 0,
    "before_act1_boss": 0,
    "partial_act1": 1,
    "partial_act2": 2,
    "perfect_run": 3,
}

RUN_ROOTS = [
    "Combat",
    "Human/Combat",
    "AI/Combat",
    "AI_Combat",
    "Macro",
    "Human/Macro",
    "AI/Macro",
]
RECENT_ROOTS = RUN_ROOTS + ["LLM_Actions"]

KNOWN_ACTIONS = {
    "play_card",
    "end_turn",
    "use_potion",
    "select_map_node",
    "claim_reward",
    "choose_card",
    "skip_reward",
    "choose_event_option",
    "choose_rest_option",
    "buy_item",
}

REWARD_LABELS = {
    "CardReward": "卡牌奖励",
    "PotionReward": "药水奖励",
    "GoldReward": "金币奖励",
    "RelicReward": "遗物奖励",
    "StolenGoldReward": "金币奖励",
}

NODE_LABELS = {
    "Monster": "普通战斗",
    "Elite": "精英",
    "Boss": "Boss",
    "RestSite": "营火",
    "Merchant": "商店",
    "Shop": "商店",
    "Treasure": "宝箱",
    "Unknown": "问号",
    "Event": "事件",
    "Ancient": "远古事件",
}


def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def iter_jsonl(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if isinstance(record, dict):
                    yield record
    except Exception:
        return


def iter_run_files(roots=RUN_ROOTS):
    files = []
    for sub in roots:
        root = DATA_DIR / sub
        if root.exists():
            files.extend(root.glob("*.jsonl"))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def iter_record_states(record):
    for key in ("state", "state_before", "state_after"):
        state = record.get(key)
        if isinstance(state, dict):
            yield state


def read_run_labels():
    data = read_json(RUN_LABELS_PATH, {"labels": {}})
    if "labels" not in data:
        data = {"labels": data if isinstance(data, dict) else {}}
    return data


def read_self_play_scores():
    data = read_json(SELF_PLAY_SCORES_PATH, {"scores": {}})
    if "scores" not in data:
        data = {"scores": data if isinstance(data, dict) else {}}
    return data


def write_self_play_scores(data):
    write_json(SELF_PLAY_SCORES_PATH, data)


def self_play_score_for(run_id):
    return read_self_play_scores().get("scores", {}).get(run_id)


def self_play_admitted_run_ids():
    scores = read_self_play_scores().get("scores", {})
    return {
        run_id
        for run_id, item in scores.items()
        if isinstance(item, dict) and item.get("admitted") is True
    }


def set_run_label(run_id, quality, note=""):
    if quality not in QUALITY_ORDER:
        quality = "unknown"
    data = read_run_labels()
    labels = data.setdefault("labels", {})
    labels[run_id] = {
        "quality": quality,
        "note": note,
        "manual": True,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(RUN_LABELS_PATH, data)
    return labels[run_id]


def set_auto_run_label(run_id, quality, note=""):
    if quality not in QUALITY_ORDER:
        quality = "unknown"
    data = read_run_labels()
    labels = data.setdefault("labels", {})
    old = labels.get(run_id, {})
    if old.get("manual"):
        return old
    if old.get("quality") == quality and old.get("note", "") == note:
        return old
    labels[run_id] = {
        "quality": quality,
        "note": note,
        "manual": False,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(RUN_LABELS_PATH, data)
    return labels[run_id]


def is_probable_act3_clear(summary):
    return (
        safe_int(summary.get("max_act")) >= 3
        and safe_int(summary.get("max_floor")) >= 48
        and safe_int(summary.get("losses")) == 0
        and safe_int(summary.get("wins")) >= 20
        and safe_int(summary.get("records")) >= 200
        and safe_int(summary.get("combat")) > 0
        and safe_int(summary.get("macro")) > 0
    )


def inferred_quality_note(summary):
    if is_probable_act3_clear(summary) and not summary.get("run_victory"):
        return (
            "auto: probable_clear "
            f"max_act={summary.get('max_act', 0)}, "
            f"max_floor={summary.get('max_floor', 0)}, "
            f"wins={summary.get('wins', 0)}, "
            f"losses={summary.get('losses', 0)}; "
            "final run_victory hook missing"
        )
    return f"auto: max_act={summary.get('max_act', 0)}, max_floor={summary.get('max_floor', 0)}"


def infer_quality(summary):
    if summary.get("losses", 0) > 0:
        return "failed_run"
    if summary.get("run_victory") or safe_int(summary.get("max_act")) >= 4 or is_probable_act3_clear(summary):
        return "perfect_run"
    if safe_int(summary.get("max_act")) >= 3:
        return "partial_act2"
    if safe_int(summary.get("max_act")) >= 2:
        return "partial_act1"
    if safe_int(summary.get("max_floor")) > 0:
        return "before_act1_boss"
    return "unknown"


def is_ai_run_summary(summary):
    run_id = str(summary.get("run_id") or "")
    if run_id.startswith("ai_"):
        return True
    return safe_int(summary.get("ai")) > 0 and safe_int(summary.get("human")) == 0


def is_stuck_run(summary):
    return safe_int(summary.get("records")) > 0 and safe_int(summary.get("max_floor")) <= 0


def collect_run_policy_signal(run_id):
    policy_counts = Counter()
    model_counts = Counter()
    source_counts = Counter()
    for path in iter_run_files():
        for record in iter_jsonl(path):
            if record.get("run_id") != run_id:
                continue
            source = str(record.get("source") or "unknown")
            source_counts[source] += 1
            action_data = record.get("action_data") if isinstance(record.get("action_data"), dict) else {}
            policy_name = record.get("policy_name") or action_data.get("policy_name")
            model_version = record.get("model_version") or action_data.get("model_version")
            if policy_name:
                policy_counts[str(policy_name)] += 1
            if model_version:
                model_counts[str(model_version)] += 1
    return {
        "policy_name": policy_counts.most_common(1)[0][0] if policy_counts else "",
        "model_version": model_counts.most_common(1)[0][0] if model_counts else "",
        "policy_counts": dict(policy_counts),
        "model_counts": dict(model_counts),
        "source_counts": dict(source_counts),
    }


def evaluate_self_play_run(summary):
    run_id = str(summary.get("run_id") or "")
    quality = summary.get("quality") or infer_quality(summary)
    probable_clear = bool(summary.get("probable_clear") or quality == "perfect_run")
    max_act = safe_int(summary.get("max_act"))
    max_floor = safe_int(summary.get("max_floor"))
    wins = safe_int(summary.get("wins"))
    losses = safe_int(summary.get("losses"))
    invalid_actions = safe_int(summary.get("invalid_actions"))
    stuck = is_stuck_run(summary)
    data_missing = summary.get("data_health") == "missing"
    policy_signal = collect_run_policy_signal(run_id)

    score = 0
    score += max_floor * 5
    score += max_act * 120
    score += wins * 20
    score -= losses * 160
    score -= invalid_actions * 400
    if probable_clear:
        score += 1000
    elif max_act >= 3:
        score += 450
    elif max_act >= 2:
        score += 250
    elif max_floor >= 18:
        score += 125
    if data_missing:
        score -= 250
    if stuck:
        score -= 300

    admitted = False
    reason = "early_failed_before_act2"
    if not is_ai_run_summary(summary):
        reason = "non_ai_run"
    elif invalid_actions > 0:
        reason = "invalid_actions"
    elif stuck:
        reason = "stuck_or_no_floor"
    elif probable_clear:
        admitted = True
        reason = "clear_or_probable_clear"
    elif max_act >= 2:
        admitted = True
        reason = "reached_act2"
    elif max_floor >= 18:
        admitted = True
        reason = "floor18_plus"

    return {
        "run_id": run_id,
        "score": int(score),
        "admitted": admitted,
        "reason": reason,
        "quality": quality,
        "probable_clear": probable_clear,
        "policy_name": policy_signal.get("policy_name", ""),
        "model_version": policy_signal.get("model_version", ""),
        "max_act": max_act,
        "max_floor": max_floor,
        "wins": wins,
        "losses": losses,
        "invalid_actions": invalid_actions,
        "data_health": summary.get("data_health"),
        "stuck": stuck,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_self_play_run_score(summary):
    result = evaluate_self_play_run(summary)
    data = read_self_play_scores()
    scores = data.setdefault("scores", {})
    scores[result["run_id"]] = result
    write_self_play_scores(data)
    return result


def empty_run_summary(run_id):
    item = {
        "run_id": run_id,
        "records": 0,
        "combat": 0,
        "macro": 0,
        "ai": 0,
        "human": 0,
        "unknown_source": 0,
        "wins": 0,
        "losses": 0,
        "game_start": 0,
        "game_resume": 0,
        "battle_start": 0,
        "battle_end": 0,
        "turn_start": 0,
        "turn_end": 0,
        "macro_actions": 0,
        "play_card": 0,
        "end_turn": 0,
        "use_potion": 0,
        "select_map_node": 0,
        "claim_reward": 0,
        "choose_card": 0,
        "skip_reward": 0,
        "choose_event_option": 0,
        "choose_rest_option": 0,
        "buy_item": 0,
        "invalid_actions": 0,
        "unknown_actions": 0,
        "max_act": 0,
        "max_floor": 0,
        "max_round": 0,
        "run_victory": False,
        "last_ts": 0,
        "first_ts": 0,
        "schema_versions": set(),
        "files": set(),
    }
    return item


def update_progress_from_state(item, state):
    run = state.get("run") if isinstance(state.get("run"), dict) else state
    item["max_act"] = max(item["max_act"], safe_int(run.get("act")), safe_int(state.get("act")))
    item["max_floor"] = max(item["max_floor"], safe_int(run.get("floor")), safe_int(state.get("floor")))
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else state
    item["max_round"] = max(item["max_round"], safe_int(battle.get("round")), safe_int(state.get("round")))


def update_run_summary(item, record, path):
    item["records"] += 1
    item["files"].add(str(path.relative_to(DATA_DIR)))

    rel = str(path.relative_to(DATA_DIR)).replace("\\", "/")
    if "Combat" in rel:
        item["combat"] += 1
    else:
        item["macro"] += 1

    source = record.get("source")
    if source == "ai":
        item["ai"] += 1
    elif source == "human":
        item["human"] += 1
    else:
        item["unknown_source"] += 1

    schema_version = record.get("schema_version")
    if schema_version is not None:
        item["schema_versions"].add(str(schema_version))

    rec_type = record.get("type")
    if rec_type in ("game_start", "game_resume", "battle_start", "battle_end", "turn_start", "turn_end"):
        item[rec_type] += 1
    if rec_type == "macro_action":
        item["macro_actions"] += 1
    if rec_type == "battle_end":
        if record.get("result") == "win":
            item["wins"] += 1
        if record.get("result") == "lose":
            item["losses"] += 1
    if rec_type in ("run_end", "game_end", "victory") and record.get("result") in ("win", "victory", "complete", True):
        item["run_victory"] = True

    action_type = record.get("action_type")
    if action_type in KNOWN_ACTIONS:
        item[action_type] += 1
    elif action_type:
        item["unknown_actions"] += 1

    status = str(record.get("status") or "").lower()
    error = str(record.get("error") or record.get("message") or "").lower()
    if status == "error" or "invalid" in error or "out of range" in error or "cannot" in error:
        item["invalid_actions"] += 1

    for state in iter_record_states(record):
        update_progress_from_state(item, state)

    ts = safe_int(record.get("timestamp"))
    if ts:
        item["last_ts"] = max(item["last_ts"], ts)
        item["first_ts"] = ts if not item["first_ts"] else min(item["first_ts"], ts)


def run_data_checks(run):
    checks = []

    def add(label, status, detail, count=None):
        checks.append({
            "label": label,
            "status": status,
            "detail": detail,
            "count": count,
        })

    records = run.get("records", 0)
    combat = run.get("combat", 0)
    macro = run.get("macro", 0)
    battle_count = run.get("battle_start", 0)
    win_or_loss = run.get("wins", 0) + run.get("losses", 0)
    reward_actions = run.get("claim_reward", 0) + run.get("choose_card", 0) + run.get("skip_reward", 0)

    add("Run 记录", "ok" if records else "missing", f"{records} 条总记录", records)
    add("宏观记录", "ok" if macro else "missing", f"{macro} 条；包含开局、地图、奖励、事件等非战斗动作", macro)
    add("战斗记录", "ok" if combat else "missing", f"{combat} 条；包含战斗开始、回合、动作和结算", combat)

    if run.get("game_start", 0) or run.get("game_resume", 0):
        add("开局/续局", "ok", f"新游戏 {run.get('game_start', 0)}，继续游戏 {run.get('game_resume', 0)}")
    else:
        add("开局/续局", "warn", "未检测到 game_start / game_resume；如果是半路接入可忽略")

    if combat:
        add("战斗开始", "ok" if battle_count else "missing", f"battle_start {battle_count} 次", battle_count)
        add("回合快照", "ok" if run.get("turn_start", 0) and run.get("turn_end", 0) else "warn", f"turn_start {run.get('turn_start', 0)}，turn_end {run.get('turn_end', 0)}")
        add("出牌样本", "ok" if run.get("play_card", 0) else "missing", f"play_card {run.get('play_card', 0)} 次", run.get("play_card", 0))
        add("结束回合", "ok" if run.get("end_turn", 0) or run.get("turn_end", 0) else "missing", f"end_turn {run.get('end_turn', 0)}，turn_end {run.get('turn_end', 0)}")
        add("战斗结算", "ok" if win_or_loss else "warn", f"胜 {run.get('wins', 0)}，败 {run.get('losses', 0)}；未结束的战斗可忽略")

    if run.get("max_floor", 0) > 0 or macro:
        add("地图选择", "ok" if run.get("select_map_node", 0) else "warn", f"select_map_node {run.get('select_map_node', 0)} 次；如果刚开局/未点地图可忽略")
    if win_or_loss or reward_actions:
        add("奖励处理", "ok" if reward_actions else "warn", f"领奖 {run.get('claim_reward', 0)}，选卡 {run.get('choose_card', 0)}，跳过 {run.get('skip_reward', 0)}")

    if run.get("run_victory"):
        add("Run 终局信号", "ok", "已采集到 run_victory")
    elif is_probable_act3_clear(run):
        add(
            "Run 终局信号",
            "warn",
            "按 Act 3 / Floor 48 / 胜场 / 0 失败判定为疑似通关，但没有采到 run_end/victory 钩子",
        )

    if run.get("invalid_actions", 0):
        add("非法动作", "warn", f"检测到 {run.get('invalid_actions', 0)} 条错误/非法动作记录", run.get("invalid_actions", 0))
    else:
        add("非法动作", "ok", "未检测到错误/非法动作记录", 0)

    versions = run.get("schema_versions", [])
    if versions:
        add("Schema", "ok" if "4" in versions else "warn", f"版本：{', '.join(versions)}")
    else:
        add("Schema", "warn", "旧数据没有 schema_version；可用于兼容训练，但新数据应为 v4")

    optional = []
    if run.get("choose_event_option", 0):
        optional.append(f"事件 {run.get('choose_event_option', 0)}")
    if run.get("choose_rest_option", 0):
        optional.append(f"营火 {run.get('choose_rest_option', 0)}")
    if run.get("buy_item", 0):
        optional.append(f"商店 {run.get('buy_item', 0)}")
    add("可选宏观", "ok" if optional else "info", "，".join(optional) if optional else "本 run 暂未出现事件/营火/商店")

    return checks


def finalize_run_summary(item, discarded, labels, self_play_scores):
    item["discarded"] = item["run_id"] in discarded
    item["probable_clear"] = is_probable_act3_clear(item)
    inferred_quality = infer_quality(item)
    inferred_note = inferred_quality_note(item)
    label = labels.get(item["run_id"])
    if label and not label.get("manual"):
        label = set_auto_run_label(item["run_id"], inferred_quality, inferred_note)
    if label:
        item["quality"] = label.get("quality", "unknown")
        item["quality_label"] = QUALITY_LABELS.get(item["quality"], item["quality"])
        item["quality_manual"] = bool(label.get("manual"))
        item["quality_note"] = label.get("note", "")
    else:
        label = set_auto_run_label(item["run_id"], inferred_quality, inferred_note)
        item["quality"] = label.get("quality", inferred_quality)
        item["quality_label"] = QUALITY_LABELS.get(item["quality"], item["quality"])
        item["quality_manual"] = False
        item["quality_note"] = label.get("note", inferred_note)
    self_play = self_play_scores.get(item["run_id"])
    if isinstance(self_play, dict):
        item["self_play_score"] = self_play.get("score")
        item["self_play_admitted"] = bool(self_play.get("admitted"))
        item["self_play_reason"] = self_play.get("reason", "")
        item["self_play_updated_at"] = self_play.get("updated_at", "")
    item["inferred_quality"] = inferred_quality
    item["inferred_quality_label"] = QUALITY_LABELS.get(inferred_quality, inferred_quality)
    item["files"] = sorted(item["files"])
    item["schema_versions"] = sorted(item["schema_versions"], key=str)
    item["duration_sec"] = int((item["last_ts"] - item["first_ts"]) / 1000) if item["first_ts"] and item["last_ts"] else 0

    checks = run_data_checks(item)
    item["data_checks"] = checks
    missing = [c for c in checks if c["status"] == "missing"]
    warnings = [c for c in checks if c["status"] == "warn"]
    item["missing_data"] = [c["label"] for c in missing]
    if missing:
        item["data_health"] = "missing"
        item["data_health_label"] = f"缺 {len(missing)} 项"
    elif warnings:
        item["data_health"] = "warn"
        item["data_health_label"] = f"需确认 {len(warnings)} 项"
    else:
        item["data_health"] = "ok"
        item["data_health_label"] = "完整"
    item["last_time"] = (
        datetime.fromtimestamp(item["last_ts"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
        if item["last_ts"]
        else ""
    )
    return item


def summarize_runs(limit=None):
    discarded = set(read_json(DISCARDED_PATH, {"discarded": []}).get("discarded", []))
    labels = read_run_labels().get("labels", {})
    self_play_scores = read_self_play_scores().get("scores", {})
    runs = {}
    for path in iter_run_files():
        for record in iter_jsonl(path):
            run_id = record.get("run_id")
            if not run_id:
                continue
            item = runs.setdefault(run_id, empty_run_summary(run_id))
            update_run_summary(item, record, path)

    result = [finalize_run_summary(item, discarded, labels, self_play_scores) for item in runs.values()]
    result.sort(key=lambda x: x["last_ts"], reverse=True)
    return result[:limit] if limit else result


def latest_runs(limit=12):
    return summarize_runs(limit=limit)


def iter_recent_records(max_files=8):
    for path in iter_run_files(RECENT_ROOTS)[:max_files]:
        for record in iter_jsonl(path):
            record["_file"] = str(path.relative_to(DATA_DIR))
            yield record


def clean_label(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text == "MegaCrit.Sts2.Core.Localization.LocString":
        return ""
    return text


def first_label(*values):
    for value in values:
        text = clean_label(value)
        if text:
            return text
    return ""


def source_label(value):
    text = clean_label(value)
    return {
        "human": "玩家",
        "ai": "AI",
        "llm": "LLM",
    }.get(text, text or "-")


def reward_label(value):
    text = clean_label(value)
    return REWARD_LABELS.get(text, text)


def node_label(value):
    text = clean_label(value)
    return NODE_LABELS.get(text, text)


def action_payload(record):
    payload = record.get("action_data")
    if isinstance(payload, dict):
        return payload
    payload = record.get("payload")
    if isinstance(payload, dict):
        return payload
    decision = record.get("decision")
    if isinstance(decision, dict):
        args = decision.get("args")
        if isinstance(args, dict):
            return args
    return {}


def preferred_state(record, *keys):
    for key in keys or ("state_before", "state", "state_after"):
        state = record.get(key)
        if isinstance(state, dict):
            return state
    return {}


def screen_state(record):
    screen = record.get("screen_state")
    return screen if isinstance(screen, dict) else {}


def state_player(state):
    player = state.get("player")
    return player if isinstance(player, dict) else state


def state_battle(state):
    battle = state.get("battle")
    return battle if isinstance(battle, dict) else state


def state_run(state):
    run = state.get("run")
    return run if isinstance(run, dict) else state


def list_field(container, key):
    value = container.get(key) if isinstance(container, dict) else None
    return value if isinstance(value, list) else []


def find_card(state, index=None, card_id=None):
    player = state_player(state)
    for card in list_field(player, "hand"):
        if not isinstance(card, dict):
            continue
        if index is not None and safe_int(card.get("index"), -999) == safe_int(index, -998):
            return card
        if card_id and clean_label(card.get("id")) == str(card_id):
            return card
    return {}


def find_potion(state, slot=None, potion_id=None):
    player = state_player(state)
    for potion in list_field(player, "potions"):
        if not isinstance(potion, dict):
            continue
        if slot is not None and safe_int(potion.get("slot"), -999) == safe_int(slot, -998):
            return potion
        if potion_id and clean_label(potion.get("id")) == str(potion_id):
            return potion
    return {}


def find_enemy_name(state, target_id):
    if target_id is None:
        return ""
    target = str(target_id)
    battle = state_battle(state)
    for enemy in list_field(battle, "enemies"):
        if not isinstance(enemy, dict):
            continue
        identifiers = [
            enemy.get("id"),
            enemy.get("entity_id"),
            enemy.get("combat_id"),
            enemy.get("name"),
        ]
        if target in {str(item) for item in identifiers if item is not None}:
            return first_label(enemy.get("name"), enemy.get("id"), enemy.get("entity_id"), target)
    return target


def record_context(record, state):
    run = state_run(state)
    battle = state_battle(state)
    act = safe_int(run.get("act"))
    floor = safe_int(record.get("floor") or run.get("floor"))
    round_no = safe_int(record.get("round") or battle.get("round") or state.get("round"))
    parts = []
    if act or floor:
        parts.append(f"Act {act or '?'} / Floor {floor or '?'}")
    if round_no:
        parts.append(f"Round {round_no}")
    return "，".join(parts)


def describe_recent_record(record):
    rec_type = record.get("type")
    action_type = record.get("action_type") or (record.get("decision") or {}).get("action")
    payload = action_payload(record)
    state = preferred_state(record, "state_before", "state", "state_after")
    context = record_context(record, state)
    source = source_label(record.get("source"))

    def desc(label, summary, detail="", tone="info", category="system"):
        return {
            "label": label,
            "summary": summary,
            "detail": detail or context,
            "tone": tone,
            "category": category,
        }

    if action_type == "play_card":
        card = find_card(state, payload.get("card_index"), payload.get("card_id"))
        name = first_label(payload.get("card_name"), card.get("name"), payload.get("card_id"), card.get("id"), "未知卡牌")
        target_id = payload.get("target_id") or payload.get("target")
        target = find_enemy_name(state, target_id)
        detail = f"目标 {target}" if target else context
        if context and target:
            detail = f"{detail}，{context}"
        return desc("出牌", f"{source} 打出 {name}", detail, "on", "combat")

    if action_type == "use_potion":
        potion = find_potion(state, payload.get("slot"), payload.get("potion_id"))
        name = first_label(payload.get("potion_name"), potion.get("name"), payload.get("potion_id"), potion.get("id"), "未知药水")
        target_id = payload.get("target_id") or payload.get("target")
        target = find_enemy_name(state, target_id)
        detail = f"目标 {target}" if target else context
        if context and target:
            detail = f"{detail}，{context}"
        return desc("药水", f"{source} 使用 {name}", detail, "warn", "combat")

    if action_type == "end_turn":
        cards_left = payload.get("cards_left")
        energy_left = payload.get("energy_left")
        extra = []
        if cards_left is not None:
            extra.append(f"剩 {cards_left} 张手牌")
        if energy_left is not None:
            extra.append(f"剩 {energy_left} 能量")
        detail = "，".join(extra + ([context] if context else []))
        return desc("回合", f"{source} 结束回合", detail, "info", "combat")

    if action_type == "choose_card":
        name = first_label(payload.get("card_name"), payload.get("card_id"), payload.get("name"), "未知卡牌")
        return desc("选卡", f"{source} 选择 {name}", context, "on", "macro")

    if action_type == "claim_reward":
        reward = first_label(
            payload.get("reward_name"),
            payload.get("item_name"),
            payload.get("relic_name"),
            payload.get("potion_name"),
            payload.get("card_name"),
            reward_label(payload.get("reward_type")),
            payload.get("type"),
            "奖励",
        )
        return desc("领奖", f"{source} 领取 {reward}", context, "on", "macro")

    if action_type == "skip_reward":
        return desc("奖励", f"{source} 跳过奖励", context, "warn", "macro")

    if action_type == "select_map_node":
        node_type = first_label(node_label(payload.get("node_type")), node_label(payload.get("type")), "地图节点")
        coord = []
        if payload.get("col") is not None:
            coord.append(f"列 {payload.get('col')}")
        if payload.get("row") is not None:
            coord.append(f"行 {payload.get('row')}")
        detail = "，".join(coord + ([context] if context else []))
        return desc("地图", f"{source} 前往 {node_type}", detail, "info", "macro")

    if action_type == "choose_event_option":
        event_screen = screen_state(record).get("event")
        event_screen = event_screen if isinstance(event_screen, dict) else {}
        options = list_field(event_screen, "options")
        chosen_options = [option for option in options if isinstance(option, dict) and option.get("was_chosen")]
        if not chosen_options and len(options) == 1 and isinstance(options[0], dict):
            chosen_options = [options[0]]
        chosen = chosen_options[0] if chosen_options else {}
        event = first_label(payload.get("event_name"), event_screen.get("event_name"))
        option = first_label(
            payload.get("option_title"),
            payload.get("option_name"),
            chosen.get("title"),
            chosen.get("relic_name"),
            payload.get("index"),
            "事件选项",
        )
        summary = f"{source} 选择 {option}"
        if event:
            summary = f"{source} 在 {event} 选择 {option}"
        return desc("事件", summary, context, "warn", "macro")

    if action_type == "choose_rest_option":
        option = first_label(payload.get("option_name"), payload.get("rest_option"), payload.get("action"), "营火选项")
        return desc("营火", f"{source} 选择 {option}", context, "warn", "macro")

    if action_type == "buy_item":
        name = first_label(payload.get("item_name"), payload.get("relic_name"), payload.get("card_name"), payload.get("potion_name"), payload.get("item_id"), "商品")
        cost = payload.get("cost")
        detail = f"花费 {cost} 金币" if cost is not None else context
        if context and cost is not None:
            detail = f"{detail}，{context}"
        return desc("商店", f"{source} 购买 {name}", detail, "warn", "macro")

    if rec_type == "battle_start":
        battle = state_battle(state)
        enemies = list_field(battle, "enemies")
        enemy_names = [first_label(e.get("name"), e.get("id"), e.get("entity_id")) for e in enemies if isinstance(e, dict)]
        detail = "，".join([x for x in [", ".join(enemy_names[:3]), context] if x])
        return desc("战斗", "战斗开始", detail, "on", "combat")

    if rec_type == "turn_start":
        return desc("回合", "玩家回合开始", context, "info", "combat")

    if rec_type == "turn_end":
        return desc("回合", "玩家回合结束", context, "info", "combat")

    if rec_type == "battle_end":
        result = "胜利" if record.get("result") == "win" else ("失败" if record.get("result") == "lose" else clean_label(record.get("result")) or "结算")
        hp = record.get("remaining_hp")
        max_hp = record.get("max_hp")
        rounds = record.get("rounds")
        detail = []
        if hp is not None and max_hp is not None:
            detail.append(f"HP {hp}/{max_hp}")
        if rounds is not None:
            detail.append(f"{rounds} 回合")
        if context:
            detail.append(context)
        return desc("结算", f"战斗{result}", "，".join(detail), "on" if record.get("result") == "win" else "warn", "combat")

    if rec_type == "game_start":
        return desc("开局", "记录到新游戏", context, "on", "system")

    if rec_type == "game_resume":
        return desc("续局", "记录到继续游戏", context, "info", "system")

    if rec_type in ("run_end", "game_end", "victory"):
        result = first_label(record.get("result"), "Run 结束")
        return desc("Run", result, context, "on", "system")

    label = first_label(action_type, rec_type, "记录")
    result = first_label(record.get("result"))
    summary = f"{source} {label}" if source != "-" else label
    if result:
        summary = f"{summary}：{result}"
    return desc("记录", summary, context, "info", "system")


def recent_records(limit=30):
    records = sorted(iter_recent_records(), key=lambda r: safe_int(r.get("timestamp")), reverse=True)[:limit]
    out = []
    for rec in records:
        ts = safe_int(rec.get("timestamp"))
        description = describe_recent_record(rec)
        out.append({
            "time": datetime.fromtimestamp(ts / 1000).strftime("%H:%M:%S") if ts else "",
            "run_id": rec.get("run_id"),
            "type": rec.get("type"),
            "source": rec.get("source"),
            "source_label": source_label(rec.get("source")),
            "action_type": rec.get("action_type") or (rec.get("decision") or {}).get("action"),
            "result": rec.get("result"),
            "round": rec.get("round") or (rec.get("state") or {}).get("round"),
            "file": rec.get("_file"),
            "label": description.get("label"),
            "summary": description.get("summary"),
            "detail": description.get("detail"),
            "tone": description.get("tone"),
            "category": description.get("category"),
        })
    return out


def current_data_summary():
    runs = latest_runs(limit=1)
    if not runs:
        return {"active_run": None, "warning": "暂无 run 数据。确认采集总开关已打开，并进入一局游戏。"}
    run = runs[0].copy()
    warnings = []
    for check in run.get("data_checks", []):
        if check.get("status") == "missing":
            warnings.append(f"未检测到{check.get('label')}：{check.get('detail')}")
    for check in run.get("data_checks", []):
        if check.get("status") == "warn":
            warnings.append(f"需要确认{check.get('label')}：{check.get('detail')}")
    if run.get("losses", 0) > 0:
        warnings.append("这个 run 有失败记录，BC 训练前建议确认是否保留。")
    if run.get("quality") == "perfect_run" and run.get("play_card", 0) == 0:
        warnings.append("这个 run 已标为完美，但没有 play_card 标签；修好 Hook 后建议再采一局高质量数据。")
    return {"active_run": run, "warnings": warnings}


def set_run_discarded(run_id, discarded):
    data = read_json(DISCARDED_PATH, {"discarded": []})
    items = set(data.get("discarded", []))
    if discarded:
        items.add(run_id)
    else:
        items.discard(run_id)
    data["discarded"] = sorted(items)
    write_json(DISCARDED_PATH, data)
    return data
