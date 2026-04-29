from collections import Counter, defaultdict
from datetime import datetime

from run_summary import DATA_DIR, iter_jsonl, iter_run_files, safe_int, summarize_runs


POLICY_LABELS = {
    "human": "Human",
    "bc_combat": "战斗 BC",
    "rule_macro": "规则宏观",
    "macro_mixed": "宏观混合",
    "bc_combat+rule_macro": "战斗 BC + 规则宏观",
    "llm_catalog_args": "LLM 动作参数",
    "llm_candidate_id": "LLM 候选 ID",
    "ai_mcp": "AI/MCP",
    "unknown": "未知",
}


def empty_signals():
    return {
        "policy_counts": Counter(),
        "model_versions": set(),
        "source_counts": Counter(),
        "ai_combat": 0,
        "ai_macro": 0,
        "human": 0,
        "llm": 0,
    }


def collect_policy_signals():
    signals = defaultdict(empty_signals)
    for path in iter_run_files():
        rel = str(path.relative_to(DATA_DIR)).replace("\\", "/")
        for record in iter_jsonl(path):
            run_id = record.get("run_id")
            if not run_id:
                continue
            item = signals[run_id]
            source = record.get("source") or "unknown"
            item["source_counts"][source] += 1
            policy_name = record.get("policy_name") or (record.get("action_data") or {}).get("policy_name")
            model_version = record.get("model_version") or (record.get("action_data") or {}).get("model_version")
            if policy_name:
                item["policy_counts"][str(policy_name)] += 1
            if model_version:
                item["model_versions"].add(str(model_version))

            if source == "human" or rel.startswith("Human/"):
                item["human"] += 1
            if source == "ai" or rel.startswith("AI/") or rel.startswith("AI_Combat"):
                if "Combat" in rel:
                    item["ai_combat"] += 1
                else:
                    item["ai_macro"] += 1
            if source == "llm":
                item["llm"] += 1
    return signals


def infer_policy(run, signal):
    if signal["policy_counts"]:
        return signal["policy_counts"].most_common(1)[0][0]
    if signal["ai_combat"] and signal["ai_macro"]:
        return "bc_combat+rule_macro"
    if signal["ai_combat"]:
        return "bc_combat"
    if signal["ai_macro"]:
        return "rule_macro"
    if run.get("ai", 0):
        return "ai_mcp"
    if run.get("human", 0) or signal["human"]:
        return "human"
    return "unknown"


def infer_failure_reason(run):
    if run.get("invalid_actions", 0):
        return "非法动作"
    if run.get("losses", 0):
        return "战斗死亡"
    if run.get("data_health") == "missing":
        return "数据不完整"
    if run.get("records", 0) and not run.get("max_floor", 0):
        return "未进入楼层"
    if run.get("data_health") == "warn":
        return "需确认"
    return "进行中/正常"


def enrich_run(run, signals):
    item = dict(run)
    signal = signals.get(run.get("run_id"), empty_signals())
    policy = infer_policy(item, signal)
    item["policy_name"] = policy
    item["policy_label"] = POLICY_LABELS.get(policy, policy)
    item["model_versions"] = sorted(signal["model_versions"])
    item["failure_reason"] = infer_failure_reason(item)
    item["stuck_count"] = 1 if item["failure_reason"] == "未进入楼层" else 0
    item["policy_source_counts"] = dict(signal["source_counts"])
    return item


def aggregate_policy_runs(runs):
    buckets = {}
    for run in runs:
        policy = run.get("policy_name", "unknown")
        bucket = buckets.setdefault(policy, {
            "policy_name": policy,
            "policy_label": POLICY_LABELS.get(policy, policy),
            "runs": 0,
            "kept_runs": 0,
            "discarded_runs": 0,
            "max_floor": 0,
            "max_act": 0,
            "floor_sum": 0,
            "wins": 0,
            "losses": 0,
            "invalid_actions": 0,
            "stuck_count": 0,
            "data_missing": 0,
            "latest_time": "",
        })
        bucket["runs"] += 1
        bucket["discarded_runs"] += 1 if run.get("discarded") else 0
        bucket["kept_runs"] += 0 if run.get("discarded") else 1
        bucket["max_floor"] = max(bucket["max_floor"], safe_int(run.get("max_floor")))
        bucket["max_act"] = max(bucket["max_act"], safe_int(run.get("max_act")))
        bucket["floor_sum"] += safe_int(run.get("max_floor"))
        bucket["wins"] += safe_int(run.get("wins"))
        bucket["losses"] += safe_int(run.get("losses"))
        bucket["invalid_actions"] += safe_int(run.get("invalid_actions"))
        bucket["stuck_count"] += safe_int(run.get("stuck_count"))
        bucket["data_missing"] += 1 if run.get("data_health") == "missing" else 0
        bucket["latest_time"] = max(bucket["latest_time"], run.get("last_time") or "")

    rows = []
    for bucket in buckets.values():
        bucket["avg_floor"] = round(bucket["floor_sum"] / bucket["runs"], 2) if bucket["runs"] else 0
        rows.append(bucket)
    rows.sort(key=lambda x: (x["latest_time"], x["max_floor"], x["runs"]), reverse=True)
    return rows


def evaluation_summary(limit=50):
    runs = summarize_runs(limit=None)
    signals = collect_policy_signals()
    enriched = [enrich_run(run, signals) for run in runs]
    policies = aggregate_policy_runs(enriched)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policies": policies,
        "recent_runs": enriched[:limit],
        "total_runs": len(enriched),
    }
