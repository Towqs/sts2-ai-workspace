import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = WORKSPACE_DIR / "RL_Datasets" / "OptionShadow"
DEFAULT_REPORT_DIR = DEFAULT_LOG_DIR / "reports"


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_nan(value):
    number = safe_float(value)
    return number is not None and math.isnan(number)


def is_inf(value):
    number = safe_float(value)
    return number is not None and math.isinf(number)


def finite_scores(options):
    scores = []
    nan_count = 0
    inf_count = 0
    for option in options or []:
        if not isinstance(option, dict):
            continue
        score = option.get("score")
        if is_nan(score):
            nan_count += 1
            continue
        if is_inf(score):
            inf_count += 1
            continue
        number = safe_float(score)
        if number is not None and math.isfinite(number):
            scores.append(number)
    return scores, nan_count, inf_count


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else 0.0


def distribution(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return {"count": 0, "mean": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0}
    avg = sum(finite) / len(finite)
    variance = sum((value - avg) ** 2 for value in finite) / len(finite)
    return {
        "count": len(finite),
        "mean": round4(avg),
        "variance": round4(variance),
        "min": round4(min(finite)),
        "max": round4(max(finite)),
    }


def pct(value):
    return round(float(value) * 100.0, 2)


def round4(value):
    return round(float(value), 4)


def old_policy_label(record):
    label = str(record.get("old_policy_action") or record.get("legacy_chosen_action") or "").strip()
    if label:
        return label
    payload = record.get("actual_payload") if isinstance(record.get("actual_payload"), dict) else {}
    action = str(payload.get("action") or record.get("actual_action") or "")
    if action == "skip_card_reward":
        return "skip_reward"
    if action == "select_card_reward":
        index = payload.get("card_index", payload.get("index"))
        if index is not None:
            return f"choose_card:index_{index}"
    return action or "unknown"


def confidence_gap(options):
    scores, nan_count, inf_count = finite_scores(options)
    if len(scores) < 2:
        return 0.0, nan_count, inf_count
    scores.sort(reverse=True)
    return scores[0] - scores[1], nan_count, inf_count


def iter_jsonl(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    yield {"_error": "json_decode", "_path": str(path), "_line": line_no}
                    continue
                if isinstance(item, dict):
                    item["_path"] = str(path)
                    item["_line"] = line_no
                    yield item
    except FileNotFoundError:
        return


def resolve_input_files(args):
    if args.files:
        return [Path(p) for p in args.files]
    log_dir = Path(args.log_dir)
    if args.all:
        return sorted(log_dir.glob("card_scorer_*.jsonl"))
    date = args.date or datetime.now().strftime("%Y-%m-%d")
    return [log_dir / f"card_scorer_{date}.jsonl"]


def summarize_records(records):
    valid = [r for r in records if r.get("type") == "card_scorer_shadow"]
    errors = [r for r in records if r.get("_error")]
    run_ids = {str(r.get("run_id") or "") for r in valid if r.get("run_id")}
    old_counts = Counter()
    scorer_counts = Counter()
    archetype_counts = Counter()
    locked_template_counts = Counter()
    run_template_counts = {}
    candidate_counts = []
    gaps = []
    consistency_values = []
    deck_sizes = []
    bloat_scores = []
    skip_scores = []
    best_card_scores = []
    reward_term_values = {}
    disagreement_examples = []
    high_confidence_examples = []
    low_confidence_examples = []
    nan_count = 0
    inf_count = 0
    reward_term_nan_count = 0
    reward_term_inf_count = 0
    agreement_count = 0
    skip_recommended_count = 0
    old_skip_count = 0
    template_locked_count = 0
    candidate_count_anomalies = 0

    for record in valid:
        options = record.get("options") if isinstance(record.get("options"), list) else []
        option_count = int(record.get("legal_option_count") or len(options))
        candidate_counts.append(option_count)
        if option_count < 2:
            candidate_count_anomalies += 1

        old_label = old_policy_label(record)
        selected = record.get("selected") if isinstance(record.get("selected"), dict) else {}
        scorer_label = str(record.get("recommended_action") or selected.get("label") or "").strip() or "unknown"
        old_counts[old_label] += 1
        scorer_counts[scorer_label] += 1
        if old_label == scorer_label:
            agreement_count += 1
        if scorer_label == "skip_reward" or bool(record.get("skip_recommended")):
            skip_recommended_count += 1
        if old_label == "skip_reward" or bool(record.get("actual_skip")):
            old_skip_count += 1

        gap, record_nan, record_inf = confidence_gap(options)
        gap = safe_float(record.get("confidence_gap")) if record.get("confidence_gap") is not None else gap
        gap = float(gap or 0.0)
        gaps.append(gap)
        nan_count += record_nan
        inf_count += record_inf

        template_id = str(record.get("template_id") or selected.get("metadata", {}).get("template_id") or "unknown")
        archetype_counts[template_id] += 1
        run_key = str(record.get("run_id") or "unknown")
        run_template_counts.setdefault(run_key, Counter())[template_id] += 1
        template_lock = record.get("template_lock") if isinstance(record.get("template_lock"), dict) else {}
        if bool(record.get("template_locked") or template_lock.get("locked")):
            template_locked_count += 1
        locked_template = str(record.get("locked_template") or template_lock.get("locked_template") or "")
        if locked_template:
            locked_template_counts[locked_template] += 1
        consistency = record.get("archetype_consistency")
        if isinstance(consistency, dict):
            value = safe_float(consistency.get("consistency"))
            if value is not None and math.isfinite(value):
                consistency_values.append(value)

        skip_score = safe_float(record.get("skip_score"))
        best_score = safe_float(record.get("best_card_score"))
        if skip_score is None:
            skip_option = next(
                (option for option in options if isinstance(option, dict) and option.get("label") == "skip_reward"),
                {},
            )
            skip_score = safe_float(skip_option.get("score"))
        if best_score is None:
            card_scores = [
                safe_float(option.get("score"))
                for option in options
                if isinstance(option, dict) and option.get("label") != "skip_reward"
            ]
            card_scores = [value for value in card_scores if value is not None and math.isfinite(value)]
            best_score = max(card_scores) if card_scores else None
        if skip_score is not None and math.isfinite(skip_score):
            skip_scores.append(skip_score)
        if best_score is not None and math.isfinite(best_score):
            best_card_scores.append(best_score)

        deck_summary = record.get("deck_summary") if isinstance(record.get("deck_summary"), dict) else {}
        deck_size = safe_float(deck_summary.get("deck_size"))
        bloat_score = safe_float(deck_summary.get("bloat_score"))
        if deck_size is not None and math.isfinite(deck_size):
            deck_sizes.append(deck_size)
        if bloat_score is not None and math.isfinite(bloat_score):
            bloat_scores.append(bloat_score)

        reward_terms = record.get("reward_terms") if isinstance(record.get("reward_terms"), dict) else {}
        for name, raw_value in reward_terms.items():
            if is_nan(raw_value):
                reward_term_nan_count += 1
                continue
            if is_inf(raw_value):
                reward_term_inf_count += 1
                continue
            value = safe_float(raw_value)
            if value is not None and math.isfinite(value):
                reward_term_values.setdefault(str(name), []).append(value)

        example = {
            "run_id": record.get("run_id"),
            "floor": record.get("floor"),
            "old": old_label,
            "scorer": scorer_label,
            "gap": round4(gap),
            "template_id": template_id,
            "template_locked": bool(record.get("template_locked") or template_lock.get("locked")),
            "locked_template": locked_template,
            "deck_size": deck_summary.get("deck_size"),
            "line": record.get("_line"),
            "file": os.path.basename(str(record.get("_path") or "")),
            "old_card": record.get("old_policy_card") or {},
            "scorer_card": record.get("scorer_card") or {},
            "selected_score": selected.get("score"),
            "selected_reasons": selected.get("reasons"),
            "score_breakdown": selected.get("score_breakdown"),
            "skip_score": skip_score,
            "best_card_score": best_score,
            "skip_reasons": record.get("skip_reasons") or [],
            "skip_score_breakdown": record.get("skip_score_breakdown") or {},
        }
        if old_label != scorer_label:
            disagreement_examples.append(example)
        if gap >= 1.0:
            high_confidence_examples.append(example)
        if gap <= 0.25:
            low_confidence_examples.append(example)

    total = len(valid)
    agreement_rate = agreement_count / total if total else 0.0
    disagreement_count = total - agreement_count
    run_consistencies = []
    for counts in run_template_counts.values():
        count_total = sum(counts.values())
        if count_total:
            run_consistencies.append(max(counts.values()) / count_total)
    metrics = {
        "total_card_reward_events": total,
        "run_count": len(run_ids),
        "avg_candidate_count": round4(mean(candidate_counts)),
        "candidate_count_anomalies": candidate_count_anomalies,
        "old_vs_scorer_agreement_rate": round4(agreement_rate),
        "scorer_disagreed_with_old_policy": disagreement_count,
        "scorer_disagreed_with_old_policy_rate": round4(1.0 - agreement_rate if total else 0.0),
        "scorer_recommended_skip_rate": round4(skip_recommended_count / total if total else 0.0),
        "old_policy_skip_rate": round4(old_skip_count / total if total else 0.0),
        "avg_skip_score": round4(mean(skip_scores)),
        "avg_best_card_score": round4(mean(best_card_scores)),
        "avg_confidence_gap": round4(mean(gaps)),
        "score_nan_count": nan_count,
        "score_inf_count": inf_count,
        "reward_term_nan_count": reward_term_nan_count,
        "reward_term_inf_count": reward_term_inf_count,
        "reward_term_distribution": {
            name: distribution(values)
            for name, values in sorted(reward_term_values.items())
        },
        "archetype_distribution": dict(archetype_counts),
        "locked_template_distribution": dict(locked_template_counts),
        "template_locked_rate": round4(template_locked_count / total if total else 0.0),
        "archetype_consistency": round4(mean(consistency_values)),
        "template_sequence_consistency": round4(mean(run_consistencies)),
        "avg_deck_size": round4(mean(deck_sizes)),
        "avg_deck_bloat_score": round4(mean(bloat_scores)),
        "json_decode_errors": len(errors),
    }
    return {
        "metrics": metrics,
        "old_policy_distribution": dict(old_counts.most_common()),
        "scorer_distribution": dict(scorer_counts.most_common()),
        "disagreement_examples": sorted(disagreement_examples, key=lambda x: x["gap"], reverse=True)[:10],
        "high_confidence_examples": sorted(high_confidence_examples, key=lambda x: x["gap"], reverse=True)[:10],
        "low_confidence_examples": sorted(low_confidence_examples, key=lambda x: x["gap"])[:10],
    }


def md_table(mapping, key_name="item", value_name="count"):
    if not mapping:
        return "| item | count |\n| --- | ---: |\n| - | 0 |\n"
    lines = [f"| {key_name} | {value_name} |", "| --- | ---: |"]
    for key, value in mapping.items():
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines) + "\n"


def reward_terms_table(distributions):
    if not distributions:
        return "| term | count | mean | variance | min | max |\n| --- | ---: | ---: | ---: | ---: | ---: |\n| - | 0 | 0 | 0 | 0 | 0 |\n"
    lines = [
        "| term | count | mean | variance | min | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for term, stats in distributions.items():
        lines.append(
            f"| `{term}` | {stats.get('count', 0)} | {stats.get('mean', 0.0)} | "
            f"{stats.get('variance', 0.0)} | {stats.get('min', 0.0)} | {stats.get('max', 0.0)} |"
        )
    return "\n".join(lines) + "\n"


def _card_label(card):
    if not isinstance(card, dict):
        return "-"
    name = card.get("name") or ""
    card_id = card.get("card_id") or ""
    index = card.get("index")
    if name or card_id:
        return f"{index}:{name or card_id}"
    return "-"


def examples_table(examples):
    if not examples:
        return "| run | floor | old | old card | scorer | scorer card | gap | template | locked | deck | skip | best | location |\n| --- | ---: | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |\n| - | 0 | - | - | - | - | 0 | - | - | 0 | 0 | 0 | - |\n"
    lines = [
        "| run | floor | old | old card | scorer | scorer card | gap | template | locked | deck | skip | best | location |",
        "| --- | ---: | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in examples:
        location = f"{item.get('file')}:{item.get('line')}"
        locked = item.get("locked_template") if item.get("template_locked") else "-"
        lines.append(
            f"| `{item.get('run_id') or ''}` | {item.get('floor') or 0} | "
            f"`{item.get('old')}` | `{_card_label(item.get('old_card'))}` | "
            f"`{item.get('scorer')}` | `{_card_label(item.get('scorer_card'))}` | {item.get('gap')} | "
            f"`{item.get('template_id')}` | `{locked}` | {item.get('deck_size') or 0} | "
            f"{item.get('skip_score') if item.get('skip_score') is not None else 0} | "
            f"{item.get('best_card_score') if item.get('best_card_score') is not None else 0} | `{location}` |"
        )
    return "\n".join(lines) + "\n"


def details_list(examples):
    if not examples:
        return "- No examples.\n"
    lines = []
    for item in examples[:5]:
        reasons = item.get("selected_reasons") or []
        breakdown = item.get("score_breakdown") or {}
        lines.append(
            f"- `{item.get('run_id')}` floor {item.get('floor')}: "
            f"{_card_label(item.get('scorer_card'))}, score={item.get('selected_score')}, "
            f"gap={item.get('gap')}, reasons={'; '.join(map(str, reasons)) or '-'}, "
            f"breakdown={json.dumps(breakdown, ensure_ascii=False, sort_keys=True)}, "
            f"skip_score={item.get('skip_score')}, best_card_score={item.get('best_card_score')}, "
            f"skip_reasons={'; '.join(map(str, item.get('skip_reasons') or [])) or '-'}, "
            f"skip_breakdown={json.dumps(item.get('skip_score_breakdown') or {}, ensure_ascii=False, sort_keys=True)}"
        )
    return "\n".join(lines) + "\n"


def render_report(summary, input_files, report_date):
    metrics = summary["metrics"]
    lines = [
        f"# Card Scorer Shadow Report {report_date}",
        "",
        "## Summary",
        "",
        f"- Runs: {metrics['run_count']}",
        f"- Card reward events: {metrics['total_card_reward_events']}",
        f"- Avg candidate count: {metrics['avg_candidate_count']}",
        f"- Agreement rate: {pct(metrics['old_vs_scorer_agreement_rate'])}%",
        f"- Disagreements: {metrics['scorer_disagreed_with_old_policy']} ({pct(metrics['scorer_disagreed_with_old_policy_rate'])}%)",
        f"- Scorer skip rate: {pct(metrics['scorer_recommended_skip_rate'])}%",
        f"- Old policy skip rate: {pct(metrics['old_policy_skip_rate'])}%",
        f"- Avg skip score: {metrics['avg_skip_score']}",
        f"- Avg best card score: {metrics['avg_best_card_score']}",
        f"- Avg confidence gap: {metrics['avg_confidence_gap']}",
        f"- Score NaN / inf: {metrics['score_nan_count']} / {metrics['score_inf_count']}",
        f"- Reward term NaN / inf: {metrics['reward_term_nan_count']} / {metrics['reward_term_inf_count']}",
        f"- Avg deck size: {metrics['avg_deck_size']}",
        f"- Avg deck bloat score: {metrics['avg_deck_bloat_score']}",
        f"- Archetype consistency: {metrics['archetype_consistency']}",
        f"- Template lock rate: {pct(metrics['template_locked_rate'])}%",
        f"- Template sequence consistency: {metrics['template_sequence_consistency']}",
        "",
        "## Input Files",
        "",
    ]
    if input_files:
        lines.extend(f"- `{path}`" for path in input_files)
    else:
        lines.append("- No input files found.")
    lines.extend([
        "",
        "## Old Policy Distribution",
        "",
        md_table(summary["old_policy_distribution"], "old_policy_action", "count"),
        "## Scorer Recommendation Distribution",
        "",
        md_table(summary["scorer_distribution"], "scorer_action", "count"),
        "## Archetype Distribution",
        "",
        md_table(metrics["archetype_distribution"], "template_id", "count"),
        "## Locked Template Distribution",
        "",
        md_table(metrics["locked_template_distribution"], "locked_template", "count"),
        "## Reward Term Distribution",
        "",
        reward_terms_table(metrics["reward_term_distribution"]),
        "## Disagreement Examples",
        "",
        examples_table(summary["disagreement_examples"]),
        "## Disagreement Details",
        "",
        details_list(summary["disagreement_examples"]),
        "## High Confidence Examples",
        "",
        examples_table(summary["high_confidence_examples"]),
        "## Low Confidence Examples",
        "",
        examples_table(summary["low_confidence_examples"]),
        "## Raw Metrics",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
    ])
    return "\n".join(lines)


def analyze(args):
    input_files = resolve_input_files(args)
    since_ms = safe_float(getattr(args, "since_ms", 0) or 0) or 0
    records = []
    for path in input_files:
        for record in iter_jsonl(path):
            if since_ms and not record.get("_error"):
                timestamp = safe_float(record.get("timestamp"))
                if timestamp is None or timestamp < since_ms:
                    continue
            records.append(record)
    summary = summarize_records(records)
    report_date = args.date or datetime.now().strftime("%Y-%m-%d")
    if args.report:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = Path(args.report) if args.report != "auto" else report_dir / f"shadow_report_{report_date}.md"
        report_path.write_text(render_report(summary, input_files, report_date), encoding="utf-8")
        summary["report_path"] = str(report_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze Ironclad card scorer shadow logs.")
    parser.add_argument("--date", default="", help="Analyze RL_Datasets/OptionShadow/card_scorer_YYYY-MM-DD.jsonl.")
    parser.add_argument("--all", action="store_true", help="Analyze all card_scorer_*.jsonl files.")
    parser.add_argument("--files", nargs="*", help="Explicit JSONL files to analyze.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--report", nargs="?", const="auto", default="", help="Write Markdown report. Pass a path or omit value for auto path.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--since-ms", type=float, default=0.0, help="Only include shadow records with timestamp >= this epoch millisecond value.")
    args = parser.parse_args()
    summary = analyze(args)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    if summary.get("report_path"):
        print(f"Report written: {summary['report_path']}")


if __name__ == "__main__":
    main()
