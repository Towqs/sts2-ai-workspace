import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from analyze_card_shadow import iter_jsonl, old_policy_label, safe_float
from run_summary import latest_runs


WORKSPACE = Path(__file__).resolve().parents[1]
SHADOW_DIR = WORKSPACE / "RL_Datasets" / "OptionShadow"
REPORT_DIR = SHADOW_DIR / "reports"


def mean(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else 0.0


def pct(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def round4(value):
    return round(float(value), 4)


def card_shadow_rows():
    for path in sorted(SHADOW_DIR.glob("card_scorer_*.jsonl")):
        yield from iter_jsonl(path)


def collect_card_metrics(run_ids):
    run_ids = {str(run_id) for run_id in run_ids}
    rows = [
        row
        for row in card_shadow_rows()
        if row.get("type") == "card_scorer_shadow" and str(row.get("run_id") or "") in run_ids
    ]
    per_run = defaultdict(list)
    for row in rows:
        per_run[str(row.get("run_id"))].append(row)

    takeover_count = 0
    scorer_skip_count = 0
    executed_skip_count = 0
    old_skip_count = 0
    deck_sizes = []
    active_runs = 0
    for run_rows in per_run.values():
        if any(bool(row.get("canary_takeover")) for row in run_rows):
            active_runs += 1
        for row in run_rows:
            takeover_count += int(bool(row.get("canary_takeover")))
            recommended = str(row.get("recommended_action") or "")
            scorer_skip_count += int(recommended == "skip_reward")
            executed_skip_count += int(bool(row.get("executed_skip")) or bool(row.get("actual_skip")))
            old_skip_count += int(old_policy_label(row) == "skip_reward")
            deck_summary = row.get("deck_summary") if isinstance(row.get("deck_summary"), dict) else {}
            deck_size = safe_float(deck_summary.get("deck_size"))
            if deck_size is not None and math.isfinite(deck_size):
                deck_sizes.append(deck_size)
    return {
        "card_reward_events": len(rows),
        "scorer_skip_rate": round4(pct(scorer_skip_count, len(rows))),
        "executed_skip_rate": round4(pct(executed_skip_count, len(rows))),
        "old_skip_rate": round4(pct(old_skip_count, len(rows))),
        "avg_deck_size": round4(mean(deck_sizes)),
        "takeover_count": takeover_count,
        "takeover_rate": round4(pct(takeover_count, len(rows))),
        "runs_with_takeover": active_runs,
    }


def runs_by_id(run_ids):
    run_ids = {str(run_id) for run_id in run_ids}
    return {
        str(run.get("run_id")): run
        for run in latest_runs(limit=100000)
        if str(run.get("run_id") or "") in run_ids
    }


def collect_outcome_metrics(run_ids):
    runs = list(runs_by_id(run_ids).values())
    max_floors = [safe_float(run.get("max_floor")) for run in runs]
    boss_damage = [safe_float(run.get("boss_damage")) for run in runs]
    illegal = [safe_float(run.get("invalid_actions")) for run in runs]
    act1_clears = sum(1 for run in runs if int(run.get("max_act") or 0) >= 2)
    return {
        "run_count": len(runs),
        "average_floor": round4(mean(max_floors)),
        "act1_clear_rate": round4(pct(act1_clears, len(runs))),
        "average_boss_damage": round4(mean(boss_damage)),
        "illegal_action_rate": round4(pct(sum(1 for value in illegal if value and value > 0), len(runs))),
        "illegal_action_total": int(sum(value or 0 for value in illegal)),
    }


def seed_outcome_rows(baseline_ids, canary_ids):
    baseline_runs = runs_by_id(baseline_ids)
    canary_runs = runs_by_id(canary_ids)
    rows_by_run = defaultdict(list)
    for row in card_shadow_rows():
        if row.get("type") == "card_scorer_shadow":
            rows_by_run[str(row.get("run_id") or "")].append(row)

    baseline_by_seed = {str(run.get("seed") or ""): run for run in baseline_runs.values()}
    canary_by_seed = {str(run.get("seed") or ""): run for run in canary_runs.values()}
    seeds = sorted(set(baseline_by_seed) | set(canary_by_seed), key=lambda value: int(value) if value.isdigit() else value)
    rows = []
    for seed in seeds:
        baseline = baseline_by_seed.get(seed, {})
        canary = canary_by_seed.get(seed, {})
        canary_rows = rows_by_run.get(str(canary.get("run_id") or ""), [])
        takeover_rows = [row for row in canary_rows if row.get("canary_takeover")]
        comparable_rows = [row for row in canary_rows if row.get("old_policy_action") and row.get("final_executed_action")]
        match_count = sum(
            1
            for row in comparable_rows
            if row.get("old_policy_action") == row.get("final_executed_action")
        )
        first_divergence = next(
            (
                row for row in canary_rows
                if row.get("old_policy_action")
                and row.get("final_executed_action")
                and row.get("old_policy_action") != row.get("final_executed_action")
            ),
            {},
        )
        rows.append({
            "seed": seed,
            "baseline_run_id": baseline.get("run_id", ""),
            "active_run_id": canary.get("run_id", ""),
            "baseline_floor": int(baseline.get("max_floor") or 0),
            "active_floor": int(canary.get("max_floor") or 0),
            "floor_delta": int(canary.get("max_floor") or 0) - int(baseline.get("max_floor") or 0),
            "baseline_act1_clear": int(baseline.get("max_act") or 0) >= 2,
            "active_act1_clear": int(canary.get("max_act") or 0) >= 2,
            "baseline_boss_damage": int(baseline.get("boss_damage") or 0),
            "active_boss_damage": int(canary.get("boss_damage") or 0),
            "takeover_count": len(takeover_rows),
            "skip_takeover_count": sum(1 for row in takeover_rows if row.get("recommended_action") == "skip_reward"),
            "pick_takeover_count": sum(1 for row in takeover_rows if row.get("recommended_action") != "skip_reward"),
            "fallback_count": sum(1 for row in canary_rows if row.get("canary_fallback_reason")),
            "executed_action_match_rate": round4(pct(match_count, len(comparable_rows))),
            "first_divergence_floor": int(first_divergence.get("floor") or 0),
            "first_divergence_screen": first_divergence.get("screen_type") or "",
            "first_divergence_action": first_divergence.get("final_executed_action") or "",
        })
    return rows


def takeover_audit_rows(canary_ids, limit=30):
    run_map = runs_by_id(canary_ids)
    run_ids = set(run_map)
    rows = []
    for row in card_shadow_rows():
        if row.get("type") != "card_scorer_shadow" or str(row.get("run_id") or "") not in run_ids:
            continue
        if not row.get("canary_takeover"):
            continue
        run = run_map.get(str(row.get("run_id") or ""), {})
        old_card = row.get("old_policy_card") if isinstance(row.get("old_policy_card"), dict) else {}
        scorer_card = row.get("scorer_card") if isinstance(row.get("scorer_card"), dict) else {}
        deck_summary = row.get("deck_summary") if isinstance(row.get("deck_summary"), dict) else {}
        rows.append({
            "seed": str(row.get("seed") or run.get("seed") or ""),
            "run_id": row.get("run_id"),
            "floor": int(row.get("floor") or 0),
            "deck_size": int(deck_summary.get("deck_size") or 0),
            "template": row.get("template_id") or "",
            "old_action": row.get("old_policy_action") or "",
            "old_card_name": old_card.get("name") or old_card.get("card_id") or "",
            "scorer_action": row.get("recommended_action") or "",
            "scorer_card_name": scorer_card.get("name") or scorer_card.get("card_id") or "",
            "confidence_gap": round4(safe_float(row.get("confidence_gap")) or 0.0),
            "final_floor": int(run.get("max_floor") or 0),
            "act1_clear": int(run.get("max_act") or 0) >= 2,
            "boss_damage": int(run.get("boss_damage") or 0),
        })
    rows.sort(key=lambda item: (item["seed"], item["floor"], item["run_id"]))
    return rows[:limit]


def summarize_arm(name, run_ids):
    run_ids = [str(run_id) for run_id in run_ids if str(run_id).strip()]
    payload = {"name": name, "run_ids": run_ids}
    payload.update(collect_outcome_metrics(run_ids))
    payload.update(collect_card_metrics(run_ids))
    return payload


def compare_arms(baseline_ids, canary_ids):
    baseline = summarize_arm("baseline", baseline_ids)
    canary = summarize_arm("active_canary", canary_ids)
    per_seed = seed_outcome_rows(baseline_ids, canary_ids)
    deltas = {}
    for key in (
        "average_floor",
        "act1_clear_rate",
        "average_boss_damage",
        "avg_deck_size",
        "executed_skip_rate",
        "illegal_action_rate",
    ):
        deltas[key] = round4(float(canary.get(key, 0.0)) - float(baseline.get(key, 0.0)))
    comparable = [row for row in per_seed if row.get("executed_action_match_rate") is not None]
    first_divergence = next(
        (
            {
                "seed": row["seed"],
                "floor": row["first_divergence_floor"],
                "screen": row["first_divergence_screen"],
                "action": row["first_divergence_action"],
            }
            for row in per_seed
            if row.get("first_divergence_floor")
        ),
        {},
    )
    final_action_first_divergence_count = sum(1 for row in per_seed if row.get("first_divergence_floor"))
    return {
        "baseline": baseline,
        "active_canary": canary,
        "delta": deltas,
        "per_seed": per_seed,
        "takeover_examples": takeover_audit_rows(canary_ids),
        "paired_seed_count": len(per_seed),
        "executed_action_match_rate": round4(mean([row["executed_action_match_rate"] for row in comparable])),
        "final_action_first_divergence_count": final_action_first_divergence_count,
        "first_divergence": first_divergence,
    }


def render_markdown(summary):
    baseline = summary["baseline"]
    canary = summary["active_canary"]
    delta = summary["delta"]
    lines = [
        "# Card Scorer A/B Outcome Report",
        "",
        "| Metric | Baseline | Active Canary | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in (
        "run_count",
        "card_reward_events",
        "average_floor",
        "act1_clear_rate",
        "average_boss_damage",
        "avg_deck_size",
        "executed_skip_rate",
        "scorer_skip_rate",
        "illegal_action_rate",
        "takeover_rate",
    ):
        lines.append(
            f"| `{key}` | {baseline.get(key, 0)} | {canary.get(key, 0)} | {delta.get(key, '')} |"
        )
    lines.extend([
        "",
        "## Runs",
        "",
        f"- Baseline: `{', '.join(baseline['run_ids'])}`",
        f"- Active canary: `{', '.join(canary['run_ids'])}`",
        "",
        "## Canary Notes",
        "",
        f"- takeover_count: `{canary['takeover_count']}`",
        f"- runs_with_takeover: `{canary['runs_with_takeover']}`",
        f"- scorer_skip_rate: `{canary['scorer_skip_rate']}`",
        f"- executed_skip_rate: `{canary['executed_skip_rate']}`",
        f"- old_skip_rate: `{canary['old_skip_rate']}`",
        f"- paired_seed_count: `{summary.get('paired_seed_count', 0)}`",
        f"- executed_action_match_rate: `{summary.get('executed_action_match_rate', 0)}`",
        f"- final_action_first_divergence_count: `{summary.get('final_action_first_divergence_count', 0)}`",
        f"- first_divergence: `{summary.get('first_divergence') or {}}`",
        "",
        "## Per-Seed Outcomes",
        "",
        "| Seed | Baseline Floor | Active Floor | Delta | Baseline A1 | Active A1 | Baseline Boss Damage | Active Boss Damage | Takeovers | Skip TO | Pick TO | Fallbacks | Action Match | First Divergence |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in summary.get("per_seed", []):
        lines.append(
            f"| `{row['seed']}` | {row['baseline_floor']} | {row['active_floor']} | {row['floor_delta']} | "
            f"{row['baseline_act1_clear']} | {row['active_act1_clear']} | {row['baseline_boss_damage']} | "
            f"{row['active_boss_damage']} | {row['takeover_count']} | {row['skip_takeover_count']} | "
            f"{row['pick_takeover_count']} | {row['fallback_count']} | {row['executed_action_match_rate']} | "
            f"{row['first_divergence_floor'] or '-'} {row['first_divergence_screen'] or ''} {row['first_divergence_action'] or ''} |"
        )
    lines.extend([
        "",
        "## Takeover Audit",
        "",
        "| Seed | Floor | Deck | Template | Old | Scorer | Gap | Final Floor | A1 Clear | Boss Damage |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |",
    ])
    for row in summary.get("takeover_examples", []):
        lines.append(
            f"| `{row['seed']}` | {row['floor']} | {row['deck_size']} | `{row['template']}` | "
            f"{row['old_card_name'] or row['old_action']} | {row['scorer_card_name'] or row['scorer_action']} | "
            f"{row['confidence_gap']} | {row['final_floor']} | {row['act1_clear']} | {row['boss_damage']} |"
        )
    return "\n".join(lines) + "\n"


def parse_ids(raw):
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs active_canary run outcomes.")
    parser.add_argument("--baseline-run-ids", required=True, help="Comma-separated baseline/shadow run ids.")
    parser.add_argument("--canary-run-ids", required=True, help="Comma-separated active_canary run ids.")
    parser.add_argument("--report", action="store_true", help="Write a markdown report under OptionShadow/reports.")
    parser.add_argument("--report-name", default="card_ab_report.md")
    args = parser.parse_args()

    summary = compare_arms(parse_ids(args.baseline_run_ids), parse_ids(args.canary_run_ids))
    if args.report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / args.report_name
        report_path.write_text(render_markdown(summary), encoding="utf-8")
        summary["report_path"] = str(report_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
