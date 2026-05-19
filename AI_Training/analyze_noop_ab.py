import argparse
import hashlib
import json
import re
from pathlib import Path

from analyze_card_ab import compare_arms, parse_ids, render_markdown
from analyze_card_shadow import iter_jsonl


WORKSPACE = Path(__file__).resolve().parents[1]
AI_DATA_DIR = WORKSPACE / "RL_Datasets" / "AI"
REPORT_DIR = WORKSPACE / "RL_Datasets" / "OptionShadow" / "reports"


ACTION_TYPES = {"action", "macro_action"}


def canonical_payload(payload):
    if not isinstance(payload, dict):
        return {}
    keys = (
        "action",
        "card_id",
        "target_id",
        "potion_id",
        "reward_type",
        "reward_name",
        "node_type",
        "col",
        "row",
        "option_index",
        "index",
        "item_id",
        "item_name",
    )
    return {key: payload.get(key) for key in keys if payload.get(key) is not None}


def action_screen(record):
    if record.get("type") == "macro_action":
        screen = record.get("screen_state")
        if isinstance(screen, dict):
            return str(screen.get("state_type") or "")
    state = record.get("state_before") or record.get("state") or {}
    if isinstance(state, dict):
        return str(state.get("state_type") or "")
    return ""


def action_floor(record):
    state = record.get("state_before") or record.get("state") or {}
    if isinstance(state, dict):
        if isinstance(state.get("run"), dict):
            return int(state["run"].get("floor") or 0)
        return int(state.get("floor") or 0)
    return 0


def action_signature(record):
    payload = record.get("action_data") if isinstance(record.get("action_data"), dict) else {}
    return {
        "type": record.get("type") or "",
        "action_type": record.get("action_type") or payload.get("action") or "",
        "screen": action_screen(record),
        "payload": canonical_payload(payload),
    }


def record_state(record):
    if record.get("type") == "macro_action" and isinstance(record.get("screen_state"), dict):
        return record["screen_state"]
    for key in ("state_before", "state", "screen_state"):
        if isinstance(record.get(key), dict):
            return record[key]
    return {}


def state_run(state):
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    return {
        "act": run.get("act", state.get("act")),
        "floor": run.get("floor", state.get("floor")),
        "ascension": run.get("ascension", state.get("ascension")),
    }


def state_player(state):
    player = state.get("player") if isinstance(state.get("player"), dict) else state
    deck = player.get("deck") if isinstance(player.get("deck"), list) else []
    relics = player.get("relics") if isinstance(player.get("relics"), list) else state.get("relics", [])
    potions = player.get("potions") if isinstance(player.get("potions"), list) else state.get("potions", [])
    return {
        "hp": player.get("hp", state.get("hp")),
        "max_hp": player.get("max_hp", state.get("max_hp")),
        "gold": player.get("gold", state.get("gold")),
        "block": player.get("block", state.get("block")),
        "energy": state.get("energy"),
        "deck": stable_sorted(card_key(card) for card in deck if isinstance(card, dict)),
        "relics": stable_sorted(item_key(item) for item in relics if isinstance(item, dict)),
        "potions": [item_key(item) for item in potions if isinstance(item, dict)],
    }


def stable_sorted(items):
    return sorted(
        list(items),
        key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def card_key(card):
    return {
        "id": card.get("id") or card.get("card_id") or card.get("name"),
        "name": card.get("name"),
        "type": card.get("type"),
        "cost": card.get("cost"),
        "upgraded": card.get("upgraded"),
        "index": card.get("index"),
    }


def item_key(item):
    return {
        "id": item.get("id") or item.get("item_id") or item.get("relic_id") or item.get("potion_id") or item.get("name"),
        "name": item.get("name") or item.get("item_name") or item.get("relic_name") or item.get("potion_name"),
        "type": item.get("type") or item.get("reward_type") or item.get("category"),
        "index": item.get("index"),
        "cost": item.get("cost"),
    }


def enemy_key(enemy):
    return {
        "id": enemy.get("id") or enemy.get("name"),
        "hp": enemy.get("hp"),
        "max_hp": enemy.get("max_hp"),
        "block": enemy.get("block"),
        "intents": enemy.get("intents") or enemy.get("intent"),
    }


def map_key(state):
    map_state = state.get("map") if isinstance(state.get("map"), dict) else {}
    return {
        "current_position": map_state.get("current_position"),
        "visited": map_state.get("visited"),
        "boss": map_state.get("boss"),
        "next_options": [
            {
                "index": opt.get("index"),
                "type": opt.get("type") or opt.get("node_type"),
                "col": opt.get("col"),
                "row": opt.get("row"),
            }
            for opt in (map_state.get("next_options") or [])
            if isinstance(opt, dict)
        ],
    }


def reward_key(state):
    rewards = state.get("rewards") if isinstance(state.get("rewards"), dict) else {}
    treasure = state.get("treasure") if isinstance(state.get("treasure"), dict) else {}
    return {
        "rewards": [item_key(item) for item in (rewards.get("items") or []) if isinstance(item, dict)],
        "treasure": [item_key(item) for item in (treasure.get("items") or []) if isinstance(item, dict)],
        "can_proceed": rewards.get("can_proceed", treasure.get("can_proceed")),
    }


def card_reward_key(state):
    reward = state.get("card_reward") if isinstance(state.get("card_reward"), dict) else {}
    return {
        "cards": [card_key(card) for card in (reward.get("cards") or []) if isinstance(card, dict)],
        "can_skip": reward.get("can_skip"),
    }


def shop_key(state):
    shop = state.get("shop") if isinstance(state.get("shop"), dict) else {}
    fake = state.get("fake_merchant") if isinstance(state.get("fake_merchant"), dict) else {}
    source = shop or (fake.get("shop") if isinstance(fake.get("shop"), dict) else {})
    return {
        "items": [item_key(item) for item in (source.get("items") or []) if isinstance(item, dict)],
        "can_proceed": source.get("can_proceed"),
    }


def combat_key(state):
    return {
        "round": state.get("round"),
        "turn": state.get("turn"),
        "hand": [card_key(card) for card in (state.get("hand") or []) if isinstance(card, dict)],
        "draw_pile_count": state.get("draw_pile_count"),
        "discard_pile_count": state.get("discard_pile_count"),
        "exhaust_pile_count": state.get("exhaust_pile_count"),
        "enemies": [enemy_key(enemy) for enemy in (state.get("enemies") or []) if isinstance(enemy, dict)],
    }


def state_signature(record):
    state = record_state(record)
    screen = action_screen(record)
    return {
        "screen": screen,
        "run": state_run(state),
        "player": state_player(state),
        "map": map_key(state),
        "reward_offer": reward_key(state),
        "card_reward": card_reward_key(state),
        "shop": shop_key(state),
        "combat": combat_key(state),
    }


def compact_component(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded) <= 220:
        return value
    return {"sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(), "bytes": len(encoded)}


def component_diffs(left, right):
    diffs = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            diffs.append({
                "component": key,
                "baseline": compact_component(left.get(key)),
                "noop": compact_component(right.get(key)),
            })
    return diffs


def state_trace_hash(run_id):
    signatures = [state_signature(row) for row in trace_for_run(run_id)]
    encoded = json.dumps(signatures, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def first_state_divergence(baseline_run_id, noop_run_id):
    baseline_rows = trace_for_run(baseline_run_id)
    noop_rows = trace_for_run(noop_run_id)
    limit = min(len(baseline_rows), len(noop_rows))
    for index in range(limit):
        baseline_sig = state_signature(baseline_rows[index])
        noop_sig = state_signature(noop_rows[index])
        if baseline_sig != noop_sig:
            return {
                "index": index,
                "baseline_floor": action_floor(baseline_rows[index]),
                "noop_floor": action_floor(noop_rows[index]),
                "baseline_screen": action_screen(baseline_rows[index]),
                "noop_screen": action_screen(noop_rows[index]),
                "components": component_diffs(baseline_sig, noop_sig),
            }
    if len(baseline_rows) != len(noop_rows):
        return {"index": limit, "length_mismatch": [len(baseline_rows), len(noop_rows)], "components": []}
    return {}


def candidate_log_paths(run_ids):
    dates = set()
    for run_id in run_ids:
        match = re.search(r"ai_(\d{8})_", str(run_id or ""))
        if match:
            raw = match.group(1)
            dates.add(f"{raw[:4]}-{raw[4:6]}-{raw[6:]}")
    paths = []
    for folder in (AI_DATA_DIR / "Macro", AI_DATA_DIR / "Combat"):
        if not folder.exists():
            continue
        if dates:
            for date in sorted(dates):
                paths.extend(sorted(folder.glob(f"*{date}.jsonl")))
        else:
            paths.extend(sorted(folder.glob("*.jsonl")))
    return paths


def iter_action_records(run_ids):
    wanted = {str(run_id) for run_id in run_ids if str(run_id).strip()}
    if not wanted:
        return
    for path in candidate_log_paths(wanted):
        for record in iter_jsonl(path):
            if str(record.get("run_id") or "") not in wanted:
                continue
            if record.get("type") not in ACTION_TYPES:
                continue
            yield record


def trace_for_run(run_id):
    rows = [row for row in iter_action_records([run_id])]
    rows.sort(key=lambda row: int(row.get("timestamp") or 0))
    return rows


def trace_hash(run_id):
    signatures = [action_signature(row) for row in trace_for_run(run_id)]
    encoded = json.dumps(signatures, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def first_trace_divergence(baseline_run_id, noop_run_id):
    baseline_rows = trace_for_run(baseline_run_id)
    noop_rows = trace_for_run(noop_run_id)
    limit = min(len(baseline_rows), len(noop_rows))
    for index in range(limit):
        baseline_sig = action_signature(baseline_rows[index])
        noop_sig = action_signature(noop_rows[index])
        if baseline_sig != noop_sig:
            return {
                "index": index,
                "baseline_floor": action_floor(baseline_rows[index]),
                "noop_floor": action_floor(noop_rows[index]),
                "baseline_screen": baseline_sig["screen"],
                "noop_screen": noop_sig["screen"],
                "baseline_action": baseline_sig["action_type"],
                "noop_action": noop_sig["action_type"],
                "baseline_payload": baseline_sig["payload"],
                "noop_payload": noop_sig["payload"],
            }
    if len(baseline_rows) != len(noop_rows):
        longer = baseline_rows if len(baseline_rows) > len(noop_rows) else noop_rows
        row = longer[limit]
        sig = action_signature(row)
        return {
            "index": limit,
            "baseline_floor": action_floor(row) if longer is baseline_rows else 0,
            "noop_floor": action_floor(row) if longer is noop_rows else 0,
            "baseline_screen": sig["screen"] if longer is baseline_rows else "",
            "noop_screen": sig["screen"] if longer is noop_rows else "",
            "baseline_action": sig["action_type"] if longer is baseline_rows else "",
            "noop_action": sig["action_type"] if longer is noop_rows else "",
            "baseline_payload": sig["payload"] if longer is baseline_rows else {},
            "noop_payload": sig["payload"] if longer is noop_rows else {},
        }
    return {}


def attach_trace_divergences(summary):
    per_seed = summary.get("per_seed", [])
    for row in per_seed:
        row["trace_first_divergence"] = first_trace_divergence(
            row.get("baseline_run_id"),
            row.get("active_run_id"),
        )
        row["state_first_divergence"] = first_state_divergence(
            row.get("baseline_run_id"),
            row.get("active_run_id"),
        )
        row["baseline_trace_hash"] = trace_hash(row.get("baseline_run_id"))
        row["noop_trace_hash"] = trace_hash(row.get("active_run_id"))
        row["trace_hash_match"] = row["baseline_trace_hash"] == row["noop_trace_hash"]
        row["baseline_state_trace_hash"] = state_trace_hash(row.get("baseline_run_id"))
        row["noop_state_trace_hash"] = state_trace_hash(row.get("active_run_id"))
        row["state_trace_hash_match"] = row["baseline_state_trace_hash"] == row["noop_state_trace_hash"]
    summary["trace_first_divergence_count"] = sum(
        1 for row in per_seed if row.get("trace_first_divergence")
    )
    summary["full_trace_first_divergence_count"] = summary["trace_first_divergence_count"]
    summary["state_first_divergence_count"] = sum(
        1 for row in per_seed if row.get("state_first_divergence")
    )
    summary["trace_first_divergence"] = next(
        (row["trace_first_divergence"] for row in per_seed if row.get("trace_first_divergence")),
        {},
    )
    summary["state_first_divergence"] = next(
        (row["state_first_divergence"] for row in per_seed if row.get("state_first_divergence")),
        {},
    )
    return summary


def render_noop_markdown(summary):
    base = render_markdown(summary).rstrip()
    lines = [
        base,
        "",
        "## No-op Integrity",
        "",
        f"- final_action_first_divergence_count: `{summary.get('final_action_first_divergence_count', 0)}`",
        f"- trace_first_divergence_count: `{summary.get('trace_first_divergence_count', 0)}`",
        f"- full_trace_first_divergence_count: `{summary.get('full_trace_first_divergence_count', 0)}`",
        f"- state_first_divergence_count: `{summary.get('state_first_divergence_count', 0)}`",
        f"- trace_first_divergence: `{summary.get('trace_first_divergence') or {}}`",
        f"- state_first_divergence: `{summary.get('state_first_divergence') or {}}`",
        "",
        "| Seed | Baseline Run | Noop Run | Trace First Divergence | State First Divergence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary.get("per_seed", []):
        div = row.get("trace_first_divergence") or {}
        if div:
            text = (
                f"#{div['index']} "
                f"{div['baseline_screen'] or div['noop_screen']} "
                f"{div['baseline_action']} -> {div['noop_action']}"
            )
        else:
            text = "-"
        state_div = row.get("state_first_divergence") or {}
        if state_div:
            components = ", ".join(
                str(item.get("component")) for item in (state_div.get("components") or [])[:4]
            )
            state_text = (
                f"#{state_div.get('index')} "
                f"{state_div.get('baseline_screen') or state_div.get('noop_screen') or ''} "
                f"{components}"
            )
        else:
            state_text = "-"
        lines.append(
            f"| `{row.get('seed', '')}` | `{row.get('baseline_run_id', '')}` | "
            f"`{row.get('active_run_id', '')}` | {text} | {state_text} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs active_canary_noop and report full trace divergence.")
    parser.add_argument("--baseline-run-ids", required=True)
    parser.add_argument("--noop-run-ids", required=True)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--report-name", default="noop_ab_report.md")
    args = parser.parse_args()

    summary = compare_arms(parse_ids(args.baseline_run_ids), parse_ids(args.noop_run_ids))
    attach_trace_divergences(summary)
    if args.report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / args.report_name
        report_path.write_text(render_noop_markdown(summary), encoding="utf-8")
        summary["report_path"] = str(report_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
