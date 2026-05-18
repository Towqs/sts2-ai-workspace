import argparse
import json
import time

import requests

from run_summary import latest_runs


PANEL_URL = "http://127.0.0.1:8765"


def request_json(method, path, body=None, timeout=20):
    response = requests.request(method, PANEL_URL + path, json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()


def configure(seed, mode, canary_overrides=None):
    scorer_cfg = {"mode": mode}
    if canary_overrides:
        scorer_cfg.update(canary_overrides)
    return request_json("POST", "/api/control", {
        "ai_enabled": True,
        "macro_enabled": True,
        "macro_shop_enabled": True,
        "collection_enabled": True,
        "record_ai_actions": True,
        "include_ai_in_training": True,
        "game_speed_enabled": True,
        "game_speed_multiplier": 4.0,
        "self_play_game_speed_multiplier": 4.0,
        "self_play_character": "IRONCLAD",
        "self_play_seed": str(seed),
        "policy_mode": "current_rl",
        "self_play_target_runs": 1,
        "self_play_train_every_admitted_runs": 999,
        "self_play_max_run_minutes": 75,
        "self_play_stall_seconds": 120,
        "option_card_scorer": scorer_cfg,
    })


def wait_for_new_run(seed, existing_ids, timeout_sec=180):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        for run in latest_runs(limit=40):
            if str(run.get("seed") or "") == str(seed) and run.get("run_id") not in existing_ids:
                return run.get("run_id")
        time.sleep(5)
    raise TimeoutError(f"new run did not appear for seed {seed}")


def wait_until_finished(run_id, timeout_sec=5400):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        status = request_json("GET", "/api/status", timeout=10)
        self_play = status.get("self_play") or {}
        if not self_play.get("running") and (
            self_play.get("current_run_id") == run_id
            or any(run.get("run_id") == run_id for run in latest_runs(limit=40))
        ):
            return status
        time.sleep(10)
    raise TimeoutError(f"run did not finish: {run_id}")


def run_seed(seed, mode, canary_overrides=None):
    existing_ids = {run.get("run_id") for run in latest_runs(limit=300)}
    request_json("POST", "/api/self-play/stop", {})
    configure(seed, mode, canary_overrides=canary_overrides)
    request_json("POST", "/api/ai/start", {})
    request_json("POST", "/api/self-play/start", {})
    run_id = wait_for_new_run(seed, existing_ids)
    wait_until_finished(run_id)
    return run_id


def main():
    parser = argparse.ArgumentParser(description="Run fixed-seed card scorer A/B batches.")
    parser.add_argument("--mode", required=True, choices=["shadow", "active_canary", "active_canary_noop"])
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds.")
    parser.add_argument("--only-when-confidence-gap-gte", type=float)
    parser.add_argument("--fallback-to-old-when-gap-lt", type=float)
    parser.add_argument("--allow-skip-when-deck-size-gte", type=int)
    parser.add_argument("--allow-skip-when-best-card-score-lte", type=float)
    parser.add_argument("--max-active-ratio-per-run", type=float)
    args = parser.parse_args()
    overrides = {
        key: value
        for key, value in {
            "only_when_confidence_gap_gte": args.only_when_confidence_gap_gte,
            "fallback_to_old_when_gap_lt": args.fallback_to_old_when_gap_lt,
            "allow_skip_when_deck_size_gte": args.allow_skip_when_deck_size_gte,
            "allow_skip_when_best_card_score_lte": args.allow_skip_when_best_card_score_lte,
            "max_active_ratio_per_run": args.max_active_ratio_per_run,
        }.items()
        if value is not None
    }
    results = []
    for seed in [item.strip() for item in args.seeds.split(",") if item.strip()]:
        run_id = run_seed(seed, args.mode, canary_overrides=overrides)
        results.append({"seed": seed, "run_id": run_id, "mode": args.mode})
        print(json.dumps(results[-1], ensure_ascii=False))
    print(json.dumps({"mode": args.mode, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
