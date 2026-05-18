import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import requests

from analyze_card_shadow import analyze, iter_jsonl


WORKSPACE = Path(__file__).resolve().parents[1]
SHADOW_DIR = WORKSPACE / "RL_Datasets" / "OptionShadow"
REPORT_DIR = SHADOW_DIR / "reports"
DEFAULT_PANEL_URL = "http://127.0.0.1:8765"
GAME_API_URL = "http://127.0.0.1:15526/api/v1/singleplayer"


class AnalyzeArgs:
    def __init__(self, files, since_ms, report=True):
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.all = False
        self.files = [str(path) for path in files]
        self.log_dir = str(SHADOW_DIR)
        self.report = "auto" if report else ""
        self.report_dir = str(REPORT_DIR)
        self.since_ms = since_ms


def request_json(method, url, body=None, timeout=10):
    response = requests.request(method, url, json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()


def shadow_files():
    return sorted(SHADOW_DIR.glob("card_scorer_*.jsonl"))


def count_shadow_events_since(start_ms, seed=""):
    total = 0
    latest_ts = 0
    files = shadow_files()
    for path in files:
        for record in iter_jsonl(path):
            if record.get("type") != "card_scorer_shadow":
                continue
            if seed and str(record.get("seed") or "") != str(seed):
                continue
            try:
                timestamp = float(record.get("timestamp") or 0)
            except (TypeError, ValueError):
                timestamp = 0
            if timestamp >= start_ms:
                total += 1
                latest_ts = max(latest_ts, int(timestamp))
    return total, latest_ts, files


def configure_panel(panel_url, args):
    seed = str(args.seed or "").strip()
    patch = {
        "ai_enabled": True,
        "macro_enabled": True,
        "macro_shop_enabled": True,
        "collection_enabled": True,
        "record_ai_actions": True,
        "include_ai_in_training": True,
        "game_speed_enabled": True,
        "game_speed_multiplier": args.game_speed,
        "self_play_game_speed_multiplier": args.game_speed,
        "self_play_character": args.character.upper(),
        "self_play_seed": seed,
        "policy_mode": args.policy_mode,
        "self_play_target_runs": 0,
        "self_play_train_every_admitted_runs": args.train_every,
        "self_play_max_run_minutes": args.max_run_minutes,
        "self_play_stall_seconds": args.stall_seconds,
        "exploration_enabled": not bool(args.deterministic_eval),
        "exploration_mode": "aggressive",
        "self_play_constraint_mode": "guarded" if args.deterministic_eval else args.constraint_mode,
        "combat_exploration_epsilon": args.combat_epsilon,
        "macro_exploration_epsilon": args.macro_epsilon,
        "exploration_top_k": args.exploration_top_k,
        "exploration_temperature": args.exploration_temperature,
        "evaluation_deterministic": bool(args.deterministic_eval),
        "option_card_scorer": {"mode": args.card_scorer_mode},
    }
    return request_json("POST", f"{panel_url}/api/control", patch, timeout=15)


def start_or_keep_running(panel_url):
    request_json("POST", f"{panel_url}/api/ai/start", {}, timeout=20)
    result = request_json("POST", f"{panel_url}/api/self-play/start", {}, timeout=20)
    return result


def stop_collection(panel_url, keep_ai_running=False):
    errors = []
    try:
        request_json("POST", f"{panel_url}/api/self-play/stop", {}, timeout=15)
    except Exception as exc:
        errors.append(f"self-play stop failed: {exc}")
    if not keep_ai_running:
        try:
            request_json("POST", f"{panel_url}/api/ai/stop", {}, timeout=15)
        except Exception as exc:
            errors.append(f"AI stop failed: {exc}")
    return errors


def compact_status(panel_url):
    try:
        status = request_json("GET", f"{panel_url}/api/status", timeout=5)
    except Exception as exc:
        return {"status_error": str(exc)}
    self_play = status.get("self_play") or {}
    return {
        "ai_pid": status.get("ai_pid"),
        "self_play_running": self_play.get("running"),
        "self_play_phase": self_play.get("phase"),
        "self_play_message": self_play.get("message"),
        "completed_runs": self_play.get("completed_runs"),
        "target_runs": self_play.get("target_runs"),
    }


def compact_game_state():
    try:
        state = request_json("GET", f"{GAME_API_URL}?format=json", timeout=5)
    except Exception as exc:
        return {"game_error": str(exc)}
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    return {
        "state_type": state.get("state_type"),
        "act": run.get("act"),
        "floor": run.get("floor"),
    }


def maybe_restart_self_play(panel_url, started_once):
    status = compact_status(panel_url)
    if status.get("self_play_running"):
        return status
    if started_once:
        try:
            result = request_json("POST", f"{panel_url}/api/self-play/start", {}, timeout=20)
            status["restart"] = result.get("message") or result.get("status")
        except Exception as exc:
            status["restart_error"] = str(exc)
    return status


def run_loop(args):
    panel_url = args.panel_url.rstrip("/")
    start_ms = int(time.time() * 1000)
    # A fixed-seed collection run must start from a fresh self-play session;
    # otherwise a previous loop can leak its current run into the next seed.
    stop_collection(panel_url, keep_ai_running=True)
    configure_panel(panel_url, args)
    start_result = start_or_keep_running(panel_url)
    print(json.dumps({
        "event": "started",
        "target_events": args.target_events,
        "seed": str(args.seed or "") or "random",
        "panel": panel_url,
        "start_result": start_result,
    }, ensure_ascii=False))

    deadline = time.time() + args.max_minutes * 60
    last_count = -1
    started_once = True
    final_count = 0
    final_files = []
    try:
        while time.time() < deadline:
            count, latest_ts, files = count_shadow_events_since(start_ms, str(args.seed or "").strip())
            final_count = count
            final_files = files
            if count != last_count or args.verbose:
                status = maybe_restart_self_play(panel_url, started_once)
                game = compact_game_state()
                print(json.dumps({
                    "event": "progress",
                    "shadow_events": count,
                    "target_events": args.target_events,
                    "latest_shadow_ts": latest_ts,
                    "game": game,
                    "status": status,
                }, ensure_ascii=False))
                last_count = count
            if count >= args.target_events:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print(json.dumps({"event": "interrupted", "shadow_events": final_count}, ensure_ascii=False))
        raise
    finally:
        if final_count >= args.target_events or args.stop_on_exit:
            errors = stop_collection(panel_url, keep_ai_running=args.keep_ai_running)
            if errors:
                print(json.dumps({"event": "stop_warnings", "errors": errors}, ensure_ascii=False))

    if final_count < args.target_events:
        raise SystemExit(f"Timed out before target: {final_count}/{args.target_events} shadow events")

    summary = analyze(AnalyzeArgs(final_files, since_ms=start_ms, report=not args.no_report))
    print(json.dumps({
        "event": "finished",
        "shadow_events": final_count,
        "report_path": summary.get("report_path"),
        "metrics": summary.get("metrics"),
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run self-play until card scorer shadow events reach a target count.")
    parser.add_argument("--target-events", type=int, default=50, help="Stop after this many new card_reward shadow events.")
    parser.add_argument("--seed", default="", help="Fixed seed. Leave empty for random runs.")
    parser.add_argument("--character", default="IRONCLAD")
    parser.add_argument("--policy-mode", default="current_rl", choices=["current_rl", "ppo_experiment", "ppo_best"])
    parser.add_argument(
        "--card-scorer-mode",
        default="shadow",
        choices=["off", "shadow", "active", "active_canary", "active_canary_noop"],
        help="Card reward scorer mode. Default stays shadow; use active_canary for guarded takeover tests.",
    )
    parser.add_argument("--panel-url", default=DEFAULT_PANEL_URL)
    parser.add_argument("--constraint-mode", default="explore", choices=["guarded", "explore", "free"])
    parser.add_argument("--train-every", type=int, default=999, help="Large default avoids training during shadow collection.")
    parser.add_argument("--max-run-minutes", type=int, default=75)
    parser.add_argument("--stall-seconds", type=int, default=120)
    parser.add_argument("--max-minutes", type=int, default=180, help="Wall-clock timeout for the collection loop.")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--game-speed", type=float, default=4.0)
    parser.add_argument("--combat-epsilon", type=float, default=0.5)
    parser.add_argument("--macro-epsilon", type=float, default=0.6)
    parser.add_argument("--exploration-top-k", type=int, default=5)
    parser.add_argument("--exploration-temperature", type=float, default=1.4)
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--keep-ai-running", action="store_true")
    parser.add_argument("--stop-on-exit", action="store_true", help="Also stop self-play/AI if the loop times out or is interrupted.")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.target_events <= 0:
        raise SystemExit("--target-events must be > 0")
    run_loop(args)


if __name__ == "__main__":
    main()
