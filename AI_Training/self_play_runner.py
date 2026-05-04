import argparse
import json
import os
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import requests

from run_summary import latest_runs, safe_int, save_self_play_run_score


WORKSPACE = Path(__file__).resolve().parents[1]
CONTROL_PATH = WORKSPACE / "AI_Training" / "control_state.json"
SELF_PLAY_STATE_PATH = WORKSPACE / "AI_Training" / "self_play_state.json"
SERVER_STATE_PATH = WORKSPACE / "AI_Training" / "control_panel_state.json"
PANEL_BASE_URL = f"http://127.0.0.1:{int(os.environ.get('STS2_AI_PANEL_PORT', '8765'))}"
GAME_API_URL = "http://127.0.0.1:15526/api/v1/singleplayer"

DEFAULT_CONFIG = {
    "self_play_character": "IRONCLAD",
    "self_play_ascension": 0,
    "self_play_target_runs": 20,
    "self_play_train_every_admitted_runs": 5,
    "self_play_max_run_minutes": 75,
    "self_play_stall_seconds": 120,
    "game_speed_enabled": True,
    "game_speed_multiplier": 3.0,
    "self_play_game_speed_multiplier": 3.0,
    "macro_enabled": True,
    "macro_shop_enabled": True,
    "collection_enabled": True,
    "include_ai_in_training": True,
    "min_training_quality": "partial_act1",
    "ai_min_training_quality": "partial_act1",
    "exploration_enabled": True,
    "exploration_mode": "aggressive",
    "combat_exploration_epsilon": 0.35,
    "macro_exploration_epsilon": 0.25,
    "exploration_top_k": 5,
    "exploration_temperature": 1.35,
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


def read_control():
    data = deepcopy(DEFAULT_CONFIG)
    data.update(read_json(CONTROL_PATH, {}))
    return data


def clamp_int(value, default, minimum, maximum):
    try:
        return max(minimum, min(int(value), maximum))
    except (TypeError, ValueError):
        return default


def clamp_float(value, default, minimum, maximum):
    try:
        return max(minimum, min(float(value), maximum))
    except (TypeError, ValueError):
        return default


def merged_config(overrides=None):
    data = read_control()
    if overrides:
        data.update(overrides)
    return {
        "self_play_character": str(data.get("self_play_character") or DEFAULT_CONFIG["self_play_character"]).upper(),
        "self_play_ascension": clamp_int(data.get("self_play_ascension"), 0, 0, 20),
        "self_play_target_runs": clamp_int(data.get("self_play_target_runs"), 20, 1, 999),
        "self_play_train_every_admitted_runs": clamp_int(data.get("self_play_train_every_admitted_runs"), 5, 1, 999),
        "self_play_max_run_minutes": clamp_int(data.get("self_play_max_run_minutes"), 75, 5, 720),
        "self_play_stall_seconds": clamp_int(data.get("self_play_stall_seconds"), 120, 15, 3600),
        "game_speed_enabled": bool(data.get("game_speed_enabled", True)),
        "game_speed_multiplier": clamp_float(
            data.get("self_play_game_speed_multiplier", data.get("game_speed_multiplier")),
            3.0,
            1.0,
            6.0,
        ),
        "macro_enabled": bool(data.get("macro_enabled", True)),
        "macro_shop_enabled": bool(data.get("macro_shop_enabled", True)),
        "collection_enabled": bool(data.get("collection_enabled", True)),
        "include_ai_in_training": bool(data.get("include_ai_in_training", True)),
        "min_training_quality": str(data.get("min_training_quality") or "partial_act1"),
        "ai_min_training_quality": str(data.get("ai_min_training_quality") or "partial_act1"),
        "exploration_enabled": bool(data.get("exploration_enabled", True)),
        "exploration_mode": str(data.get("exploration_mode") or "aggressive"),
        "combat_exploration_epsilon": clamp_float(data.get("combat_exploration_epsilon"), 0.35, 0.0, 1.0),
        "macro_exploration_epsilon": clamp_float(data.get("macro_exploration_epsilon"), 0.25, 0.0, 1.0),
        "exploration_top_k": clamp_int(data.get("exploration_top_k"), 5, 1, 12),
        "exploration_temperature": clamp_float(data.get("exploration_temperature"), 1.35, 0.1, 5.0),
    }


def next_training_in(admitted_runs, interval):
    if interval <= 0:
        return 0
    if admitted_runs <= 0:
        return interval
    remainder = admitted_runs % interval
    return 0 if remainder == 0 else interval - remainder


def default_self_play_state():
    return {
        "running": False,
        "phase": "idle",
        "message": "",
        "started_at": "",
        "finished_at": "",
        "last_updated_at": "",
        "current_run_id": "",
        "current_state_type": "",
        "completed_runs": 0,
        "admitted_runs": 0,
        "pending_training_runs": 0,
        "target_runs": 0,
        "train_every_admitted_runs": 0,
        "recent_scores": [],
        "last_score": None,
        "last_reason": "",
        "last_model_id": "",
        "last_train_at": "",
        "loop_count": 0,
    }


class SelfPlayManager:
    def __init__(self, panel_base_url=PANEL_BASE_URL, game_api_url=GAME_API_URL):
        self.panel_base_url = panel_base_url.rstrip("/")
        self.game_api_url = game_api_url
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._status = {
            "running": False,
            "stop_requested": False,
            "started": None,
            "finished": None,
            "current_state": "idle",
            "current_run_id": "",
            "current_floor": 0,
            "completed_runs": 0,
            "admitted_runs": 0,
            "target_runs": 0,
            "recent_scores": [],
            "last_score": None,
            "next_training_in": 0,
            "training_running": False,
            "exploration": {},
            "config": {},
            "last_message": "",
            "last_error": "",
        }
        self._persist_status()

    def get_status(self):
        with self._lock:
            status = deepcopy(self._status)
            status["thread_alive"] = bool(self._thread and self._thread.is_alive())
            return status

    def _persist_status(self):
        payload = default_self_play_state()
        payload.update({
            "running": bool(self._status.get("running")),
            "phase": self._status.get("current_state") or "idle",
            "message": self._status.get("last_message") or self._status.get("last_error") or "",
            "started_at": self._status.get("started") or "",
            "finished_at": self._status.get("finished") or "",
            "last_updated_at": datetime.now().isoformat(timespec="seconds"),
            "current_run_id": self._status.get("current_run_id") or "",
            "current_state_type": self._status.get("current_state_type") or "",
            "completed_runs": int(self._status.get("completed_runs") or 0),
            "admitted_runs": int(self._status.get("admitted_runs") or 0),
            "pending_training_runs": int(self._status.get("next_training_in") or 0),
            "target_runs": int(self._status.get("target_runs") or 0),
            "train_every_admitted_runs": int((self._status.get("config") or {}).get("self_play_train_every_admitted_runs") or 0),
            "recent_scores": deepcopy(self._status.get("recent_scores") or []),
            "last_score": deepcopy(self._status.get("last_score")),
            "last_reason": str(
                self._status.get("last_reason")
                or ((self._status.get("last_score") or {}).get("reason") if isinstance(self._status.get("last_score"), dict) else "")
                or ""
            ),
            "last_model_id": str(self._status.get("last_model_id") or ""),
            "last_train_at": str(self._status.get("last_train_at") or ""),
            "loop_count": int(self._status.get("completed_runs") or 0) + (1 if self._status.get("current_run_id") else 0),
            "current_floor": int(self._status.get("current_floor") or 0),
            "run_started_at": self._status.get("run_started_at") or 0,
        })
        write_json(SELF_PLAY_STATE_PATH, payload)

    def start(self, overrides=None):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {"status": "busy", "message": "自训练已经在运行中。", "self_play": deepcopy(self._status)}
            config = merged_config(overrides)
            self._stop_event.clear()
            self._status = {
                "running": True,
                "stop_requested": False,
                "started": datetime.now().isoformat(timespec="seconds"),
                "finished": None,
                "current_state": "starting",
                "current_run_id": "",
                "current_floor": 0,
                "completed_runs": 0,
                "admitted_runs": 0,
                "target_runs": int(config["self_play_target_runs"]),
                "recent_scores": [],
                "last_score": None,
                "next_training_in": int(config["self_play_train_every_admitted_runs"]),
                "training_running": False,
                "exploration": {
                    "enabled": bool(config["exploration_enabled"]),
                    "mode": config["exploration_mode"],
                    "combat_epsilon": config["combat_exploration_epsilon"],
                    "macro_epsilon": config["macro_exploration_epsilon"],
                    "top_k": config["exploration_top_k"],
                    "temperature": config["exploration_temperature"],
                },
                "config": deepcopy(config),
                "last_message": "准备启动自训练循环。",
                "last_error": "",
            }
            self._persist_status()
            self._thread = threading.Thread(target=self._worker, args=(deepcopy(config),), daemon=True, name="STS2SelfPlay")
            self._thread.start()
            return {"status": "ok", "message": "自训练已启动。", "self_play": deepcopy(self._status)}

    def stop(self):
        self._stop_event.set()
        with self._lock:
            self._status["stop_requested"] = True
            self._status["last_message"] = "正在请求停止自训练。"
        return {"status": "ok", "message": "已请求停止自训练。", "self_play": self.get_status()}

    def _set_status(self, **patch):
        with self._lock:
            self._status.update(patch)
            self._persist_status()

    def _panel_get(self, path, timeout=10):
        resp = requests.get(self.panel_base_url + path, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _panel_post(self, path, body=None, timeout=15):
        resp = requests.post(self.panel_base_url + path, json=body or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _game_get(self, timeout=8):
        resp = requests.get(self.game_api_url + "?format=json", timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _game_post(self, body, timeout=15):
        resp = requests.post(self.game_api_url, json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _ensure_control(self, config):
        control_patch = {
            "ai_enabled": True,
            "macro_enabled": bool(config["macro_enabled"]),
            "macro_shop_enabled": bool(config["macro_shop_enabled"]),
            "collection_enabled": bool(config["collection_enabled"]),
            "include_ai_in_training": bool(config["include_ai_in_training"]),
            "min_training_quality": config["min_training_quality"],
            "ai_min_training_quality": config["ai_min_training_quality"],
            "game_speed_enabled": bool(config["game_speed_enabled"]),
            "game_speed_multiplier": float(config["game_speed_multiplier"]),
            "next_run_mode": "new",
            "exploration_enabled": bool(config["exploration_enabled"]),
            "exploration_mode": config["exploration_mode"],
            "combat_exploration_epsilon": float(config["combat_exploration_epsilon"]),
            "macro_exploration_epsilon": float(config["macro_exploration_epsilon"]),
            "exploration_top_k": int(config["exploration_top_k"]),
            "exploration_temperature": float(config["exploration_temperature"]),
            "self_play_character": config["self_play_character"],
            "self_play_ascension": int(config["self_play_ascension"]),
            "self_play_target_runs": int(config["self_play_target_runs"]),
            "self_play_train_every_admitted_runs": int(config["self_play_train_every_admitted_runs"]),
            "self_play_max_run_minutes": int(config["self_play_max_run_minutes"]),
            "self_play_stall_seconds": int(config["self_play_stall_seconds"]),
        }
        self._panel_post("/api/control", control_patch, timeout=20)
        if config["game_speed_enabled"]:
            self._panel_post("/api/game-speed", {"enabled": True, "multiplier": config["game_speed_multiplier"]}, timeout=10)

    def _wait_for_ai(self, timeout_sec=45):
        self._panel_post("/api/ai/start", {}, timeout=15)
        deadline = time.time() + timeout_sec
        while time.time() < deadline and not self._stop_event.is_set():
            status = self._panel_get("/api/status", timeout=15)
            ai_pid = status.get("ai_pid")
            game = status.get("game") or {}
            if ai_pid:
                self._set_status(last_message=f"AI 已托管，当前界面：{game.get('state_type') or 'unknown'}")
                return
            time.sleep(1.0)
        raise RuntimeError("AI 进程未能在预期时间内启动。")

    def _existing_run_ids(self):
        return {run.get("run_id") for run in latest_runs(limit=200) if run.get("run_id")}

    def _find_new_run(self, existing_run_ids, started_ms):
        for run in latest_runs(limit=50):
            run_id = run.get("run_id")
            if not run_id or run_id in existing_run_ids or not str(run_id).startswith("ai_"):
                continue
            first_ts = safe_int(run.get("first_ts") or run.get("last_ts"))
            if first_ts and first_ts + 5000 < started_ms:
                continue
            return run
        return None

    def _launch_run(self, config, existing_run_ids):
        started_ms = int(time.time() * 1000)
        deadline = time.time() + 90
        last_message = ""
        start_attempted = False
        transitional_since = None
        while time.time() < deadline and not self._stop_event.is_set():
            state = self._game_get(timeout=10)
            state_type = str(state.get("state_type") or "").lower()

            if state_type == "menu":
                transitional_since = None
                self._panel_post("/api/control", {"next_run_mode": "new"}, timeout=10)
                result = self._game_post({
                    "action": "start_new_run",
                    "character": config["self_play_character"],
                    "ascension": config["self_play_ascension"],
                    "seed": None,
                }, timeout=20)
                if result.get("status") == "error" or result.get("error"):
                    error_text = str(result.get("error") or result.get("message") or "")
                    if error_text == "No run in progress":
                        raise RuntimeError(
                            "游戏正在运行旧版 STS2_MCP，尚未加载 start_new_run。"
                            "请关闭游戏，安装新版 Mod DLL 后重启游戏。"
                        )
                    last_message = error_text or "start_new_run 返回错误。"
                else:
                    last_message = str(result.get("message") or "正在尝试开新局。")
                    start_attempted = True
                self._set_status(last_message=last_message, current_state="launching")
            elif start_attempted:
                # 角色已选择，游戏正在加载/过渡（overlay / card_select 等）
                # AI agent 需要处理这些界面，runner 只需等待
                if transitional_since is None:
                    transitional_since = time.time()
                elapsed = time.time() - transitional_since
                self._set_status(
                    last_message=f"等待游戏初始化：{state_type}（已等 {int(elapsed)} 秒）",
                    current_state="launching",
                    current_state_type=state_type,
                )
                # 如果过渡状态持续超过 60 秒，尝试恢复到菜单重试
                if elapsed > 60:
                    self._set_status(last_message=f"游戏在 {state_type} 状态卡住超过 60 秒，尝试恢复到菜单。")
                    self._recover_to_menu("launch_stuck_" + state_type)
                    transitional_since = None
                    start_attempted = False

            new_run = self._find_new_run(existing_run_ids, started_ms)
            if new_run:
                return new_run
            time.sleep(2.0)
        raise RuntimeError(last_message or "无法启动新的 AI run。")

    def _latest_run(self, run_id):
        for run in latest_runs(limit=80):
            if run.get("run_id") == run_id:
                return run
        return None

    def _recover_to_menu(self, reason):
        deadline = time.time() + 45
        while time.time() < deadline and not self._stop_event.is_set():
            for action in ("return_to_menu", "abandon_run"):
                try:
                    self._set_status(last_message=f"正在恢复到主菜单：{reason} ({action})", current_state="recovering")
                    self._game_post({"action": action}, timeout=20)
                    time.sleep(3.0)
                    state = self._game_get(timeout=10)
                    if str(state.get("state_type") or "").lower() == "menu":
                        return True
                except Exception:
                    continue
            time.sleep(2.0)
        return False

    def _monitor_run(self, run_id, config):
        started_at = time.time()
        last_progress_ts = time.time()
        last_seen_marker = None
        stable_menu_polls = 0
        menu_since = None
        while not self._stop_event.is_set():
            run = self._latest_run(run_id)
            state = self._game_get(timeout=10)
            state_type = str(state.get("state_type") or "").lower()
            if run:
                marker = (
                    safe_int(run.get("last_ts")),
                    safe_int(run.get("records")),
                    safe_int(run.get("max_floor")),
                    safe_int(run.get("wins")),
                    safe_int(run.get("losses")),
                    safe_int(run.get("invalid_actions")),
                )
                if marker != last_seen_marker:
                    last_seen_marker = marker
                    last_progress_ts = time.time()
                self._set_status(
                    current_state="running",
                    current_run_id=run_id,
                    current_state_type=state_type,
                    current_floor=safe_int(run.get("max_floor")),
                    last_message=f"运行中：floor {safe_int(run.get('max_floor'))} / act {safe_int(run.get('max_act'))}",
                )

            if state_type == "menu":
                # Fast path: data already flushed
                if run and safe_int(run.get("records")) > 0:
                    stable_menu_polls += 1
                    if stable_menu_polls >= 2:
                        return run
                # Slow path: menu持续超过10秒，即使数据还没flush也认为局已结束
                if menu_since is None:
                    menu_since = time.time()
                elif time.time() - menu_since > 10:
                    self._set_status(last_message=f"菜单持续超过10秒，认定局 {run_id} 已结束。")
                    return run or self._latest_run(run_id)
            else:
                stable_menu_polls = 0
                menu_since = None

            if time.time() - started_at > config["self_play_max_run_minutes"] * 60:
                self._recover_to_menu("run_timeout")
                return self._latest_run(run_id)
            if time.time() - last_progress_ts > config["self_play_stall_seconds"]:
                self._recover_to_menu("stalled")
                return self._latest_run(run_id)
            time.sleep(3.0)
        return self._latest_run(run_id)

    def _wait_for_training(self, triggered_at, timeout_sec=3600):
        deadline = time.time() + timeout_sec
        saw_current_training = False
        while time.time() < deadline and not self._stop_event.is_set():
            status = self._panel_get("/api/status", timeout=20)
            training = status.get("training") or {}
            started = str(training.get("started") or "")
            finished = str(training.get("finished") or "")
            if started >= triggered_at or finished >= triggered_at:
                saw_current_training = True
            self._set_status(training_running=bool(training.get("running")) or not saw_current_training)
            if saw_current_training and not training.get("running") and finished >= triggered_at:
                return training
            time.sleep(4.0)
        raise RuntimeError("训练等待超时。")

    def _trigger_training(self):
        self._set_status(current_state="training", training_running=True, last_message="达到入训批次，开始自动训练。")
        triggered_at = datetime.now().isoformat(timespec="seconds")
        response = self._panel_post("/api/train", {}, timeout=20)
        if response.get("status") not in {"ok", "busy"}:
            raise RuntimeError(response.get("message") or response.get("error") or "训练启动失败。")
        training = self._wait_for_training(triggered_at)
        output = str(training.get("output") or "")
        if "ERROR:" in output:
            raise RuntimeError("训练过程中出现错误。")
        self._panel_post("/api/ai/restart", {}, timeout=20)
        status = self._panel_get("/api/status", timeout=20)
        registry = ((status.get("models") or {}).get("registry") or {})
        self._set_status(
            training_running=False,
            last_message="训练完成，AI 已重启加载新模型。",
            last_model_id=str(registry.get("active_model_id") or ""),
            last_train_at=str(training.get("finished") or datetime.now().isoformat(timespec="seconds")),
        )

    def _worker(self, config):
        try:
            self._ensure_control(config)
            self._wait_for_ai()
            existing_run_ids = self._existing_run_ids()
            admitted_runs = 0
            completed_runs = 0
            recent_scores = []
            target = int(config["self_play_target_runs"])
            while (target <= 0 or completed_runs < target) and not self._stop_event.is_set():
                new_run = self._launch_run(config, existing_run_ids)
                run_id = str(new_run.get("run_id"))
                existing_run_ids.add(run_id)
                run_started_at = time.time()
                self._set_status(current_run_id=run_id, current_floor=0, current_state="launching", run_started_at=run_started_at)
                run_summary = self._monitor_run(run_id, config)
                # 等待数据 flush：JSONL 可能还没写完，重试几次避免误判
                for _retry in range(3):
                    if run_summary and safe_int(run_summary.get("records")) > 0:
                        break
                    time.sleep(3)
                    run_summary = self._latest_run(run_id) or run_summary
                if not run_summary:
                    raise RuntimeError(f"无法读取 run 结果：{run_id}")
                score = save_self_play_run_score(run_summary)
                completed_runs += 1
                admitted_runs += 1 if score.get("admitted") else 0
                recent_scores = ([score] + recent_scores)[:8]
                self._set_status(
                    completed_runs=completed_runs,
                    admitted_runs=admitted_runs,
                    last_score=score,
                    recent_scores=recent_scores,
                    next_training_in=next_training_in(admitted_runs, config["self_play_train_every_admitted_runs"]),
                    last_message=f"Run {run_id} 结束，score={score.get('score')}，admitted={score.get('admitted')}",
                )
                if score.get("admitted") and admitted_runs % config["self_play_train_every_admitted_runs"] == 0:
                    self._trigger_training()
            self._set_status(
                running=False,
                training_running=False,
                finished=datetime.now().isoformat(timespec="seconds"),
                current_state="finished",
                last_message="自训练循环已完成。",
            )
        except Exception as exc:
            self._set_status(
                running=False,
                training_running=False,
                finished=datetime.now().isoformat(timespec="seconds"),
                current_state="error",
                last_error=str(exc),
                last_message=str(exc),
            )
        finally:
            self._set_status(stop_requested=False)
            state = read_json(SERVER_STATE_PATH, {})
            current_pid = state.get("self_play_pid")
            if current_pid == os.getpid():
                state.pop("self_play_pid", None)
                write_json(SERVER_STATE_PATH, state)


def main():
    parser = argparse.ArgumentParser(description="Run STS2 self-play loop through the local control panel.")
    parser.add_argument("--target-runs", type=int, default=None)
    parser.add_argument("--train-every", type=int, default=None)
    parser.add_argument("--max-run-minutes", type=int, default=None)
    parser.add_argument("--stall-seconds", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.target_runs is not None:
        overrides["self_play_target_runs"] = args.target_runs
    if args.train_every is not None:
        overrides["self_play_train_every_admitted_runs"] = args.train_every
    if args.max_run_minutes is not None:
        overrides["self_play_max_run_minutes"] = args.max_run_minutes
    if args.stall_seconds is not None:
        overrides["self_play_stall_seconds"] = args.stall_seconds

    manager = SelfPlayManager()
    result = manager.start(overrides or None)
    if result.get("status") != "ok":
        raise SystemExit(result.get("message") or result.get("error") or "self-play start failed")
    try:
        while True:
            status = manager.get_status()
            print(json.dumps(status, ensure_ascii=False))
            if not status.get("running"):
                break
            time.sleep(10)
    except KeyboardInterrupt:
        manager.stop()
        raise SystemExit(130)


if __name__ == "__main__":
    main()
