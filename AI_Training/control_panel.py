import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from evaluation_summary import evaluation_summary
from run_summary import (
    QUALITY_ORDER,
    current_data_summary,
    latest_runs,
    recent_records,
    set_run_discarded,
    set_run_label,
)

WORKSPACE = Path(__file__).resolve().parents[1]
AI_DIR = WORKSPACE / "AI_Training"
DATA_DIR = WORKSPACE / "RL_Datasets"
EXPORT_DIR = WORKSPACE / "Data_Packages"
ASSETS_DIR = AI_DIR / "assets"
CONTROL_PATH = AI_DIR / "control_state.json"
AI_LOGIC_PATH = AI_DIR / "ai_logic_state.json"
LLM_CONFIG_PATH = AI_DIR / "model_config.json"
LLM_LOGIC_PATH = AI_DIR / "llm_logic_state.json"
DISCARDED_PATH = DATA_DIR / "discarded_runs.json"
SERVER_STATE_PATH = AI_DIR / "control_panel_state.json"
DEFAULT_PYTHON_EXE = WORKSPACE / ".venv" / "Scripts" / "python.exe"
AGENT_PATH = AI_DIR / "ai_agent.py"
LLM_AGENT_PATH = AI_DIR / "llm_agent.py"
API_URL = "http://localhost:15526/api/v1/singleplayer"


def python_exe_runs(candidate):
    if not candidate:
        return False
    try:
        proc = subprocess.run([str(candidate), "--version"], capture_output=True, text=True, timeout=5)
        return proc.returncode == 0 and "Python" in ((proc.stdout or "") + (proc.stderr or ""))
    except Exception:
        return False


def resolve_python_exe():
    candidates = [
        os.environ.get("STS2_AI_PYTHON"),
        str(DEFAULT_PYTHON_EXE),
        str(Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "python" / "python.exe"),
        sys.executable,
        shutil.which("python"),
        shutil.which("py"),
    ]
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if python_exe_runs(candidate):
            return Path(candidate)
    return Path(sys.executable)


PYTHON_EXE = resolve_python_exe()

DEFAULT_CONTROL = {
    "ai_enabled": False,
    "macro_enabled": False,
    "macro_shop_enabled": False,
    "record_ai_actions": True,
    "include_ai_in_training": False,
    "next_run_mode": "auto",
    "collection_enabled": True,
    "collection_disabled_since": None,
    "collection_disabled_ranges": [],
    "min_training_quality": "unknown",
}
DEFAULT_LLM_CONFIG = {
    "enabled": False,
    "mode": "advisor",
    "provider": "openai_compatible",
    "base_url": "https://api.openai.com/v1",
    "api_key": "",
    "model": "",
    "temperature": 0.2,
    "max_tokens": 700,
    "decision_interval_sec": 3.0,
    "execute_combat": False,
    "confirm_shop": True,
    "max_actions_per_turn": 12,
    "action_selection_mode": "candidate_id",
    "profiles": [],
    "active_profile_id": "",
}
TRAIN_LOCK = threading.Lock()
LLM_TEST_LOCK = threading.Lock()
LLM_TEST_COOLDOWN_SEC = 60
LAST_TRAIN = {"running": False, "started": None, "finished": None, "output": ""}
LAST_EXPORT = {"path": None, "filename": None, "created": None, "size": 0, "file_count": 0}
LLM_TEST_STATE = {"running": False, "last_started": 0.0, "last_finished": 0.0}
GAME_CACHE = {"state": None, "ts": 0.0}
PYTHON_RUNTIME_CACHE = {"ts": 0.0, "data": None}
REQUIRED_AGENT_MODULES = ["requests", "torch", "numpy", "colorama"]
REQUIRED_TRAINING_MODULES = ["numpy", "torch"]


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
    data = DEFAULT_CONTROL.copy()
    data.update(read_json(CONTROL_PATH, {}))
    return data


def update_control(patch):
    data = read_control()
    for key in DEFAULT_CONTROL:
        if key in patch:
            if key == "next_run_mode":
                data[key] = patch[key] if patch[key] in ("auto", "new", "continue") else "auto"
            elif key == "min_training_quality":
                data[key] = patch[key] if patch[key] in QUALITY_ORDER else "unknown"
            elif key == "collection_enabled":
                enabled = bool(patch[key])
                was_enabled = bool(data.get("collection_enabled", True))
                now = int(time.time() * 1000)
                if not enabled and was_enabled:
                    data["collection_disabled_since"] = now
                elif enabled and not was_enabled:
                    start = data.get("collection_disabled_since")
                    if start:
                        ranges = list(data.get("collection_disabled_ranges", []))
                        ranges.append([int(start), now])
                        data["collection_disabled_ranges"] = ranges[-100:]
                    data["collection_disabled_since"] = None
                data[key] = enabled
            else:
                data[key] = bool(patch[key])
    write_json(CONTROL_PATH, data)
    return data


def read_llm_config(mask_key=True):
    data = DEFAULT_LLM_CONFIG.copy()
    data.update(read_json(LLM_CONFIG_PATH, {}))
    if data.get("mode") not in ("advisor", "combat_auto"):
        data["mode"] = "advisor"
    if data.get("action_selection_mode") not in ("catalog_args", "candidate_id"):
        data["action_selection_mode"] = "candidate_id"
    data["provider"] = "openai_compatible"
    profiles = data.get("profiles") if isinstance(data.get("profiles"), list) else []
    data["profiles"] = [_public_llm_profile(p) for p in profiles]
    if mask_key and data.get("api_key"):
        data["api_key"] = "********"
        data["has_api_key"] = True
    else:
        data["has_api_key"] = bool(data.get("api_key"))
    return data


def _api_key_tail(api_key):
    key = str(api_key or "")
    return key[-4:] if key else ""


def _public_llm_profile(profile):
    api_key = profile.get("api_key", "")
    return {
        "id": profile.get("id", ""),
        "name": profile.get("name") or profile.get("model") or profile.get("base_url") or "Unnamed",
        "provider": profile.get("provider", "openai_compatible"),
        "base_url": profile.get("base_url", ""),
        "model": profile.get("model", ""),
        "has_api_key": bool(api_key),
        "key_tail": _api_key_tail(api_key),
        "updated_at": profile.get("updated_at", ""),
    }


def _upsert_current_llm_profile(data, profile_name=""):
    if not data.get("api_key"):
        return data
    profiles = data.get("profiles")
    if not isinstance(profiles, list):
        profiles = []

    active_id = data.get("active_profile_id") or ""
    profile = None
    for item in profiles:
        if item.get("id") == active_id:
            profile = item
            break
    if profile is None:
        profile = {"id": uuid.uuid4().hex[:10], "created_at": datetime.now().isoformat(timespec="seconds")}
        profiles.append(profile)

    profile.update({
        "name": profile_name or data.get("model") or data.get("base_url") or "OpenAI Compatible",
        "provider": "openai_compatible",
        "base_url": data.get("base_url", ""),
        "api_key": data.get("api_key", ""),
        "model": data.get("model", ""),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    data["profiles"] = profiles
    data["active_profile_id"] = profile["id"]
    return data


def update_llm_config(patch):
    old = read_json(LLM_CONFIG_PATH, {})
    data = DEFAULT_LLM_CONFIG.copy()
    data.update(old)
    save_profile = bool(patch.get("save_profile"))

    for key in DEFAULT_LLM_CONFIG:
        if key not in patch:
            continue
        if key in ("profiles", "active_profile_id"):
            continue
        if key == "api_key" and patch[key] == "********":
            continue
        if key in ("enabled", "execute_combat", "confirm_shop"):
            data[key] = bool(patch[key])
        elif key == "mode":
            data[key] = patch[key] if patch[key] in ("advisor", "combat_auto") else "advisor"
        elif key == "action_selection_mode":
            data[key] = patch[key] if patch[key] in ("catalog_args", "candidate_id") else "candidate_id"
        elif key in ("temperature", "decision_interval_sec"):
            try:
                data[key] = float(patch[key])
            except (TypeError, ValueError):
                pass
        elif key in ("max_tokens", "max_actions_per_turn"):
            try:
                data[key] = int(patch[key])
            except (TypeError, ValueError):
                pass
        else:
            data[key] = str(patch[key] or "").strip()

    if save_profile or ("api_key" in patch and data.get("api_key")):
        data = _upsert_current_llm_profile(data, str(patch.get("profile_name") or ""))

    write_json(LLM_CONFIG_PATH, data)
    return read_llm_config(mask_key=True)


def use_llm_profile(profile_id):
    data = DEFAULT_LLM_CONFIG.copy()
    data.update(read_json(LLM_CONFIG_PATH, {}))
    profiles = data.get("profiles") if isinstance(data.get("profiles"), list) else []
    for profile in profiles:
        if profile.get("id") == profile_id:
            data.update({
                "provider": "openai_compatible",
                "base_url": profile.get("base_url", ""),
                "api_key": profile.get("api_key", ""),
                "model": profile.get("model", ""),
                "active_profile_id": profile_id,
            })
            write_json(LLM_CONFIG_PATH, data)
            return read_llm_config(mask_key=True)
    raise ValueError("profile not found")


def delete_llm_profile(profile_id):
    data = DEFAULT_LLM_CONFIG.copy()
    data.update(read_json(LLM_CONFIG_PATH, {}))
    profiles = data.get("profiles") if isinstance(data.get("profiles"), list) else []
    data["profiles"] = [p for p in profiles if p.get("id") != profile_id]
    if data.get("active_profile_id") == profile_id:
        data["active_profile_id"] = ""
    write_json(LLM_CONFIG_PATH, data)
    return read_llm_config(mask_key=True)


def update_llm_profile(patch):
    profile_id = str(patch.get("profile_id") or "")
    if not profile_id:
        raise ValueError("profile_id is required")

    data = DEFAULT_LLM_CONFIG.copy()
    data.update(read_json(LLM_CONFIG_PATH, {}))
    profiles = data.get("profiles") if isinstance(data.get("profiles"), list) else []
    profile = None
    for item in profiles:
        if item.get("id") == profile_id:
            profile = item
            break
    if profile is None:
        raise ValueError("profile not found")

    for key in ("name", "base_url", "model"):
        if key in patch:
            profile[key] = str(patch.get(key) or "").strip()
    api_key = str(patch.get("api_key") or "").strip()
    if api_key and api_key != "********":
        profile["api_key"] = api_key
    profile["provider"] = "openai_compatible"
    profile["updated_at"] = datetime.now().isoformat(timespec="seconds")

    if data.get("active_profile_id") == profile_id:
        data.update({
            "provider": "openai_compatible",
            "base_url": profile.get("base_url", ""),
            "api_key": profile.get("api_key", ""),
            "model": profile.get("model", ""),
        })

    data["profiles"] = profiles
    write_json(LLM_CONFIG_PATH, data)
    return read_llm_config(mask_key=True)


def _llm_chat_url(base_url):
    base = str(base_url or "").rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def _chat_response_content(result):
    choices = result.get("choices") if isinstance(result, dict) else None
    if not choices:
        return None
    choice = choices[0] if choices else {}
    message = choice.get("message") if isinstance(choice, dict) else None
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(part))
            return "\n".join(p for p in parts if p)
        if content is not None:
            return str(content)
    if isinstance(choice, dict) and choice.get("text") is not None:
        return str(choice.get("text"))
    return None


def test_llm_connection():
    data = DEFAULT_LLM_CONFIG.copy()
    data.update(read_json(LLM_CONFIG_PATH, {}))
    if not data.get("api_key"):
        return {"status": "error", "message": "Missing API key."}
    if not data.get("model"):
        return {"status": "error", "message": "Missing model name."}

    now = time.time()
    with LLM_TEST_LOCK:
        if LLM_TEST_STATE.get("running"):
            return {
                "status": "rejected",
                "message": "连接测试正在进行中，本次点击已拒绝，未请求模型。",
            }
        wait = LLM_TEST_COOLDOWN_SEC - (now - float(LLM_TEST_STATE.get("last_started") or 0))
        if wait > 0:
            return {
                "status": "rejected",
                "message": f"连接测试冷却中，还需 {int(wait) + 1} 秒。本次点击已拒绝，未请求模型。",
                "retry_after_sec": int(wait) + 1,
            }
        LLM_TEST_STATE["running"] = True
        LLM_TEST_STATE["last_started"] = now

    body = {
        "model": data["model"],
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": "{\"ok\": true}"},
        ],
        "temperature": 0,
        "max_tokens": 32,
    }
    req = Request(
        _llm_chat_url(data.get("base_url")),
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {data['api_key']}",
        },
    )
    try:
        with urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        result = json.loads(raw)
        content = _chat_response_content(result)
        if content is None:
            preview = json.dumps(result, ensure_ascii=False)[:500]
            return {"status": "error", "message": f"API reached, but response has no chat content: {preview}"}
        return {"status": "ok", "message": f"Model connection OK. Response: {content[:120]}"}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {"status": "error", "message": f"HTTP {exc.code}: {raw[:500]}"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
    finally:
        with LLM_TEST_LOCK:
            LLM_TEST_STATE["running"] = False
            LLM_TEST_STATE["last_finished"] = time.time()


def ensure_llm_profiles_initialized():
    data = DEFAULT_LLM_CONFIG.copy()
    data.update(read_json(LLM_CONFIG_PATH, {}))
    profiles = data.get("profiles") if isinstance(data.get("profiles"), list) else []
    if data.get("api_key") and not profiles:
        write_json(LLM_CONFIG_PATH, _upsert_current_llm_profile(data))


def python_runtime_status(force=False):
    now = time.time()
    cached = PYTHON_RUNTIME_CACHE.get("data")
    if cached and not force and now - float(PYTHON_RUNTIME_CACHE.get("ts") or 0) < 60:
        return cached

    modules = sorted(set(REQUIRED_AGENT_MODULES + REQUIRED_TRAINING_MODULES))
    code = (
        "import importlib, json, sys;"
        f"mods={modules!r};"
        "missing=[];"
        "\nfor m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception:\n"
        "        missing.append(m)\n"
        "print(json.dumps({'executable': sys.executable, 'version': sys.version.split()[0], 'missing': missing}))"
    )
    status = {
        "executable": str(PYTHON_EXE),
        "version": "",
        "ok": False,
        "missing": modules,
        "agent_ready": False,
        "training_ready": False,
        "message": "",
    }
    try:
        proc = subprocess.run(
            [str(PYTHON_EXE), "-c", code],
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            timeout=12,
        )
        if proc.returncode == 0:
            payload = json.loads((proc.stdout or "{}").strip().splitlines()[-1])
            missing = payload.get("missing") or []
            status.update({
                "executable": payload.get("executable") or str(PYTHON_EXE),
                "version": payload.get("version") or "",
                "ok": True,
                "missing": missing,
                "agent_ready": not any(m in missing for m in REQUIRED_AGENT_MODULES),
                "training_ready": not any(m in missing for m in REQUIRED_TRAINING_MODULES),
                "message": "ok" if not missing else "缺少模块：" + ", ".join(missing),
            })
        else:
            detail = (proc.stderr or proc.stdout or "").strip()
            status["message"] = detail[-500:] if detail else f"Python exited {proc.returncode}"
    except Exception as exc:
        status["message"] = str(exc)

    PYTHON_RUNTIME_CACHE.update({"ts": now, "data": status})
    return status


def post_game_action(payload):
    req = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=2) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_game_state():
    try:
        with urlopen(API_URL + "?format=json", timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def get_game_state_for_dashboard(control):
    cached = GAME_CACHE.get("state")
    cached_type = str((cached or {}).get("state_type") or "").lower()
    shop_guard = cached_type in ("shop", "fake_merchant") and not control.get("macro_shop_enabled", False)
    if shop_guard and time.time() - float(GAME_CACHE.get("ts") or 0) < 12:
        state = dict(cached)
        state["shop_poll_guard"] = True
        return state
    state = get_game_state()
    if "error" not in state:
        GAME_CACHE["state"] = state
        GAME_CACHE["ts"] = time.time()
    return state


def pid_is_running(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def get_ai_pid():
    state = read_json(SERVER_STATE_PATH, {})
    pid = state.get("ai_pid")
    return int(pid) if pid and pid_is_running(pid) else None


def get_llm_pid():
    state = read_json(SERVER_STATE_PATH, {})
    pid = state.get("llm_pid")
    return int(pid) if pid and pid_is_running(pid) else None


def start_ai():
    pid = get_ai_pid()
    if pid:
        return {"status": "ok", "message": f"AI already running, pid={pid}", "pid": pid}
    runtime = python_runtime_status()
    if not runtime.get("agent_ready"):
        missing = [m for m in REQUIRED_AGENT_MODULES if m in (runtime.get("missing") or [])] or REQUIRED_AGENT_MODULES
        return {
            "status": "error",
            "message": "AI 未启动：Python 环境缺少 " + ", ".join(missing),
            "python_runtime": runtime,
        }
    proc = subprocess.Popen(
        [str(PYTHON_EXE), str(AGENT_PATH)],
        cwd=str(WORKSPACE),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    state = read_json(SERVER_STATE_PATH, {})
    state["ai_pid"] = proc.pid
    state["ai_started_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(SERVER_STATE_PATH, state)
    update_control({"ai_enabled": True})
    return {"status": "ok", "message": f"AI started, pid={proc.pid}", "pid": proc.pid}


def stop_ai():
    pid = get_ai_pid()
    update_control({"ai_enabled": False})
    if not pid:
        return {"status": "ok", "message": "AI disabled. No managed AI process is running."}
    try:
        os.kill(pid, signal.CTRL_BREAK_EVENT)
        time.sleep(0.5)
    except Exception:
        pass
    if pid_is_running(pid):
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, timeout=5)
        except Exception:
            pass
    state = read_json(SERVER_STATE_PATH, {})
    state.pop("ai_pid", None)
    write_json(SERVER_STATE_PATH, state)
    return {"status": "ok", "message": "AI stopped/disabled."}


def start_llm():
    pid = get_llm_pid()
    if pid:
        return {"status": "ok", "message": f"LLM already running, pid={pid}", "pid": pid}
    proc = subprocess.Popen(
        [str(PYTHON_EXE), str(LLM_AGENT_PATH)],
        cwd=str(WORKSPACE),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    state = read_json(SERVER_STATE_PATH, {})
    state["llm_pid"] = proc.pid
    state["llm_started_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(SERVER_STATE_PATH, state)
    update_llm_config({"enabled": True})
    return {"status": "ok", "message": f"LLM agent started, pid={proc.pid}", "pid": proc.pid}


def stop_llm():
    pid = get_llm_pid()
    update_llm_config({"enabled": False})
    if not pid:
        return {"status": "ok", "message": "LLM disabled. No managed LLM process is running."}
    try:
        os.kill(pid, signal.CTRL_BREAK_EVENT)
        time.sleep(0.5)
    except Exception:
        pass
    if pid_is_running(pid):
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, timeout=5)
        except Exception:
            pass
    state = read_json(SERVER_STATE_PATH, {})
    state.pop("llm_pid", None)
    write_json(SERVER_STATE_PATH, state)
    return {"status": "ok", "message": "LLM agent stopped/disabled."}


def llm_logic_snapshot():
    data = read_json(LLM_LOGIC_PATH, {})
    ts = int(data.get("timestamp") or 0)
    if ts:
        data["time"] = datetime.fromtimestamp(ts / 1000).strftime("%H:%M:%S")
    compact = data.get("compact_state")
    if isinstance(compact, dict):
        player = compact.get("player", {})
        hand = player.get("hand", [])
        data["hand_summary"] = [c.get("id") for c in hand]
        data.setdefault("combat_eval", compact.get("combat_eval"))
        data.setdefault("pile_summary", player.get("pile_summary"))
        combat_eval = compact.get("combat_eval") if isinstance(compact.get("combat_eval"), dict) else {}
        data.setdefault("potion_opportunities", combat_eval.get("potion_opportunities", []))
        data.pop("compact_state", None)
    return data


def ai_logic_snapshot():
    data = read_json(AI_LOGIC_PATH, {})
    ts = int(data.get("timestamp") or 0)
    if ts:
        data["time"] = datetime.fromtimestamp(ts / 1000).strftime("%H:%M:%S")
    return data


def run_training_background():
    runtime = python_runtime_status()
    if not runtime.get("training_ready"):
        missing = [m for m in REQUIRED_TRAINING_MODULES if m in (runtime.get("missing") or [])] or REQUIRED_TRAINING_MODULES
        message = "训练未启动：Python 环境缺少 " + ", ".join(missing)
        LAST_TRAIN.update({
            "running": False,
            "started": None,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "output": message,
        })
        return {"status": "error", "message": message, "python_runtime": runtime}

    def worker():
        with TRAIN_LOCK:
            LAST_TRAIN.update({"running": True, "started": datetime.now().isoformat(timespec="seconds"), "finished": None, "output": ""})
            output = []
            try:
                for cmd in [
                    [str(PYTHON_EXE), str(AI_DIR / "monster_profile_builder.py")],
                    [str(PYTHON_EXE), str(AI_DIR / "data_pipeline.py")],
                    [str(PYTHON_EXE), str(AI_DIR / "train_bc.py")],
                    [str(PYTHON_EXE), str(AI_DIR / "train_candidate_bc.py")],
                    [str(PYTHON_EXE), str(AI_DIR / "macro_data_pipeline.py")],
                    [str(PYTHON_EXE), str(AI_DIR / "train_macro_bc.py")],
                ]:
                    proc = subprocess.run(cmd, cwd=str(WORKSPACE), capture_output=True, text=True, timeout=600)
                    output.append("> " + " ".join(cmd))
                    output.append(proc.stdout)
                    if proc.stderr:
                        output.append(proc.stderr)
                    if proc.returncode != 0:
                        break
            except Exception as exc:
                output.append(f"ERROR: {exc}")
            LAST_TRAIN.update({"running": False, "finished": datetime.now().isoformat(timespec="seconds"), "output": "\n".join(output)[-12000:]})

    if LAST_TRAIN.get("running"):
        return {"status": "busy", "message": "Training is already running."}
    threading.Thread(target=worker, daemon=True).start()
    return {"status": "ok", "message": "Training started."}


def export_database_package():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sts2_dataset_package_{stamp}.zip"
    zip_path = EXPORT_DIR / filename

    include_roots = ["Human", "AI", "Combat", "Macro", "Monster"]
    include_files = [
        "discarded_runs.json",
        "run_labels.json",
        "rl_monitor.log",
        "scan_report.txt",
        "README_DATA_FORMAT.md",
    ]
    top_patterns = ["action_logs_*.jsonl"]
    file_count = 0

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "workspace": str(WORKSPACE),
        "control": read_control(),
        "runs": latest_runs(limit=100000),
        "notes": [
            "Raw dataset package for STS2 AI training.",
            "Processed training arrays and model weights are intentionally excluded; they can be regenerated.",
            "Manual run labels are stored in RL_Datasets/run_labels.json.",
        ],
    }

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root_name in include_roots:
            root = DATA_DIR / root_name
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                zf.write(path, path.relative_to(WORKSPACE))
                file_count += 1

        for name in include_files:
            path = DATA_DIR / name
            if path.exists() and path.is_file():
                zf.write(path, path.relative_to(WORKSPACE))
                file_count += 1

        for pattern in top_patterns:
            for path in DATA_DIR.glob(pattern):
                if path.is_file():
                    zf.write(path, path.relative_to(WORKSPACE))
                    file_count += 1

        zf.writestr("PACKAGE_SUMMARY.json", json.dumps(summary, ensure_ascii=False, indent=2))
        zf.writestr(
            "README_PACKAGE.txt",
            "STS2 AI dataset package\n"
            "Send this zip to the trainer/collector owner.\n"
            "It contains raw jsonl logs, run labels, discard list, and a package summary.\n"
            "Processed arrays and model weights are not included by default.\n",
        )
        file_count += 2

    LAST_EXPORT.update({
        "path": str(zip_path),
        "filename": filename,
        "created": datetime.now().isoformat(timespec="seconds"),
        "size": zip_path.stat().st_size,
        "file_count": file_count,
    })
    return {
        "status": "ok",
        "export": LAST_EXPORT,
        "download_url": f"/exports/{filename}",
    }


def file_status(path):
    exists = path.exists()
    data = {
        "exists": exists,
        "path": str(path),
        "name": path.name,
        "size": path.stat().st_size if exists and path.is_file() else 0,
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if exists else "",
    }
    return data


def models_status():
    combat_dir = AI_DIR / "ProcessedParams"
    macro_dir = AI_DIR / "ProcessedMacroParams"
    combat_model = combat_dir / "bc_model_best.pth"
    combat_vocab = combat_dir / "vocab.json"
    combat_metadata = read_json(combat_dir / "metadata.json", {})
    macro_model = macro_dir / "macro_bc_model_best.pth"
    macro_vocab = macro_dir / "vocab.json"
    macro_summary = read_json(macro_dir / "training_summary.json", {})
    macro_metadata = read_json(macro_dir / "metadata.json", {})
    return {
        "combat": {
            "ready": combat_model.exists() and combat_vocab.exists(),
            "model": file_status(combat_model),
            "vocab": file_status(combat_vocab),
            "metadata": combat_metadata,
        },
        "macro": {
            "ready": macro_model.exists() and macro_vocab.exists(),
            "model": file_status(macro_model),
            "vocab": file_status(macro_vocab),
            "summary": macro_summary,
            "metadata": {
                "samples": macro_metadata.get("samples", 0),
                "features": macro_metadata.get("features", 0),
                "classes": len(macro_metadata.get("classes", {})) if isinstance(macro_metadata.get("classes"), dict) else 0,
            },
        },
    }


def monster_status():
    monster_dir = DATA_DIR / "Monster"
    summary = read_json(monster_dir / "monster_build_summary.json", {})
    profiles = read_json(monster_dir / "monster_profiles.json", {})
    monsters = profiles.get("monsters") if isinstance(profiles.get("monsters"), dict) else {}
    top_monsters = []
    for key, profile in sorted(
        monsters.items(),
        key=lambda item: int((item[1] or {}).get("turn_observations") or 0),
        reverse=True,
    )[:8]:
        top_monsters.append({
            "key": key,
            "name": (profile or {}).get("display_name") or key,
            "turn_observations": (profile or {}).get("turn_observations", 0),
            "encounters_seen": (profile or {}).get("encounters_seen", 0),
            "avg_hp_lost": (profile or {}).get("avg_hp_lost"),
            "max_incoming_damage": (profile or {}).get("max_incoming_damage", 0),
        })
    return {
        "ready": bool(monsters),
        "dir": file_status(monster_dir),
        "summary": summary,
        "profiles": file_status(monster_dir / "monster_profiles.json"),
        "encounters": file_status(monster_dir / "encounter_profiles.json"),
        "vocab": file_status(DATA_DIR / "Processed" / "monster_vocab.json"),
        "top_monsters": top_monsters,
    }


def script_newer_than_start(script_path, started_at):
    if not started_at or not script_path.exists():
        return False
    try:
        started_ts = datetime.fromisoformat(started_at).timestamp()
    except Exception:
        return False
    return script_path.stat().st_mtime > started_ts + 1


def ai_process_status():
    state = read_json(SERVER_STATE_PATH, {})
    pid = get_ai_pid()
    started_at = state.get("ai_started_at", "") if pid else ""
    return {
        "pid": pid,
        "started_at": started_at,
        "needs_restart": bool(pid and script_newer_than_start(AGENT_PATH, started_at)),
        "script_mtime": datetime.fromtimestamp(AGENT_PATH.stat().st_mtime).isoformat(timespec="seconds") if AGENT_PATH.exists() else "",
    }


def restart_ai():
    stop_ai()
    return start_ai()


def status_payload():
    control = read_control()
    game = get_game_state_for_dashboard(control)
    ai_process = ai_process_status()
    return {
        "control": control,
        "ai_pid": ai_process.get("pid"),
        "ai_process": ai_process,
        "models": models_status(),
        "monster_profiles": monster_status(),
        "game": {
            "online": "error" not in game,
            "error": game.get("error"),
            "state_type": game.get("state_type"),
            "character": game.get("player", {}).get("character"),
            "hp": game.get("player", {}).get("hp"),
            "max_hp": game.get("player", {}).get("max_hp"),
            "energy": game.get("player", {}).get("energy"),
            "shop_poll_guard": game.get("shop_poll_guard", False),
        },
        "runs": latest_runs(),
        "current_data": current_data_summary(),
        "recent_records": recent_records(),
        "evaluation": evaluation_summary(limit=50),
        "python_runtime": python_runtime_status(),
        "ai_logic": ai_logic_snapshot(),
        "llm": {
            "config": read_llm_config(mask_key=True),
            "pid": get_llm_pid(),
            "logic": llm_logic_snapshot(),
        },
        "training": LAST_TRAIN,
        "export": LAST_EXPORT,
    }


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STS2 AI 控制台</title>
  <link rel="icon" type="image/png" href="/assets/sts2_ai_logo.png">
  <link rel="shortcut icon" type="image/png" href="/assets/sts2_ai_logo.png">
  <link rel="apple-touch-icon" href="/assets/sts2_ai_logo.png">
  <style>
    :root {
      color-scheme: light;
      --ink:#172126;
      --muted:#617079;
      --soft:#8b9aa3;
      --bg:#f4f7f2;
      --surface:#fffefa;
      --surface-soft:#f7faf6;
      --surface-tint:#edf5f2;
      --line:#d9e1dc;
      --line-strong:#a7b6af;
      --primary:#2f6f78;
      --primary-strong:#245760;
      --primary-bg:#e5f2f3;
      --good:#248360;
      --good-bg:#e8f6ef;
      --warn:#a76815;
      --warn-bg:#fff3dd;
      --bad:#b6463e;
      --bad-bg:#fff0ef;
      --blue:#34699a;
      --blue-bg:#e9f2fb;
      --accent:#d1a23a;
      --accent-bg:#fbf4df;
      --shadow:0 18px 46px rgba(38,55,58,.08);
      --shadow-soft:0 8px 24px rgba(38,55,58,.06);
    }
    * { box-sizing:border-box; }
    body {
      margin:0;
      position:relative;
      font-family:"Segoe UI", system-ui, sans-serif;
      color:var(--ink);
      background:
        radial-gradient(circle at 12% 8%, rgba(209,162,58,.12), transparent 26%),
        radial-gradient(circle at 90% 3%, rgba(47,111,120,.10), transparent 22%),
        linear-gradient(90deg, rgba(23,33,38,.035) 1px, transparent 1px) 0 0 / 44px 44px,
        linear-gradient(0deg, rgba(23,33,38,.028) 1px, transparent 1px) 0 0 / 44px 44px,
        var(--bg);
      font-size:14px;
      line-height:1.45;
      overflow-x:hidden;
    }
    body::before {
      content:"";
      position:fixed;
      right:-72px;
      top:150px;
      width:280px;
      height:280px;
      background:url("/assets/sts2_ai_logo.png") center / contain no-repeat;
      opacity:.035;
      transform:rotate(-10deg);
      pointer-events:none;
      z-index:0;
    }
    body::after {
      content:"";
      position:fixed;
      left:28px;
      bottom:32px;
      width:120px;
      height:92px;
      background:
        linear-gradient(135deg, transparent 0 37%, rgba(47,111,120,.13) 38% 40%, transparent 41%),
        linear-gradient(90deg, rgba(47,111,120,.11) 0 28px, transparent 28px 44px, rgba(209,162,58,.14) 44px 72px, transparent 72px);
      clip-path:polygon(0 30%, 34% 0, 72% 10%, 100% 46%, 78% 100%, 28% 88%);
      opacity:.55;
      pointer-events:none;
      z-index:0;
    }
    header {
      position:relative;
      z-index:1;
      color:var(--ink);
      background:linear-gradient(135deg, rgba(255,254,250,.98), rgba(239,247,244,.96));
      border-bottom:1px solid var(--line);
      padding:28px 32px 24px;
      overflow:hidden;
    }
    header::before {
      content:"";
      position:absolute;
      left:0;
      right:0;
      top:0;
      height:4px;
      background:linear-gradient(90deg, var(--primary), var(--accent), rgba(47,111,120,.35));
      pointer-events:none;
    }
    header::after {
      content:"";
      position:absolute;
      right:34px;
      top:34px;
      width:148px;
      height:104px;
      background:
        linear-gradient(135deg, transparent 0 34%, rgba(47,111,120,.10) 35% 37%, transparent 38%),
        linear-gradient(90deg, rgba(47,111,120,.08) 0 34px, transparent 34px 52px, rgba(209,162,58,.10) 52px 86px, transparent 86px);
      clip-path:polygon(8% 38%, 36% 8%, 72% 16%, 94% 42%, 78% 82%, 36% 94%, 10% 70%);
      pointer-events:none;
    }
    .topbar {
      position:relative;
      z-index:1;
      display:grid;
      grid-template-columns:minmax(0, 1fr) auto;
      align-items:center;
      gap:22px;
    }
    .brand { display:flex; align-items:center; gap:20px; min-width:0; }
    .brand-mark {
      position:relative;
      flex:0 0 auto;
      width:104px;
      height:104px;
      display:grid;
      place-items:center;
    }
    .brand-mark::before,
    .brand-mark::after {
      content:"";
      position:absolute;
      border:1px solid rgba(47,111,120,.34);
      pointer-events:none;
    }
    .brand-mark::before {
      inset:10px;
      transform:rotate(45deg);
      background:rgba(47,111,120,.035);
    }
    .brand-mark::after {
      right:0;
      top:4px;
      width:24px;
      height:24px;
      background:var(--accent-bg);
      transform:rotate(12deg);
    }
    .brand-logo {
      position:relative;
      z-index:1;
      width:86px;
      height:86px;
      object-fit:contain;
      filter:none;
      background:#fff;
      border:1px solid var(--line-strong);
      border-radius:10px;
      padding:10px;
      box-shadow:0 10px 26px rgba(38,55,58,.13);
    }
    .brand-copy { min-width:0; }
    h1 { margin:0; font-size:29px; letter-spacing:0; line-height:1.1; }
    h2 { margin:0; font-size:16px; letter-spacing:0; }
    h3 { margin:0; font-size:13px; color:var(--muted); font-weight:650; }
    .subtitle { margin-top:7px; color:var(--muted); max-width:720px; }
    .status-grid {
      display:grid;
      position:relative;
      z-index:1;
      grid-template-columns:repeat(4, minmax(170px, 1fr));
      gap:14px;
      margin-top:24px;
      align-items:stretch;
    }
    .status-card {
      position:relative;
      display:grid;
      grid-template-rows:18px 34px minmax(32px, auto);
      gap:6px;
      border:1px solid var(--line);
      background:rgba(255,254,250,.94);
      color:var(--ink);
      border-radius:14px;
      padding:15px 16px 14px;
      min-height:112px;
      box-shadow:var(--shadow-soft);
      overflow:hidden;
    }
    .status-card::before {
      content:"";
      position:absolute;
      left:0;
      top:0;
      bottom:0;
      width:4px;
      background:linear-gradient(180deg, var(--primary), rgba(47,111,120,.28));
    }
    .status-card::after {
      content:"";
      position:absolute;
      right:12px;
      top:12px;
      width:24px;
      height:18px;
      border:1px solid rgba(47,111,120,.24);
      border-left:0;
      border-bottom:0;
      transform:skew(-16deg);
      opacity:.9;
    }
    .status-title {
      color:var(--muted);
      font-size:12px;
      font-weight:800;
      display:flex;
      align-items:center;
    }
    .status-main {
      min-height:34px;
      display:flex;
      align-items:center;
      font-size:23px;
      font-weight:850;
      line-height:1.1;
      font-variant-numeric:tabular-nums;
    }
    .status-main.on, .status-main.off, .status-main.warn, .status-main.info {
      background:transparent;
      border-color:transparent;
      padding:0;
    }
    .status-main.on { color:var(--good); }
    .status-main.off { color:var(--bad); }
    .status-main.warn { color:var(--warn); }
    .status-main.info { color:var(--blue); }
    .status-sub {
      color:var(--muted);
      font-size:12px;
      min-height:32px;
      overflow-wrap:anywhere;
    }
    .live-panel {
      position:relative;
      z-index:1;
      margin-top:18px;
      border:1px solid var(--line);
      border-radius:16px;
      background:rgba(255,254,250,.96);
      padding:16px;
      box-shadow:var(--shadow);
      overflow:hidden;
    }
    .live-panel::before {
      content:"";
      position:absolute;
      left:16px;
      top:0;
      width:84px;
      height:4px;
      background:linear-gradient(90deg, var(--primary), var(--accent));
      border-radius:0 0 999px 999px;
    }
    .live-panel .section-head { margin-bottom:13px; }
    .activity-feed {
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(196px, 1fr));
      gap:12px;
    }
    .activity-item {
      position:relative;
      display:grid;
      gap:8px;
      min-height:100px;
      padding:13px 14px;
      border:1px solid var(--line);
      border-radius:12px;
      background:var(--surface-soft);
      color:var(--ink);
      overflow:hidden;
    }
    .activity-item::after {
      content:"";
      position:absolute;
      right:-8px;
      bottom:-8px;
      width:36px;
      height:36px;
      border:1px solid rgba(47,111,120,.16);
      transform:rotate(45deg);
    }
    .activity-item.tone-on { background:linear-gradient(180deg, var(--good-bg), #fff); }
    .activity-item.tone-warn { background:linear-gradient(180deg, var(--warn-bg), #fff); }
    .activity-item.tone-off { background:linear-gradient(180deg, var(--bad-bg), #fff); }
    .activity-item.tone-info { background:linear-gradient(180deg, var(--blue-bg), #fff); }
    .activity-meta { display:flex; justify-content:space-between; align-items:center; gap:8px; }
    .activity-label {
      color:var(--primary-strong);
      font-size:12px;
      font-weight:800;
      white-space:nowrap;
      background:var(--primary-bg);
      border:1px solid rgba(47,111,120,.20);
      padding:2px 8px;
      border-radius:999px;
    }
    .activity-title { font-weight:800; overflow-wrap:anywhere; line-height:1.35; }
    .activity-detail { color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
    .activity-time { color:var(--muted); font-size:12px; white-space:nowrap; font-variant-numeric:tabular-nums; }
    main {
      position:relative;
      z-index:1;
      padding:24px 32px 36px;
      display:grid;
      grid-template-columns:336px minmax(0, 1fr);
      gap:22px;
      align-items:start;
    }
    .stack { display:grid; gap:16px; min-width:0; }
    .sidebar { min-width:0; max-width:336px; }
    .content { min-width:0; overflow:hidden; }
    section {
      position:relative;
      min-width:0;
      background:rgba(255,254,250,.96);
      border:1px solid var(--line);
      border-radius:16px;
      padding:17px;
      box-shadow:var(--shadow-soft);
      overflow:hidden;
    }
    section::after {
      content:"";
      position:absolute;
      right:14px;
      top:14px;
      width:18px;
      height:18px;
      border:1px solid rgba(47,111,120,.18);
      border-left:0;
      border-bottom:0;
      transform:skew(-12deg);
      pointer-events:none;
    }
    .priority-grid {
      display:grid;
      grid-template-columns:minmax(0, 1.15fr) minmax(280px, .85fr);
      gap:16px;
      min-width:0;
    }
    .fold-panel { padding:0; }
    .fold-panel details { padding:0; }
    .fold-panel summary {
      cursor:pointer;
      display:grid;
      grid-template-columns:minmax(0, 1fr) 88px 28px;
      align-items:center;
      gap:10px;
      padding:14px 15px;
      list-style:none;
      background:rgba(255,254,250,.96);
      transition:background .15s ease, border-color .15s ease;
    }
    .fold-panel summary:hover { background:var(--surface-tint); }
    .fold-panel summary::-webkit-details-marker { display:none; }
    .fold-title {
      min-width:0;
      font-size:15px;
      font-weight:850;
      display:flex;
      align-items:center;
      gap:8px;
    }
    .fold-title::before {
      content:"";
      width:11px;
      height:11px;
      flex:0 0 auto;
      border:1px solid var(--primary);
      background:linear-gradient(135deg, #fff 0 48%, var(--primary-bg) 49%);
      transform:rotate(45deg);
    }
    .fold-panel summary .pill {
      width:88px;
      justify-content:center;
    }
    .fold-panel summary::after, details.more-panel summary::after {
      content:"+";
      color:var(--primary-strong);
      border:1px solid rgba(47,111,120,.34);
      background:var(--primary-bg);
      border-radius:8px;
      width:28px;
      height:28px;
      display:inline-grid;
      place-items:center;
      font-size:18px;
      line-height:1;
      font-weight:850;
      white-space:nowrap;
      flex:0 0 auto;
    }
    .fold-panel details[open] > summary::after, details.more-panel[open] > summary::after {
      content:"-";
      background:var(--primary);
      color:#fff;
      border-color:var(--primary);
    }
    .fold-body {
      border-top:1px solid var(--line);
      padding:14px 16px 16px;
      background:linear-gradient(180deg, #fff, var(--surface-soft));
    }
    .section-head {
      display:grid;
      grid-template-columns:minmax(0, 1fr) auto;
      align-items:center;
      gap:12px;
      margin-bottom:13px;
    }
    .section-head h2 {
      display:flex;
      align-items:center;
      gap:8px;
      min-width:0;
    }
    .section-head h2::before {
      content:"";
      width:18px;
      height:13px;
      border:1px solid rgba(47,111,120,.40);
      transform:skew(-18deg);
      background:linear-gradient(135deg, transparent 0 48%, rgba(209,162,58,.22) 49%);
    }
    .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    .field { display:grid; gap:6px; margin-top:12px; }
    .field span { color:var(--muted); font-size:12px; font-weight:750; }
    .mode-help {
      display:grid;
      gap:8px;
      margin-top:8px;
      padding:10px 12px;
      border:1px solid var(--line);
      border-radius:10px;
      background:var(--surface-soft);
    }
    .mode-help-row { display:grid; gap:3px; }
    .mode-help-row b { color:var(--primary-strong); font-size:12px; }
    .mode-help-row span { color:var(--muted); font-size:12px; line-height:1.45; }
    .kv {
      display:grid;
      grid-template-columns:104px minmax(0, 1fr);
      gap:10px;
      padding:9px 0;
      border-bottom:1px solid var(--line);
    }
    .kv:last-child { border-bottom:0; }
    .kv span:first-child { color:var(--muted); font-weight:750; }
    .muted { color:var(--muted); }
    .fine { color:var(--muted); font-size:12px; }
    .strong { font-weight:850; }
    button, select, input[type=text], input[type=password], input[type=number] {
      border:1px solid var(--line-strong);
      background:#fff;
      border-radius:9px;
      padding:8px 11px;
      min-height:37px;
      font:inherit;
    }
    input[type=text], input[type=password], input[type=number] { width:100%; }
    button {
      cursor:pointer;
      font-weight:800;
      color:var(--ink);
      box-shadow:none;
      transition:background .15s ease, border-color .15s ease, transform .15s ease;
    }
    button:hover { border-color:var(--primary); background:var(--primary-bg); }
    button:active { transform:translateY(1px); }
    button:disabled { cursor:not-allowed; opacity:.55; }
    button.primary { background:var(--primary); color:#fff; border-color:var(--primary); }
    button.primary:hover { background:var(--primary-strong); }
    button.good { background:var(--good-bg); color:var(--good); border-color:#a8d9c2; }
    button.bad { background:var(--bad-bg); color:var(--bad); border-color:#efb8b4; }
    .button-row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .button-row button, .field select { width:100%; }
    .segmented {
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(96px, 1fr));
      gap:4px;
      background:#eaf1ed;
      padding:4px;
      border:1px solid var(--line);
      border-radius:12px;
    }
    .segmented button { border:0; background:transparent; color:var(--muted); box-shadow:none; }
    .segmented button.active { background:#fff; color:var(--primary-strong); box-shadow:0 4px 12px rgba(38,55,58,.08); }
    .switch {
      display:grid;
      grid-template-columns:1fr auto;
      gap:12px;
      align-items:center;
      padding:12px 0;
      border-top:1px solid var(--line);
    }
    .switch:first-of-type { border-top:0; }
    .switch-title { font-weight:850; }
    .switch-note { color:var(--muted); font-size:12px; margin-top:2px; }
    input[type=checkbox] { width:20px; height:20px; accent-color:var(--primary); }
    .pill {
      display:inline-flex;
      align-items:center;
      justify-content:center;
      gap:6px;
      min-width:70px;
      min-height:26px;
      padding:3px 9px;
      border-radius:999px;
      font-size:12px;
      border:1px solid var(--line);
      background:#fff;
      white-space:nowrap;
      font-weight:850;
      font-variant-numeric:tabular-nums;
    }
    .section-head .pill { min-width:86px; }
    .on { color:var(--good); border-color:#a8d9c2; background:var(--good-bg); }
    .off { color:var(--bad); border-color:#efb8b4; background:var(--bad-bg); }
    .warn { color:var(--warn); border-color:#edce92; background:var(--warn-bg); }
    .info { color:var(--blue); border-color:#b8d0ea; background:var(--blue-bg); }
    .metric-grid {
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(120px, 1fr));
      gap:10px;
      margin-top:12px;
    }
    .metric {
      background:#fff;
      border:1px solid var(--line);
      border-radius:12px;
      padding:11px;
    }
    .metric-value { font-size:22px; font-weight:900; color:var(--primary-strong); }
    .metric-label { color:var(--muted); font-size:12px; font-weight:750; }
    .warning-list { display:grid; gap:8px; margin-top:12px; }
    .notice {
      border:1px solid var(--line);
      border-radius:12px;
      background:var(--surface-tint);
      padding:10px;
      margin-top:10px;
    }
    .notice.warn { background:var(--warn-bg); border-color:#edce92; color:var(--warn); }
    .notice.bad { background:var(--bad-bg); border-color:#efb8b4; color:var(--bad); }
    .notice.good { background:var(--good-bg); border-color:#a8d9c2; color:var(--good); }
    .check-list {
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(210px, 1fr));
      gap:9px;
      margin-top:12px;
    }
    .check-item {
      border:1px solid var(--line);
      border-radius:12px;
      padding:11px;
      background:#fff;
      min-height:80px;
    }
    .check-top { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:6px; }
    .check-title { font-weight:850; }
    .check-detail { color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
    .compact-actions { display:flex; gap:8px; flex-wrap:wrap; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { border-bottom:1px solid var(--line); text-align:left; padding:10px 9px; vertical-align:top; }
    th { color:var(--muted); font-weight:850; background:#f1f6f3; }
    tr:hover td { background:#f8fbf9; }
    code, pre { font-family:Consolas, "Cascadia Mono", monospace; }
    code { overflow-wrap:anywhere; }
    pre {
      white-space:pre-wrap;
      max-height:260px;
      overflow:auto;
      background:#233036;
      color:#eef6f2;
      padding:12px;
      border-radius:12px;
      margin:10px 0 0;
      font-size:12px;
      border:1px solid rgba(35,48,54,.34);
    }
    .table-wrap { overflow:auto; border:1px solid var(--line); border-radius:12px; background:#fff; }
    .table-wrap table th, .table-wrap table td { white-space:nowrap; }
    .run-id { max-width:260px; white-space:normal; }
    .compact-card {
      position:relative;
      border:1px solid var(--line);
      border-radius:12px;
      background:linear-gradient(135deg, #fff, var(--surface-tint));
      padding:13px;
      display:grid;
      gap:8px;
      overflow:hidden;
    }
    .compact-card::after {
      content:"";
      position:absolute;
      right:12px;
      bottom:10px;
      width:26px;
      height:20px;
      border:1px solid rgba(47,111,120,.18);
      transform:skew(-18deg);
    }
    .compact-card-title { font-weight:850; overflow-wrap:anywhere; }
    details.more-panel { margin-top:10px; }
    details.more-panel summary {
      cursor:pointer;
      display:grid;
      grid-template-columns:minmax(0, 1fr) 28px;
      align-items:center;
      gap:10px;
      color:var(--ink);
      font-weight:850;
      margin-bottom:10px;
      border:1px solid var(--line);
      padding:9px 10px;
      background:#fff;
      border-radius:12px;
    }
    .modal-backdrop {
      display:none;
      position:fixed;
      inset:0;
      z-index:20;
      background:rgba(23,33,38,.36);
      padding:20px;
      align-items:center;
      justify-content:center;
    }
    .modal-backdrop.open { display:flex; }
    .modal {
      width:min(560px, 100%);
      background:#fff;
      border:1px solid var(--line-strong);
      border-radius:16px;
      padding:16px;
      box-shadow:0 22px 54px rgba(38,55,58,.24);
    }
    .module-library {
      padding:0;
      background:linear-gradient(135deg, rgba(255,254,250,.98), rgba(237,245,242,.96));
    }
    .module-library::after { display:none; }
    .module-library .fold-body {
      background:linear-gradient(180deg, #fff, rgba(237,245,242,.72));
    }
    .module-groups { display:grid; gap:12px; }
    .module-group { display:grid; gap:7px; }
    .module-group-title {
      color:var(--muted);
      font-size:12px;
      font-weight:850;
      padding-left:2px;
    }
    .module-item {
      width:100%;
      display:grid;
      grid-template-columns:18px minmax(0, 1fr) 28px;
      align-items:center;
      gap:9px;
      min-height:42px;
      padding:9px 10px;
      text-align:left;
      border-color:var(--line);
      background:#fff;
      color:var(--ink);
      border-radius:12px;
    }
    .module-item:hover { background:var(--surface-tint); }
    .module-item.active {
      border-color:rgba(47,111,120,.38);
      background:var(--primary-bg);
    }
    .module-item.collapsed {
      background:#fff;
      border-style:dashed;
    }
    .module-item.dragging { opacity:.58; }
    .module-glyph {
      width:16px;
      height:16px;
      border:1px solid rgba(47,111,120,.44);
      background:linear-gradient(135deg, #fff 0 48%, rgba(209,162,58,.24) 49%);
      transform:rotate(45deg);
    }
    .module-label { min-width:0; font-weight:850; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .module-state {
      width:26px;
      height:26px;
      display:inline-grid;
      place-items:center;
      color:var(--primary-strong);
      font-size:18px;
      line-height:1;
      font-weight:900;
      border:1px solid rgba(47,111,120,.28);
      background:#fff;
      border-radius:8px;
    }
    .workspace {
      min-height:0;
      border-radius:18px;
      transition:background .15s ease, outline-color .15s ease;
    }
    .workspace.drop-ready {
      outline:2px dashed rgba(47,111,120,.45);
      outline-offset:8px;
      background:rgba(229,242,243,.45);
    }
    .module-card {
      align-self:start;
      transition:opacity .15s ease, transform .15s ease, box-shadow .15s ease;
    }
    .module-card.is-hidden { display:none; }
    .module-card.is-collapsed {
      padding-bottom:14px;
      min-height:0;
    }
    .module-card.is-collapsed > :not(.section-head) { display:none; }
    .module-card.flash {
      box-shadow:0 0 0 3px rgba(209,162,58,.34), var(--shadow-soft);
    }
    .module-actions {
      display:flex;
      align-items:center;
      justify-content:flex-end;
      gap:7px;
      min-width:0;
    }
    .module-action {
      min-width:30px;
      width:30px;
      height:30px;
      min-height:30px;
      padding:0;
      display:inline-grid;
      place-items:center;
      border-radius:9px;
      font-size:16px;
      line-height:1;
      color:var(--primary-strong);
      border-color:rgba(47,111,120,.30);
      background:#fff;
    }
    .module-action:hover { background:var(--primary-bg); }
    .priority-grid { align-items:start; }
    .priority-grid.single-visible { grid-template-columns:1fr; }
    @media (max-width: 1120px) {
      .status-grid { grid-template-columns:repeat(2, minmax(160px, 1fr)); }
      main { grid-template-columns:1fr; }
      .priority-grid { grid-template-columns:1fr; }
      .sidebar { max-width:none; }
    }
    @media (max-width: 720px) {
      header { padding:18px 14px 14px; }
      main { padding:14px; }
      .topbar { display:block; }
      .brand { gap:12px; align-items:flex-start; }
      .brand-mark { width:74px; height:74px; }
      .brand-logo { width:62px; height:62px; padding:7px; border-radius:8px; }
      h1 { font-size:22px; }
      #lastRefresh { margin-top:10px; }
      .status-grid, .metric-grid { grid-template-columns:1fr; }
      .activity-feed { grid-template-columns:1fr; }
      .button-row { grid-template-columns:1fr; }
      .fold-panel summary { grid-template-columns:minmax(0, 1fr) 78px 28px; }
      .fold-panel summary .pill { width:78px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div class="brand">
        <div class="brand-mark">
          <img class="brand-logo" src="/assets/sts2_ai_logo.png" alt="STS2 AI">
        </div>
        <div class="brand-copy">
          <h1>STS2 AI 控制台</h1>
          <div class="subtitle">战斗托管、数据采集、BC 重训和 run 质量管理</div>
        </div>
      </div>
      <span id="lastRefresh" class="pill info">读取中</span>
    </div>
    <div class="status-grid">
      <div class="status-card">
        <div class="status-title">游戏连接</div>
        <div id="gamePhase" class="status-main">读取中</div>
        <div id="gameDetail" class="status-sub">-</div>
      </div>
      <div class="status-card">
        <div class="status-title">AI 接管</div>
        <div id="aiStatus" class="status-main">读取中</div>
        <div id="aiDetail" class="status-sub">-</div>
      </div>
      <div class="status-card">
        <div class="status-title">采集总开关</div>
        <div id="collectStatus" class="status-main">读取中</div>
        <div id="collectDetail" class="status-sub">-</div>
      </div>
      <div class="status-card">
        <div class="status-title">当前 Run</div>
        <div id="runQuality" class="status-main">读取中</div>
        <div id="runDetail" class="status-sub">-</div>
      </div>
    </div>
    <div class="live-panel">
      <div class="section-head">
        <h2>实时采集动态</h2>
        <span id="activityBadge" class="pill info">读取中</span>
      </div>
      <div id="liveActivity" class="activity-feed">
        <div class="muted">读取中</div>
      </div>
    </div>
  </header>

  <main>
    <div class="stack sidebar">
      <section class="fold-panel module-library">
        <details open>
          <summary>
            <span class="fold-title">工作区</span>
            <span id="openModuleCount" class="pill info">-</span>
          </summary>
          <div class="fold-body">
            <div class="module-groups">
              <div class="module-group">
                <div class="module-group-title">决策</div>
                <button class="module-item" data-module-target="ai_logic" draggable="true" onclick="openModule('ai_logic')" ondragstart="beginModuleDrag(event, 'ai_logic')" ondragend="endModuleDrag(event)">
                  <span class="module-glyph"></span><span class="module-label">AI 出牌逻辑</span><span class="module-state">+</span>
                </button>
                <button class="module-item" data-module-target="llm_logic" draggable="true" onclick="openModule('llm_logic')" ondragstart="beginModuleDrag(event, 'llm_logic')" ondragend="endModuleDrag(event)">
                  <span class="module-glyph"></span><span class="module-label">LLM 决策</span><span class="module-state">+</span>
                </button>
              </div>
              <div class="module-group">
                <div class="module-group-title">数据</div>
                <button class="module-item" data-module-target="current_data" draggable="true" onclick="openModule('current_data')" ondragstart="beginModuleDrag(event, 'current_data')" ondragend="endModuleDrag(event)">
                  <span class="module-glyph"></span><span class="module-label">Run 数据体检</span><span class="module-state">+</span>
                </button>
                <button class="module-item" data-module-target="runs" draggable="true" onclick="openModule('runs')" ondragstart="beginModuleDrag(event, 'runs')" ondragend="endModuleDrag(event)">
                  <span class="module-glyph"></span><span class="module-label">最近 Run</span><span class="module-state">+</span>
                </button>
                <button class="module-item" data-module-target="records" draggable="true" onclick="openModule('records')" ondragstart="beginModuleDrag(event, 'records')" ondragend="endModuleDrag(event)">
                  <span class="module-glyph"></span><span class="module-label">最近采集记录</span><span class="module-state">+</span>
                </button>
              </div>
              <div class="module-group">
                <div class="module-group-title">训练</div>
                <button class="module-item" data-module-target="evaluation" draggable="true" onclick="openModule('evaluation')" ondragstart="beginModuleDrag(event, 'evaluation')" ondragend="endModuleDrag(event)">
                  <span class="module-glyph"></span><span class="module-label">策略评测</span><span class="module-state">+</span>
                </button>
                <button class="module-item" data-module-target="training" draggable="true" onclick="openModule('training')" ondragstart="beginModuleDrag(event, 'training')" ondragend="endModuleDrag(event)">
                  <span class="module-glyph"></span><span class="module-label">训练输出</span><span class="module-state">+</span>
                </button>
              </div>
            </div>
          </div>
        </details>
      </section>

      <section class="fold-panel">
        <details open>
          <summary>
            <span class="fold-title">战斗 AI</span>
            <span id="aiProcessBadge" class="pill">-</span>
          </summary>
          <div class="fold-body">
        <div class="button-row">
          <button class="good" onclick="startAI()">启动 AI</button>
          <button class="bad" onclick="stopAI()">停止 AI</button>
        </div>
        <div class="row" style="margin-top:8px">
          <button onclick="restartAI()">重启 AI</button>
        </div>
        <div class="switch">
          <div><div class="switch-title">允许 AI 出牌</div><div class="switch-note">只影响战斗自动打牌</div></div>
          <input id="ai_enabled" type="checkbox" onchange="saveControl()">
        </div>
        <div class="switch">
          <div><div class="switch-title">允许 AI 宏观操作</div><div class="switch-note">地图、奖励、选卡、事件、休息点；默认关闭</div></div>
          <input id="macro_enabled" type="checkbox" onchange="saveControl()">
        </div>
        <div class="switch">
          <div><div class="switch-title">允许 AI 商店购买</div><div class="switch-note">独立开关；默认关闭，避免和手动商店操作冲突</div></div>
          <input id="macro_shop_enabled" type="checkbox" onchange="saveControl()">
        </div>
        <div class="switch">
          <div><div class="switch-title">记录 AI 战斗动作</div><div class="switch-note">控制台镜像写入 AI_Combat；Mod 正式记录写入 AI/Combat</div></div>
          <input id="record_ai_actions" type="checkbox" onchange="saveControl()">
        </div>
        <div class="switch">
          <div><div class="switch-title">AI 数据进入 BC</div><div class="switch-note">默认关闭，避免自举污染</div></div>
          <input id="include_ai_in_training" type="checkbox" onchange="saveControl()">
        </div>
          </div>
        </details>
      </section>

      <section class="fold-panel">
        <details>
          <summary>
            <span class="fold-title">模型与进程</span>
            <span id="modelBadge" class="pill">-</span>
          </summary>
          <div class="fold-body">
            <div id="modelHealth">读取中</div>
          </div>
        </details>
      </section>

      <section id="llmConfigSection" class="fold-panel">
        <details>
          <summary>
            <span class="fold-title">LLM 模型接入</span>
            <span id="llmProcessBadge" class="pill">-</span>
          </summary>
          <div class="fold-body">
        <div class="button-row">
          <button class="good" onclick="startLLM()">启动 LLM</button>
          <button class="bad" onclick="stopLLM()">停止 LLM</button>
        </div>
        <div class="switch">
          <div><div class="switch-title">启用模型决策</div><div class="switch-note">关闭后 LLM 进程待机，不请求模型</div></div>
          <input id="llm_enabled" type="checkbox" onchange="saveLLMConfig()">
        </div>
        <div class="field">
          <span>模式</span>
          <select id="llm_mode" onchange="saveLLMConfig()">
            <option value="advisor">Advisor：只给建议</option>
            <option value="combat_auto">Combat Auto：战斗可执行</option>
          </select>
        </div>
        <div class="field">
          <span>动作选择</span>
          <select id="llm_action_selection_mode" onchange="saveLLMConfig()">
            <option value="candidate_id">推荐：只从合法候选动作里选</option>
            <option value="catalog_args">兼容：让模型填写动作参数</option>
          </select>
          <div class="mode-help">
            <div class="mode-help-row">
              <b>推荐模式</b>
              <span>系统先列出当前能做的合法动作，例如出哪张牌、打哪个怪、用哪个药水、是否结束回合。LLM 只能选其中一个，不允许自己编动作。演示和实战优先用这个。</span>
            </div>
            <div class="mode-help-row">
              <b>兼容模式</b>
              <span>LLM 自己填写动作和参数，例如 card_index、target、potion slot。系统仍会校验和兜底，但更容易因为参数理解错被拦截。这个主要保留给对比和调试。</span>
            </div>
          </div>
        </div>
        <div class="switch">
          <div><div class="switch-title">允许 LLM 自动战斗</div><div class="switch-note">只执行战斗出牌/结束回合；宏观仍只建议</div></div>
          <input id="llm_execute_combat" type="checkbox" onchange="saveLLMConfig()">
        </div>
        <div class="field">
          <span>Base URL</span>
          <input id="llm_base_url" type="text" placeholder="https://api.openai.com/v1" oninput="markLLMDirty()">
        </div>
        <div class="field">
          <span>API Key</span>
          <input id="llm_api_key" type="password" placeholder="留空表示不修改已保存 key" oninput="markLLMDirty()">
        </div>
        <div class="field">
          <span>Model</span>
          <input id="llm_model" type="text" placeholder="例如 gpt-4.1-mini" oninput="markLLMDirty()">
        </div>
        <div class="row" style="margin-top:12px">
          <button class="primary" onclick="saveLLMConfig()">保存模型配置</button>
          <button onclick="saveLLMProfile()">保存为 API 配置</button>
          <button id="llmTestButton" onclick="testLLM()">测试连接</button>
        </div>
        <div id="llmTestResult" class="fine" style="margin-top:8px">未测试</div>
        <div class="fine" style="margin-top:10px">API Key 只保存在本机 <code>AI_Training/model_config.json</code>，不会进入 Git。</div>
        <div class="table-wrap" style="margin-top:10px">
          <table>
            <thead><tr><th>名称</th><th>Base URL</th><th>Model</th><th>Key</th><th>操作</th></tr></thead>
            <tbody id="llmProfiles"></tbody>
          </table>
        </div>
          </div>
        </details>
      </section>

      <section class="fold-panel">
        <details>
          <summary>
            <span class="fold-title">采集与训练</span>
            <span id="collectBadge" class="pill">-</span>
          </summary>
          <div class="fold-body">
        <div class="switch">
          <div><div class="switch-title">采集总开关</div><div class="switch-note">打开才写入战斗/宏观日志；关闭后不采集本局后续动作</div></div>
          <input id="collection_enabled" type="checkbox" onchange="saveControl()">
        </div>
        <div class="field">
          <span>最低训练质量</span>
          <select id="min_training_quality" onchange="saveControl()">
            <option value="failed_run">失败也要</option>
            <option value="unknown">未知及以上</option>
            <option value="before_act1_boss">一关Boss前及以上</option>
            <option value="partial_act1">一关Boss及以上</option>
            <option value="partial_act2">二关Boss及以上</option>
            <option value="perfect_run">只用通关完美</option>
          </select>
        </div>
        <div class="row" style="margin-top:12px">
          <button class="primary" onclick="train()">重建数据 + 重训战斗/候选/宏观 BC</button>
          <button onclick="proceed()">Proceed</button>
        </div>
          </div>
        </details>
      </section>

      <section class="fold-panel">
        <details>
          <summary>
            <span class="fold-title">数据打包</span>
            <span id="exportBadge" class="pill info">未导出</span>
          </summary>
          <div class="fold-body">
        <button class="primary" onclick="exportData()">一键打包数据库</button>
        <div id="exportInfo" class="fine" style="margin-top:10px">生成 zip 后，直接把这个文件发给你。</div>
          </div>
        </details>
      </section>

      <section class="fold-panel">
        <details>
          <summary>
            <span class="fold-title">主菜单标记</span>
            <span id="nextRunBadge" class="pill info">-</span>
          </summary>
          <div class="fold-body">
        <div class="segmented">
          <button id="modeAuto" onclick="setRunMode('auto')">自动检测</button>
          <button id="modeNew" onclick="setRunMode('new')">强制新局一次</button>
          <button id="modeContinue" onclick="setRunMode('continue')">续接旧 Run</button>
        </div>
        <div class="fine" style="margin-top:10px">推荐用自动检测。手动按钮仍保留：如果你在主菜单明确要重开，点“强制新局一次”；如果是中途接回旧档，点“续接旧 Run”。</div>
          </div>
        </details>
      </section>
    </div>

    <div id="workspace" class="stack content workspace" ondragover="handleWorkspaceDragOver(event)" ondragleave="handleWorkspaceDragLeave(event)" ondrop="handleWorkspaceDrop(event)">
      <div class="priority-grid">
        <section id="module-ai-logic" class="module-card" data-module="ai_logic">
          <div class="section-head">
            <h2>AI 出牌逻辑</h2>
            <div class="module-actions">
              <span id="aiDecisionBadge" class="pill">-</span>
              <button class="module-action" onclick="toggleModuleCollapse('ai_logic')" data-collapse-for="ai_logic" title="收起">-</button>
              <button class="module-action" onclick="closeModule('ai_logic')" title="关闭">×</button>
            </div>
          </div>
          <div id="aiLogic" class="muted">暂无 AI 决策</div>
          <div class="table-wrap" style="margin-top:10px">
            <table>
              <thead><tr><th>候选动作</th><th>概率</th><th>状态</th></tr></thead>
              <tbody id="aiTopActions"></tbody>
            </table>
          </div>
        </section>

        <section id="module-llm-logic" class="module-card" data-module="llm_logic">
          <div class="section-head">
            <h2>LLM 决策</h2>
            <div class="module-actions">
              <span id="llmDecisionBadge" class="pill">-</span>
              <button class="module-action" onclick="toggleModuleCollapse('llm_logic')" data-collapse-for="llm_logic" title="收起">-</button>
              <button class="module-action" onclick="closeModule('llm_logic')" title="关闭">×</button>
            </div>
          </div>
          <div id="llmLogic" class="muted">暂无 LLM 决策</div>
        </section>
      </div>

      <section id="module-current-data" class="module-card" data-module="current_data">
        <div class="section-head">
          <h2>Run 数据体检</h2>
          <div class="module-actions">
            <span id="currentDataBadge" class="pill">读取中</span>
            <button class="module-action" onclick="toggleModuleCollapse('current_data')" data-collapse-for="current_data" title="收起">-</button>
            <button class="module-action" onclick="closeModule('current_data')" title="关闭">×</button>
          </div>
        </div>
        <div id="currentData">读取中</div>
      </section>

      <section id="module-runs" class="module-card" data-module="runs">
        <div class="section-head">
          <h2>最近 Run</h2>
          <div class="module-actions">
            <span id="runsBadge" class="pill info">-</span>
            <button class="module-action" onclick="toggleModuleCollapse('runs')" data-collapse-for="runs" title="收起">-</button>
            <button class="module-action" onclick="closeModule('runs')" title="关闭">×</button>
          </div>
        </div>
        <div id="latestRunCard" class="compact-card">读取中</div>
        <details class="more-panel">
          <summary>更多 Run</summary>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Run</th><th>进度</th><th>动作</th><th>来源</th><th>结果</th><th>质量</th><th>数据</th><th>保留</th><th></th></tr></thead>
              <tbody id="runs"></tbody>
            </table>
          </div>
        </details>
      </section>

      <section id="module-evaluation" class="module-card" data-module="evaluation">
        <div class="section-head">
          <h2>策略评测</h2>
          <div class="module-actions">
            <span id="evalBadge" class="pill info">读取中</span>
            <button class="module-action" onclick="toggleModuleCollapse('evaluation')" data-collapse-for="evaluation" title="收起">-</button>
            <button class="module-action" onclick="closeModule('evaluation')" title="关闭">×</button>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>策略</th><th>Run</th><th>平均/最高楼层</th><th>结果</th><th>问题</th><th>最近</th></tr></thead>
            <tbody id="policyEval"></tbody>
          </table>
        </div>
      </section>

      <section id="module-records" class="module-card" data-module="records">
        <div class="section-head">
          <h2>最近采集记录</h2>
          <div class="module-actions">
            <span class="fine">原始事件流</span>
            <button class="module-action" onclick="toggleModuleCollapse('records')" data-collapse-for="records" title="收起">-</button>
            <button class="module-action" onclick="closeModule('records')" title="关闭">×</button>
          </div>
        </div>
        <details class="more-panel">
          <summary>原始事件表</summary>
          <div class="notice">这里不是“每个 Run 一行”，而是 Mod/AI 写入的原始事件流：战斗开始、回合开始、出牌、回合结束、奖励、地图等都会各占一条。顶部“实时采集动态”是给人看的摘要，这里保留给排查字段。</div>
          <div class="table-wrap">
            <table>
              <thead><tr><th>时间</th><th>记录类型</th><th>来源</th><th>摘要</th><th>文件</th></tr></thead>
              <tbody id="recentRecords"></tbody>
            </table>
          </div>
        </details>
      </section>

      <section id="module-training" class="module-card" data-module="training">
        <div class="section-head">
          <h2>训练输出</h2>
          <div class="module-actions">
            <span id="trainStatus" class="pill">-</span>
            <button class="module-action" onclick="toggleModuleCollapse('training')" data-collapse-for="training" title="收起">-</button>
            <button class="module-action" onclick="closeModule('training')" title="关闭">×</button>
          </div>
        </div>
        <pre id="trainOutput">暂无输出</pre>
      </section>
    </div>
  </main>
  <div id="llmProfileEditor" class="modal-backdrop" onclick="closeLLMProfileEditor(event)">
    <div class="modal" onclick="event.stopPropagation()">
      <div class="section-head">
        <h2>修改 API 配置</h2>
        <button onclick="closeLLMProfileEditor()">关闭</button>
      </div>
      <input id="edit_profile_id" type="hidden">
      <div class="field">
        <span>名称</span>
        <input id="edit_profile_name" type="text">
      </div>
      <div class="field">
        <span>Base URL</span>
        <input id="edit_base_url" type="text">
      </div>
      <div class="field">
        <span>Model</span>
        <input id="edit_model" type="text">
      </div>
      <div class="field">
        <span>API Key</span>
        <input id="edit_api_key" type="password" placeholder="留空表示不修改已保存 key">
      </div>
      <div class="row" style="margin-top:14px">
        <button class="primary" onclick="saveLLMProfileEdit()">保存修改</button>
        <button onclick="closeLLMProfileEditor()">取消</button>
      </div>
    </div>
  </div>
<script>
async function api(path, body) {
  const opts = body ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)} : {};
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return await r.json();
}
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&":"&amp;",
    "<":"&lt;",
    ">":"&gt;",
    '"':"&quot;",
    "'":"&#39;"
  }[ch]));
}
function phaseInfo(game) {
  if (!game.online) return {label:"未连接", cls:"off", detail:`游戏 API 离线：${game.error || "无响应"}`};
  const raw = String(game.state_type || "unknown");
  const lower = raw.toLowerCase();
  if (lower.includes("menu")) return {label:"主菜单", cls:"warn", detail:"游戏已连接，当前不在一局游戏内"};
  if (lower.includes("monster") || lower.includes("combat")) return {label:"战斗中", cls:"on", detail:`${game.character || ""} HP ${game.hp ?? "?"}/${game.max_hp ?? "?"} 能量 ${game.energy ?? "?"}`};
  if (lower.includes("shop") || lower.includes("merchant")) {
    return {label:"商店中", cls:"warn", detail:game.shop_poll_guard ? `${raw}；商店保护中，减少自动刷新` : raw};
  }
  if (lower.includes("map")) return {label:"地图中", cls:"info", detail:raw};
  if (lower.includes("event")) return {label:"事件中", cls:"info", detail:raw};
  if (lower.includes("rest")) return {label:"营火中", cls:"info", detail:raw};
  return {label:"游戏中", cls:"info", detail:raw};
}
function qualityOptions(selected) {
  const opts = [
    ["failed_run", "失败"],
    ["unknown", "未知"],
    ["before_act1_boss", "一关Boss前"],
    ["partial_act1", "一关Boss"],
    ["partial_act2", "二关Boss"],
    ["perfect_run", "通关完美"]
  ];
  return opts.map(([v, label]) => `<option value="${v}" ${v === selected ? "selected" : ""}>${label}</option>`).join("");
}
function setPill(id, text, cls) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = `pill ${cls || ""}`;
}
function checkClass(status) {
  if (status === "ok") return "on";
  if (status === "missing") return "off";
  if (status === "warn") return "warn";
  return "info";
}
function checkLabel(status) {
  if (status === "ok") return "正常";
  if (status === "missing") return "缺失";
  if (status === "warn") return "确认";
  return "可选";
}
function dataHealthClass(health) {
  if (health === "ok") return "on";
  if (health === "missing") return "off";
  if (health === "warn") return "warn";
  return "info";
}
const MODULE_IDS = ["ai_logic", "llm_logic", "current_data", "runs", "records", "evaluation", "training"];
const MODULE_STORAGE_KEY = "sts2_control_panel_modules";
function defaultModuleState() {
  return Object.fromEntries(MODULE_IDS.map(id => [id, {open:true, collapsed:false}]));
}
function readModuleState() {
  const state = defaultModuleState();
  try {
    const saved = JSON.parse(localStorage.getItem(MODULE_STORAGE_KEY) || "{}");
    for (const id of MODULE_IDS) {
      if (saved[id]) {
        state[id].open = saved[id].open !== false;
        state[id].collapsed = !!saved[id].collapsed;
      }
    }
  } catch (_) {}
  return state;
}
let moduleState = readModuleState();
let draggingModuleId = "";
function saveModuleState() {
  localStorage.setItem(MODULE_STORAGE_KEY, JSON.stringify(moduleState));
}
function moduleElement(id) {
  return document.querySelector(`.module-card[data-module="${id}"]`);
}
function syncModuleUI() {
  let openCount = 0;
  let visibleDecisionCards = 0;
  for (const id of MODULE_IDS) {
    const state = moduleState[id] || {open:true, collapsed:false};
    const card = moduleElement(id);
    if (card) {
      card.classList.toggle("is-hidden", !state.open);
      card.classList.toggle("is-collapsed", !!state.collapsed);
    }
    const dock = document.querySelector(`.module-item[data-module-target="${id}"]`);
    if (dock) {
      dock.classList.toggle("active", !!state.open);
      dock.classList.toggle("collapsed", !!state.open && !!state.collapsed);
      dock.title = !state.open ? "点击打开；也可以拖到右侧工作区" : (state.collapsed ? "已收起，点击展开并定位" : "已打开，点击定位");
      const label = dock.querySelector(".module-state");
      if (label) label.textContent = state.open && !state.collapsed ? "-" : "+";
    }
    const collapseButton = document.querySelector(`[data-collapse-for="${id}"]`);
    if (collapseButton) {
      collapseButton.textContent = state.collapsed ? "+" : "-";
      collapseButton.title = state.collapsed ? "展开" : "收起";
    }
    if (state.open) {
      openCount++;
      if (id === "ai_logic" || id === "llm_logic") visibleDecisionCards++;
    }
  }
  const decisionGrid = document.querySelector(".priority-grid");
  if (decisionGrid) decisionGrid.classList.toggle("single-visible", visibleDecisionCards === 1);
  const count = document.getElementById("openModuleCount");
  if (count) count.textContent = `${openCount}/${MODULE_IDS.length}`;
}
function flashModule(card) {
  if (!card) return;
  card.classList.remove("flash");
  void card.offsetWidth;
  card.classList.add("flash");
  window.setTimeout(() => card.classList.remove("flash"), 900);
}
function openModule(id, opts = {}) {
  if (!MODULE_IDS.includes(id)) return;
  moduleState[id] = {open:true, collapsed:false};
  saveModuleState();
  syncModuleUI();
  const card = moduleElement(id);
  flashModule(card);
  if (opts.scroll !== false && card) {
    card.scrollIntoView({behavior:"smooth", block:"start"});
  }
}
function closeModule(id) {
  if (!MODULE_IDS.includes(id)) return;
  moduleState[id] = {...(moduleState[id] || {}), open:false};
  saveModuleState();
  syncModuleUI();
}
function toggleModuleCollapse(id) {
  if (!MODULE_IDS.includes(id)) return;
  const current = moduleState[id] || {open:true, collapsed:false};
  moduleState[id] = {open:true, collapsed:!current.collapsed};
  saveModuleState();
  syncModuleUI();
  flashModule(moduleElement(id));
}
function beginModuleDrag(event, id) {
  draggingModuleId = id;
  event.dataTransfer.setData("text/plain", id);
  event.dataTransfer.effectAllowed = "copy";
  event.currentTarget.classList.add("dragging");
}
function endModuleDrag(event) {
  draggingModuleId = "";
  event.currentTarget.classList.remove("dragging");
  const workspace = document.getElementById("workspace");
  if (workspace) workspace.classList.remove("drop-ready");
}
function handleWorkspaceDragOver(event) {
  const id = draggingModuleId || event.dataTransfer.getData("text/plain");
  if (!MODULE_IDS.includes(id)) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = "copy";
  event.currentTarget.classList.add("drop-ready");
}
function handleWorkspaceDragLeave(event) {
  if (!event.currentTarget.contains(event.relatedTarget)) {
    event.currentTarget.classList.remove("drop-ready");
  }
}
function handleWorkspaceDrop(event) {
  const id = draggingModuleId || event.dataTransfer.getData("text/plain");
  if (!MODULE_IDS.includes(id)) return;
  event.preventDefault();
  event.currentTarget.classList.remove("drop-ready");
  draggingModuleId = "";
  openModule(id, {scroll:false});
}
let llmFormDirty = false;
let llmProfilesCache = [];
let refreshInFlight = false;
let refreshPending = false;
function markLLMDirty() {
  llmFormDirty = true;
}
function isLLMFormEditing() {
  const section = document.getElementById("llmConfigSection");
  return llmFormDirty || (section && section.contains(document.activeElement));
}
function applyLLMConfigToForm(llmCfg) {
  document.getElementById("llm_enabled").checked = !!llmCfg.enabled;
  document.getElementById("llm_mode").value = llmCfg.mode || "advisor";
  document.getElementById("llm_action_selection_mode").value = llmCfg.action_selection_mode || "candidate_id";
  document.getElementById("llm_execute_combat").checked = !!llmCfg.execute_combat;
  document.getElementById("llm_base_url").value = llmCfg.base_url || "";
  document.getElementById("llm_model").value = llmCfg.model || "";
  document.getElementById("llm_api_key").placeholder = llmCfg.has_api_key ? "已保存，留空不修改" : "未配置 API Key";
}
async function refresh() {
  if (refreshInFlight) {
    refreshPending = true;
    return;
  }
  refreshInFlight = true;
  try {
    const s = await api("/api/status");
    renderStatus(s);
  } catch (err) {
    setPill("lastRefresh", "刷新失败", "off");
    const detail = document.getElementById("gameDetail");
    if (detail) detail.textContent = `控制台刷新失败：${err.message || err}`;
  } finally {
    refreshInFlight = false;
    if (refreshPending) {
      refreshPending = false;
      window.setTimeout(refresh, 100);
    }
  }
}
function renderStatus(s) {
  const active = s.current_data && s.current_data.active_run;
  const phase = phaseInfo(s.game);

  document.getElementById("ai_enabled").checked = !!s.control.ai_enabled;
  document.getElementById("macro_enabled").checked = !!s.control.macro_enabled;
  document.getElementById("macro_shop_enabled").checked = !!s.control.macro_shop_enabled;
  document.getElementById("collection_enabled").checked = !!s.control.collection_enabled;
  document.getElementById("record_ai_actions").checked = !!s.control.record_ai_actions;
  document.getElementById("include_ai_in_training").checked = !!s.control.include_ai_in_training;
  document.getElementById("min_training_quality").value = s.control.min_training_quality || "unknown";
  const llmCfg = (s.llm && s.llm.config) || {};
  if (!isLLMFormEditing()) {
    applyLLMConfigToForm(llmCfg);
  }
  renderLLMProfiles(llmCfg.profiles || [], llmCfg.active_profile_id || "");

  document.getElementById("gamePhase").textContent = phase.label;
  document.getElementById("gamePhase").className = `status-main ${phase.cls}`;
  document.getElementById("gameDetail").textContent = phase.detail;
  document.getElementById("aiStatus").textContent = s.control.ai_enabled ? (s.control.macro_enabled ? "战斗+宏观" : "只管战斗") : "手动模式";
  document.getElementById("aiStatus").className = `status-main ${s.control.ai_enabled ? "on" : "warn"}`;
  document.getElementById("aiDetail").textContent = s.ai_pid ? `托管进程 PID ${s.ai_pid}；宏观 ${s.control.macro_enabled ? "开启" : "关闭"}；商店 ${s.control.macro_shop_enabled ? "允许" : "保护"}` : "AI 进程未由控制台托管";
  document.getElementById("collectStatus").textContent = s.control.collection_enabled ? "采集中" : "已暂停";
  document.getElementById("collectStatus").className = `status-main ${s.control.collection_enabled ? "on" : "off"}`;
  document.getElementById("collectDetail").textContent = s.control.collection_enabled
    ? (s.control.include_ai_in_training ? "写入 Human + AI 数据" : "写入 Human 数据，AI 不进 BC")
    : "不会写入新的战斗/宏观日志";
  document.getElementById("runQuality").textContent = active ? (active.quality_label || active.quality || "-") : "无 run";
  document.getElementById("runQuality").className = `status-main ${active && active.discarded ? "off" : "info"}`;
  document.getElementById("runDetail").textContent = active ? `Act ${active.max_act || 0} / Floor ${active.max_floor || 0}，${active.records || 0} 条` : "尚未读取到采集数据";

  setPill("lastRefresh", `已刷新 ${new Date().toLocaleTimeString()}`, "info");
  setPill("aiProcessBadge", s.ai_pid ? (s.ai_process && s.ai_process.needs_restart ? "需重启" : "运行中") : "未启动", s.ai_pid ? ((s.ai_process && s.ai_process.needs_restart) ? "warn" : "on") : "warn");
  setPill("llmProcessBadge", s.llm && s.llm.pid ? "运行中" : "未启动", s.llm && s.llm.pid ? "on" : "warn");
  setPill("collectBadge", s.control.collection_enabled ? "启用" : "暂停", s.control.collection_enabled ? "on" : "off");
  const runMode = s.control.next_run_mode || "auto";
  const runModeLabel = runMode === "new" ? "强制新局一次" : (runMode === "continue" ? "续接旧 Run" : "自动检测");
  setPill("nextRunBadge", runModeLabel, runMode === "new" ? "warn" : "info");
  document.getElementById("modeAuto").className = runMode === "auto" ? "active" : "";
  document.getElementById("modeNew").className = runMode === "new" ? "active" : "";
  document.getElementById("modeContinue").className = runMode === "continue" ? "active" : "";
  renderExport(s.export);

  renderCurrentData(s.current_data);
  renderRuns(s.runs || []);
  renderPolicyEvaluation(s.evaluation || {});
  renderLiveActivity(s.recent_records || []);
  renderRecentRecords(s.recent_records || []);
  renderModelHealth(s.models || {}, s.ai_process || {}, s.control || {}, s.python_runtime || {}, s.monster_profiles || {});
  renderAiLogic(s.ai_logic);
  renderLLMLogic(s.llm && s.llm.logic, llmCfg);

  setPill("trainStatus", s.training.running ? `训练中 ${s.training.started || ""}` : (s.training.finished ? `完成 ${s.training.finished}` : "未运行"), s.training.running ? "warn" : "info");
  document.getElementById("trainOutput").textContent = s.training.output || "暂无输出";
}
function formatBytes(n) {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let v = Number(n), i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i ? 1 : 0)} ${units[i]}`;
}
function renderExport(info) {
  if (!info || !info.filename) {
    setPill("exportBadge", "未导出", "info");
    document.getElementById("exportInfo").textContent = "生成 zip 后，直接把这个文件发给你。";
    return;
  }
  setPill("exportBadge", "已生成", "on");
  document.getElementById("exportInfo").innerHTML =
    `<div><b>${info.filename}</b></div>
     <div>${formatBytes(info.size)}，${info.file_count || 0} 个文件，${info.created || ""}</div>
     <div><a href="/exports/${info.filename}">下载这个数据包</a></div>
     <div class="muted">${info.path || ""}</div>`;
}
function renderCurrentData(data) {
  const run = data && data.active_run;
  if (!run) {
    document.getElementById("currentData").innerHTML = `<div class="muted">${(data && data.warning) || "暂无数据"}</div>`;
    setPill("currentDataBadge", "无数据", "warn");
    return;
  }
  const schemaVersions = (run.schema_versions || []).join(",") || "旧";
  const warnings = (data.warnings || []).map(w => `<div><span class="pill off">注意</span> ${w}</div>`).join("");
  const checks = (run.data_checks || []).map(c => `
    <div class="check-item">
      <div class="check-top">
        <span class="check-title">${c.label || "-"}</span>
        <span class="pill ${checkClass(c.status)}">${checkLabel(c.status)}</span>
      </div>
      <div class="check-detail">${c.detail || "-"}</div>
    </div>
  `).join("");
  setPill("currentDataBadge", run.data_health_label || "体检", dataHealthClass(run.data_health));
  document.getElementById("currentData").innerHTML = `
    <div class="kv"><span>Run</span><code>${run.run_id}</code></div>
    <div class="kv"><span>时间</span><span>${run.last_time || "-"}</span></div>
    <div class="metric-grid">
      <div class="metric"><div class="metric-value">${run.records || 0}</div><div class="metric-label">总记录</div></div>
      <div class="metric"><div class="metric-value">${run.combat || 0}</div><div class="metric-label">战斗记录</div></div>
      <div class="metric"><div class="metric-value">${run.macro || 0}</div><div class="metric-label">宏观记录</div></div>
      <div class="metric"><div class="metric-value">${run.play_card || 0}</div><div class="metric-label">出牌样本</div></div>
      <div class="metric"><div class="metric-value">${run.end_turn || 0}</div><div class="metric-label">结束回合</div></div>
      <div class="metric"><div class="metric-value">A${run.max_act || 0}/F${run.max_floor || 0}</div><div class="metric-label">${run.quality_label || "未知"}</div></div>
      <div class="metric"><div class="metric-value">${run.duration_sec || 0}s</div><div class="metric-label">持续时间</div></div>
      <div class="metric"><div class="metric-value">${run.invalid_actions || 0}</div><div class="metric-label">非法动作</div></div>
      <div class="metric"><div class="metric-value">v${schemaVersions}</div><div class="metric-label">Schema</div></div>
    </div>
    <div class="check-list">${checks}</div>
    <div class="warning-list">${warnings || '<div><span class="pill on">正常</span> 最近 run 有数据写入</div>'}</div>`;
}
function renderModelHealth(models, aiProcess, control, runtime, monsterProfiles) {
  const combat = models.combat || {};
  const macro = models.macro || {};
  const monster = monsterProfiles || {};
  const monsterSummary = monster.summary || {};
  const combatMeta = combat.metadata || {};
  const macroSummary = macro.summary || {};
  const macroMeta = macro.metadata || {};
  const ready = !!combat.ready && !!macro.ready;
  const needsRestart = !!aiProcess.needs_restart;
  const warnings = [];
  if (!combat.ready) warnings.push("战斗 BC 模型缺失，需要重训。");
  if (!macro.ready) warnings.push("宏观 BC 模型缺失，需要先训练宏观模型。");
  if (needsRestart) warnings.push("AI 进程早于当前 ai_agent.py，必须重启 AI 后宏观执行才会生效。");
  if (runtime && runtime.agent_ready === false) warnings.push(`当前 Python 缺少 AI 依赖：${(runtime.missing || []).join(", ")}。网页能开，但启动 AI / 重训会失败。`);
  if (control.macro_enabled && !macro.ready) warnings.push("宏观开关已打开，但宏观模型不可用。");
  if (control.macro_enabled && !control.macro_shop_enabled) warnings.push("商店保护已开启：AI 不会买东西，也不会自动离开商店。");
  setPill("modelBadge", ready ? (needsRestart ? "需重启" : "可用") : "缺模型", ready ? (needsRestart ? "warn" : "on") : "off");

  const restartNotice = needsRestart
    ? `<div class="notice warn"><b>需要重启 AI。</b>当前 AI 进程仍可能在跑旧代码，点击左侧“重启 AI”后宏观模型才会进入运行时。</div>`
    : "";
  const warningHtml = warnings.length
    ? `<div class="warning-list">${warnings.map(w => `<div><span class="pill warn">注意</span> ${w}</div>`).join("")}</div>`
    : `<div class="notice good">战斗模型和宏观模型都已就绪。</div>`;

  document.getElementById("modelHealth").innerHTML = `
    <div class="kv"><span>战斗模型</span><span><span class="pill ${combat.ready ? "on" : "off"}">${combat.ready ? "可用" : "缺失"}</span> 样本 ${combatMeta.samples || "-"}，特征 ${combatMeta.features || "旧版"} ${combat.model && combat.model.mtime ? combat.model.mtime : ""}</span></div>
    <div class="kv"><span>宏观模型</span><span><span class="pill ${macro.ready ? "on" : "off"}">${macro.ready ? "可用" : "缺失"}</span> 样本 ${macroSummary.samples || macroMeta.samples || 0}，动作 ${macroSummary.actions || macroMeta.classes || 0}</span></div>
    <div class="kv"><span>怪物画像</span><span><span class="pill ${monster.ready ? "on" : "warn"}">${monster.ready ? "可用" : "待生成"}</span> 怪物 ${monsterSummary.monsters || 0}，战斗 ${monsterSummary.encounters || 0}，回合样本 ${monsterSummary.monster_turn_rows || 0}</span></div>
    <div class="kv"><span>Python</span><span><span class="pill ${runtime && runtime.agent_ready ? "on" : "warn"}">${runtime && runtime.agent_ready ? "依赖可用" : "缺依赖"}</span> ${escapeHtml((runtime && runtime.executable) || "-")} ${runtime && runtime.version ? `(${runtime.version})` : ""}</span></div>
    <div class="kv"><span>AI 进程</span><span>${aiProcess.pid ? `PID ${aiProcess.pid}` : "未启动"}${aiProcess.started_at ? `，启动 ${aiProcess.started_at}` : ""}</span></div>
    <div class="kv"><span>宏观执行</span><span><span class="pill ${control.macro_enabled ? "warn" : "info"}">${control.macro_enabled ? "开启" : "关闭"}</span> ${control.macro_enabled ? "会自动点地图/奖励/选卡" : "只显示战斗托管"}</span></div>
    <div class="kv"><span>商店保护</span><span><span class="pill ${control.macro_shop_enabled ? "warn" : "on"}">${control.macro_shop_enabled ? "允许购买" : "保护中"}</span> ${control.macro_shop_enabled ? "AI 可买明确商品；不自动删牌" : "AI 不碰商店，避免抢操作"}</span></div>
    ${restartNotice}
    ${warningHtml}`;
}
function renderRuns(runs) {
  setPill("runsBadge", `${runs.length || 0} 条`, runs.length ? "info" : "warn");
  const latest = runs[0];
  if (!latest) {
    document.getElementById("latestRunCard").innerHTML = '<div class="muted">暂无 Run 数据</div>';
    document.getElementById("runs").innerHTML = "<tr><td colspan=9>暂无数据</td></tr>";
    return;
  }
  document.getElementById("latestRunCard").innerHTML = `
    <div class="compact-card-title"><code>${latest.run_id}</code></div>
    <div class="row">
      <span class="pill ${dataHealthClass(latest.data_health)}">${latest.data_health_label || "-"}</span>
      <span class="pill ${latest.discarded ? "off" : "on"}">${latest.discarded ? "丢弃" : "保留"}</span>
      <span class="fine">${latest.last_time || ""}</span>
    </div>
    <div class="fine">Act ${latest.max_act || 0} / Floor ${latest.max_floor || 0}，记录 ${latest.records || 0}，战斗 ${latest.combat || 0}，宏观 ${latest.macro || 0}</div>
    <div class="fine">出牌 ${latest.play_card || 0}，结束回合 ${latest.end_turn || 0}，非法动作 ${latest.invalid_actions || 0}</div>`;
  const rows = runs.map(run => `
    <tr>
      <td class="run-id"><code>${run.run_id}</code><br><span class="fine">${run.last_time || ""}</span></td>
      <td>Act ${run.max_act || 0} / Floor ${run.max_floor || 0}<br><span class="fine">${run.records || 0} 条，C ${run.combat || 0} / M ${run.macro || 0}</span></td>
      <td>出牌 ${run.play_card || 0}<br><span class="fine">回合 ${run.end_turn || 0}，非法 ${run.invalid_actions || 0}</span></td>
      <td>Human ${run.human || 0}<br><span class="fine">AI ${run.ai || 0}</span></td>
      <td>胜 ${run.wins || 0}<br><span class="fine">败 ${run.losses || 0}，v${(run.schema_versions || []).join("/") || "旧"}</span></td>
      <td>
        <select onchange="setQuality('${run.run_id}', this.value)">${qualityOptions(run.quality)}</select>
        <br><span class="fine">${run.quality_manual ? "手动标签" : "自动推断"}</span>
      </td>
      <td><span class="pill ${dataHealthClass(run.data_health)}">${run.data_health_label || "-"}</span><br><span class="fine">${(run.missing_data || []).slice(0, 2).join("、") || "关键项正常"}</span></td>
      <td>${run.discarded ? '<span class="pill off">丢弃</span>' : '<span class="pill on">保留</span>'}</td>
      <td><button onclick="markRun('${run.run_id}', ${!run.discarded})">${run.discarded ? "保留" : "丢弃"}</button></td>
    </tr>`).join("");
  document.getElementById("runs").innerHTML = rows || "<tr><td colspan=9>暂无数据</td></tr>";
}
function renderPolicyEvaluation(evaluation) {
  const policies = evaluation.policies || [];
  setPill("evalBadge", `${evaluation.total_runs || 0} 个 run`, policies.length ? "info" : "warn");
  const rows = policies.map(p => `
    <tr>
      <td>${p.policy_label || p.policy_name || "-"}</td>
      <td>${p.runs || 0}<br><span class="fine">保留 ${p.kept_runs || 0} / 丢弃 ${p.discarded_runs || 0}</span></td>
      <td>${p.avg_floor || 0} / ${p.max_floor || 0}<br><span class="fine">最高 Act ${p.max_act || 0}</span></td>
      <td>胜 ${p.wins || 0}<br><span class="fine">败 ${p.losses || 0}</span></td>
      <td>非法 ${p.invalid_actions || 0}<br><span class="fine">卡住 ${p.stuck_count || 0}，缺数据 ${p.data_missing || 0}</span></td>
      <td><span class="fine">${p.latest_time || "-"}</span></td>
    </tr>`).join("");
  document.getElementById("policyEval").innerHTML = rows || "<tr><td colspan=6>暂无可评测 run</td></tr>";
}
function renderLiveActivity(records) {
  const items = (records || []).slice(0, 5);
  setPill("activityBadge", items.length ? `最近 ${items.length} 条` : "暂无记录", items.length ? "info" : "warn");
  const root = document.getElementById("liveActivity");
  if (!items.length) {
    root.innerHTML = '<div class="muted">暂无采集记录。进入游戏后这里会显示出牌、药水、选卡、领奖和地图动作。</div>';
    return;
  }
  root.innerHTML = items.map(r => {
    const tone = ["on", "warn", "off", "info"].includes(r.tone) ? r.tone : "info";
    const toneClass = `tone-${tone}`;
    const label = escapeHtml(r.label || r.action_type || r.type || "记录");
    const summary = escapeHtml(r.summary || r.action_type || r.type || "记录");
    const detail = escapeHtml(r.detail || r.file || "");
    return `
      <div class="activity-item ${toneClass}">
        <div class="activity-meta">
          <span class="activity-label">${label}</span>
          <span class="activity-time">${escapeHtml(r.time || "")}</span>
        </div>
        <div class="activity-title">${summary}</div>
        <div class="activity-detail">${detail}</div>
      </div>`;
  }).join("");
}
function renderRecentRecords(records) {
  const typeLabels = {
    game_start: "开局",
    game_resume: "续局",
    battle_start: "战斗开始",
    turn_start: "回合开始",
    action: "战斗动作",
    macro_action: "宏观动作",
    turn_end: "回合结束",
    battle_end: "战斗结算",
    run_end: "Run 结束"
  };
  document.getElementById("recentRecords").innerHTML = records.slice(0, 12).map(r => `
    <tr><td>${escapeHtml(r.time || "")}</td><td>${escapeHtml(typeLabels[r.type] || r.type || "")}</td><td>${escapeHtml(r.source_label || r.source || "")}</td><td>${escapeHtml(r.summary || r.action_type || r.result || "")}<br><span class="fine">${escapeHtml(r.detail || "")}</span></td><td>${escapeHtml(r.file || "")}</td></tr>
  `).join("") || "<tr><td colspan=5>暂无记录</td></tr>";
}
function renderAiLogic(logic) {
  if (!logic || !logic.timestamp) {
    document.getElementById("aiLogic").innerHTML = '<div class="muted">暂无 AI 决策。启动 AI 并进入战斗后，这里会显示候选动作、概率和为什么没打出去。</div>';
    document.getElementById("aiTopActions").innerHTML = "<tr><td colspan=3>暂无概率</td></tr>";
    setPill("aiDecisionBadge", "无决策", "warn");
    return;
  }
  const mode = logic.mode === "macro" ? "macro" : "combat";
  const payload = logic.payload ? JSON.stringify(logic.payload) : "-";
  setPill("aiDecisionBadge", mode === "macro" ? `宏观 ${logic.chosen_action || "-"}` : (logic.chosen_action || "有决策"), mode === "macro" ? "warn" : "info");
  if (mode === "macro") {
    document.getElementById("aiLogic").innerHTML = `
      <div class="kv"><span>类型</span><span class="strong">宏观决策</span></div>
      <div class="kv"><span>时间</span><span>${logic.time || "-"}</span></div>
      <div class="kv"><span>场景</span><span>${logic.state_type || "-"}</span></div>
      <div class="kv"><span>执行</span><span class="strong">${logic.chosen_action || "-"}</span></div>
      <div class="kv"><span>Payload</span><code>${payload}</code></div>
      <div class="kv"><span>原因</span><span>${logic.reason || "-"}</span></div>`;
  } else {
    document.getElementById("aiLogic").innerHTML = `
      <div class="kv"><span>类型</span><span class="strong">战斗决策</span></div>
      <div class="kv"><span>时间</span><span>${logic.time || "-"}</span></div>
      <div class="kv"><span>执行</span><span class="strong">${logic.chosen_action || "-"}</span></div>
      <div class="kv"><span>手牌</span><span>${(logic.hand || []).join(", ") || "-"}</span></div>
      <div class="kv"><span>Payload</span><code>${payload}</code></div>
      <div class="kv"><span>原因</span><span>${logic.reason || "-"}</span></div>`;
  }
  document.getElementById("aiTopActions").innerHTML = (logic.top_actions || []).map(a => `
    <tr><td>${a.action}</td><td>${a.confidence}%</td><td>${a.marker || ""}</td></tr>
  `).join("") || "<tr><td colspan=3>暂无概率</td></tr>";
}
function renderLLMLogic(logic, cfg) {
  if (!logic || !logic.timestamp) {
    setPill("llmDecisionBadge", cfg && cfg.enabled ? "等待决策" : "未启用", cfg && cfg.enabled ? "warn" : "info");
    document.getElementById("llmLogic").innerHTML = '<div class="muted">暂无 LLM 决策。配置模型并启动 LLM 后，这里会显示建议、校验结果和执行结果。</div>';
    return;
  }
  if (logic.status === "error") {
    setPill("llmDecisionBadge", "错误", "off");
    document.getElementById("llmLogic").innerHTML = `
      <div class="kv"><span>时间</span><span>${logic.time || "-"}</span></div>
      <div class="kv"><span>错误</span><span>${logic.error || "-"}</span></div>`;
    return;
  }
  if (logic.status === "cooldown") {
    setPill("llmDecisionBadge", "冷却", "warn");
    document.getElementById("llmLogic").innerHTML = `
      <div class="kv"><span>重试</span><span>${logic.retry_after_sec || "-"} 秒</span></div>
      <div class="kv"><span>原因</span><span>${logic.error || logic.message || "-"}</span></div>`;
    return;
  }
  if (logic.status === "waiting") {
    setPill("llmDecisionBadge", "等待战斗", "info");
    document.getElementById("llmLogic").innerHTML = `
      <div class="kv"><span>模式</span><span>${logic.mode || "-"}</span></div>
      <div class="kv"><span>场景</span><span>${logic.state_type || "-"}</span></div>
      <div class="kv"><span>状态</span><span>${logic.message || "等待玩家战斗行动阶段，不请求模型。"}</span></div>`;
    return;
  }
  const d = logic.decision || {};
  const payload = logic.payload ? JSON.stringify(logic.payload) : "-";
  const selected = logic.selected_candidate || {};
  const suggestion = d.candidate_id || d.action || "-";
  const args = d.candidate_id ? {candidate_id: d.candidate_id} : (d.args || {});
  const raw = logic.raw_response ? escapeHtml(String(logic.raw_response).slice(0, 1200)) : "-";
  const combatEval = logic.combat_eval || {};
  const pile = logic.pile_summary || {};
  const potionOps = logic.potion_opportunities || [];
  const potionText = potionOps.map(p => `${p.name || p.id || "药水"}:${p.trigger || "-"}`).join(", ") || "-";
  setPill("llmDecisionBadge", logic.executed ? "已执行" : "建议", logic.executed ? "on" : "info");
  document.getElementById("llmLogic").innerHTML = `
    <div class="kv"><span>时间</span><span>${logic.time || "-"}</span></div>
    <div class="kv"><span>模式</span><span>${logic.mode || "-"}</span></div>
    <div class="kv"><span>动作选择</span><span>${logic.action_selection_mode || (cfg && cfg.action_selection_mode) || "-"}</span></div>
    <div class="kv"><span>模型</span><span>${logic.model || "-"}</span></div>
    <div class="kv"><span>场景</span><span>${logic.state_type || "-"}</span></div>
    <div class="kv"><span>建议</span><span class="strong">${suggestion}</span></div>
    <div class="kv"><span>候选</span><span>${selected.id || "-"} / ${(logic.candidate_actions || []).length} 个</span></div>
    <div class="kv"><span>战斗判断</span><span>${combatEval.priority || "-"} / ${combatEval.risk_level || "-"}</span></div>
    <div class="kv"><span>攻防估算</span><span>伤害 ${combatEval.turn_max_damage ?? "-"}，格挡 ${combatEval.turn_max_block ?? "-"}，斩杀差 ${combatEval.lethal_gap ?? "-"}</span></div>
    <div class="kv"><span>药水机会</span><span>${escapeHtml(potionText)}</span></div>
    <div class="kv"><span>牌堆</span><span>抽 ${pile.draw_count ?? "-"} / 弃 ${pile.discard_count ?? "-"} / 消耗 ${pile.exhaust_count ?? "-"}</span></div>
    <div class="kv"><span>参数</span><code>${JSON.stringify(args)}</code></div>
    <div class="kv"><span>校验</span><span>${logic.validation || "-"}</span></div>
    <div class="kv"><span>执行</span><span>${logic.executed ? "是" : "否"} ${logic.ok === false ? "(失败)" : ""}</span></div>
    <div class="kv"><span>Payload</span><code>${payload}</code></div>
    <div class="kv"><span>手牌</span><span>${(logic.hand_summary || []).join(", ") || "-"}</span></div>
    <div class="kv"><span>理由</span><span>${d.reason || "-"}</span></div>
    <div class="kv"><span>原始回复</span><code>${raw}</code></div>`;
}
function renderLLMProfiles(profiles, activeId) {
  llmProfilesCache = profiles || [];
  const rows = (profiles || []).map(p => `
    <tr>
      <td>${p.id === activeId ? '<span class="pill on">当前</span> ' : ''}${escapeHtml(p.name || "-")}</td>
      <td><code>${escapeHtml(p.base_url || "-")}</code></td>
      <td>${escapeHtml(p.model || "-")}</td>
      <td>${p.has_api_key ? `****${escapeHtml(p.key_tail || "")}` : "-"}</td>
      <td class="row">
        <button onclick="useLLMProfile('${p.id}')">使用</button>
        <button onclick="openLLMProfileEditor('${p.id}')">修改</button>
        <button onclick="deleteLLMProfile('${p.id}')">删除</button>
      </td>
    </tr>`).join("");
  document.getElementById("llmProfiles").innerHTML = rows || "<tr><td colspan=5>暂无保存的 API 配置</td></tr>";
}
async function saveControl() {
  await api("/api/control", {
    ai_enabled: document.getElementById("ai_enabled").checked,
    macro_enabled: document.getElementById("macro_enabled").checked,
    macro_shop_enabled: document.getElementById("macro_shop_enabled").checked,
    collection_enabled: document.getElementById("collection_enabled").checked,
    record_ai_actions: document.getElementById("record_ai_actions").checked,
    include_ai_in_training: document.getElementById("include_ai_in_training").checked,
    min_training_quality: document.getElementById("min_training_quality").value
  });
  refresh();
}
async function saveLLMConfig() {
  const apiKey = document.getElementById("llm_api_key").value.trim();
  const body = {
    enabled: document.getElementById("llm_enabled").checked,
    mode: document.getElementById("llm_mode").value,
    action_selection_mode: document.getElementById("llm_action_selection_mode").value,
    execute_combat: document.getElementById("llm_execute_combat").checked,
    base_url: document.getElementById("llm_base_url").value.trim(),
    model: document.getElementById("llm_model").value.trim()
  };
  if (apiKey) body.api_key = apiKey;
  const result = await api("/api/llm/config", body);
  llmFormDirty = false;
  document.getElementById("llm_api_key").value = "";
  if (result.config) applyLLMConfigToForm(result.config);
  await refresh();
}
async function saveLLMProfile() {
  const apiKey = document.getElementById("llm_api_key").value.trim();
  const body = {
    enabled: document.getElementById("llm_enabled").checked,
    mode: document.getElementById("llm_mode").value,
    action_selection_mode: document.getElementById("llm_action_selection_mode").value,
    execute_combat: document.getElementById("llm_execute_combat").checked,
    base_url: document.getElementById("llm_base_url").value.trim(),
    model: document.getElementById("llm_model").value.trim(),
    save_profile: true
  };
  if (apiKey) body.api_key = apiKey;
  const result = await api("/api/llm/config", body);
  llmFormDirty = false;
  document.getElementById("llm_api_key").value = "";
  if (result.config) applyLLMConfigToForm(result.config);
  await refresh();
}
async function useLLMProfile(profileId) {
  await api("/api/llm/profile/use", {profile_id: profileId});
  llmFormDirty = false;
  await refresh();
}
async function deleteLLMProfile(profileId) {
  await api("/api/llm/profile/delete", {profile_id: profileId});
  await refresh();
}
function openLLMProfileEditor(profileId) {
  const profile = llmProfilesCache.find(p => p.id === profileId);
  if (!profile) return;
  document.getElementById("edit_profile_id").value = profile.id || "";
  document.getElementById("edit_profile_name").value = profile.name || "";
  document.getElementById("edit_base_url").value = profile.base_url || "";
  document.getElementById("edit_model").value = profile.model || "";
  document.getElementById("edit_api_key").value = "";
  document.getElementById("llmProfileEditor").classList.add("open");
}
function closeLLMProfileEditor(event) {
  if (event && event.target && event.target.id !== "llmProfileEditor") return;
  document.getElementById("llmProfileEditor").classList.remove("open");
}
async function saveLLMProfileEdit() {
  const body = {
    profile_id: document.getElementById("edit_profile_id").value,
    name: document.getElementById("edit_profile_name").value.trim(),
    base_url: document.getElementById("edit_base_url").value.trim(),
    model: document.getElementById("edit_model").value.trim()
  };
  const apiKey = document.getElementById("edit_api_key").value.trim();
  if (apiKey) body.api_key = apiKey;
  await api("/api/llm/profile/update", body);
  closeLLMProfileEditor();
  llmFormDirty = false;
  await refresh();
}
async function testLLM() {
  const button = document.getElementById("llmTestButton");
  button.disabled = true;
  try {
    await saveLLMConfig();
    document.getElementById("llmTestResult").textContent = "正在测试...";
    const result = await api("/api/llm/test", {});
    document.getElementById("llmTestResult").textContent = result.message || result.status || "测试完成";
  } finally {
    button.disabled = false;
  }
}
async function setRunMode(mode) { await api("/api/control", {next_run_mode: mode}); refresh(); }
async function startAI(){ await api("/api/ai/start", {}); refresh(); }
async function stopAI(){ await api("/api/ai/stop", {}); refresh(); }
async function restartAI(){ await api("/api/ai/restart", {}); refresh(); }
async function startLLM(){ await saveLLMConfig(); await api("/api/llm/start", {}); refresh(); }
async function stopLLM(){ await api("/api/llm/stop", {}); refresh(); }
async function train(){ await api("/api/train", {}); refresh(); }
async function proceed(){ await api("/api/proceed", {}); refresh(); }
async function markRun(run_id, discarded){ await api("/api/run", {run_id, discarded}); refresh(); }
async function setQuality(run_id, quality){ await api("/api/quality", {run_id, quality}); refresh(); }
async function exportData(){
  setPill("exportBadge", "打包中", "warn");
  document.getElementById("exportInfo").textContent = "正在压缩数据，请稍等...";
  const result = await api("/api/export", {});
  if (result.status === "ok") {
    renderExport(result.export);
  } else {
    setPill("exportBadge", "失败", "off");
    document.getElementById("exportInfo").textContent = result.error || "导出失败";
  }
}
syncModuleUI();
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def _file(self, path, content_type="application/octet-stream", download=True):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload, ensure_ascii=False), "application/json")

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        if self.path == "/":
            self._send(200, INDEX_HTML, "text/html")
        elif self.path == "/api/ping":
            self._json(200, {"status": "ok", "time": datetime.now().isoformat(timespec="seconds")})
        elif self.path == "/api/status":
            self._json(200, status_payload())
        elif self.path.startswith("/assets/"):
            name = Path(self.path.split("?", 1)[0].split("/assets/", 1)[1]).name
            path = ASSETS_DIR / name
            content_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}
            if path.exists() and path.is_file() and path.suffix.lower() in content_types:
                self._file(path, content_types[path.suffix.lower()], download=False)
            else:
                self._json(404, {"error": "asset not found"})
        elif self.path.startswith("/exports/"):
            name = Path(self.path.split("/exports/", 1)[1]).name
            path = EXPORT_DIR / name
            if path.exists() and path.is_file() and path.suffix == ".zip":
                self._file(path, "application/zip")
            else:
                self._json(404, {"error": "export not found"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/api/control":
                self._json(200, {"control": update_control(body)})
            elif self.path == "/api/llm/config":
                self._json(200, {"config": update_llm_config(body)})
            elif self.path == "/api/llm/profile/use":
                self._json(200, {"config": use_llm_profile(body["profile_id"])})
            elif self.path == "/api/llm/profile/delete":
                self._json(200, {"config": delete_llm_profile(body["profile_id"])})
            elif self.path == "/api/llm/profile/update":
                self._json(200, {"config": update_llm_profile(body)})
            elif self.path == "/api/llm/test":
                self._json(200, test_llm_connection())
            elif self.path == "/api/llm/start":
                self._json(200, start_llm())
            elif self.path == "/api/llm/stop":
                self._json(200, stop_llm())
            elif self.path == "/api/ai/start":
                self._json(200, start_ai())
            elif self.path == "/api/ai/stop":
                self._json(200, stop_ai())
            elif self.path == "/api/ai/restart":
                self._json(200, restart_ai())
            elif self.path == "/api/train":
                self._json(200, run_training_background())
            elif self.path == "/api/export":
                self._json(200, export_database_package())
            elif self.path == "/api/run":
                self._json(200, set_run_discarded(body["run_id"], bool(body.get("discarded"))))
            elif self.path == "/api/quality":
                self._json(200, set_run_label(body["run_id"], body.get("quality", "unknown"), body.get("note", "")))
            elif self.path == "/api/proceed":
                try:
                    self._json(200, post_game_action({"action": "proceed"}))
                except URLError as exc:
                    self._json(500, {"error": str(exc)})
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def log_message(self, fmt, *args):
        return


def main():
    CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONTROL_PATH.exists():
        write_json(CONTROL_PATH, DEFAULT_CONTROL)
    if not LLM_CONFIG_PATH.exists():
        write_json(LLM_CONFIG_PATH, DEFAULT_LLM_CONFIG)
    ensure_llm_profiles_initialized()
    if not DISCARDED_PATH.exists():
        write_json(DISCARDED_PATH, {"discarded": []})
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("STS2 AI control panel: http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
