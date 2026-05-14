import base64
import json
import io
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
from pathlib import Path, PurePosixPath
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
DOCS_DIR = WORKSPACE / "docs"
MODEL_ZOO_DIR = AI_DIR / "ModelZoo"
AUTO_MODEL_KEEP_LIMIT = 5
CONTROL_PATH = AI_DIR / "control_state.json"
AI_LOGIC_PATH = AI_DIR / "ai_logic_state.json"
LLM_CONFIG_PATH = AI_DIR / "model_config.json"
LLM_LOGIC_PATH = AI_DIR / "llm_logic_state.json"
SELF_PLAY_RUNNER_PATH = AI_DIR / "self_play_runner.py"
SELF_PLAY_STATE_PATH = AI_DIR / "self_play_state.json"
SELF_PLAY_SCORES_PATH = DATA_DIR / "self_play_scores.json"
PPO_DIR = AI_DIR / "ProcessedPPOParams"
PPO_ROLLOUT_DIR = DATA_DIR / "PPO"
DISCARDED_PATH = DATA_DIR / "discarded_runs.json"
SERVER_STATE_PATH = AI_DIR / "control_panel_state.json"
DEFAULT_PYTHON_EXE = WORKSPACE / ".venv" / "Scripts" / "python.exe"
AGENT_PATH = AI_DIR / "ai_agent.py"
LLM_AGENT_PATH = AI_DIR / "llm_agent.py"
API_URL = "http://127.0.0.1:15526/api/v1/singleplayer"

MODEL_ARTIFACTS = {
    "ProcessedParams": [
        "bc_model_best.pth",
        "candidate_bc_model_best.pth",
        "vocab.json",
        "metadata.json",
        "candidate_metadata.json",
    ],
    "ProcessedMacroParams": [
        "macro_bc_model_best.pth",
        "vocab.json",
        "metadata.json",
        "training_summary.json",
    ],
}
OPTIONAL_MODEL_ARTIFACTS = {
    "ProcessedParams": [
        "candidate_rl_model_best.pth",
        "candidate_rl_metadata.json",
    ],
}

MODEL_IMPORT_MAX_BYTES = 128 * 1024 * 1024


def resolve_panel_port(default=8765):
    for value in (os.environ.get("STS2_AI_PANEL_PORT"), sys.argv[1] if len(sys.argv) > 1 else ""):
        try:
            port = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= port <= 65535:
            return port
    return default


def python_exe_runs(candidate):
    if not candidate:
        return False
    try:
        proc = subprocess.run([str(candidate), "--version"], capture_output=True, text=True, timeout=5)
        return proc.returncode == 0 and "Python" in ((proc.stdout or "") + (proc.stderr or ""))
    except Exception:
        return False


def resolve_venv_base_python():
    cfg_path = WORKSPACE / ".venv" / "pyvenv.cfg"
    try:
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().lower() == "executable":
                return value.strip()
    except Exception:
        pass
    return ""


def read_venv_version():
    cfg_path = WORKSPACE / ".venv" / "pyvenv.cfg"
    try:
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().lower() == "version":
                parts = value.strip().split(".")
                if len(parts) >= 2:
                    return ".".join(parts[:2])
    except Exception:
        pass
    return ""


def resolve_python_exe():
    candidates = [
        os.environ.get("STS2_AI_PYTHON"),
        str(DEFAULT_PYTHON_EXE),
        resolve_venv_base_python(),
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


def python_subprocess_env():
    env = os.environ.copy()
    venv_python = DEFAULT_PYTHON_EXE.resolve()
    venv_base_raw = resolve_venv_base_python()
    try:
        venv_base_python = Path(venv_base_raw).resolve() if venv_base_raw else None
    except Exception:
        venv_base_python = None
    try:
        selected_python = PYTHON_EXE.resolve()
    except Exception:
        selected_python = PYTHON_EXE
    venv_site = WORKSPACE / ".venv" / "Lib" / "site-packages"
    selected_version = f"{sys.version_info.major}.{sys.version_info.minor}" if selected_python == Path(sys.executable) else ""
    venv_version = read_venv_version()
    use_venv_site = (
        selected_python == venv_python
        or (venv_base_python is not None and selected_python == venv_base_python)
        or (selected_python == Path(sys.executable) and bool(venv_version) and selected_version == venv_version)
    )
    if selected_python != venv_python and venv_site.exists() and use_venv_site:
        existing = env.get("PYTHONPATH", "")
        parts = [str(venv_site)]
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env

DEFAULT_CONTROL = {
    "ai_enabled": False,
    "macro_enabled": False,
    "macro_shop_enabled": False,
    "macro_card_reward_weight": 0.35,
    "record_ai_actions": True,
    "include_ai_in_training": False,
    "game_speed_enabled": False,
    "game_speed_multiplier": 2.0,
    "ai_min_training_quality": "partial_act1",
    "ai_accept_failed_after_act1": True,
    "ai_require_no_invalid_actions": True,
    "active_model_id": "local",
    "next_run_mode": "auto",
    "collection_enabled": True,
    "collection_disabled_since": None,
    "collection_disabled_ranges": [],
    "min_training_quality": "unknown",
    "exploration_enabled": False,
    "exploration_mode": "aggressive",
    "self_play_constraint_mode": "explore",
    "combat_exploration_epsilon": 0.35,
    "macro_exploration_epsilon": 0.25,
    "exploration_top_k": 5,
    "exploration_temperature": 1.35,
    "self_play_character": "IRONCLAD",
    "self_play_ascension": 0,
    "self_play_seed": "",
    "policy_mode": "current_rl",
    "ppo_seed_mode": "fixed",
    "ppo_fixed_seed": "101",
    "self_play_target_runs": 0,
    "self_play_train_every_admitted_runs": 5,
    "self_play_max_run_minutes": 75,
    "self_play_stall_seconds": 120,
    "self_play_game_speed_multiplier": 3.0,
    "option_card_scorer": {"mode": "shadow"},
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
UPDATE_LOCK = threading.Lock()
LLM_TEST_LOCK = threading.Lock()
LLM_TEST_COOLDOWN_SEC = 60
LAST_TRAIN = {"running": False, "started": None, "finished": None, "output": ""}
LAST_PPO_TRAIN = {"running": False, "started": None, "finished": None, "output": ""}
LAST_EXPORT = {"path": None, "filename": None, "created": None, "size": 0, "file_count": 0}
LAST_UPDATE = {"running": False, "started": None, "finished": None, "status": "idle", "output": ""}
LLM_TEST_STATE = {"running": False, "last_started": 0.0, "last_finished": 0.0}
GAME_CACHE = {"state": None, "ts": 0.0}
PYTHON_RUNTIME_CACHE = {"ts": 0.0, "data": None}
PYTHON_RUNTIME_LOCK = threading.Lock()
DASHBOARD_DATA_CACHE = {"ts": 0.0, "data": None}
DASHBOARD_DATA_LOCK = threading.Lock()
APP_VERSION_CACHE = {"ts": 0.0, "data": None}
PYTHON_RUNTIME_CACHE_TTL_SEC = 300
DASHBOARD_DATA_CACHE_TTL_SEC = 8
REQUIRED_AGENT_MODULES = ["requests", "torch", "numpy", "colorama"]
REQUIRED_TRAINING_MODULES = ["numpy", "torch"]
UPDATE_SCRIPT_PATH = WORKSPACE / "tools" / "update_workspace.ps1"


def clamp_int(value, default, minimum, maximum):
    try:
        return max(minimum, min(int(value), maximum))
    except (TypeError, ValueError):
        return default


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


def clamp_game_speed_multiplier(value):
    try:
        return max(1.0, min(float(value), 6.0))
    except (TypeError, ValueError):
        return DEFAULT_CONTROL["game_speed_multiplier"]


def update_control(patch):
    data = read_control()
    for key in DEFAULT_CONTROL:
        if key in patch:
            if key == "next_run_mode":
                data[key] = patch[key] if patch[key] in ("auto", "new", "continue") else "auto"
            elif key == "min_training_quality":
                data[key] = patch[key] if patch[key] in QUALITY_ORDER else "unknown"
            elif key == "ai_min_training_quality":
                data[key] = patch[key] if patch[key] in QUALITY_ORDER else "partial_act1"
            elif key == "exploration_mode":
                mode = str(patch[key] or "aggressive").strip().lower()
                data[key] = mode if mode in ("aggressive", "off") else "aggressive"
            elif key == "self_play_constraint_mode":
                mode = str(patch[key] or "explore").strip().lower()
                data[key] = mode if mode in ("guarded", "explore", "free") else "explore"
            elif key == "self_play_character":
                character = str(patch[key] or "IRONCLAD").strip().upper()
                data[key] = character if character == "IRONCLAD" else "IRONCLAD"
            elif key == "self_play_seed":
                data[key] = str(patch[key] or "").strip().upper()
            elif key == "policy_mode":
                mode = str(patch[key] or "current_rl").strip().lower()
                data[key] = mode if mode in ("current_rl", "ppo_experiment", "ppo_best") else "current_rl"
            elif key == "ppo_seed_mode":
                mode = str(patch[key] or "fixed").strip().lower()
                data[key] = mode if mode in ("fixed", "random") else "fixed"
            elif key == "ppo_fixed_seed":
                data[key] = str(patch[key] or "101").strip().upper() or "101"
            elif key == "option_card_scorer":
                setting = patch[key] if isinstance(patch[key], dict) else {"mode": patch[key]}
                mode = str(setting.get("mode") or "shadow").strip().lower()
                data[key] = {"mode": mode if mode in ("off", "shadow", "active", "active_canary") else "shadow"}
            elif key == "active_model_id":
                data[key] = str(patch[key] or "local")
            elif key == "macro_card_reward_weight":
                try:
                    data[key] = max(0.0, min(float(patch[key]), 2.0))
                except (TypeError, ValueError):
                    data[key] = DEFAULT_CONTROL[key]
            elif key == "game_speed_multiplier":
                data[key] = clamp_game_speed_multiplier(patch[key])
            elif key == "self_play_game_speed_multiplier":
                data[key] = clamp_game_speed_multiplier(patch[key])
            elif key == "combat_exploration_epsilon":
                try:
                    data[key] = max(0.0, min(float(patch[key]), 1.0))
                except (TypeError, ValueError):
                    data[key] = DEFAULT_CONTROL[key]
            elif key == "macro_exploration_epsilon":
                try:
                    data[key] = max(0.0, min(float(patch[key]), 1.0))
                except (TypeError, ValueError):
                    data[key] = DEFAULT_CONTROL[key]
            elif key == "exploration_top_k":
                data[key] = clamp_int(patch[key], DEFAULT_CONTROL[key], 1, 12)
            elif key == "exploration_temperature":
                try:
                    data[key] = max(0.1, min(float(patch[key]), 5.0))
                except (TypeError, ValueError):
                    data[key] = DEFAULT_CONTROL[key]
            elif key == "self_play_ascension":
                data[key] = clamp_int(patch[key], DEFAULT_CONTROL[key], 0, 20)
            elif key == "self_play_target_runs":
                data[key] = clamp_int(patch[key], DEFAULT_CONTROL[key], 0, 500)
            elif key == "self_play_train_every_admitted_runs":
                data[key] = clamp_int(patch[key], DEFAULT_CONTROL[key], 1, 100)
            elif key == "self_play_max_run_minutes":
                data[key] = clamp_int(patch[key], DEFAULT_CONTROL[key], 5, 240)
            elif key == "self_play_stall_seconds":
                data[key] = clamp_int(patch[key], DEFAULT_CONTROL[key], 15, 1800)
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


def provisional_python_runtime_status():
    modules = sorted(set(REQUIRED_AGENT_MODULES + REQUIRED_TRAINING_MODULES))
    return {
        "executable": str(PYTHON_EXE),
        "version": "",
        "ok": False,
        "checking": True,
        "missing": [],
        "agent_ready": None,
        "training_ready": None,
        "message": "正在检查 Python 环境",
    }


def _refresh_python_runtime_cache(blocking=True):
    acquired = PYTHON_RUNTIME_LOCK.acquire(blocking=blocking)
    if not acquired:
        return False
    try:
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
            "checking": False,
            "missing": modules,
            "agent_ready": False,
            "training_ready": False,
            "message": "",
        }
        try:
            env = python_subprocess_env()
            proc = subprocess.run(
                [str(PYTHON_EXE), "-c", code],
                cwd=str(WORKSPACE),
                capture_output=True,
                text=True,
                timeout=12,
                env=env,
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

        PYTHON_RUNTIME_CACHE.update({"ts": time.time(), "data": status})
        return True
    finally:
        PYTHON_RUNTIME_LOCK.release()


def python_runtime_status(force=False, background=False):
    now = time.time()
    cached = PYTHON_RUNTIME_CACHE.get("data")
    if cached and not force and now - float(PYTHON_RUNTIME_CACHE.get("ts") or 0) < PYTHON_RUNTIME_CACHE_TTL_SEC:
        return cached
    if background:
        if not PYTHON_RUNTIME_LOCK.locked():
            threading.Thread(target=_refresh_python_runtime_cache, kwargs={"blocking": False}, daemon=True).start()
        return cached or provisional_python_runtime_status()

    _refresh_python_runtime_cache(blocking=True)
    return PYTHON_RUNTIME_CACHE.get("data") or provisional_python_runtime_status()

def post_game_action(payload):
    req = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=2) as resp:
        return json.loads(resp.read().decode("utf-8"))


def apply_game_speed(body=None):
    control = read_control()
    enabled = bool(control.get("game_speed_enabled", False))
    multiplier = clamp_game_speed_multiplier(control.get("game_speed_multiplier", 2.0))

    if isinstance(body, dict):
        if "enabled" in body:
            enabled = bool(body.get("enabled"))
        if "multiplier" in body:
            multiplier = clamp_game_speed_multiplier(body.get("multiplier"))
        elif "speed" in body:
            multiplier = clamp_game_speed_multiplier(body.get("speed"))

    speed = multiplier if enabled else 1.0
    try:
        result = post_game_action({"action": "set_game_speed", "enabled": enabled, "speed": speed})
        if isinstance(result, dict):
            result["configured_enabled"] = enabled
            result["configured_multiplier"] = multiplier
        return result
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "configured_enabled": enabled,
            "configured_multiplier": multiplier,
            "speed": speed,
        }


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
        "history_completed_runs": 0,
        "history_admitted_runs": 0,
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


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_seed(value):
    return str(value or "").strip().upper()


def _normalize_policy_mode(value):
    mode = str(value or "current_rl").strip().lower()
    return mode if mode in ("current_rl", "ppo_experiment", "ppo_best") else "current_rl"


def _normalize_ppo_seed_mode(value):
    mode = str(value or "fixed").strip().lower()
    return mode if mode in ("fixed", "random") else "fixed"


def _effective_self_play_seed(control):
    control = control or {}
    policy_mode = _normalize_policy_mode(control.get("policy_mode"))
    ppo_seed_mode = _normalize_ppo_seed_mode(control.get("ppo_seed_mode"))
    if policy_mode in ("ppo_experiment", "ppo_best") and ppo_seed_mode == "fixed":
        return _normalize_seed(control.get("ppo_fixed_seed") or DEFAULT_CONTROL["ppo_fixed_seed"])
    return _normalize_seed(control.get("self_play_seed"))


def _next_training_in(admitted_runs, interval):
    interval = max(1, _safe_int(interval, 5))
    admitted_runs = max(0, _safe_int(admitted_runs, 0))
    if admitted_runs <= 0:
        return interval
    remainder = admitted_runs % interval
    return 0 if remainder == 0 else interval - remainder


def self_play_history_summary(control):
    seed = _effective_self_play_seed(control)
    interval = _safe_int((control or {}).get("self_play_train_every_admitted_runs"), DEFAULT_CONTROL["self_play_train_every_admitted_runs"])
    raw = read_json(SELF_PLAY_SCORES_PATH, {})
    scores = raw.get("scores") if isinstance(raw, dict) else {}
    if not isinstance(scores, dict):
        return {
            "completed_runs": 0,
            "admitted_runs": 0,
            "recent_scores": [],
            "last_score": None,
            "pending_training_runs": _next_training_in(0, interval),
        }
    rows = []
    for score in scores.values():
        if not isinstance(score, dict):
            continue
        if seed and _normalize_seed(score.get("seed")) != seed:
            continue
        rows.append(score)
    rows.sort(key=lambda item: str(item.get("updated_at") or ""))
    admitted = sum(1 for item in rows if item.get("admitted"))
    use_completed_interval = _normalize_policy_mode((control or {}).get("policy_mode")) in ("ppo_experiment", "ppo_best")
    return {
        "completed_runs": len(rows),
        "admitted_runs": admitted,
        "recent_scores": list(reversed(rows[-8:])),
        "last_score": rows[-1] if rows else None,
        "pending_training_runs": _next_training_in(len(rows) if use_completed_interval else admitted, interval),
    }


def read_self_play_state():
    data = default_self_play_state()
    data.update(read_json(SELF_PLAY_STATE_PATH, {}))
    return data


def get_self_play_pid():
    state = read_json(SERVER_STATE_PATH, {})
    pid = state.get("self_play_pid")
    return int(pid) if pid and pid_is_running(pid) else None


def self_play_status_payload():
    state = read_self_play_state()
    server_state = read_json(SERVER_STATE_PATH, {})
    pid = get_self_play_pid()
    control = read_control()
    history = self_play_history_summary(control)
    state_seed = _normalize_seed(state.get("current_seed") or _effective_self_play_seed(control))
    control_seed = _effective_self_play_seed(control)
    if state_seed == control_seed:
        state["history_completed_runs"] = int(history.get("completed_runs") or 0)
        state["history_admitted_runs"] = int(history.get("admitted_runs") or 0)
    if (
        state_seed == control_seed
        and not state.get("started_at")
        and _safe_int(history.get("completed_runs")) > _safe_int(state.get("completed_runs"))
    ):
        state["recent_scores"] = history.get("recent_scores") or []
        state["last_score"] = history.get("last_score")
    if state_seed == control_seed and not state.get("running") and not state.get("started_at"):
        state["pending_training_runs"] = int(history.get("pending_training_runs") or 0)
    if not pid and state.get("running"):
        state["running"] = False
        if not state.get("finished_at"):
            state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    state["pid"] = pid
    state["started_at"] = server_state.get("self_play_started_at", state.get("started_at", ""))
    state["config"] = {
        "character": control.get("self_play_character", DEFAULT_CONTROL["self_play_character"]),
        "ascension": control.get("self_play_ascension", DEFAULT_CONTROL["self_play_ascension"]),
        "policy_mode": _normalize_policy_mode(control.get("policy_mode")),
        "ppo_seed_mode": _normalize_ppo_seed_mode(control.get("ppo_seed_mode")),
        "ppo_fixed_seed": _normalize_seed(control.get("ppo_fixed_seed") or DEFAULT_CONTROL["ppo_fixed_seed"]),
        "target_runs": control.get("self_play_target_runs", DEFAULT_CONTROL["self_play_target_runs"]),
        "train_every_admitted_runs": control.get("self_play_train_every_admitted_runs", DEFAULT_CONTROL["self_play_train_every_admitted_runs"]),
        "max_run_minutes": control.get("self_play_max_run_minutes", DEFAULT_CONTROL["self_play_max_run_minutes"]),
        "stall_seconds": control.get("self_play_stall_seconds", DEFAULT_CONTROL["self_play_stall_seconds"]),
        "game_speed_multiplier": control.get("self_play_game_speed_multiplier", DEFAULT_CONTROL["self_play_game_speed_multiplier"]),
        "exploration_enabled": control.get("exploration_enabled", DEFAULT_CONTROL["exploration_enabled"]),
        "combat_exploration_epsilon": control.get("combat_exploration_epsilon", DEFAULT_CONTROL["combat_exploration_epsilon"]),
        "macro_exploration_epsilon": control.get("macro_exploration_epsilon", DEFAULT_CONTROL["macro_exploration_epsilon"]),
        "exploration_top_k": control.get("exploration_top_k", DEFAULT_CONTROL["exploration_top_k"]),
        "exploration_temperature": control.get("exploration_temperature", DEFAULT_CONTROL["exploration_temperature"]),
    }
    return state


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
    log_path = AI_DIR / "ai_agent_stdout.log"
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] starting ai_agent\n")
    log_file.flush()
    try:
        proc = subprocess.Popen(
            [str(PYTHON_EXE), str(AGENT_PATH)],
            cwd=str(WORKSPACE),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            env=python_subprocess_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_file.close()
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
        env=python_subprocess_env(),
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


def start_self_play():
    pid = get_self_play_pid()
    if pid:
        return {"status": "ok", "message": f"Self-play already running, pid={pid}", "pid": pid}
    if not SELF_PLAY_RUNNER_PATH.exists():
        return {"status": "error", "message": f"Self-play runner not found: {SELF_PLAY_RUNNER_PATH}"}

    runtime = python_runtime_status()
    if not runtime.get("agent_ready"):
        missing = [m for m in REQUIRED_AGENT_MODULES if m in (runtime.get("missing") or [])] or REQUIRED_AGENT_MODULES
        return {
            "status": "error",
            "message": "自训练未启动：Python 环境缺少 " + ", ".join(missing),
            "python_runtime": runtime,
        }

    control = update_control({
        "ai_enabled": True,
        "macro_enabled": True,
        "macro_shop_enabled": True,
        "collection_enabled": True,
        "record_ai_actions": True,
        "include_ai_in_training": True,
        "exploration_enabled": True,
        "exploration_mode": "aggressive",
        "game_speed_enabled": True,
        "game_speed_multiplier": read_control().get("self_play_game_speed_multiplier", DEFAULT_CONTROL["self_play_game_speed_multiplier"]),
    })
    apply_game_speed({
        "enabled": True,
        "multiplier": control.get("game_speed_multiplier", DEFAULT_CONTROL["game_speed_multiplier"]),
    })

    history = self_play_history_summary(control)
    state_payload = default_self_play_state()
    state_payload.update({
        "running": True,
        "phase": "starting",
        "message": "正在启动自训练 runner...",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "last_updated_at": datetime.now().isoformat(timespec="seconds"),
        "current_seed": _effective_self_play_seed(control),
        "completed_runs": 0,
        "admitted_runs": 0,
        "history_completed_runs": int(history.get("completed_runs") or 0),
        "history_admitted_runs": int(history.get("admitted_runs") or 0),
        "pending_training_runs": int(history.get("pending_training_runs") or 0),
        "target_runs": int(control.get("self_play_target_runs", DEFAULT_CONTROL["self_play_target_runs"])),
        "train_every_admitted_runs": int(control.get("self_play_train_every_admitted_runs", DEFAULT_CONTROL["self_play_train_every_admitted_runs"])),
        "recent_scores": history.get("recent_scores") or [],
        "last_score": history.get("last_score"),
    })
    write_json(SELF_PLAY_STATE_PATH, state_payload)

    env = python_subprocess_env()
    env["STS2_AI_PANEL_PORT"] = str(resolve_panel_port())
    proc = subprocess.Popen(
        [str(PYTHON_EXE), str(SELF_PLAY_RUNNER_PATH)],
        cwd=str(WORKSPACE),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        env=env,
    )
    state = read_json(SERVER_STATE_PATH, {})
    state["self_play_pid"] = proc.pid
    state["self_play_started_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(SERVER_STATE_PATH, state)
    return {"status": "ok", "message": f"Self-play started, pid={proc.pid}", "pid": proc.pid}


def stop_self_play():
    pid = get_self_play_pid()
    if not pid:
        state = read_self_play_state()
        state["running"] = False
        state["phase"] = "stopped"
        state["message"] = "自训练已停止。"
        state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        state["last_updated_at"] = state["finished_at"]
        write_json(SELF_PLAY_STATE_PATH, state)
        return {"status": "ok", "message": "Self-play stopped. No managed runner is running."}
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
    state.pop("self_play_pid", None)
    write_json(SERVER_STATE_PATH, state)
    payload = read_self_play_state()
    payload["running"] = False
    payload["phase"] = "stopped"
    payload["message"] = "自训练已停止。"
    payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
    payload["last_updated_at"] = payload["finished_at"]
    write_json(SELF_PLAY_STATE_PATH, payload)
    return {"status": "ok", "message": "Self-play stopped."}


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
            success = True
            try:
                for cmd in [
                    [str(PYTHON_EXE), str(AI_DIR / "monster_profile_builder.py")],
                    [str(PYTHON_EXE), str(AI_DIR / "data_pipeline.py")],
                    [str(PYTHON_EXE), str(AI_DIR / "train_bc.py")],
                    [str(PYTHON_EXE), str(AI_DIR / "train_candidate_bc.py")],
                    [str(PYTHON_EXE), str(AI_DIR / "train_rl_finetune.py")],
                    [str(PYTHON_EXE), str(AI_DIR / "macro_data_pipeline.py")],
                    [str(PYTHON_EXE), str(AI_DIR / "train_macro_bc.py")],
                ]:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(WORKSPACE),
                        capture_output=True,
                        text=True,
                        timeout=600,
                        env=python_subprocess_env(),
                    )
                    output.append("> " + " ".join(cmd))
                    output.append(proc.stdout)
                    if proc.stderr:
                        output.append(proc.stderr)
                    if proc.returncode != 0:
                        success = False
                        output.append(f"ERROR: command exited with {proc.returncode}")
                        break
                if success:
                    snapshot = create_model_snapshot(retention="auto", activate=True)
                    if snapshot.get("status") == "ok":
                        output.append(f"模型快照已自动保存：{snapshot.get('label')} ({snapshot.get('model_id')})")
                        cleanup = cleanup_auto_model_snapshots()
                        if cleanup.get("deleted"):
                            output.append("自动模型已清理旧版本：" + ", ".join(cleanup["deleted"]))
                    else:
                        output.append("ERROR: 自动保存模型快照失败：" + json.dumps(snapshot, ensure_ascii=False))
                        success = False
            except Exception as exc:
                success = False
                output.append(f"ERROR: {exc}")
            LAST_TRAIN.update({"running": False, "finished": datetime.now().isoformat(timespec="seconds"), "output": "\n".join(output)[-12000:]})

    if LAST_TRAIN.get("running"):
        return {"status": "busy", "message": "Training is already running."}
    threading.Thread(target=worker, daemon=True).start()
    return {"status": "ok", "message": "Training started."}


def run_ppo_training_background():
    runtime = python_runtime_status()
    if not runtime.get("training_ready"):
        missing = [m for m in REQUIRED_TRAINING_MODULES if m in (runtime.get("missing") or [])] or REQUIRED_TRAINING_MODULES
        message = "PPO training not started: Python environment missing " + ", ".join(missing)
        LAST_PPO_TRAIN.update({
            "running": False,
            "started": None,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "output": message,
        })
        return {"status": "error", "message": message, "python_runtime": runtime}

    def worker():
        with TRAIN_LOCK:
            LAST_PPO_TRAIN.update({"running": True, "started": datetime.now().isoformat(timespec="seconds"), "finished": None, "output": ""})
            output = []
            try:
                cmd = [str(PYTHON_EXE), str(AI_DIR / "train_ppo.py")]
                proc = subprocess.run(
                    cmd,
                    cwd=str(WORKSPACE),
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    env=python_subprocess_env(),
                )
                output.append("> " + " ".join(cmd))
                if proc.stdout:
                    output.append(proc.stdout)
                if proc.stderr:
                    output.append(proc.stderr)
                if proc.returncode != 0:
                    output.append(f"ERROR: command exited with {proc.returncode}")
            except Exception as exc:
                output.append(f"ERROR: {exc}")
            LAST_PPO_TRAIN.update({
                "running": False,
                "finished": datetime.now().isoformat(timespec="seconds"),
                "output": "\n".join(output)[-12000:],
            })

    if LAST_PPO_TRAIN.get("running"):
        return {"status": "busy", "message": "PPO training is already running."}
    threading.Thread(target=worker, daemon=True).start()
    return {"status": "ok", "message": "PPO training started."}


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


def run_workspace_update_background():
    if LAST_UPDATE.get("running"):
        return {"status": "busy", "message": "更新已经在运行中。"}
    if not UPDATE_SCRIPT_PATH.exists():
        LAST_UPDATE.update({
            "running": False,
            "started": None,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "status": "error",
            "output": f"ERROR: 更新脚本不存在：{UPDATE_SCRIPT_PATH}",
        })
        return {"status": "error", "message": "更新脚本不存在。"}

    def worker():
        with UPDATE_LOCK:
            LAST_UPDATE.update({
                "running": True,
                "started": datetime.now().isoformat(timespec="seconds"),
                "finished": None,
                "status": "running",
                "output": "",
            })
            try:
                cmd = [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(UPDATE_SCRIPT_PATH),
                    "-NoPause",
                ]
                proc = subprocess.run(
                    cmd,
                    cwd=str(WORKSPACE),
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )
                output_parts = ["> " + " ".join(cmd)]
                if proc.stdout:
                    output_parts.append(proc.stdout)
                if proc.stderr:
                    output_parts.append(proc.stderr)
                status = "ok" if proc.returncode == 0 else "error"
                tail = "\n".join(output_parts)[-16000:]
                if status == "ok":
                    tail = (
                        tail
                        + "\n\n更新脚本已完成。为了确保网页加载到最新控制台代码，请关闭当前控制台并重新运行 start_all.bat。"
                    )
                LAST_UPDATE.update({
                    "running": False,
                    "finished": datetime.now().isoformat(timespec="seconds"),
                    "status": status,
                    "output": tail,
                })
                APP_VERSION_CACHE["ts"] = 0.0
            except Exception as exc:
                LAST_UPDATE.update({
                    "running": False,
                    "finished": datetime.now().isoformat(timespec="seconds"),
                    "status": "error",
                    "output": f"ERROR: {exc}",
                })

    threading.Thread(target=worker, daemon=True).start()
    return {"status": "ok", "message": "仓库更新已开始。"}


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


def safe_model_id(value):
    model_id = str(value or "").strip()
    if not model_id or model_id != Path(model_id).name:
        return ""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(ch not in allowed for ch in model_id):
        return ""
    return model_id


def model_package_status(model_id):
    model_id = safe_model_id(model_id)
    if not model_id:
        return None
    root = MODEL_ZOO_DIR / model_id
    manifest = read_json(root / "manifest.json", {})
    missing = []
    artifacts = {}
    total_size = 0
    artifact_map = {
        dirname: list(filenames) + list(OPTIONAL_MODEL_ARTIFACTS.get(dirname, []))
        for dirname, filenames in MODEL_ARTIFACTS.items()
    }
    for dirname, filenames in artifact_map.items():
        artifacts[dirname] = {}
        for filename in filenames:
            path = root / dirname / filename
            status = file_status(path)
            artifacts[dirname][filename] = status
            total_size += int(status.get("size") or 0)
            if filename not in OPTIONAL_MODEL_ARTIFACTS.get(dirname, []) and not status.get("exists"):
                missing.append(f"{dirname}/{filename}")

    combat_meta = read_json(root / "ProcessedParams" / "metadata.json", {})
    candidate_meta = read_json(root / "ProcessedParams" / "candidate_metadata.json", {})
    candidate_rl_meta = read_json(root / "ProcessedParams" / "candidate_rl_metadata.json", {})
    macro_meta = read_json(root / "ProcessedMacroParams" / "metadata.json", {})
    macro_summary = read_json(root / "ProcessedMacroParams" / "training_summary.json", {})
    retention = manifest.get("retention") or ("auto" if model_id.startswith("auto_train_") else "manual")
    pinned = bool(manifest.get("pinned") or retention == "manual")
    return {
        "id": model_id,
        "label": manifest.get("label") or model_id,
        "description": manifest.get("description", ""),
        "created_at": manifest.get("created_at", ""),
        "source": manifest.get("source", ""),
        "retention": retention,
        "pinned": pinned,
        "complete": not missing,
        "missing": missing,
        "size": total_size,
        "artifacts": artifacts,
        "summary": {
            "combat_samples": combat_meta.get("samples", 0),
            "combat_features": combat_meta.get("features", 0),
            "candidate_rows": candidate_meta.get("samples", combat_meta.get("candidate_rows", 0)),
            "candidate_features": candidate_meta.get("features", combat_meta.get("candidate_total_features", 0)),
            "candidate_rl": candidate_rl_meta,
            "macro_samples": macro_summary.get("samples", macro_meta.get("samples", 0)),
            "macro_features": macro_summary.get("features", macro_meta.get("features", 0)),
            "include_ai": combat_meta.get("include_ai", False) or macro_meta.get("include_ai", False),
            "accepted_sources": {
                "combat": combat_meta.get("accepted_sources", {}),
                "macro": macro_meta.get("accepted_sources", {}),
            },
        },
    }


def copy_current_model_artifacts(target_root):
    missing = []
    copied = []
    for dirname, filenames in MODEL_ARTIFACTS.items():
        dst_dir = target_root / dirname
        dst_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            src = AI_DIR / dirname / filename
            if not src.exists():
                missing.append(f"{dirname}/{filename}")
                continue
            dst = dst_dir / filename
            shutil.copy2(src, dst)
            copied.append(str(dst.relative_to(target_root)))
        for filename in OPTIONAL_MODEL_ARTIFACTS.get(dirname, []):
            src = AI_DIR / dirname / filename
            if not src.exists():
                continue
            dst = dst_dir / filename
            shutil.copy2(src, dst)
            copied.append(str(dst.relative_to(target_root)))
    return missing, copied


def set_active_model_id(model_id):
    control = read_control()
    control["active_model_id"] = model_id
    write_json(CONTROL_PATH, control)


def create_model_snapshot(label="", retention="auto", description="", activate=False):
    retention = "manual" if retention == "manual" else "auto"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_id = f"{'manual_keep' if retention == 'manual' else 'auto_train'}_{stamp}_{uuid.uuid4().hex[:6]}"
    root = MODEL_ZOO_DIR / package_id
    MODEL_ZOO_DIR.mkdir(parents=True, exist_ok=True)
    missing, copied = copy_current_model_artifacts(root)
    if missing:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        return {"status": "error", "error": "model artifacts missing", "missing": missing}

    created_at = datetime.now().isoformat(timespec="seconds")
    default_label = f"{'永久保留模型' if retention == 'manual' else '自动训练模型'} {created_at.replace('T', ' ')}"
    manifest = {
        "id": package_id,
        "label": str(label or default_label).strip()[:80],
        "description": str(description or ("手动永久保留的本地训练模型。" if retention == "manual" else "训练完成后自动保存的模型快照。")).strip()[:300],
        "created_at": created_at,
        "source": "training_snapshot",
        "retention": retention,
        "pinned": retention == "manual",
        "artifacts": copied,
    }
    write_json(root / "manifest.json", manifest)
    if activate:
        set_active_model_id(package_id)
    return {
        "status": "ok",
        "model_id": package_id,
        "label": manifest["label"],
        "retention": retention,
        "pinned": manifest["pinned"],
        "package": model_package_status(package_id),
    }


def cleanup_auto_model_snapshots(keep_limit=AUTO_MODEL_KEEP_LIMIT):
    active = safe_model_id(read_control().get("active_model_id"))
    autos = []
    if not MODEL_ZOO_DIR.exists():
        return {"deleted": [], "kept": 0, "limit": keep_limit}
    for item in MODEL_ZOO_DIR.iterdir():
        if not item.is_dir():
            continue
        manifest = read_json(item / "manifest.json", {})
        retention = manifest.get("retention") or ("auto" if item.name.startswith("auto_train_") else "manual")
        if retention != "auto" or manifest.get("pinned"):
            continue
        created_at = str(manifest.get("created_at") or "")
        autos.append((created_at, item.stat().st_mtime, item))

    autos.sort(key=lambda row: (row[0], row[1]), reverse=True)
    deleted = []
    kept_ids = []
    for _created, _mtime, item in autos:
        if item.name == active or len(kept_ids) < keep_limit:
            kept_ids.append(item.name)
            continue
        shutil.rmtree(item, ignore_errors=True)
        deleted.append(item.name)
    return {"deleted": deleted, "kept": len(kept_ids), "limit": keep_limit}


def update_model_package(model_id, label=None, description=None, pin=False):
    model_id = safe_model_id(model_id)
    if not model_id:
        return {"status": "error", "error": "invalid model_id"}
    root = MODEL_ZOO_DIR / model_id
    if not root.exists() or not root.is_dir():
        return {"status": "error", "error": "model package not found"}
    manifest = read_json(root / "manifest.json", {})
    manifest.setdefault("id", model_id)
    manifest.setdefault("created_at", datetime.fromtimestamp(root.stat().st_mtime).isoformat(timespec="seconds"))
    if label is not None:
        next_label = str(label or "").strip()
        if not next_label:
            return {"status": "error", "error": "label cannot be empty"}
        manifest["label"] = next_label[:80]
    if description is not None:
        manifest["description"] = str(description or "").strip()[:300]
    if pin:
        manifest["pinned"] = True
        manifest["retention"] = "manual"
    write_json(root / "manifest.json", manifest)
    return {"status": "ok", "package": model_package_status(model_id)}


def delete_model_package(model_id):
    model_id = safe_model_id(model_id)
    if not model_id:
        return {"status": "error", "error": "invalid model_id"}
    root = MODEL_ZOO_DIR / model_id
    if not root.exists() or not root.is_dir():
        return {"status": "error", "error": "未找到模型包"}

    control = read_control()
    active = safe_model_id(control.get("active_model_id"))
    active_reset = False
    if model_id == active:
        control["active_model_id"] = "local"
        write_json(CONTROL_PATH, control)
        active_reset = True

    try:
        shutil.rmtree(root)
    except Exception as exc:
        if active_reset:
            control["active_model_id"] = model_id
            write_json(CONTROL_PATH, control)
        return {"status": "error", "error": f"删除模型包失败: {exc}"}

    if root.exists():
        if active_reset:
            control["active_model_id"] = model_id
            write_json(CONTROL_PATH, control)
        return {"status": "error", "error": "删除模型包失败: 目录仍然存在"}

    return {
        "status": "ok",
        "deleted_model_id": model_id,
        "active_reset": active_reset,
        "active_model_id": "local" if active_reset else active,
    }


def infer_model_id_from_manifest(manifest, fallback=""):
    if isinstance(manifest, dict):
        manifest_id = safe_model_id(manifest.get("id"))
        if manifest_id:
            return manifest_id
    return safe_model_id(fallback)


def normalize_import_member(name):
    path = PurePosixPath(str(name or ""))
    if path.is_absolute():
        return None
    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    return parts


def import_model_package_archive(filename, content_b64):
    raw_name = Path(str(filename or "")).name
    if not raw_name or Path(raw_name).suffix.lower() != ".zip":
        return {"status": "error", "error": "only .zip model packages are supported"}
    if not content_b64:
        return {"status": "error", "error": "missing archive content"}

    try:
        archive_bytes = base64.b64decode(content_b64.encode("ascii"), validate=True)
    except Exception:
        return {"status": "error", "error": "invalid archive encoding"}

    if not archive_bytes:
        return {"status": "error", "error": "archive is empty"}
    if len(archive_bytes) > MODEL_IMPORT_MAX_BYTES:
        return {
            "status": "error",
            "error": "archive is too large",
            "limit_mb": round(MODEL_IMPORT_MAX_BYTES / (1024 * 1024)),
        }

    try:
        zf = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile:
        return {"status": "error", "error": "invalid zip archive"}

    with zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        if not members:
            return {"status": "error", "error": "archive has no files"}

        normalized = []
        for info in members:
            parts = normalize_import_member(info.filename)
            if not parts:
                return {"status": "error", "error": f"invalid archive entry: {info.filename}"}
            normalized.append((info, parts))

        manifest_parts = None
        for _info, parts in normalized:
            if parts[-1] == "manifest.json":
                manifest_parts = parts
                break
        if not manifest_parts:
            return {"status": "error", "error": "manifest.json not found in archive"}

        root_prefix = manifest_parts[:-1]
        try:
            manifest = json.loads(zf.read("/".join(manifest_parts)).decode("utf-8"))
        except Exception as exc:
            return {"status": "error", "error": f"manifest.json is invalid: {exc}"}

        inferred_id = infer_model_id_from_manifest(manifest, fallback=root_prefix[-1] if root_prefix else Path(raw_name).stem)
        if not inferred_id:
            return {"status": "error", "error": "cannot determine model package id"}

        root = MODEL_ZOO_DIR / inferred_id
        if root.exists():
            return {"status": "error", "error": f"model package already exists: {inferred_id}"}

        MODEL_ZOO_DIR.mkdir(parents=True, exist_ok=True)
        try:
            for info, parts in normalized:
                relative_parts = parts[len(root_prefix):] if root_prefix and parts[:len(root_prefix)] == root_prefix else parts
                if not relative_parts:
                    continue
                target = root.joinpath(*relative_parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            manifest_path = root / "manifest.json"
            imported_manifest = read_json(manifest_path, {})
            imported_manifest["id"] = inferred_id
            imported_manifest.setdefault("label", imported_manifest.get("label") or inferred_id)
            imported_manifest.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
            imported_manifest.setdefault("source", imported_manifest.get("source") or "imported_zip")
            imported_manifest["imported_at"] = datetime.now().isoformat(timespec="seconds")
            imported_manifest["import_filename"] = raw_name
            write_json(manifest_path, imported_manifest)

            status = model_package_status(inferred_id)
            if not status:
                raise ValueError("imported package cannot be indexed")
            if not status.get("complete"):
                missing = status.get("missing", [])
                raise ValueError("imported package is incomplete: " + ", ".join(missing))
            return {
                "status": "ok",
                "model_id": inferred_id,
                "package": status,
            }
        except Exception as exc:
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
            return {"status": "error", "error": str(exc)}


def model_registry_status(active_model_id=None):
    packages = []
    if MODEL_ZOO_DIR.exists():
        for item in sorted(MODEL_ZOO_DIR.iterdir(), key=lambda p: p.name.lower()):
            if item.is_dir():
                status = model_package_status(item.name)
                if status:
                    packages.append(status)
    active = safe_model_id(active_model_id or read_control().get("active_model_id")) or "local"
    active_package = next((pkg for pkg in packages if pkg["id"] == active), None)
    return {
        "active_model_id": active,
        "active_label": active_package["label"] if active_package else ("本地当前模型" if active == "local" else active),
        "auto_keep_limit": AUTO_MODEL_KEEP_LIMIT,
        "packages": packages,
    }


def activate_model_package(model_id):
    model_id = safe_model_id(model_id)
    if not model_id:
        return {"status": "error", "error": "invalid model_id"}
    package = model_package_status(model_id)
    if not package:
        return {"status": "error", "error": "model package not found"}
    if not package.get("complete"):
        return {"status": "error", "error": "model package is incomplete", "missing": package.get("missing", [])}

    root = MODEL_ZOO_DIR / model_id
    for dirname, filenames in MODEL_ARTIFACTS.items():
        dst_dir = AI_DIR / dirname
        dst_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            shutil.copy2(root / dirname / filename, dst_dir / filename)
        for filename in OPTIONAL_MODEL_ARTIFACTS.get(dirname, []):
            src = root / dirname / filename
            dst = dst_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
            elif dst.exists():
                dst.unlink()

    control = read_control()
    control["active_model_id"] = model_id
    write_json(CONTROL_PATH, control)
    return {
        "status": "ok",
        "active_model_id": model_id,
        "active_label": package.get("label", model_id),
        "needs_restart": bool(get_ai_pid()),
    }


def active_model_artifacts_complete():
    for dirname, filenames in MODEL_ARTIFACTS.items():
        active_dir = AI_DIR / dirname
        for filename in filenames:
            if not (active_dir / filename).exists():
                return False
    return True


def ensure_active_model_available():
    if active_model_artifacts_complete():
        return
    registry = model_registry_status()
    for package in registry.get("packages", []):
        if package.get("complete"):
            activate_model_package(package["id"])
            return


def app_version_status():
    now = time.time()
    cached = APP_VERSION_CACHE.get("data")
    if cached and now - APP_VERSION_CACHE.get("ts", 0.0) < 30:
        return cached

    label = ""
    commit = ""
    dirty = False
    try:
        desc = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            timeout=3,
        )
        if desc.returncode == 0:
            label = desc.stdout.strip()
            dirty = label.endswith("-dirty")
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            timeout=3,
        )
        if rev.returncode == 0:
            commit = rev.stdout.strip()
    except Exception:
        pass

    if not label:
        label = f"local-{datetime.fromtimestamp(Path(__file__).stat().st_mtime).strftime('%Y%m%d-%H%M')}"
    data = {
        "label": label,
        "commit": commit,
        "dirty": dirty,
        "control_panel_mtime": datetime.fromtimestamp(Path(__file__).stat().st_mtime).isoformat(timespec="seconds"),
    }
    APP_VERSION_CACHE.update({"ts": now, "data": data})
    return data


def _parse_iso_ts(value):
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return 0.0


def training_version_status(models):
    combat = models.get("combat", {})
    candidate = models.get("candidate", {})
    macro = models.get("macro", {})
    model_items = [
        ("combat", combat.get("model", {}), combat.get("metadata", {})),
        ("candidate", candidate.get("model", {}), candidate.get("metadata", {})),
        ("macro", macro.get("model", {}), macro.get("summary", {}) or macro.get("metadata", {})),
    ]
    ready_items = [item for item in model_items if item[1].get("exists")]
    if not ready_items:
        return {"label": "未训练", "detail": "模型文件缺失", "ready": False}

    latest_name, latest_file, _ = max(ready_items, key=lambda item: _parse_iso_ts(item[1].get("mtime")))
    latest_mtime = latest_file.get("mtime", "")
    label_time = latest_mtime.replace("-", "").replace(":", "").replace("T", "-")[:13] if latest_mtime else "unknown"
    combat_meta = combat.get("metadata", {}) or {}
    candidate_meta = candidate.get("metadata", {}) or {}
    macro_summary = macro.get("summary", {}) or {}
    detail = (
        f"战斗 {combat_meta.get('samples', 0)}/{combat_meta.get('features', '-')}"
        f"，候选 {candidate_meta.get('samples', 0)}/{candidate_meta.get('features', '-')}"
        f"，宏观 {macro_summary.get('samples', 0)}/{macro_summary.get('features', '-')}"
    )
    return {
        "label": f"train-{label_time}",
        "latest_model": latest_name,
        "latest_mtime": latest_mtime,
        "detail": detail,
        "ready": True,
    }


def models_status():
    control = read_control()
    combat_dir = AI_DIR / "ProcessedParams"
    macro_dir = AI_DIR / "ProcessedMacroParams"
    combat_model = combat_dir / "bc_model_best.pth"
    combat_vocab = combat_dir / "vocab.json"
    combat_metadata = read_json(combat_dir / "metadata.json", {})
    candidate_model = combat_dir / "candidate_bc_model_best.pth"
    candidate_metadata = read_json(combat_dir / "candidate_metadata.json", {})
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
        "candidate": {
            "ready": candidate_model.exists(),
            "model": file_status(candidate_model),
            "metadata": candidate_metadata,
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
        "registry": model_registry_status(control.get("active_model_id", "local")),
    }


def ppo_status():
    control = read_control()
    metadata = read_json(PPO_DIR / "ppo_metadata.json", {})
    rollout_files = sorted(PPO_ROLLOUT_DIR.glob("ppo_rollouts_*.jsonl")) if PPO_ROLLOUT_DIR.exists() else []
    scores_raw = read_json(SELF_PLAY_SCORES_PATH, {})
    scores = scores_raw.get("scores") if isinstance(scores_raw, dict) else {}
    rows = [row for row in (scores or {}).values() if isinstance(row, dict)]
    rows.sort(key=lambda item: str(item.get("updated_at") or ""))
    recent5 = rows[-5:]
    recent20 = rows[-20:]
    fixed_seed = _normalize_seed(control.get("ppo_fixed_seed") or DEFAULT_CONTROL["ppo_fixed_seed"])
    fixed_rows = [row for row in rows if _normalize_seed(row.get("seed")) == fixed_seed]
    fixed_best = {}
    if fixed_rows:
        fixed_best = max(
            fixed_rows,
            key=lambda row: (
                1 if int(row.get("max_act") or 0) >= 2 or row.get("reason_group") == "clear" else 0,
                int(row.get("max_floor") or 0),
                float(row.get("boss_damage") or 0.0),
                float(row.get("score") or 0.0),
            ),
        )

    def avg(items, key):
        if not items:
            return 0.0
        return round(sum(float(item.get(key) or 0.0) for item in items) / len(items), 3)

    def clear_count(items):
        return sum(1 for item in items if int(item.get("max_act") or 0) >= 2 or item.get("reason_group") == "clear")

    return {
        "mode": _normalize_policy_mode(control.get("policy_mode")),
        "seed_mode": _normalize_ppo_seed_mode(control.get("ppo_seed_mode")),
        "fixed_seed": fixed_seed,
        "latest": file_status(PPO_DIR / "ppo_policy_latest.pth"),
        "best": file_status(PPO_DIR / "ppo_policy_best.pth"),
        "metadata": metadata,
        "rollout_file_count": len(rollout_files),
        "latest_rollout": file_status(rollout_files[-1]) if rollout_files else file_status(PPO_ROLLOUT_DIR / "ppo_rollouts_none.jsonl"),
        "avg_floor_5": avg(recent5, "max_floor"),
        "avg_floor_20": avg(recent20, "max_floor"),
        "avg_boss_damage_5": avg(recent5, "boss_damage"),
        "avg_boss_damage_20": avg(recent20, "boss_damage"),
        "act1_clear_count": clear_count(rows),
        "act1_clear_count_20": clear_count(recent20),
        "fixed_seed_clear": bool(fixed_best and (int(fixed_best.get("max_act") or 0) >= 2 or fixed_best.get("reason_group") == "clear")),
        "fixed_seed_best": fixed_best,
        "loss": (metadata.get("metrics") or {}).get("loss"),
        "value_loss": (metadata.get("metrics") or {}).get("value_loss"),
        "entropy": (metadata.get("metrics") or {}).get("entropy"),
    }


def training_composition():
    """Return structured data composition info for the control panel."""
    control = read_control()
    include_ai = bool(control.get("include_ai_in_training", False))
    min_quality = control.get("min_training_quality", "unknown")
    ai_min_quality = control.get("ai_min_training_quality", "partial_act1")

    combat_meta = read_json(AI_DIR / "ProcessedParams" / "metadata.json", {})
    macro_meta = read_json(AI_DIR / "ProcessedMacroParams" / "metadata.json", {})

    combat_runs = combat_meta.get("accepted_runs", [])
    macro_runs = macro_meta.get("accepted_runs", [])

    combat_human_runs = [r for r in combat_runs if r.get("source") == "human"]
    combat_ai_runs = [r for r in combat_runs if r.get("source") == "ai"]
    macro_human_runs = [r for r in macro_runs if r.get("source") == "human"]
    macro_ai_runs = [r for r in macro_runs if r.get("source") == "ai"]
    combat_human_samples = combat_meta.get("human_samples", combat_meta.get("accepted_sources", {}).get("human", 0))
    combat_ai_samples = combat_meta.get("ai_samples", combat_meta.get("accepted_sources", {}).get("ai", 0))
    macro_human_samples = macro_meta.get("human_samples", macro_meta.get("accepted_sources", {}).get("human", 0))
    macro_ai_samples = macro_meta.get("ai_samples", macro_meta.get("accepted_sources", {}).get("ai", 0))

    return {
        "settings": {
            "include_ai": include_ai,
            "min_quality": min_quality,
            "ai_min_quality": ai_min_quality,
        },
        "combat": {
            "total_samples": combat_meta.get("samples", 0),
            "human_samples": combat_human_samples,
            "ai_samples": combat_ai_samples,
            "human_ratio": combat_meta.get("human_ratio", 0),
            "ai_ratio": combat_meta.get("ai_ratio", 0),
            "include_ai": bool(combat_meta.get("include_ai", combat_ai_samples > 0)),
            "run_count": combat_meta.get("accepted_run_count", len(combat_runs)),
            "human_run_count": len(combat_human_runs),
            "ai_run_count": len(combat_ai_runs),
            "runs": combat_runs[:20],
            "build_timestamp": combat_meta.get("build_timestamp", ""),
            "build_elapsed_sec": combat_meta.get("build_elapsed_sec"),
            "data_file_count": combat_meta.get("data_file_count"),
            "candidate_groups": combat_meta.get("candidate_groups", 0),
            "candidate_match_misses": combat_meta.get("candidate_match_misses", 0),
        },
        "macro": {
            "total_samples": macro_meta.get("samples", 0),
            "human_samples": macro_human_samples,
            "ai_samples": macro_ai_samples,
            "human_ratio": macro_meta.get("human_ratio", 0),
            "ai_ratio": macro_meta.get("ai_ratio", 0),
            "include_ai": bool(macro_meta.get("include_ai", macro_ai_samples > 0)),
            "run_count": macro_meta.get("accepted_run_count", len(macro_runs)),
            "human_run_count": len(macro_human_runs),
            "ai_run_count": len(macro_ai_runs),
            "runs": macro_runs[:20],
            "build_timestamp": macro_meta.get("build_timestamp", ""),
            "build_elapsed_sec": macro_meta.get("build_elapsed_sec"),
            "data_file_count": len(macro_meta.get("files", [])) if macro_meta.get("files") else macro_meta.get("data_file_count"),
        },
        "has_data": bool(combat_meta.get("samples", 0) or macro_meta.get("samples", 0)),
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


def provisional_dashboard_data(message="正在读取 run 数据..."):
    return {
        "runs": [],
        "current_data": {"active_run": None, "warning": message},
        "recent_records": [],
        "evaluation": {"generated_at": "", "policies": [], "recent_runs": [], "total_runs": 0},
        "dashboard_loading": True,
    }


def _build_dashboard_data():
    return {
        "runs": latest_runs(),
        "current_data": current_data_summary(),
        "recent_records": recent_records(),
        "evaluation": evaluation_summary(limit=50),
        "dashboard_loading": False,
    }


def _refresh_dashboard_data_cache(blocking=True):
    acquired = DASHBOARD_DATA_LOCK.acquire(blocking=blocking)
    if not acquired:
        return False
    try:
        try:
            data = _build_dashboard_data()
        except Exception as exc:
            data = DASHBOARD_DATA_CACHE.get("data") or provisional_dashboard_data(f"run 数据读取失败：{exc}")
            data = dict(data)
            data["dashboard_error"] = str(exc)
        DASHBOARD_DATA_CACHE.update({"ts": time.time(), "data": data})
        return True
    finally:
        DASHBOARD_DATA_LOCK.release()


def dashboard_data_status(force=False, background=False):
    now = time.time()
    cached = DASHBOARD_DATA_CACHE.get("data")
    if cached and not force and now - float(DASHBOARD_DATA_CACHE.get("ts") or 0) < DASHBOARD_DATA_CACHE_TTL_SEC:
        return cached
    if background:
        if not DASHBOARD_DATA_LOCK.locked():
            threading.Thread(target=_refresh_dashboard_data_cache, kwargs={"blocking": False}, daemon=True).start()
        return cached or provisional_dashboard_data()

    _refresh_dashboard_data_cache(blocking=True)
    return DASHBOARD_DATA_CACHE.get("data") or provisional_dashboard_data()


def invalidate_dashboard_data_cache():
    DASHBOARD_DATA_CACHE["ts"] = 0.0


def status_payload():
    control = read_control()
    game = get_game_state_for_dashboard(control)
    ai_process = ai_process_status()
    models = models_status()
    dashboard = dashboard_data_status(background=True)
    return {
        "control": control,
        "ai_pid": ai_process.get("pid"),
        "ai_process": ai_process,
        "app_version": app_version_status(),
        "training_version": training_version_status(models),
        "models": models,
        "monster_profiles": monster_status(),
        "game": {
            "online": "error" not in game,
            "error": game.get("error"),
            "state_type": game.get("state_type"),
            "message": game.get("message"),
            "character": game.get("player", {}).get("character"),
            "hp": game.get("player", {}).get("hp"),
            "max_hp": game.get("player", {}).get("max_hp"),
            "energy": game.get("player", {}).get("energy"),
            "shop_poll_guard": game.get("shop_poll_guard", False),
        },
        "runs": dashboard.get("runs", []),
        "current_data": dashboard.get("current_data") or provisional_dashboard_data()["current_data"],
        "recent_records": dashboard.get("recent_records", []),
        "evaluation": dashboard.get("evaluation", {}),
        "dashboard_loading": dashboard.get("dashboard_loading", False),
        "dashboard_error": dashboard.get("dashboard_error", ""),
        "python_runtime": python_runtime_status(background=True),
        "ai_logic": ai_logic_snapshot(),
        "llm": {
            "config": read_llm_config(mask_key=True),
            "pid": get_llm_pid(),
            "logic": llm_logic_snapshot(),
        },
        "training": LAST_TRAIN,
        "ppo": ppo_status(),
        "ppo_training": LAST_PPO_TRAIN,
        "training_composition": training_composition(),
        "export": LAST_EXPORT,
        "update": LAST_UPDATE,
        "self_play": self_play_status_payload(),
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
      --ease-smooth:cubic-bezier(.2,.8,.2,1);
      --ease-settle:cubic-bezier(.16,1,.3,1);
      --parallax-back:0px;
      --parallax-soft:0px;
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
        linear-gradient(90deg, rgba(23,33,38,.075) 1px, transparent 1px) 0 0 / 44px 44px,
        linear-gradient(0deg, rgba(23,33,38,.060) 1px, transparent 1px) 0 0 / 44px 44px,
        var(--bg);
      font-size:14px;
      line-height:1.45;
      overflow-x:hidden;
    }
    body.is-dragging-card,
    body.is-dragging-module,
    body.is-resizing-module,
    body.is-dragging-card *,
    body.is-dragging-module * {
      cursor:grabbing !important;
      user-select:none !important;
      -webkit-user-select:none !important;
    }
    body.is-resizing-module,
    body.is-resizing-module * {
      cursor:nwse-resize !important;
      user-select:none !important;
      -webkit-user-select:none !important;
    }
    body::before {
      content:"";
      position:fixed;
      left:50%;
      top:382px;
      width:310px;
      height:310px;
      background:url("/assets/sts2_ai_logo.png") center / contain no-repeat;
      opacity:.042;
      transform:translateX(-50%) translateY(var(--parallax-back)) rotate(-10deg);
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
      left:50%;
      top:28px;
      width:170px;
      height:120px;
      background:
        linear-gradient(135deg, transparent 0 34%, rgba(47,111,120,.13) 35% 37%, transparent 38%),
        linear-gradient(90deg, rgba(47,111,120,.10) 0 34px, transparent 34px 52px, rgba(209,162,58,.12) 52px 86px, transparent 86px);
      clip-path:polygon(8% 38%, 36% 8%, 72% 16%, 94% 42%, 78% 82%, 36% 94%, 10% 70%);
      transform:translateX(-50%) translateY(var(--parallax-soft));
      opacity:.85;
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
    .top-actions {
      display:grid;
      justify-items:end;
      gap:8px;
      min-width:min(610px, 48vw);
    }
    .top-action-row {
      display:flex;
      align-items:center;
      justify-content:flex-end;
      gap:10px;
      flex-wrap:wrap;
    }
    .refresh-stack {
      display:grid;
      justify-items:end;
      gap:7px;
      min-width:0;
    }
    .version-info {
      display:grid;
      grid-template-columns:repeat(2, minmax(190px, auto));
      justify-content:end;
      gap:7px;
      color:var(--muted);
      font-size:12px;
      line-height:1.35;
      text-align:left;
      max-width:100%;
    }
    .version-item {
      display:grid;
      grid-template-columns:auto minmax(0, 1fr);
      align-items:baseline;
      gap:6px;
      min-height:28px;
      padding:5px 8px;
      border:1px solid rgba(47,111,120,.16);
      border-radius:9px;
      background:rgba(255,254,250,.72);
      box-shadow:0 4px 14px rgba(38,55,58,.04);
      transition:transform .20s var(--ease-smooth), box-shadow .20s var(--ease-smooth), border-color .20s var(--ease-smooth);
    }
    .version-item:hover {
      transform:translateY(-1px);
      border-color:rgba(47,111,120,.24);
      box-shadow:0 8px 18px rgba(38,55,58,.07);
    }
    .version-item b {
      color:var(--primary-strong);
      font-weight:850;
      white-space:nowrap;
    }
    .version-value {
      min-width:0;
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
      font-variant-numeric:tabular-nums;
    }
    .version-detail {
      grid-column:1 / -1;
      color:var(--soft);
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
      font-size:11px;
    }
    .formula-tuned {
      transition-timing-function:var(--ease-smooth);
    }
    button:focus-visible,
    input:focus-visible,
    select:focus-visible,
    .module-action:focus-visible,
    .module-size-button:focus-visible {
      outline:0;
      box-shadow:
        0 0 0 2px rgba(255,254,250,.96),
        0 0 0 5px rgba(47,111,120,.24);
    }
    .guide-button {
      min-height:32px;
      padding:7px 12px;
      border-color:rgba(47,111,120,.36);
      background:linear-gradient(135deg, #fff, var(--surface-tint));
      color:var(--primary-strong);
    }
    .project-button {
      background:#fff;
      color:var(--ink);
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
      inset:8px;
      transform:rotate(45deg);
      border-color:rgba(47,111,120,.28);
      border-radius:15px;
      background:linear-gradient(135deg, rgba(255,254,250,.82), rgba(237,245,242,.72));
      box-shadow:inset 0 0 0 1px rgba(255,255,255,.70), 0 12px 28px rgba(38,55,58,.08);
    }
    .brand-mark::after {
      right:5px;
      top:6px;
      width:26px;
      height:26px;
      border-color:rgba(209,162,58,.34);
      background:rgba(251,244,223,.72);
      transform:rotate(12deg);
      opacity:.92;
    }
    .brand-logo {
      position:relative;
      z-index:1;
      width:86px;
      height:86px;
      object-fit:contain;
      filter:drop-shadow(0 2px 0 rgba(255,255,255,.72)) drop-shadow(0 8px 14px rgba(38,55,58,.12));
      background:transparent;
      border:0;
      border-radius:0;
      padding:6px;
      box-shadow:none;
      opacity:.94;
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
      transition:transform .22s var(--ease-smooth), box-shadow .22s var(--ease-smooth), border-color .22s var(--ease-smooth);
    }
    .status-card:hover {
      transform:translateY(-2px);
      border-color:rgba(47,111,120,.28);
      box-shadow:0 14px 32px rgba(38,55,58,.10);
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
      transition:transform .22s var(--ease-smooth), box-shadow .22s var(--ease-smooth), border-color .22s var(--ease-smooth);
    }
    .live-panel:hover {
      transform:translateY(-1px);
      border-color:rgba(47,111,120,.24);
      box-shadow:0 20px 48px rgba(38,55,58,.10);
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
      transition:transform .20s var(--ease-smooth), box-shadow .20s var(--ease-smooth), border-color .20s var(--ease-smooth);
    }
    .activity-item:hover {
      transform:translateY(-2px);
      border-color:rgba(47,111,120,.24);
      box-shadow:0 10px 24px rgba(38,55,58,.08);
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
    .content { min-width:0; overflow:visible; }
    section {
      position:relative;
      min-width:0;
      background:rgba(255, 255, 255, 0.75);
      backdrop-filter:blur(16px) saturate(120%);
      -webkit-backdrop-filter:blur(16px) saturate(120%);
      border:1px solid rgba(47,111,120,.14);
      border-top-color:rgba(255,255,255,.8);
      border-left-color:rgba(255,255,255,.6);
      border-radius:18px;
      padding:17px;
      box-shadow:0 8px 32px rgba(47,111,120,.06), 0 2px 8px rgba(47,111,120,.03);
      overflow:hidden;
      transition:transform .3s var(--ease-smooth), box-shadow .3s var(--ease-smooth), border-color .3s var(--ease-smooth), background .3s var(--ease-smooth);
    }
    section:hover {
      border-color:rgba(47,111,120,.22);
      border-top-color:rgba(255,255,255,.9);
      background:rgba(255, 255, 255, 0.85);
      box-shadow:0 16px 40px rgba(47,111,120,.12), 0 4px 14px rgba(47,111,120,.05);
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
      transition:background .18s var(--ease-smooth), border-color .18s var(--ease-smooth);
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
    .composition-card {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px 16px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .composition-card .comp-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
      font-weight: 600;
      font-size: 13px;
      color: var(--ink);
    }
    .composition-card .comp-bar {
      height: 10px;
      border-radius: 5px;
      background: var(--line);
      overflow: hidden;
      margin-bottom: 12px;
      display: flex;
    }
    .composition-card .comp-bar .bar-human {
      background: linear-gradient(90deg, #3b82f6, #60a5fa);
      height: 100%;
      transition: width 0.4s ease;
    }
    .composition-card .comp-bar .bar-ai {
      background: linear-gradient(90deg, #f59e0b, #fbbf24);
      height: 100%;
      transition: width 0.4s ease;
    }
    .composition-card .comp-stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }
    .composition-card .comp-stat {
      text-align: center;
      padding: 6px 4px;
      border-radius: 6px;
      background: var(--surface-soft);
      border: 1px solid var(--line);
    }
    .composition-card .comp-stat .stat-value {
      font-size: 18px;
      font-weight: 700;
      color: var(--ink);
    }
    .composition-card .comp-stat .stat-label {
      font-size: 11px;
      color: var(--muted);
      margin-top: 2px;
    }
    .composition-card .comp-runs {
      max-height: none;
      overflow: visible;
      font-size: 12px;
    }
    .composition-card .comp-runs table {
      width: 100%;
      border-collapse: collapse;
    }
    .composition-card .comp-runs th,
    .composition-card .comp-runs td {
      padding: 3px 6px;
      text-align: left;
      border-bottom: 1px solid var(--line);
    }
    .composition-card .comp-runs th {
      font-weight: 600;
      color: var(--muted);
      font-size: 11px;
      background: var(--surface);
    }
    .composition-card .comp-empty {
      text-align: center;
      color: var(--muted);
      padding: 18px 0;
      font-size: 13px;
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
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
    }
    .section-head h2::before {
      content:"";
      width:18px;
      flex:0 0 18px;
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
    .kv > * {
      min-width:0;
      overflow-wrap:anywhere;
    }
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
      transition:background .18s var(--ease-smooth), border-color .18s var(--ease-smooth), transform .18s var(--ease-smooth);
    }
    button:hover { border-color:var(--primary); background:var(--primary-bg); transform:translateY(-1px); }
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
      max-height:none;
      overflow:visible;
      background:#233036;
      color:#eef6f2;
      padding:12px;
      border-radius:12px;
      margin:10px 0 0;
      font-size:12px;
      border:1px solid rgba(35,48,54,.34);
    }
    .table-wrap {
      overflow-x:auto;
      overflow-y:visible;
      border:1px solid var(--line);
      border-radius:12px;
      background:#fff;
    }
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
      overscroll-behavior:contain;
    }
    .modal-backdrop.open { display:flex; }
    html.modal-lock, body.modal-lock {
      overflow:hidden;
      overscroll-behavior:none;
    }
    .modal {
      width:min(560px, 100%);
      background:#fff;
      border:1px solid var(--line-strong);
      border-radius:16px;
      padding:16px;
      box-shadow:0 22px 54px rgba(38,55,58,.24);
    }
    .project-modal {
      width:min(1120px, calc(100vw - 32px));
      height:min(820px, calc(100vh - 32px));
      padding:0;
      overflow:hidden;
      display:grid;
      grid-template-rows:auto minmax(0, 1fr);
    }
    .project-modal-head {
      display:grid;
      grid-template-columns:minmax(0, 1fr) auto;
      gap:18px;
      padding:22px 24px 20px;
      border-bottom:1px solid var(--line);
      background:linear-gradient(135deg, rgba(255,255,255,.96), rgba(237,245,242,.90));
    }
    .project-head-copy { align-self:center; }
    .project-kicker {
      color:var(--primary-strong);
      font-size:12px;
      font-weight:900;
      margin-bottom:7px;
    }
    .project-title {
      font-size:26px;
      line-height:1.15;
      margin:0;
      color:var(--ink);
    }
    .project-lead {
      color:var(--muted);
      line-height:1.65;
      margin:10px 0 0;
      max-width:820px;
    }
    .project-modal-body {
      overflow:auto;
      padding:22px 24px 26px;
      background:#fff;
      overscroll-behavior:contain;
    }
    .project-section {
      display:grid;
      gap:11px;
      margin-top:18px;
    }
    .project-section:first-child { margin-top:0; }
    .project-section-title {
      font-weight:900;
      color:var(--primary-strong);
      font-size:15px;
    }
    .project-panel {
      border:1px solid var(--line);
      border-radius:16px;
      background:linear-gradient(135deg, #fff, var(--surface-soft));
      padding:15px;
      overflow:hidden;
    }
    .project-panel p {
      margin:8px 0 0;
      color:var(--muted);
      line-height:1.65;
    }
    .project-copy {
      display:grid;
      gap:10px;
      color:var(--muted);
      line-height:1.7;
    }
    .project-copy p { margin:0; }
    .project-grid {
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));
      gap:10px;
    }
    .project-doc-list, .project-path-list {
      display:grid;
      gap:8px;
    }
    .project-doc, .project-path {
      display:grid;
      grid-template-columns:minmax(128px, .42fr) minmax(0, 1fr);
      gap:10px;
      padding:12px;
      border:1px solid var(--line);
      border-radius:12px;
      background:#fff;
      color:inherit;
      text-decoration:none;
    }
    .project-doc:hover, .project-inline-link:hover {
      border-color:rgba(47,111,120,.42);
    }
    .project-doc b, .project-path b { color:var(--ink); }
    .project-doc span, .project-path span {
      color:var(--muted);
      font-size:12px;
      line-height:1.5;
    }
    .project-inline-link {
      color:var(--primary-strong);
      text-decoration:underline;
      text-underline-offset:3px;
      font-weight:800;
    }
    .project-item {
      border:1px solid var(--line);
      border-radius:12px;
      padding:12px;
      background:var(--surface-soft);
      display:grid;
      gap:5px;
    }
    .project-item b { color:var(--ink); }
    .project-item span {
      color:var(--muted);
      font-size:12px;
      line-height:1.5;
    }
    .project-flow {
      margin:0;
      padding-left:20px;
      color:var(--muted);
      line-height:1.7;
    }
    .project-flow li { padding:3px 0; }
    .guide-overlay {
      position:fixed;
      inset:0;
      z-index:30;
      background:rgba(23,33,38,.42);
      backdrop-filter:blur(1px);
    }
    .guide-overlay.is-hidden { display:none; }
    .guide-spotlight {
      position:fixed;
      border:2px solid var(--accent);
      border-radius:14px;
      box-shadow:0 0 0 9999px rgba(23,33,38,.28), 0 0 0 6px rgba(209,162,58,.20);
      pointer-events:none;
      transition:left .18s ease, top .18s ease, width .18s ease, height .18s ease;
    }
    .guide-arrow {
      position:fixed;
      height:2px;
      background:var(--accent);
      transform-origin:left center;
      pointer-events:none;
      box-shadow:0 1px 4px rgba(38,55,58,.18);
    }
    .guide-arrow::after {
      content:"";
      position:absolute;
      right:-1px;
      top:-5px;
      width:0;
      height:0;
      border-top:6px solid transparent;
      border-bottom:6px solid transparent;
      border-left:10px solid var(--accent);
    }
    .guide-card {
      position:fixed;
      width:min(380px, calc(100vw - 28px));
      background:#fff;
      border:1px solid var(--line-strong);
      border-radius:14px;
      padding:15px;
      box-shadow:0 22px 48px rgba(38,55,58,.22);
    }
    .guide-kicker {
      display:flex;
      justify-content:space-between;
      gap:12px;
      color:var(--primary-strong);
      font-size:12px;
      font-weight:850;
      margin-bottom:8px;
    }
    .guide-card h2 {
      font-size:17px;
      line-height:1.25;
      margin:0;
      color:var(--ink);
    }
    .guide-card p {
      margin:9px 0 13px;
      color:var(--muted);
      line-height:1.6;
      font-size:13px;
    }
    .guide-actions {
      display:grid;
      grid-template-columns:1fr 1fr auto;
      gap:8px;
      align-items:center;
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
      user-select:none;
      -webkit-user-select:none;
      transition:background .18s var(--ease-smooth), border-color .18s var(--ease-smooth), opacity .18s var(--ease-smooth), transform .18s var(--ease-smooth);
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
      position:relative;
      display:flex;
      flex-wrap:wrap;
      gap:16px;
      min-height:0;
      border-radius:18px;
      align-items:start;
      align-content:flex-start;
      transition:background .18s var(--ease-smooth), outline-color .18s var(--ease-smooth);
    }
    .workspace.drop-ready {
      outline:2px dashed rgba(47,111,120,.45);
      outline-offset:8px;
      background:rgba(229,242,243,.45);
    }
    .module-card {
      position:relative;
      flex:0 0 100%;
      width:100%;
      max-width:100%;
      min-width:min(360px, 100%);
      min-height:96px;
      align-self:start;
      container-type:inline-size;
      will-change:transform;
      user-select:none;
      -webkit-user-select:none;
      transition:opacity .18s var(--ease-smooth), transform .22s var(--ease-smooth), box-shadow .22s var(--ease-smooth), border-color .22s var(--ease-smooth);
    }
    .module-card input,
    .module-card textarea {
      user-select:text;
      -webkit-user-select:text;
    }
    .module-card[data-size="compact"] {
      flex-basis:calc((100% - 32px) / 3);
      width:calc((100% - 32px) / 3);
    }
    .module-card[data-size="normal"] {
      flex-basis:calc((100% - 16px) / 2);
      width:calc((100% - 16px) / 2);
    }
    .module-card[data-size="wide"] {
      flex-basis:100%;
      width:100%;
    }
    .module-card.has-custom-width {
      flex-basis:min(var(--module-width), 100%);
      width:min(var(--module-width), 100%);
    }
    .module-card.has-custom-height {
      height:auto;
      min-height:var(--module-height);
      overflow:visible;
      padding-bottom:26px;
    }
    .module-card[data-size="compact"] {
      padding:13px;
    }
    .module-card[data-size="compact"] .section-head {
      gap:8px;
      margin-bottom:9px;
    }
    .module-card[data-size="compact"] h2 {
      font-size:14px;
    }
    .module-card[data-size="compact"] .kv {
      grid-template-columns:86px minmax(0, 1fr);
      gap:8px;
      padding:7px 0;
    }
    .module-card[data-size="compact"] .table-wrap,
    .module-card[data-size="compact"] pre {
      max-height:none;
    }
    .module-card[data-size="wide"] .table-wrap {
      max-height:none;
    }
    .module-card[data-size="wide"] pre {
      max-height:none;
    }
    .model-panel {
      display:grid;
      gap:14px;
    }
    .model-summary-grid {
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));
      gap:10px;
    }
    .model-manager-grid {
      display:grid;
      grid-template-columns:minmax(280px, 360px) minmax(0, 1fr);
      gap:14px;
      align-items:start;
    }
    .model-panel-block {
      border:1px solid var(--line);
      border-radius:14px;
      background:linear-gradient(135deg, #fff, var(--surface-soft));
      padding:14px;
      display:grid;
      gap:10px;
      min-width:0;
    }
    .model-panel-block h3 {
      margin:0;
      font-size:14px;
      color:var(--ink);
      font-weight:850;
    }
    .model-panel-copy {
      color:var(--muted);
      font-size:12px;
      line-height:1.6;
    }
    .model-actions-grid {
      display:grid;
      gap:10px;
    }
    .model-import-row {
      display:grid;
      gap:8px;
    }
    .model-import-row input[type="file"] {
      width:100%;
      min-width:0;
    }
    .model-table-wrap {
      display:grid;
      gap:10px;
    }
    .model-pkg-card {
      border:1px solid var(--line);
      border-radius:14px;
      background:linear-gradient(135deg, #fff 60%, var(--surface-soft));
      padding:14px 16px;
      display:grid;
      gap:8px;
      transition:border-color .2s, box-shadow .2s;
    }
    .model-pkg-card:hover {
      border-color:var(--primary);
      box-shadow:0 4px 16px rgba(47,111,120,.08);
    }
    .model-pkg-card.is-active {
      border-color:var(--good);
      background:linear-gradient(135deg, var(--good-bg) 30%, #fff);
    }
    .model-pkg-card-head {
      display:flex;
      align-items:center;
      gap:8px;
      flex-wrap:wrap;
    }
    .model-pkg-card-head .pkg-label {
      font-weight:700;
      font-size:14px;
      color:var(--ink);
      flex:1;
      min-width:0;
    }
    .model-pkg-card-head .pill {
      flex-shrink:0;
    }
    .model-pkg-card-meta {
      display:flex;
      flex-wrap:wrap;
      gap:6px 16px;
      font-size:12px;
      color:var(--muted);
    }
    .model-pkg-card-meta span {
      white-space:nowrap;
    }
    .model-pkg-card-actions {
      display:flex;
      flex-wrap:wrap;
      gap:6px;
      align-items:center;
      padding-top:4px;
      border-top:1px solid var(--line);
    }
    .model-pkg-card-actions input[type="text"] {
      flex:1;
      min-width:120px;
      max-width:260px;
    }
    .model-pkg-card-actions button {
      font-size:12px;
      padding:4px 10px;
    }
    .model-pkg-desc {
      font-size:11px;
      color:var(--soft);
      line-height:1.5;
    }
    .model-table th,
    .model-table td {
      white-space:normal;
      overflow-wrap:anywhere;
    }
    .model-table td code {
      display:inline-block;
      max-width:100%;
      white-space:normal;
    }
    .model-row-actions {
      display:flex;
      flex-wrap:wrap;
      gap:8px;
      align-items:center;
    }
    .model-label-input {
      width:100%;
      min-width:0;
    }
    .model-source-lines {
      display:grid;
      gap:4px;
    }
    .model-status-stack {
      display:grid;
      gap:6px;
      justify-items:start;
    }
    .model-table-empty {
      padding:16px;
      color:var(--muted);
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
    .module-card.dragging-card {
      opacity:.62;
      transform:scale(.99);
      box-shadow:0 18px 40px rgba(38,55,58,.15);
    }
    .module-card.drop-target {
      border-color:rgba(47,111,120,.42);
      box-shadow:0 0 0 3px rgba(47,111,120,.18), var(--shadow-soft);
    }
    .module-card.drop-target::before {
      content:"";
      position:absolute;
      left:12px;
      right:12px;
      height:4px;
      border-radius:999px;
      background:linear-gradient(90deg, var(--primary), var(--accent));
      box-shadow:0 4px 12px rgba(47,111,120,.18);
      pointer-events:none;
      z-index:3;
    }
    .module-card.drop-target.drop-before::before {
      top:-10px;
    }
    .module-card.drop-target.drop-after::before {
      bottom:-10px;
    }
    .module-card.is-animating {
      z-index:2;
    }
    .module-card.resizing-card {
      z-index:4;
      transition:none;
      box-shadow:0 0 0 3px rgba(209,162,58,.32), 0 18px 42px rgba(38,55,58,.18);
    }
    .module-card.resizing-card:hover {
      transform:none;
    }
    .module-card > .section-head {
      cursor:grab;
      user-select:none;
      -webkit-user-select:none;
    }
    .module-card > .section-head:active {
      cursor:grabbing;
    }
    .module-actions {
      display:flex;
      align-items:center;
      justify-content:flex-end;
      flex-wrap:wrap;
      gap:7px;
      min-width:0;
      user-select:none;
      -webkit-user-select:none;
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
    .module-size-controls {
      display:flex;
      align-items:center;
      gap:3px;
      padding:2px;
      border:1px solid var(--line);
      border-radius:10px;
      background:#fff;
      user-select:none;
      -webkit-user-select:none;
    }
    .module-size-button {
      width:26px;
      min-width:26px;
      height:26px;
      min-height:26px;
      border:0;
      border-radius:7px;
      padding:0;
      font-size:12px;
      color:var(--muted);
      background:transparent;
      transition:background .18s var(--ease-smooth), color .18s var(--ease-smooth), transform .18s var(--ease-smooth);
    }
    .module-size-button:hover { transform:translateY(-1px); }
    .module-size-button.active {
      color:#fff;
      background:var(--primary);
    }
    .module-resize-handle {
      position:absolute;
      right:4px;
      bottom:4px;
      width:22px;
      height:22px;
      border:0;
      background:
        linear-gradient(135deg, transparent 0 54%, rgba(47,111,120,.55) 55% 60%, transparent 61%),
        linear-gradient(135deg, transparent 0 68%, rgba(47,111,120,.36) 69% 74%, transparent 75%);
      cursor:nwse-resize;
      opacity:.58;
      z-index:5;
      user-select:none;
      -webkit-user-select:none;
    }
    .module-resize-handle:hover {
      opacity:1;
    }
    @container (max-width: 520px) {
      .module-card > .section-head {
        grid-template-columns:1fr;
        align-items:start;
        gap:9px;
      }
      .module-card > .section-head h2 {
        max-width:100%;
      }
      .module-card .module-actions {
        width:100%;
        justify-content:flex-start;
      }
      .module-card .section-head .pill {
        min-width:0;
        max-width:100%;
      }
      .module-card .kv {
        grid-template-columns:1fr;
        gap:4px;
      }
      .model-manager-grid {
        grid-template-columns:1fr;
      }
      .module-card .kv span:first-child {
        font-size:12px;
      }
    }
    .drag-ghost {
      position:fixed;
      left:-9999px;
      top:-9999px;
      z-index:9999;
      display:grid;
      gap:2px;
      min-width:150px;
      max-width:240px;
      padding:10px 12px;
      border:1px solid rgba(47,111,120,.36);
      border-radius:12px;
      background:rgba(255,254,250,.96);
      color:var(--ink);
      box-shadow:0 18px 38px rgba(38,55,58,.18);
      pointer-events:none;
      user-select:none;
      -webkit-user-select:none;
    }
    .drag-ghost b {
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
      font-size:13px;
    }
    .drag-ghost span {
      color:var(--muted);
      font-size:12px;
      font-weight:800;
    }
    .priority-grid { align-items:start; }
    .priority-grid.single-visible { grid-template-columns:1fr; }
    @media (max-width: 1120px) {
      .status-grid { grid-template-columns:repeat(2, minmax(160px, 1fr)); }
      main { grid-template-columns:1fr; }
      .priority-grid { grid-template-columns:1fr; }
      .module-card,
      .module-card[data-size="compact"],
      .module-card[data-size="normal"],
      .module-card[data-size="wide"],
      .module-card.has-custom-width {
        flex-basis:100%;
        width:100%;
      }
      .module-card.has-custom-height { height:auto; }
      .module-resize-handle { display:none; }
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
      .top-actions { justify-items:start; min-width:0; margin-top:10px; }
      .top-action-row { justify-content:flex-start; }
      .refresh-stack { justify-items:start; }
      .version-info { grid-template-columns:1fr; width:100%; }
      .version-item { width:100%; }
      #lastRefresh { margin-top:0; }
      .status-grid, .metric-grid { grid-template-columns:1fr; }
      .activity-feed { grid-template-columns:1fr; }
      .button-row { grid-template-columns:1fr; }
      .project-modal { width:calc(100vw - 18px); height:calc(100vh - 18px); }
      .project-modal-head, .project-modal-body { padding-left:14px; padding-right:14px; }
      .project-title { font-size:21px; }
      .project-modal-head { grid-template-columns:1fr auto; }
      .project-doc, .project-path { grid-template-columns:1fr; }
      .guide-actions { grid-template-columns:1fr; }
      .fold-panel summary { grid-template-columns:minmax(0, 1fr) 78px 28px; }
      .fold-panel summary .pill { width:78px; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        transition-duration:.01ms !important;
        animation-duration:.01ms !important;
        scroll-behavior:auto !important;
      }
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
      <div class="top-actions">
        <div class="top-action-row">
          <button class="guide-button project-button" onclick="openProjectGuide()">项目说明</button>
          <button class="guide-button" onclick="startGuide()">新手引导</button>
        </div>
        <div class="refresh-stack">
          <span id="lastRefresh" class="pill info">读取中</span>
          <div id="versionInfo" class="version-info">
            <div class="version-item"><b>当前版本</b><span class="version-value">-</span></div>
            <div class="version-item"><b>训练版本</b><span class="version-value">-</span></div>
          </div>
        </div>
      </div>
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
                <button class="module-item" data-module-target="model_status" draggable="true" onclick="openModule('model_status')" ondragstart="beginModuleDrag(event, 'model_status')" ondragend="endModuleDrag(event)">
                  <span class="module-glyph"></span><span class="module-label">AI 模型状态</span><span class="module-state">+</span>
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
        <div class="switch">
          <div><div class="switch-title">测试加速</div><div class="switch-note">加快游戏时间，并缩短 AI 固定等待；出异常先关掉</div></div>
          <input id="game_speed_enabled" type="checkbox" onchange="saveControl(true)">
        </div>
        <div class="field">
          <span>加速倍率</span>
          <select id="game_speed_multiplier" onchange="saveControl(true)">
            <option value="1.5">1.5x：稳妥</option>
            <option value="2">2x：推荐测试</option>
            <option value="3">3x：快速跑图</option>
            <option value="4">4x：高风险</option>
            <option value="6">6x：只做极限测试</option>
          </select>
        </div>
        <div id="speedDetail" class="fine" style="margin-top:8px">正常速度</div>
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
        <div id="trainingComposition" class="composition-card" style="margin-top:14px"></div>
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
            <span class="fold-title">自训练</span>
            <span id="selfPlayBadge" class="pill info">未启动</span>
          </summary>
          <div class="fold-body">
        <div class="button-row">
          <button class="good" onclick="startSelfPlay()">启动自训练</button>
          <button class="bad" onclick="stopSelfPlay()">停止自训练</button>
        </div>
        <div id="selfPlaySummary" class="fine" style="margin-top:10px">等待启动</div>
        <div id="selfPlayProgress" style="margin-top:6px"></div>
        <div id="selfPlayRecentScores" style="margin-top:8px"></div>
        <div class="field">
          <span>角色</span>
          <select id="self_play_character" onchange="saveSelfPlayConfig()">
            <option value="IRONCLAD">IRONCLAD / 铁甲战士</option>
          </select>
        </div>
        <div class="field">
          <span>目标局数 <small style="opacity:0.6">(0=无限)</small></span>
          <input id="self_play_target_runs" type="number" min="0" max="999" step="1" onchange="saveSelfPlayConfig()">
        </div>
        <div class="field">
          <span>入训批次</span>
          <input id="self_play_train_every_admitted_runs" type="number" min="1" max="100" step="1" onchange="saveSelfPlayConfig()">
        </div>
        <div class="field">
          <span>单局超时(分钟)</span>
          <input id="self_play_max_run_minutes" type="number" min="5" max="240" step="1" onchange="saveSelfPlayConfig()">
        </div>
        <div class="field">
          <span>卡死判定(秒)</span>
          <input id="self_play_stall_seconds" type="number" min="15" max="1800" step="1" onchange="saveSelfPlayConfig()">
        </div>
        <div class="field">
          <span>自训练速度</span>
          <select id="self_play_game_speed_multiplier" onchange="saveSelfPlayConfig()">
            <option value="1.5">1.5x</option>
            <option value="2">2x</option>
            <option value="3">3x</option>
            <option value="4">4x</option>
            <option value="6">6x</option>
          </select>
        </div>
        <div class="switch">
          <div><div class="switch-title">激进探索</div><div class="switch-note">始终只从合法候选动作里抽样</div></div>
          <input id="exploration_enabled" type="checkbox" onchange="saveSelfPlayConfig()">
        </div>
        <div class="field">
          <span>行为约束模式</span>
          <select id="self_play_constraint_mode" onchange="saveSelfPlayConfig()">
            <option value="guarded">保守：完整规则护栏</option>
            <option value="explore">探索：弱化策略护栏</option>
            <option value="free">自由：只保留合法动作</option>
          </select>
        </div>
        <div class="field">
          <span>战斗探索</span>
          <input id="combat_exploration_epsilon" type="number" min="0" max="1" step="0.05" onchange="saveSelfPlayConfig()">
        </div>
        <div class="field">
          <span>宏观探索</span>
          <input id="macro_exploration_epsilon" type="number" min="0" max="1" step="0.05" onchange="saveSelfPlayConfig()">
        </div>
        <div class="field">
          <span>Top K</span>
          <input id="exploration_top_k" type="number" min="1" max="12" step="1" onchange="saveSelfPlayConfig()">
        </div>
        <div class="field">
          <span>温度</span>
          <input id="exploration_temperature" type="number" min="0.1" max="5" step="0.05" onchange="saveSelfPlayConfig()">
        </div>
        <details class="more-panel">
          <summary>自训练状态</summary>
          <pre id="selfPlayState">暂无自训练状态</pre>
        </details>
          </div>
        </details>
      </section>

      <section class="fold-panel">
        <details>
          <summary>
            <span class="fold-title">仓库更新</span>
            <span id="updateBadge" class="pill info">未运行</span>
          </summary>
          <div class="fold-body">
        <div class="button-row">
          <button class="primary" onclick="runWorkspaceUpdate()">一键更新仓库</button>
        </div>
        <div class="fine" style="margin-top:10px">会调用本机 Git 执行安全的快进更新；如果有已跟踪的本地改动，会先自动做 stash 备份。更新完请关闭当前控制台，再重新运行 start_all.bat。</div>
        <details class="more-panel">
          <summary>更新日志</summary>
          <pre id="updateOutput">暂无更新记录</pre>
        </details>
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

      <section id="module-model-status" class="module-card" data-module="model_status">
        <div class="section-head">
          <h2>AI 模型状态</h2>
          <div class="module-actions">
            <span id="modelBadge" class="pill">-</span>
            <button class="module-action" onclick="toggleModuleCollapse('model_status')" data-collapse-for="model_status" title="收起">-</button>
            <button class="module-action" onclick="closeModule('model_status')" title="关闭">×</button>
          </div>
        </div>
        <div id="modelHealth">读取中</div>
      </section>

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
          <h2>训练</h2>
          <div class="module-actions">
            <span id="trainStatus" class="pill">-</span>
            <button class="module-action" onclick="toggleModuleCollapse('training')" data-collapse-for="training" title="收起">-</button>
            <button class="module-action" onclick="closeModule('training')" title="关闭">×</button>
          </div>
        </div>
        <div id="trainCompositionMain" class="composition-card" style="margin-bottom:12px"></div>
        <div class="row" style="margin-bottom:12px">
          <button class="primary" onclick="train()">重建数据 + 重训战斗/候选/宏观 BC</button>
        </div>
        <details class="more-panel">
          <summary>训练日志</summary>
          <pre id="trainOutput">暂无输出</pre>
        </details>
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
  <div id="projectGuideModal" class="modal-backdrop" aria-hidden="true">
    <div class="modal project-modal" onclick="event.stopPropagation()">
      <div class="project-modal-head">
        <div class="project-head-copy">
          <div class="project-kicker">项目说明</div>
          <h2 class="project-title">STS2 AI 工作区说明</h2>
          <p class="project-lead">这不是单个自动出牌脚本，而是一套本地实验工作区：Mod 负责读游戏和写日志，控制台负责观察和操作，BC/LLM 负责给动作建议，数据再回到训练流程里。</p>
        </div>
        <button onclick="closeProjectGuide()">关闭</button>
      </div>
      <div class="project-modal-body">
        <div class="project-section">
          <div class="project-section-title">项目现在到底在做什么</div>
          <div class="project-copy">
            <p>当前目标是让杀戮尖塔 2 的对局可以被本地 AI 系统读取、记录、复盘和训练。游戏里的状态先通过 Mod API 进入工作区，控制台把这些状态整理成玩家能看懂的页面，同时把玩家或 AI 的动作写入本地数据集。</p>
            <p>基础 BC AI 负责快速执行已经训练出来的策略；LLM 负责更像“大脑”的判断，但它不应该绕过系统校验直接乱构造动作。推荐模式下，系统先生成合法候选动作，LLM 只能从候选动作里选一个。</p>
            <p>这个项目目前是半成品：链路已经跑通，但数据量和策略稳定性还不够。现在最重要的是稳定采集高质量 Run，补齐怪物、药水、牌堆和失败局的记录，然后再继续训练。</p>
          </div>
        </div>
        <div class="project-section">
          <div class="project-section-title">真实模块怎么分工</div>
          <div class="project-grid">
            <div class="project-item"><b>游戏 Mod</b><span><code>训练脚本/STS2MCP/</code> 提供本地 API，读取场景、手牌、敌人、药水、奖励、地图，并写入 Run 数据。</span></div>
            <div class="project-item"><b>控制台</b><span><code>AI_Training/control_panel.py</code> 是你现在看到的网页，负责启动进程、显示状态、打包数据、触发训练。</span></div>
            <div class="project-item"><b>基础 AI</b><span><code>AI_Training/ai_agent.py</code> 执行 BC 模型或规则给出的战斗/宏观动作。</span></div>
            <div class="project-item"><b>LLM Agent</b><span><code>AI_Training/llm_agent.py</code> 调用 OpenAI-compatible 接口，让模型在合法候选动作里选择。</span></div>
            <div class="project-item"><b>数据管线</b><span><code>AI_Training/data_pipeline.py</code> 把采集到的 jsonl 转成训练样本。</span></div>
            <div class="project-item"><b>本地数据</b><span><code>RL_Datasets/</code> 保存原始数据，<code>Data_Packages/</code> 保存一键打包后的 zip。</span></div>
          </div>
        </div>
        <div class="project-section">
          <div class="project-section-title">控制台页面怎么读</div>
          <div class="project-path-list">
            <div class="project-path"><b>顶部状态</b><span>先看“游戏连接”“AI 接管”“采集总开关”“当前 Run”。这里能判断游戏 API 是否在线、AI 是否会动、数据是否会写入。</span></div>
            <div class="project-path"><b>实时采集动态</b><span>只显示最近几条动作，比如出牌、用药水、选卡、奖励、地图动作。它是观察窗口，不是完整日志。</span></div>
            <div class="project-path"><b>战斗 AI</b><span>控制基础 AI 进程和“允许 AI 出牌”。如果只想手动玩并采数据，保持关闭。</span></div>
            <div class="project-path"><b>LLM 模型接入</b><span>配置 Base URL、API Key、Model，控制是否启用模型决策，以及 LLM 是否真的执行战斗动作。</span></div>
            <div class="project-path"><b>采集与训练</b><span>控制是否写入数据，选择最低训练质量，触发重建数据和重训。</span></div>
            <div class="project-path"><b>右侧工作区</b><span>显示 AI 出牌逻辑、LLM 决策、Run 数据体检、最近 Run、采集记录、评测和训练输出。</span></div>
          </div>
        </div>
        <div class="project-section">
          <div class="project-section-title">推荐演示流程</div>
          <ol class="project-flow">
            <li>双击 <code>start_all.bat</code> 或中文的一键启动脚本，打开控制台、日志窗口、BC AI 和 LLM。</li>
            <li>进游戏后先看顶部“游戏连接”。如果显示未连接，优先检查游戏、Mod 和本地 API。</li>
            <li>如果只是采集人类数据：打开“采集总开关”，关闭“允许 AI 出牌”和“允许 LLM 自动战斗”。</li>
            <li>如果演示基础 AI：打开“允许 AI 出牌”，先不要打开宏观操作和商店购买，避免战斗外误操作影响展示。</li>
            <li>如果演示 LLM：先测试连接，再启用模型决策；动作选择用“只从合法候选动作里选”；确认建议靠谱后再打开自动战斗。</li>
            <li>一局结束后看“最近 Run”和质量标记。明显坏局要丢弃，正常局可以一键打包数据库提交给维护者。</li>
          </ol>
        </div>
        <div class="project-section">
          <div class="project-section-title">关键开关说明</div>
          <div class="project-grid">
            <div class="project-item"><b>允许 AI 出牌</b><span>基础 AI 是否可以在战斗中自动出牌。这个开关不等于 LLM。</span></div>
            <div class="project-item"><b>允许 AI 宏观操作</b><span>地图、奖励、选卡、事件、营火等战斗外行为。演示前期建议关闭。</span></div>
            <div class="project-item"><b>采集总开关</b><span>是否把后续动作写入训练日志。调 UI 时可以关，正式采样要开。</span></div>
            <div class="project-item"><b>启用模型决策</b><span>是否请求 LLM。关闭后 API 配置保留，但不会继续消耗请求。</span></div>
            <div class="project-item"><b>动作选择</b><span>推荐“只从合法候选动作里选”。兼容模式保留给调试，不适合演示稳定性。</span></div>
            <div class="project-item"><b>允许 LLM 自动战斗</b><span>打开后 LLM 建议会被执行；关闭时只显示建议和理由。</span></div>
          </div>
        </div>
        <div class="project-section">
          <div class="project-section-title">数据和训练现在缺什么</div>
          <div class="project-copy">
            <p>现在最缺的是大量高质量 Run，尤其是一关 Boss 前后的不同牌组、不同药水、不同怪物组合。失败局也有价值，但必须标清楚质量，不能把卡死或误操作当成正常样本。</p>
            <p>怪物数据已经开始采集，后续会把怪物名称、意图、攻击模式和打法经验接入训练。这样同类怪物可以共享打法，而不是每次都只靠当前手牌做短视判断。</p>
            <p>LLM 强化的重点不是让它自由发挥，而是让它读懂当前局面：手牌、弃牌堆、抽牌堆、药水、敌人意图、斩杀差、当前风险，然后在合法动作中选择更合理的一步。</p>
          </div>
        </div>
        <div class="project-section">
          <div class="project-section-title">Git 上公开文档</div>
          <div class="project-doc-list">
            <div class="project-doc"><b><a class="project-inline-link" href="/docs/project_guide.md" target="_blank" rel="noopener">docs/project_guide.md</a></b><span>项目目标、模块分工、推荐流程和限制。</span></div>
            <div class="project-doc"><b><a class="project-inline-link" href="/docs/startup.md" target="_blank" rel="noopener">docs/startup.md</a></b><span>一键启动控制台、日志窗口、BC AI 和 LLM。</span></div>
            <div class="project-doc"><b><a class="project-inline-link" href="/docs/data_contribution.md" target="_blank" rel="noopener">docs/data_contribution.md</a></b><span>如何打包本地数据并提交给维护者。</span></div>
            <div class="project-doc"><b><a class="project-inline-link" href="/docs/public_roadmap.md" target="_blank" rel="noopener">docs/public_roadmap.md</a></b><span>公开路线图、近期目标和后续发展。</span></div>
            <div class="project-doc"><b><a class="project-inline-link" href="/docs/monster_data.md" target="_blank" rel="noopener">docs/monster_data.md</a></b><span>怪物数据采集字段和训练接入计划。</span></div>
          </div>
        </div>
        <div class="project-section">
          <div class="project-section-title">本机开发文档</div>
          <div class="project-doc-list">
            <div class="project-doc"><b><a class="project-inline-link" href="/docs/ai_training_roadmap.md" target="_blank" rel="noopener">docs/ai_training_roadmap.md</a></b><span>训练路线、阶段目标和数据策略，本机保留。</span></div>
            <div class="project-doc"><b><a class="project-inline-link" href="/docs/llm_combat_decision_mode.md" target="_blank" rel="noopener">docs/llm_combat_decision_mode.md</a></b><span>LLM 战斗决策输入、候选动作和校验逻辑。</span></div>
            <div class="project-doc"><b><a class="project-inline-link" href="/docs/fair_rewind_training_protocol.md" target="_blank" rel="noopener">docs/fair_rewind_training_protocol.md</a></b><span>公平回溯训练协议和使用边界。</span></div>
            <div class="project-doc"><b><a class="project-inline-link" href="/docs/rewind_training_mode.md" target="_blank" rel="noopener">docs/rewind_training_mode.md</a></b><span>早期回溯训练模式讨论稿。</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div id="guideOverlay" class="guide-overlay is-hidden" aria-hidden="true">
    <div id="guideSpotlight" class="guide-spotlight"></div>
    <div id="guideArrow" class="guide-arrow"></div>
    <div id="guideCard" class="guide-card" onclick="event.stopPropagation()">
      <div class="guide-kicker">
        <span>新手引导</span>
        <span id="guideProgress">1/1</span>
      </div>
      <h2 id="guideTitle">控制台入口</h2>
      <p id="guideText">按下一步查看每个区域的作用。</p>
      <div class="guide-actions">
        <button onclick="prevGuideStep()">上一步</button>
        <button id="guideNextButton" class="primary" onclick="nextGuideStep()">下一步</button>
        <button onclick="closeGuide()">结束</button>
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
function jsString(value) {
  return JSON.stringify(String(value ?? ""));
}
function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#96;");
}
function phaseInfo(game, activeRun) {
  if (!game.online) return {label:"未连接", cls:"off", detail:`游戏 API 离线：${game.error || "无响应"}`};
  const raw = String(game.state_type || "unknown");
  const lower = raw.toLowerCase();
  if (lower.includes("menu")) {
    const recent = activeRun && activeRun.run_id
      ? `；最近 Run ${activeRun.run_id} 最后更新 ${activeRun.last_time || "-"}`
      : "";
    return {
      label:"主菜单",
      cls:"warn",
      detail:`Mod API 返回主菜单，当前没有可读取的玩家/楼层数据${recent}`
    };
  }
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
const MODULE_IDS = ["ai_logic", "llm_logic", "model_status", "current_data", "runs", "records", "evaluation", "training"];
const MODULE_DEFAULT_SIZES = {
  ai_logic: "normal",
  llm_logic: "normal",
  model_status: "wide",
  current_data: "wide",
  runs: "wide",
  records: "wide",
  evaluation: "wide",
  training: "wide"
};
const MODULE_SIZE_LABELS = {
  compact: "小",
  normal: "中",
  wide: "大"
};
const MODULE_MIN_WIDTH = 360;
const MODULE_MIN_HEIGHT = 96;
const MODULE_MAX_HEIGHT = 900;
const MODULE_STORAGE_KEY = "sts2_control_panel_modules";
const MODULE_ORDER_STORAGE_KEY = "sts2_control_panel_module_order";
const GUIDE_STEPS = [
  {
    target: "#gamePhase",
    title: "先看游戏是否连接",
    text: "这里显示当前游戏状态。主菜单、地图、商店、战斗都会分开显示；如果是未连接，先启动游戏和 Mod API。"
  },
  {
    target: "#liveActivity",
    title: "确认系统正在读数据",
    text: "这里只显示最近几条采集动态，例如出牌、用药水、选卡和奖励。它不是完整日志，只是让你确认控制台正在跟游戏同步。"
  },
  {
    target: "#ai_enabled",
    title: "允许 AI 出牌",
    text: "这是战斗自动出牌总开关。只想看建议时关闭；要让基础 AI 在战斗里代打时打开。"
  },
  {
    target: "#macro_enabled",
    title: "宏观操作要单独开",
    text: "这个开关控制地图、奖励、选卡、事件、休息点等战斗外动作。演示时建议先关着，确认战斗稳定后再开。"
  },
  {
    target: "#collection_enabled",
    title: "采集总开关",
    text: "打开后才写入训练数据。临时测试、不想污染数据时可以关闭；正式打样本时保持打开。"
  },
  {
    target: "#llm_enabled",
    title: "启用模型决策",
    text: "这个开关只决定是否请求大模型。接口、Key、Model 可以保留，关掉后 LLM 进程待机，不会继续消耗请求。"
  },
  {
    target: "#llm_action_selection_mode",
    title: "动作选择模式",
    text: "推荐用候选动作模式：系统先生成合法动作，LLM 只能从里面选，不能自由编参数。兼容模式保留给对比和调试。"
  },
  {
    target: "#llm_execute_combat",
    title: "允许 LLM 自动战斗",
    text: "打开后 LLM 的战斗建议会被执行；关闭时只显示建议和理由。第一次演示建议先关着看几轮判断。"
  },
  {
    target: ".module-item[data-module-target='ai_logic']",
    title: "左侧工作区可以管理面板",
    text: "这些卡片可以点击打开，也可以拖到右侧工作区。右侧卡片的标题可以拖动排序，小/中/大按钮可以调整宽度。"
  },
  {
    target: ".module-item[data-module-target='model_status']",
    title: "模型状态也在决策区",
    text: "AI 模型状态现在归到决策分类里。打开后可以直接看到战斗模型、候选动作模型、宏观模型和 Python 依赖是否齐。"
  },
  {
    target: "#module-runs",
    module: "runs",
    title: "最近 Run 和数据检查",
    text: "这里看最近一次运行、质量标记和是否保留。训练前先检查这里，避免把明显坏数据混进训练。"
  }
];
function openProjectGuide() {
  const modal = document.getElementById("projectGuideModal");
  modal.classList.add("open");
  modal.setAttribute("aria-hidden", "false");
  const body = modal.querySelector(".project-modal-body");
  if (body) body.scrollTop = 0;
  updateModalLock();
}
function closeProjectGuide() {
  const modal = document.getElementById("projectGuideModal");
  modal.classList.remove("open");
  modal.setAttribute("aria-hidden", "true");
  updateModalLock();
}
function anyModalOpen() {
  const project = document.getElementById("projectGuideModal");
  const editor = document.getElementById("llmProfileEditor");
  const guide = document.getElementById("guideOverlay");
  return !!(
    (project && project.classList.contains("open")) ||
    (editor && editor.classList.contains("open")) ||
    (guide && !guide.classList.contains("is-hidden"))
  );
}
function updateModalLock() {
  const locked = anyModalOpen();
  document.documentElement.classList.toggle("modal-lock", locked);
  document.body.classList.toggle("modal-lock", locked);
}
function defaultModuleState() {
  return Object.fromEntries(MODULE_IDS.map(id => [id, {
    open:true,
    collapsed:false,
    size: MODULE_DEFAULT_SIZES[id] || "wide",
    width:null,
    height:null
  }]));
}
function readModuleState() {
  const state = defaultModuleState();
  try {
    const saved = JSON.parse(localStorage.getItem(MODULE_STORAGE_KEY) || "{}");
    for (const id of MODULE_IDS) {
      if (saved[id]) {
        state[id].open = saved[id].open !== false;
        state[id].collapsed = !!saved[id].collapsed;
        if (MODULE_SIZE_LABELS[saved[id].size]) state[id].size = saved[id].size;
        if (Number.isFinite(saved[id].width)) state[id].width = Math.max(MODULE_MIN_WIDTH, saved[id].width);
        if (Number.isFinite(saved[id].height)) state[id].height = Math.max(MODULE_MIN_HEIGHT, saved[id].height);
      }
    }
  } catch (_) {}
  return state;
}
function readModuleOrder() {
  try {
    const saved = JSON.parse(localStorage.getItem(MODULE_ORDER_STORAGE_KEY) || "[]");
    if (Array.isArray(saved)) {
      const filtered = saved.filter(id => MODULE_IDS.includes(id));
      return [...filtered, ...MODULE_IDS.filter(id => !filtered.includes(id))];
    }
  } catch (_) {}
  return [...MODULE_IDS];
}
let moduleState = readModuleState();
let moduleOrder = readModuleOrder();
let draggingModuleId = "";
let draggingCardId = "";
let dragGhost = null;
let lastDropTargetId = "";
let lastDropPlaceAfter = false;
let resizingModuleId = "";
let resizeState = null;
let lastSpeedControlKey = "";
function saveModuleState() {
  localStorage.setItem(MODULE_STORAGE_KEY, JSON.stringify(moduleState));
}
function saveModuleOrder() {
  localStorage.setItem(MODULE_ORDER_STORAGE_KEY, JSON.stringify(moduleOrder));
}
function moduleElement(id) {
  return document.querySelector(`.module-card[data-module="${id}"]`);
}
function clampNumber(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
function smoothstep01(value) {
  const t = clampNumber(value, 0, 1);
  return t * t * (3 - 2 * t);
}
function motionDurationFromDistance(dx, dy) {
  const distance = Math.sqrt(dx * dx + dy * dy);
  return Math.round(150 + smoothstep01(distance / 900) * 170);
}
function workspaceInnerWidth() {
  const workspace = document.getElementById("workspace");
  return workspace ? Math.max(MODULE_MIN_WIDTH, workspace.clientWidth) : 1200;
}
function applyModuleLayout(card, state) {
  if (!card) return;
  const customWidth = Number.isFinite(state.width);
  const customHeight = Number.isFinite(state.height) && !state.collapsed;
  card.classList.toggle("has-custom-width", customWidth);
  card.classList.toggle("has-custom-height", customHeight);
  if (customWidth) {
    const width = clampNumber(state.width, MODULE_MIN_WIDTH, workspaceInnerWidth());
    card.style.setProperty("--module-width", `${Math.round(width)}px`);
  } else {
    card.style.removeProperty("--module-width");
  }
  if (customHeight) {
    const height = clampNumber(state.height, MODULE_MIN_HEIGHT, MODULE_MAX_HEIGHT);
    card.style.setProperty("--module-height", `${Math.round(height)}px`);
  } else {
    card.style.removeProperty("--module-height");
  }
}
function workspaceCards() {
  return Array.from(document.querySelectorAll("#workspace .module-card"))
    .filter(card => !card.classList.contains("is-hidden"));
}
function captureWorkspaceRects() {
  const rects = new Map();
  for (const card of workspaceCards()) {
    rects.set(card.dataset.module, card.getBoundingClientRect());
  }
  return rects;
}
function animateWorkspaceFrom(rects) {
  window.requestAnimationFrame(() => {
    for (const card of workspaceCards()) {
      const first = rects.get(card.dataset.module);
      if (!first) continue;
      const last = card.getBoundingClientRect();
      const dx = first.left - last.left;
      const dy = first.top - last.top;
      if (Math.abs(dx) < 1 && Math.abs(dy) < 1) continue;
      card.classList.add("is-animating");
      const animation = card.animate(
        [
          {transform:`translate(${dx}px, ${dy}px)`},
          {transform:"translate(0, 0)"}
        ],
        {duration:motionDurationFromDistance(dx, dy), easing:"cubic-bezier(.2,.8,.2,1)"}
      );
      animation.onfinish = () => card.classList.remove("is-animating");
      animation.oncancel = () => card.classList.remove("is-animating");
    }
  });
}
function animateWorkspaceChange(mutator) {
  const rects = captureWorkspaceRects();
  const result = mutator();
  animateWorkspaceFrom(rects);
  return result;
}
function clearDragSelection() {
  const selection = window.getSelection && window.getSelection();
  if (selection) selection.removeAllRanges();
}
function setDragUi(kind) {
  document.body.classList.toggle("is-dragging-card", kind === "card");
  document.body.classList.toggle("is-dragging-module", kind === "module");
}
function clearDragUi() {
  setDragUi("");
  clearDragSelection();
  removeDropMarkers();
  if (dragGhost) {
    dragGhost.remove();
    dragGhost = null;
  }
}
function moduleDisplayName(id) {
  const card = moduleElement(id);
  const title = card && card.querySelector(":scope > .section-head h2");
  if (title && title.textContent.trim()) return title.textContent.trim();
  const dock = document.querySelector(`.module-item[data-module-target="${id}"] .module-label`);
  return dock && dock.textContent.trim() ? dock.textContent.trim() : id;
}
function createDragGhost(event, id, actionLabel) {
  if (!event.dataTransfer || !event.dataTransfer.setDragImage) return;
  if (dragGhost) dragGhost.remove();
  const size = (moduleState[id] && moduleState[id].size) || MODULE_DEFAULT_SIZES[id] || "wide";
  const ghost = document.createElement("div");
  ghost.className = "drag-ghost";
  ghost.innerHTML = `<b>${escapeHtml(moduleDisplayName(id))}</b><span>${escapeHtml(actionLabel)} / ${escapeHtml(MODULE_SIZE_LABELS[size] || "")}</span>`;
  document.body.appendChild(ghost);
  dragGhost = ghost;
  event.dataTransfer.setDragImage(ghost, 22, 20);
  window.setTimeout(() => {
    if (dragGhost === ghost) dragGhost = null;
    ghost.remove();
  }, 0);
}
function applyModuleOrder() {
  const workspace = document.getElementById("workspace");
  if (!workspace) return;
  for (const id of moduleOrder) {
    const card = moduleElement(id);
    if (card) workspace.appendChild(card);
  }
}
function injectModuleControls() {
  for (const id of MODULE_IDS) {
    const card = moduleElement(id);
    if (!card) continue;
    const head = card.querySelector(":scope > .section-head");
    const actions = card.querySelector(":scope > .section-head .module-actions");
    if (head && !head.dataset.dragReady) {
      head.dataset.dragReady = "1";
      head.draggable = true;
      head.addEventListener("dragstart", event => beginCardDrag(event, id));
      head.addEventListener("dragend", endCardDrag);
      head.addEventListener("selectstart", event => event.preventDefault());
      head.addEventListener("mousedown", event => {
        if (!event.target.closest("button,input,select,a,textarea")) clearDragSelection();
      });
      head.title = "拖动标题可以调整卡片顺序";
    }
    if (actions && !actions.querySelector(".module-size-controls")) {
      const group = document.createElement("span");
      group.className = "module-size-controls";
      group.setAttribute("aria-label", "卡片大小");
      for (const size of Object.keys(MODULE_SIZE_LABELS)) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "module-size-button";
        button.dataset.sizeControl = size;
        button.textContent = MODULE_SIZE_LABELS[size];
        button.title = `切到${MODULE_SIZE_LABELS[size]}卡片`;
        button.onclick = event => {
          event.stopPropagation();
          setModuleSize(id, size);
        };
        group.appendChild(button);
      }
      const firstActionButton = actions.querySelector(".module-action");
      actions.insertBefore(group, firstActionButton || null);
    }
    if (!card.querySelector(":scope > .module-resize-handle")) {
      const handle = document.createElement("span");
      handle.className = "module-resize-handle";
      handle.setAttribute("role", "separator");
      handle.setAttribute("aria-label", "拖动调整卡片大小");
      handle.title = "拖动调整卡片大小；双击恢复自动高度";
      handle.addEventListener("pointerdown", event => beginModuleResize(event, id));
      handle.addEventListener("dblclick", event => {
        event.preventDefault();
        event.stopPropagation();
        resetModuleDimensions(id);
      });
      card.appendChild(handle);
    }
  }
}
function syncModuleUI() {
  let openCount = 0;
  applyModuleOrder();
  injectModuleControls();
  for (const id of MODULE_IDS) {
    const state = moduleState[id] || {open:true, collapsed:false};
    const card = moduleElement(id);
    if (card) {
      card.classList.toggle("is-hidden", !state.open);
      card.classList.toggle("is-collapsed", !!state.collapsed);
      card.dataset.size = MODULE_SIZE_LABELS[state.size] ? state.size : (MODULE_DEFAULT_SIZES[id] || "wide");
      applyModuleLayout(card, state);
      for (const button of card.querySelectorAll(".module-size-button")) {
        button.classList.toggle("active", !Number.isFinite(state.width) && button.dataset.sizeControl === card.dataset.size);
      }
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
    }
  }
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
  const update = () => {
    moduleState[id] = {...(moduleState[id] || {}), open:true, collapsed:false};
    saveModuleState();
    syncModuleUI();
  };
  if (opts.animate === false) update();
  else animateWorkspaceChange(update);
  const card = moduleElement(id);
  flashModule(card);
  if (opts.scroll !== false && card) {
    card.scrollIntoView({behavior:"smooth", block:"start"});
  }
}
function setModuleSize(id, size) {
  if (!MODULE_IDS.includes(id) || !MODULE_SIZE_LABELS[size]) return;
  animateWorkspaceChange(() => {
    moduleState[id] = {...(moduleState[id] || {}), open:true, size, width:null};
    saveModuleState();
    syncModuleUI();
  });
  flashModule(moduleElement(id));
}
function resetModuleDimensions(id) {
  if (!MODULE_IDS.includes(id)) return;
  animateWorkspaceChange(() => {
    moduleState[id] = {...(moduleState[id] || {}), width:null, height:null};
    saveModuleState();
    syncModuleUI();
  });
  flashModule(moduleElement(id));
}
function beginModuleResize(event, id) {
  if (!MODULE_IDS.includes(id) || event.button !== 0) return;
  const card = moduleElement(id);
  if (!card || card.classList.contains("is-hidden")) return;
  event.preventDefault();
  event.stopPropagation();
  clearDragSelection();
  const rect = card.getBoundingClientRect();
  const workspace = document.getElementById("workspace");
  const workspaceRect = workspace ? workspace.getBoundingClientRect() : {right:window.innerWidth};
  resizingModuleId = id;
  resizeState = {
    pointerId:event.pointerId,
    startX:event.clientX,
    startY:event.clientY,
    startWidth:rect.width,
    startHeight:rect.height,
    maxWidth:Math.max(MODULE_MIN_WIDTH, workspaceRect.right - rect.left)
  };
  document.body.classList.add("is-resizing-module");
  card.classList.add("resizing-card", "has-custom-width", "has-custom-height");
  card.style.setProperty("--module-width", `${Math.round(rect.width)}px`);
  card.style.setProperty("--module-height", `${Math.round(rect.height)}px`);
  event.currentTarget.setPointerCapture && event.currentTarget.setPointerCapture(event.pointerId);
  document.addEventListener("pointermove", handleModuleResizeMove);
  document.addEventListener("pointerup", endModuleResize, {once:true});
  document.addEventListener("pointercancel", endModuleResize, {once:true});
}
function handleModuleResizeMove(event) {
  if (!resizingModuleId || !resizeState) return;
  event.preventDefault();
  const card = moduleElement(resizingModuleId);
  if (!card) return;
  const width = clampNumber(resizeState.startWidth + event.clientX - resizeState.startX, MODULE_MIN_WIDTH, resizeState.maxWidth);
  const height = clampNumber(resizeState.startHeight + event.clientY - resizeState.startY, MODULE_MIN_HEIGHT, MODULE_MAX_HEIGHT);
  card.style.setProperty("--module-width", `${Math.round(width)}px`);
  card.style.setProperty("--module-height", `${Math.round(height)}px`);
}
function endModuleResize() {
  if (!resizingModuleId || !resizeState) return;
  document.removeEventListener("pointermove", handleModuleResizeMove);
  const card = moduleElement(resizingModuleId);
  if (card) {
    const rect = card.getBoundingClientRect();
    moduleState[resizingModuleId] = {
      ...(moduleState[resizingModuleId] || {}),
      open:true,
      width:Math.round(clampNumber(rect.width, MODULE_MIN_WIDTH, workspaceInnerWidth())),
      height:Math.round(clampNumber(rect.height, MODULE_MIN_HEIGHT, MODULE_MAX_HEIGHT))
    };
    card.classList.remove("resizing-card");
  }
  saveModuleState();
  syncModuleUI();
  document.body.classList.remove("is-resizing-module");
  flashModule(moduleElement(resizingModuleId));
  resizingModuleId = "";
  resizeState = null;
}
let guideIndex = 0;
function clampGuide(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
function startGuide() {
  guideIndex = 0;
  showGuideStep();
}
function closeGuide() {
  const overlay = document.getElementById("guideOverlay");
  if (!overlay) return;
  overlay.classList.add("is-hidden");
  overlay.setAttribute("aria-hidden", "true");
  updateModalLock();
}
function nextGuideStep() {
  if (guideIndex >= GUIDE_STEPS.length - 1) {
    closeGuide();
    return;
  }
  guideIndex += 1;
  showGuideStep();
}
function prevGuideStep() {
  guideIndex = Math.max(0, guideIndex - 1);
  showGuideStep();
}
function revealGuideTarget(step) {
  if (step.module) openModule(step.module, {scroll:false});
  const target = document.querySelector(step.target);
  if (!target) return null;
  const details = target.closest("details");
  if (details) details.open = true;
  target.scrollIntoView({behavior:"smooth", block:"center", inline:"nearest"});
  return target;
}
function showGuideStep() {
  const overlay = document.getElementById("guideOverlay");
  const step = GUIDE_STEPS[guideIndex];
  if (!overlay || !step) return;
  overlay.classList.remove("is-hidden");
  overlay.setAttribute("aria-hidden", "false");
  updateModalLock();
  document.getElementById("guideProgress").textContent = `${guideIndex + 1}/${GUIDE_STEPS.length}`;
  document.getElementById("guideTitle").textContent = step.title;
  document.getElementById("guideText").textContent = step.text;
  document.getElementById("guideNextButton").textContent = guideIndex >= GUIDE_STEPS.length - 1 ? "完成" : "下一步";
  revealGuideTarget(step);
  window.setTimeout(positionGuide, 90);
  window.setTimeout(positionGuide, 360);
}
function positionGuide() {
  const overlay = document.getElementById("guideOverlay");
  if (!overlay || overlay.classList.contains("is-hidden")) return;
  const step = GUIDE_STEPS[guideIndex];
  const target = document.querySelector(step.target);
  if (!target) return;
  const rect = target.getBoundingClientRect();
  const pad = 8;
  const spot = document.getElementById("guideSpotlight");
  spot.style.left = `${Math.max(8, rect.left - pad)}px`;
  spot.style.top = `${Math.max(8, rect.top - pad)}px`;
  spot.style.width = `${Math.max(28, rect.width + pad * 2)}px`;
  spot.style.height = `${Math.max(28, rect.height + pad * 2)}px`;

  const card = document.getElementById("guideCard");
  const viewportW = window.innerWidth;
  const viewportH = window.innerHeight;
  const initialCardRect = card.getBoundingClientRect();
  let left = rect.right + 28;
  if (left + initialCardRect.width > viewportW - 14) {
    left = rect.left - initialCardRect.width - 28;
  }
  if (left < 14) {
    left = clampGuide(rect.left, 14, viewportW - initialCardRect.width - 14);
  }
  let top = clampGuide(rect.top - 18, 14, viewportH - initialCardRect.height - 14);
  if (viewportH < initialCardRect.height + 40) top = 14;
  card.style.left = `${left}px`;
  card.style.top = `${top}px`;
  window.requestAnimationFrame(() => positionGuideArrow(rect, card.getBoundingClientRect()));
}
function positionGuideArrow(targetRect, cardRect) {
  const arrow = document.getElementById("guideArrow");
  const targetX = targetRect.left + targetRect.width / 2;
  const targetY = targetRect.top + targetRect.height / 2;
  const startsOnLeft = cardRect.left > targetX;
  const startX = startsOnLeft ? cardRect.left : cardRect.right;
  const startY = clampGuide(targetY, cardRect.top + 18, cardRect.bottom - 18);
  const deltaX = targetX - startX;
  const deltaY = targetY - startY;
  const length = Math.max(24, Math.sqrt(deltaX * deltaX + deltaY * deltaY));
  arrow.style.left = `${startX}px`;
  arrow.style.top = `${startY}px`;
  arrow.style.width = `${length}px`;
  arrow.style.transform = `rotate(${Math.atan2(deltaY, deltaX)}rad)`;
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
  moduleState[id] = {...current, open:true, collapsed:!current.collapsed};
  saveModuleState();
  syncModuleUI();
  flashModule(moduleElement(id));
}
function removeDropMarkers() {
  document.querySelectorAll(".module-card.drop-target").forEach(item => {
    item.classList.remove("drop-target", "drop-before", "drop-after");
  });
  const workspace = document.getElementById("workspace");
  if (workspace) {
    workspace.classList.remove("drop-ready");
    workspace.removeAttribute("data-drop-size");
  }
  lastDropTargetId = "";
  lastDropPlaceAfter = false;
}
function pointerPlacesAfter(event, card) {
  const rect = card.getBoundingClientRect();
  const sameRow = event.clientY >= rect.top && event.clientY <= rect.bottom;
  return sameRow
    ? event.clientX > rect.left + rect.width / 2
    : event.clientY > rect.top + rect.height / 2;
}
function dropTargetInfo(event, id) {
  const targetCard = event.target.closest(".module-card");
  if (!targetCard || targetCard.dataset.module === id || targetCard.classList.contains("is-hidden")) {
    return {targetCard:null, targetId:"", placeAfter:true};
  }
  return {
    targetCard,
    targetId:targetCard.dataset.module,
    placeAfter:pointerPlacesAfter(event, targetCard)
  };
}
function suggestedDropSize(event, id, targetCard) {
  if (window.matchMedia("(max-width: 1120px)").matches) return "wide";
  const targetSize = targetCard && targetCard.dataset.size;
  if (MODULE_SIZE_LABELS[targetSize]) return targetSize;
  const currentSize = moduleState[id] && moduleState[id].size;
  if (MODULE_SIZE_LABELS[currentSize]) return currentSize;
  return MODULE_DEFAULT_SIZES[id] || "wide";
}
function suggestedDropDimensions(id, targetCard) {
  const size = suggestedDropSize(null, id, targetCard);
  if (!targetCard) return {size, width:null, height:null};
  const targetState = moduleState[targetCard.dataset.module] || {};
  return {
    size,
    width:Number.isFinite(targetState.width) ? targetState.width : null,
    height:Number.isFinite(targetState.height) ? targetState.height : null
  };
}
function markDropTarget(event, id) {
  removeDropMarkers();
  const workspace = document.getElementById("workspace");
  if (!workspace) return {targetCard:null, targetId:"", placeAfter:true, size:MODULE_DEFAULT_SIZES[id] || "wide", width:null, height:null};
  workspace.classList.add("drop-ready");
  const info = dropTargetInfo(event, id);
  const dimensions = suggestedDropDimensions(id, info.targetCard);
  workspace.dataset.dropSize = dimensions.size;
  if (info.targetCard) {
    info.targetCard.classList.add("drop-target", info.placeAfter ? "drop-after" : "drop-before");
    lastDropTargetId = info.targetId;
    lastDropPlaceAfter = info.placeAfter;
  }
  return {...info, ...dimensions};
}
function beginModuleDrag(event, id) {
  draggingModuleId = id;
  draggingCardId = "";
  clearDragSelection();
  setDragUi("module");
  event.dataTransfer.setData("text/plain", id);
  event.dataTransfer.setData("application/x-sts2-module-open", id);
  event.dataTransfer.effectAllowed = "copy";
  createDragGhost(event, id, "打开卡片");
  event.currentTarget.classList.add("dragging");
}
function endModuleDrag(event) {
  draggingModuleId = "";
  if (event && event.currentTarget) event.currentTarget.classList.remove("dragging");
  clearDragUi();
}
function beginCardDrag(event, id) {
  if (!MODULE_IDS.includes(id)) return;
  if (event.target.closest("button,input,select,a,textarea")) {
    event.preventDefault();
    return;
  }
  draggingCardId = id;
  draggingModuleId = "";
  clearDragSelection();
  setDragUi("card");
  event.dataTransfer.setData("text/plain", id);
  event.dataTransfer.setData("application/x-sts2-module-card", id);
  event.dataTransfer.effectAllowed = "move";
  createDragGhost(event, id, "移动卡片");
  const card = moduleElement(id);
  if (card) card.classList.add("dragging-card");
}
function endCardDrag() {
  const card = moduleElement(draggingCardId);
  if (card) card.classList.remove("dragging-card");
  draggingCardId = "";
  clearDragUi();
}
function reorderModuleCard(id, targetId, placeAfter) {
  if (!MODULE_IDS.includes(id)) return;
  const nextOrder = moduleOrder.filter(item => item !== id);
  const targetIndex = nextOrder.indexOf(targetId);
  if (targetIndex < 0) {
    nextOrder.push(id);
  } else {
    nextOrder.splice(targetIndex + (placeAfter ? 1 : 0), 0, id);
  }
  moduleOrder = nextOrder;
  saveModuleOrder();
  applyModuleOrder();
}
function handleWorkspaceDragOver(event) {
  const id = draggingModuleId || draggingCardId || event.dataTransfer.getData("text/plain");
  if (!MODULE_IDS.includes(id)) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = draggingCardId ? "move" : "copy";
  markDropTarget(event, id);
}
function handleWorkspaceDragLeave(event) {
  const rect = event.currentTarget.getBoundingClientRect();
  const inside = event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
  if (!inside) {
    removeDropMarkers();
  }
}
function handleWorkspaceDrop(event) {
  const id = draggingModuleId || draggingCardId || event.dataTransfer.getData("text/plain");
  if (!MODULE_IDS.includes(id)) return;
  event.preventDefault();
  const info = markDropTarget(event, id);
  removeDropMarkers();
  if (draggingCardId) {
    const movedId = draggingCardId;
    animateWorkspaceChange(() => {
      if (info.targetId) reorderModuleCard(movedId, info.targetId, info.placeAfter);
      else reorderModuleCard(movedId, "", true);
      const movedCard = moduleElement(movedId);
      if (movedCard) movedCard.classList.remove("dragging-card");
      draggingCardId = "";
      syncModuleUI();
    });
    clearDragUi();
    flashModule(moduleElement(movedId));
    return;
  }
  draggingModuleId = "";
  animateWorkspaceChange(() => {
    moduleState[id] = {
      ...(moduleState[id] || {}),
      open:true,
      collapsed:false,
      size:info.size,
      width:info.width,
      height:info.height
    };
    saveModuleState();
    if (info.targetId) reorderModuleCard(id, info.targetId, info.placeAfter);
    else reorderModuleCard(id, "", true);
    syncModuleUI();
  });
  clearDragUi();
  flashModule(moduleElement(id));
}
let llmFormDirty = false;
let llmProfilesCache = [];
let refreshInFlight = false;
let refreshPending = false;
let forceModelHealthRefreshUntil = 0;
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
function compactAppVersion(label, commit, dirty) {
  if (commit) return `${commit}${dirty ? "*" : ""}`;
  const text = String(label || "-");
  return text.length > 18 ? `${text.slice(0, 15)}...` : text;
}
function compactTrainingDetail(detail) {
  const text = String(detail || "");
  return text.replace(/，/g, " / ");
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
  const phase = phaseInfo(s.game, active);
  const appVersion = s.app_version || {};
  const trainingVersion = s.training_version || {};

  document.getElementById("ai_enabled").checked = !!s.control.ai_enabled;
  document.getElementById("macro_enabled").checked = !!s.control.macro_enabled;
  document.getElementById("macro_shop_enabled").checked = !!s.control.macro_shop_enabled;
  document.getElementById("collection_enabled").checked = !!s.control.collection_enabled;
  document.getElementById("record_ai_actions").checked = !!s.control.record_ai_actions;
  document.getElementById("include_ai_in_training").checked = !!s.control.include_ai_in_training;
  document.getElementById("game_speed_enabled").checked = !!s.control.game_speed_enabled;
  document.getElementById("game_speed_multiplier").value = String(s.control.game_speed_multiplier || 2);
  document.getElementById("min_training_quality").value = s.control.min_training_quality || "unknown";
  ensureSelfPlaySeedField();
  ensurePPOFields();
  document.getElementById("self_play_character").value = s.control.self_play_character || "IRONCLAD";
  document.getElementById("self_play_seed").value = s.control.self_play_seed || "";
  document.getElementById("policy_mode").value = s.control.policy_mode || "current_rl";
  document.getElementById("ppo_seed_mode").value = s.control.ppo_seed_mode || "fixed";
  document.getElementById("ppo_fixed_seed").value = s.control.ppo_fixed_seed || "101";
  document.getElementById("self_play_target_runs").value = Number(s.control.self_play_target_runs ?? 0);
  document.getElementById("self_play_train_every_admitted_runs").value = Number(s.control.self_play_train_every_admitted_runs || 5);
  document.getElementById("self_play_max_run_minutes").value = Number(s.control.self_play_max_run_minutes || 75);
  document.getElementById("self_play_stall_seconds").value = Number(s.control.self_play_stall_seconds || 120);
  document.getElementById("self_play_game_speed_multiplier").value = String(s.control.self_play_game_speed_multiplier || 3);
  document.getElementById("exploration_enabled").checked = !!s.control.exploration_enabled;
  document.getElementById("self_play_constraint_mode").value = s.control.self_play_constraint_mode || "explore";
  document.getElementById("combat_exploration_epsilon").value = Number(s.control.combat_exploration_epsilon ?? 0.35);
  document.getElementById("macro_exploration_epsilon").value = Number(s.control.macro_exploration_epsilon ?? 0.25);
  document.getElementById("exploration_top_k").value = Number(s.control.exploration_top_k || 5);
  document.getElementById("exploration_temperature").value = Number(s.control.exploration_temperature || 1.35);
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
  const speedLabel = s.control.game_speed_enabled ? `${s.control.game_speed_multiplier || 2}x` : "1x";
  lastSpeedControlKey = `${!!s.control.game_speed_enabled}:${Number(s.control.game_speed_multiplier || 2)}`;
  document.getElementById("aiDetail").textContent = s.ai_pid ? `托管进程 PID ${s.ai_pid}；宏观 ${s.control.macro_enabled ? "开启" : "关闭"}；商店 ${s.control.macro_shop_enabled ? "允许" : "保护"}；速度 ${speedLabel}` : `AI 进程未由控制台托管；速度 ${speedLabel}`;
  const speedDetail = document.getElementById("speedDetail");
  if (speedDetail) speedDetail.textContent = s.control.game_speed_enabled
    ? `已配置 ${speedLabel}：游戏时间加速，AI 等待同步缩短`
    : "正常速度：游戏 1x，AI 使用标准等待";
  document.getElementById("collectStatus").textContent = s.control.collection_enabled ? "采集中" : "已暂停";
  document.getElementById("collectStatus").className = `status-main ${s.control.collection_enabled ? "on" : "off"}`;
  document.getElementById("collectDetail").textContent = s.control.collection_enabled
    ? (s.control.include_ai_in_training ? "采集开启；下次重训会纳入合格 AI 样本" : "采集开启；下次重训只使用 Human 样本")
    : "不会写入新的战斗/宏观日志";
  document.getElementById("runQuality").textContent = active ? (active.quality_label || active.quality || "-") : "无 run";
  document.getElementById("runQuality").className = `status-main ${active && active.discarded ? "off" : "info"}`;
  document.getElementById("runDetail").textContent = active ? `Act ${active.max_act || 0} / Floor ${active.max_floor || 0}，${active.records || 0} 条` : "尚未读取到采集数据";

  setPill("lastRefresh", `已刷新 ${new Date().toLocaleTimeString()}`, "info");
  const versionInfo = document.getElementById("versionInfo");
  if (versionInfo) {
    const appFull = appVersion.label || "-";
    const appShort = compactAppVersion(appFull, appVersion.commit, appVersion.dirty);
    const trainLabel = trainingVersion.label || "-";
    const trainDetail = compactTrainingDetail(trainingVersion.detail || "");
    versionInfo.innerHTML = `
      <div class="version-item">
        <b>当前版本</b><span class="version-value">${escapeHtml(appShort)}</span>
        <span class="version-detail">${escapeHtml(appVersion.dirty ? "本地有修改" : "工作区干净")}</span>
      </div>
      <div class="version-item">
        <b>训练版本</b><span class="version-value">${escapeHtml(trainLabel)}</span>
        <span class="version-detail">${escapeHtml(trainDetail || "暂无训练摘要")}</span>
      </div>`;
    versionInfo.title = [
      `当前版本：${appFull}`,
      appVersion.control_panel_mtime ? `控制台文件：${appVersion.control_panel_mtime}` : "",
      `训练版本：${trainingVersion.label || "-"}`,
      trainingVersion.latest_model ? `最新模型：${trainingVersion.latest_model}` : "",
      trainingVersion.latest_mtime ? `训练时间：${trainingVersion.latest_mtime}` : "",
      trainingVersion.detail || ""
    ].filter(Boolean).join("\n");
  }
  setPill("aiProcessBadge", s.ai_pid ? (s.ai_process && s.ai_process.needs_restart ? "需重启" : "运行中") : "未启动", s.ai_pid ? ((s.ai_process && s.ai_process.needs_restart) ? "warn" : "on") : "warn");
  setPill("llmProcessBadge", s.llm && s.llm.pid ? "运行中" : "未启动", s.llm && s.llm.pid ? "on" : "warn");
  setPill(
    "collectBadge",
    s.control.collection_enabled ? (s.control.include_ai_in_training ? "AI入训" : "AI不入训") : "暂停",
    s.control.collection_enabled ? (s.control.include_ai_in_training ? "warn" : "on") : "off"
  );
  const runMode = s.control.next_run_mode || "auto";
  const runModeLabel = runMode === "new" ? "强制新局一次" : (runMode === "continue" ? "续接旧 Run" : "自动检测");
  setPill("nextRunBadge", runModeLabel, runMode === "new" ? "warn" : "info");
  document.getElementById("modeAuto").className = runMode === "auto" ? "active" : "";
  document.getElementById("modeNew").className = runMode === "new" ? "active" : "";
  document.getElementById("modeContinue").className = runMode === "continue" ? "active" : "";
  renderExport(s.export);
  renderUpdate(s.update);

  renderCurrentData(s.current_data);
  renderRuns(s.runs || []);
  renderPolicyEvaluation(s.evaluation || {});
  renderLiveActivity(s.recent_records || []);
  renderRecentRecords(s.recent_records || []);
  renderModelHealthV2(s.models || {}, s.ai_process || {}, s.control || {}, s.python_runtime || {}, s.monster_profiles || {});
  renderAiLogic(s.ai_logic);
  renderLLMLogic(s.llm && s.llm.logic, llmCfg);
  renderTrainingComposition(s.training_composition || {});
  renderSelfPlay(s.self_play || {}, s.control || {}, s.ppo || {}, s.ppo_training || {});

  setPill("trainStatus", s.training.running ? `训练中 ${s.training.started || ""}` : (s.training.finished ? `完成 ${s.training.finished}` : "未运行"), s.training.running ? "warn" : "info");
  document.getElementById("trainOutput").textContent = s.training.output || "暂无输出";
}
function renderTrainingComposition(comp) {
  const targets = ["trainingComposition", "trainCompositionMain"]
    .map(id => document.getElementById(id))
    .filter(Boolean);
  if (!targets.length) return;
  const emptyHtml = `<div class="comp-empty">尚未构建训练数据。点击"重建数据 + 重训"生成。</div>`;
  if (!comp || !comp.has_data) {
    targets.forEach(el => { el.innerHTML = emptyHtml; });
    return;
  }
  const settings = comp.settings || {};
  const combat = comp.combat || {};
  const macro = comp.macro || {};
  const nextIncludeAi = !!settings.include_ai;
  const combatIncludeAi = !!combat.include_ai;
  const macroIncludeAi = !!macro.include_ai;
  const lastIncludeAi = combatIncludeAi || macroIncludeAi;
  const mixedLastIncludeAi = combatIncludeAi !== macroIncludeAi;
  const totalCombat = combat.total_samples || 0;
  const humanCombat = combat.human_samples || 0;
  const aiCombat = combat.ai_samples || 0;
  const humanPct = totalCombat ? Math.round(humanCombat * 100 / totalCombat) : 0;
  const aiPct = totalCombat ? Math.round(aiCombat * 100 / totalCombat) : 0;
  const totalMacro = macro.total_samples || 0;
  const lastAiPill = mixedLastIncludeAi
    ? '<span class="pill warn">上次部分含 AI</span>'
    : (lastIncludeAi ? '<span class="pill on">上次含 AI</span>' : '<span class="pill off">上次未含 AI</span>');
  const nextAiPill = nextIncludeAi
    ? '<span class="pill warn">下次含 AI</span>'
    : '<span class="pill info">下次仅 Human</span>';
  const configPill = nextIncludeAi === lastIncludeAi
    ? '<span class="pill info">设置已对齐</span>'
    : '<span class="pill warn">开关已变更</span>';
  const qualityLabels = {
    "failed_run": "失败也要", "unknown": "未知及以上", "before_act1_boss": "一关Boss前",
    "partial_act1": "一关Boss", "partial_act2": "二关Boss", "perfect_run": "通关完美"
  };
  const minQLabel = qualityLabels[settings.min_quality] || settings.min_quality || "-";
  const aiMinQLabel = qualityLabels[settings.ai_min_quality] || settings.ai_min_quality || "-";
  const buildTimestamp = combat.build_timestamp || macro.build_timestamp || "";
  const buildTime = buildTimestamp ? buildTimestamp.replace("T", " ") : "";
  const buildNotes = [
    combat.data_file_count ? `战斗文件 ${combat.data_file_count}` : "",
    combat.build_elapsed_sec != null ? `战斗耗时 ${combat.build_elapsed_sec}s` : "",
    macro.data_file_count ? `宏观文件 ${macro.data_file_count}` : "",
    macro.build_elapsed_sec != null ? `宏观耗时 ${macro.build_elapsed_sec}s` : "",
  ].filter(Boolean).join(" · ");
  const combatRuns = combat.runs || [];
  const macroRuns = macro.runs || [];
  const legacyMetaNote = (!combatRuns.length && !macroRuns.length && (totalCombat || totalMacro))
    ? "当前模型 metadata 是旧格式；重建数据后会显示每个 Run 的来源、质量和样本数。"
    : "";
  const runRows = combatRuns.map(r => {
    const srcIcon = r.source === "ai" ? "🤖" : "🧑";
    const qLabel = qualityLabels[r.quality] || r.quality || "-";
    return `<tr><td>${srcIcon} ${escapeHtml((r.run_id||"").substring(0,28))}</td><td>${escapeHtml(qLabel)}</td><td>${r.samples}</td></tr>`;
  }).join("");
  const macroRunRows = macroRuns.map(r => {
    const srcIcon = r.source === "ai" ? "🤖" : "🧑";
    const qLabel = qualityLabels[r.quality] || r.quality || "-";
    return `<tr><td>${srcIcon} ${escapeHtml((r.run_id||"").substring(0,28))}</td><td>${escapeHtml(qLabel)}</td><td>${r.samples}</td></tr>`;
  }).join("");
  const html = `
    <div class="comp-header">
      <span>📊 上次训练数据配比</span>
      ${lastAiPill}
      ${nextAiPill}
      ${configPill}
      <span class="pill info">Human 最低 ${escapeHtml(minQLabel)}</span>
      <span class="pill info">AI 最低 ${escapeHtml(aiMinQLabel)}</span>
      ${buildTime ? `<span class="fine" style="margin-left:auto">${escapeHtml(buildTime)}</span>` : ""}
    </div>
    ${buildNotes ? `<div class="fine" style="margin-bottom:8px">${escapeHtml(buildNotes)}</div>` : ""}
    ${legacyMetaNote ? `<div class="fine" style="margin-bottom:8px">${escapeHtml(legacyMetaNote)}</div>` : ""}
    <div class="comp-bar">
      <div class="bar-human" style="width:${humanPct}%" title="人类 ${humanPct}%"></div>
      <div class="bar-ai" style="width:${aiPct}%" title="AI ${aiPct}%"></div>
    </div>
    <div class="comp-stats">
      <div class="comp-stat">
        <div class="stat-value">${totalCombat}</div>
        <div class="stat-label">战斗样本</div>
      </div>
      <div class="comp-stat">
        <div class="stat-value" style="color:#3b82f6">${humanCombat}</div>
        <div class="stat-label">🧑 人类 (${humanPct}%)</div>
      </div>
      <div class="comp-stat">
        <div class="stat-value" style="color:#f59e0b">${aiCombat}</div>
        <div class="stat-label">🤖 AI (${aiPct}%)</div>
      </div>
      <div class="comp-stat">
        <div class="stat-value">${totalMacro}</div>
        <div class="stat-label">宏观样本</div>
      </div>
      <div class="comp-stat">
        <div class="stat-value">${combat.run_count || 0}</div>
        <div class="stat-label">战斗 Run</div>
      </div>
      <div class="comp-stat">
        <div class="stat-value">${macro.run_count || 0}</div>
        <div class="stat-label">宏观 Run</div>
      </div>
    </div>
    ${combatRuns.length ? `
    <details style="margin-top:4px">
      <summary style="font-size:12px;cursor:pointer;color:var(--muted)">▶ 战斗 Run 明细（${combatRuns.length} 个）</summary>
      <div class="comp-runs">
        <table><thead><tr><th>Run</th><th>质量</th><th>样本数</th></tr></thead>
        <tbody>${runRows}</tbody></table>
      </div>
    </details>` : ""}
    ${macroRuns.length ? `
    <details style="margin-top:4px">
      <summary style="font-size:12px;cursor:pointer;color:var(--muted)">▶ 宏观 Run 明细（${macroRuns.length} 个）</summary>
      <div class="comp-runs">
        <table><thead><tr><th>Run</th><th>质量</th><th>样本数</th></tr></thead>
        <tbody>${macroRunRows}</tbody></table>
      </div>
    </details>` : ""}
  `;
  targets.forEach(el => { el.innerHTML = html; });
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
function renderUpdate(info) {
  const output = document.getElementById("updateOutput");
  if (!output) return;
  if (!info || (!info.running && !info.finished && !info.output)) {
    setPill("updateBadge", "未运行", "info");
    output.textContent = "暂无更新记录";
    return;
  }
  if (info.running) {
    setPill("updateBadge", "更新中", "warn");
  } else if (info.status === "ok") {
    setPill("updateBadge", "已完成", "on");
  } else if (info.status === "error") {
    setPill("updateBadge", "失败", "off");
  } else {
    setPill("updateBadge", "已结束", "info");
  }
  output.textContent = info.output || (info.finished ? `更新结束：${info.finished}` : "暂无更新输出");
}
function selfPlayReasonBucket(score) {
  const group = String((score && score.reason_group) || "").toLowerCase();
  if (group) return group;
  const reason = String((score && score.reason) || "").toLowerCase();
  if (reason.startsWith("death")) return "death";
  if (reason === "invalid_actions") return "illegal_action";
  if (reason === "stuck_or_no_floor") return "stuck";
  if (reason === "clear_or_probable_clear") return "clear";
  if (reason === "reached_act2" || reason === "floor18_plus") return "progress";
  if (reason === "non_ai_run") return "non_ai";
  return "early_failure";
}
function selfPlayReasonLabel(code) {
  const map = {
    non_ai_run: "非 AI 局",
    invalid_actions: "非法动作",
    stuck_or_no_floor: "卡死 / 无进度",
    death: "死亡",
    death_before_act2: "死亡（A1）",
    death_after_act2: "死亡（A2+）",
    clear_or_probable_clear: "通关 / 疑似通关",
    reached_act2: "到达 A2",
    floor18_plus: "F18+",
    early_failed_before_act2: "早期失败",
  };
  return map[code] || code || "-";
}
function selfPlayReasonGroupLabel(code) {
  const map = {
    non_ai: "非 AI",
    illegal_action: "非法动作",
    stuck: "卡死",
    death: "死亡",
    early_failure: "早期失败",
    progress: "进度",
    clear: "通关",
  };
  return map[code] || code || "-";
}
function selfPlayTrendText(scores) {
  const recent = (scores || []).slice(0, 6);
  const counts = {};
  recent.forEach(score => {
    const bucket = selfPlayReasonBucket(score);
    counts[bucket] = (counts[bucket] || 0) + 1;
  });
  const order = ["death", "stuck", "illegal_action", "early_failure", "progress", "clear", "non_ai"];
  const parts = order
    .map(key => counts[key] ? `${selfPlayReasonGroupLabel(key)} ${counts[key]}` : "")
    .filter(Boolean);
  return parts.length ? `近 ${recent.length} 局：${parts.join(" / ")}` : "近 6 局：暂无";
}
function ensureSelfPlaySeedField() {
  if (document.getElementById("self_play_seed")) return;
  const characterField = document.getElementById("self_play_character")?.closest(".field");
  if (!characterField || !characterField.parentElement) return;
  const field = document.createElement("div");
  field.className = "field";
  field.innerHTML = '<span>固定 Seed</span><input id="self_play_seed" type="text" maxlength="64" onchange="saveSelfPlayConfig()">';
  characterField.parentElement.insertBefore(field, characterField.nextSibling);
}
function ensurePPOFields() {
  const characterField = document.getElementById("self_play_character")?.closest(".field");
  if (!characterField || !characterField.parentElement) return;
  const parent = characterField.parentElement;
  const summary = document.getElementById("selfPlaySummary");
  let ppoBlock = document.getElementById("ppoConfigBlock");
  if (!ppoBlock && summary && summary.parentElement) {
    ppoBlock = document.createElement("div");
    ppoBlock.id = "ppoConfigBlock";
    ppoBlock.style.marginTop = "10px";
    ppoBlock.innerHTML = `
      <div class="field">
        <span>策略模式</span>
        <select id="policy_mode" onchange="saveSelfPlayConfig()">
          <option value="current_rl">当前 RL</option>
          <option value="ppo_experiment">PPO 实验</option>
          <option value="ppo_best">PPO best</option>
        </select>
      </div>
      <div class="field">
        <span>PPO Seed 模式</span>
        <select id="ppo_seed_mode" onchange="saveSelfPlayConfig()">
          <option value="fixed">固定 seed</option>
          <option value="random">随机 seed</option>
        </select>
      </div>
      <div class="field">
        <span>PPO 固定 Seed</span>
        <input id="ppo_fixed_seed" type="text" maxlength="64" onchange="saveSelfPlayConfig()">
      </div>`;
    summary.parentElement.insertBefore(ppoBlock, summary);
  }
  if (ppoBlock) {
    ["policy_mode", "ppo_seed_mode", "ppo_fixed_seed"].forEach(id => {
      const control = document.getElementById(id);
      const field = control?.closest(".field");
      if (field && field.parentElement !== ppoBlock) {
        ppoBlock.appendChild(field);
      }
    });
  }
  if (!document.getElementById("policy_mode")) {
    const field = document.createElement("div");
    field.className = "field";
    field.innerHTML = '<span>策略模式</span><select id="policy_mode" onchange="saveSelfPlayConfig()"><option value="current_rl">当前 RL</option><option value="ppo_experiment">PPO 实验</option><option value="ppo_best">PPO best</option></select>';
    parent.insertBefore(field, characterField);
  }
  if (!document.getElementById("ppo_seed_mode")) {
    const field = document.createElement("div");
    field.className = "field";
    field.innerHTML = '<span>PPO Seed 模式</span><select id="ppo_seed_mode" onchange="saveSelfPlayConfig()"><option value="fixed">固定 seed</option><option value="random">随机 seed</option></select>';
    parent.insertBefore(field, characterField.nextSibling);
  }
  if (!document.getElementById("ppo_fixed_seed")) {
    const field = document.createElement("div");
    field.className = "field";
    field.innerHTML = '<span>PPO 固定 Seed</span><input id="ppo_fixed_seed" type="text" maxlength="64" onchange="saveSelfPlayConfig()">';
    const seedField = document.getElementById("self_play_seed")?.closest(".field");
    parent.insertBefore(field, seedField ? seedField.nextSibling : characterField.nextSibling);
  }
  if (!document.getElementById("ppoManualTrainBtn")) {
    const buttonRow = summary?.parentElement?.querySelector(".button-row");
    if (buttonRow) {
      const btn = document.createElement("button");
      btn.id = "ppoManualTrainBtn";
      btn.className = "primary";
      btn.type = "button";
      btn.textContent = "PPO update";
      btn.onclick = trainPPO;
      buttonRow.appendChild(btn);
    }
  }
}
function renderSelfPlay(info, control, ppo, ppoTraining) {
  const summary = document.getElementById("selfPlaySummary");
  const state = document.getElementById("selfPlayState");
  const progressEl = document.getElementById("selfPlayProgress");
  const scoresEl = document.getElementById("selfPlayRecentScores");
  if (!summary || !state) return;

  const running = !!(info && info.running);
  const phase = (info && info.phase) || "idle";
  const message = (info && info.message) || "";
  const completed = Number((info && info.completed_runs) || 0);
  const admitted = Number((info && info.admitted_runs) || 0);
  const rawTarget = Number((info && info.target_runs) || (control && control.self_play_target_runs) || 0);
  const targetLabel = rawTarget <= 0 ? "∞" : String(rawTarget);
  const pending = Number((info && info.pending_training_runs) || 0);
  const currentRun = (info && info.current_run_id) || "-";
  const currentSeed = (info && info.current_seed) || (control && control.self_play_seed) || "";
  const currentFloor = Number((info && info.current_floor) || 0);
  const lastScore = info && info.last_score;
  const lastReasonLabel = lastScore ? (lastScore.reason_label || selfPlayReasonLabel(lastScore.reason)) : "-";
  const lastStageLabel = lastScore ? (lastScore.failure_stage_label || "") : "";
  const lastAdmissionLabel = lastScore ? (lastScore.admission_reason_label || selfPlayReasonLabel(lastScore.admission_reason)) : "-";
  const scoreText = lastScore
    ? `score ${lastScore.score ?? "-"} / ${lastScore.admitted ? "入训" : "拒绝"} / ${lastReasonLabel}${lastStageLabel ? ` · ${lastStageLabel}` : ""}${lastScore.admission_reason && lastScore.admission_reason !== lastScore.reason ? ` / 判定 ${lastAdmissionLabel}` : ""}`
    : "暂无评分";
  const trendText = selfPlayTrendText((info && info.recent_scores) || []);
  const explore = [
    control && control.exploration_enabled ? "探索开" : "探索关",
    `约束 ${((control && control.self_play_constraint_mode) || "explore")}`,
    `战斗 ${Number((control && control.combat_exploration_epsilon) ?? 0.35).toFixed(2)}`,
    `宏观 ${Number((control && control.macro_exploration_epsilon) ?? 0.25).toFixed(2)}`,
    `TopK ${Number((control && control.exploration_top_k) || 5)}`,
    `T ${Number((control && control.exploration_temperature) || 1.35).toFixed(2)}`,
  ].join(" · ");

  const ppoMeta = (ppo && ppo.metadata) || {};
  const ppoLoss = ppo && ppo.loss != null ? `loss ${Number(ppo.loss).toFixed(3)}` : "loss -";
  const ppoEntropy = ppo && ppo.entropy != null ? `entropy ${Number(ppo.entropy).toFixed(3)}` : "entropy -";
  const ppoBest = (ppo && ppo.best && ppo.best.exists) ? (ppo.best.mtime || "best saved") : "best missing";
  const ppoLatest = (ppo && ppo.latest && ppo.latest.exists) ? (ppo.latest.mtime || "latest saved") : "latest missing";
  const fixedBest = (ppo && ppo.fixed_seed_best) || {};
  const ppoText = [
    `mode ${(control && control.policy_mode) || "current_rl"}`,
    `seed ${(control && control.ppo_seed_mode) || "fixed"}:${(control && control.ppo_fixed_seed) || "101"}`,
    `avgF5 ${Number((ppo && ppo.avg_floor_5) || 0).toFixed(1)}`,
    `bossD20 ${Number((ppo && ppo.avg_boss_damage_20) || 0).toFixed(1)}`,
    `clear ${Number((ppo && ppo.act1_clear_count) || 0)}`,
    `fixedBest A${fixedBest.max_act || 0}/F${fixedBest.max_floor || 0}`,
    ppoLoss,
    ppoEntropy,
    ppoTraining && ppoTraining.running ? "training running" : (ppoMeta.status ? `status ${ppoMeta.status}` : "status idle"),
    `${ppoLatest}`,
    `${ppoBest}`,
  ].join(" | ");

  const ppoHint = ((control && control.policy_mode) || "current_rl") === "current_rl"
    ? "当前未启用 PPO：请把策略模式从“当前 RL”切到“PPO 实验”，然后重启自训练。"
    : "";

  if (running) {
    setPill("selfPlayBadge", `${phase || "运行中"}`, "warn");
  } else if (phase === "finished") {
    setPill("selfPlayBadge", "已完成", "on");
  } else if (phase === "error") {
    setPill("selfPlayBadge", "错误", "off");
  } else {
    setPill("selfPlayBadge", "未启动", "info");
  }

  summary.innerHTML = `
    <div class="kv"><span>进度</span><span>${completed}/${targetLabel} 局，入训 ${admitted} 局</span></div>
    <div class="kv"><span>当前 Run</span><code>${escapeHtml(currentRun)}</code></div>
    <div class="kv"><span>下次重训</span><span>${pending ? `还差 ${pending} 个入训 run` : "达到批次或等待新样本"}</span></div>
    <div class="kv"><span>最近评分</span><span>${escapeHtml(scoreText)}</span></div>
    <div class="kv"><span>最近趋势</span><span>${escapeHtml(trendText)}</span></div>
    <div class="kv"><span>探索强度</span><span>${escapeHtml(explore)}</span></div>
    <div class="kv"><span>PPO</span><span>${escapeHtml(ppoText)}</span></div>
    ${ppoHint ? `<div class="fine" style="margin-top:4px;color:#b45309">${escapeHtml(ppoHint)}</div>` : ""}
    ${lastScore && lastScore.reason_detail ? `<div class="fine" style="margin-top:4px">${escapeHtml(lastScore.reason_detail)}</div>` : ""}
    ${message ? `<div class="fine" style="margin-top:8px">${escapeHtml(message)}</div>` : ""}
  `;

  // 实时进度：当前局 floor / 已跑时间
  if (progressEl) {
    if (running && currentRun && currentRun !== "-") {
      const runStarted = Number(info.run_started_at || 0);
      let elapsed = "";
      if (runStarted > 0) {
        const mins = Math.floor((Date.now() / 1000 - runStarted) / 60);
        elapsed = ` · 已跑 ${mins} 分钟`;
      }
      const stType = info.current_state_type || "";
      progressEl.innerHTML = `<div class="kv"><span>当前局进度</span><span>Floor ${currentFloor}${elapsed}${stType ? " · " + escapeHtml(stType) : ""}</span></div>`;
    } else {
      progressEl.innerHTML = "";
    }
  }

  // 最近 N 局评分表格
  if (scoresEl) {
    const scores = (info && info.recent_scores) || [];
    if (scores.length > 0) {
      const rows = scores.slice(0, 6).map(s => {
        const rid = (s.run_id || "").slice(-8);
        const adm = s.admitted ? '<span class="pill on" style="font-size:11px">入训</span>' : '<span class="pill off" style="font-size:11px">拒绝</span>';
        const reasonLabel = s.reason_label || selfPlayReasonLabel(s.reason);
        const reasonDetail = s.reason_detail ? `<div class="fine">${escapeHtml(s.reason_detail)}</div>` : "";
        const admissionLabel = s.admission_reason_label || selfPlayReasonLabel(s.admission_reason || "");
        const admissionDetail = admissionLabel ? `<div class="fine">${escapeHtml(admissionLabel)}</div>` : "";
        return `<tr><td><code>${rid}</code></td><td>A${s.max_act||0}/F${s.max_floor||0}</td><td>${s.score??"?"}</td><td>${escapeHtml(reasonLabel)}${reasonDetail}</td><td>${adm}${admissionDetail}</td></tr>`;
      }).join("");
      scoresEl.innerHTML = `
        <details open style="margin-top:4px">
          <summary style="cursor:pointer;font-size:13px;opacity:0.8">最近 ${scores.length} 局评分</summary>
          <table style="width:100%;font-size:12px;border-collapse:collapse;margin-top:4px">
            <thead><tr style="opacity:0.7;text-align:left"><th>Run</th><th>进度</th><th>Score</th><th>原因</th><th>结果</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </details>`;
    } else {
      scoresEl.innerHTML = "";
    }
  }

  state.textContent = JSON.stringify(info || {}, null, 2);
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
    ${(run.self_play_reason_label || run.self_play_reason_group_label || run.self_play_failure_stage_label) ? `<div class="kv"><span>自训结论</span><span>${escapeHtml([run.self_play_reason_group_label, run.self_play_reason_label, run.self_play_failure_stage_label].filter(Boolean).join(" / "))}</span></div>` : ""}
    ${run.self_play_reason_detail ? `<div class="fine" style="margin:4px 0 8px">${escapeHtml(run.self_play_reason_detail)}</div>` : ""}
    <div class="check-list">${checks}</div>
    <div class="warning-list">${warnings || '<div><span class="pill on">正常</span> 最近 run 有数据写入</div>'}</div>`;
}
function renderModelHealth(models, aiProcess, control, runtime, monsterProfiles) {
  const modelHealthDiv = document.getElementById("modelHealth");
  const forceRefresh = Date.now() < forceModelHealthRefreshUntil;
  if (!forceRefresh && modelHealthDiv && (modelHealthDiv.contains(document.activeElement) || modelHealthDiv.matches(':hover') || modelHealthDiv.querySelector('input:focus, select:focus, [data-editing="1"]'))) {
    return;
  }
  const combat = models.combat || {};
  const candidate = models.candidate || {};
  const macro = models.macro || {};
  const monster = monsterProfiles || {};
  const monsterSummary = monster.summary || {};
  const combatMeta = combat.metadata || {};
  const candidateMeta = candidate.metadata || {};
  const macroSummary = macro.summary || {};
  const macroMeta = macro.metadata || {};
  const ready = !!combat.ready && !!candidate.ready && !!macro.ready;
  const needsRestart = !!aiProcess.needs_restart;
  const warnings = [];
  if (!combat.ready) warnings.push("战斗 BC 模型缺失，需要重训。");
  if (!candidate.ready) warnings.push("候选动作评分模型缺失：AI 会回退到旧战斗 BC。");
  if (!macro.ready) warnings.push("宏观 BC 模型缺失，需要先训练宏观模型。");
  if (needsRestart) warnings.push("AI 进程早于当前 ai_agent.py，必须重启 AI 后宏观执行才会生效。");
  if (runtime && runtime.agent_ready === false) warnings.push(`当前 Python 缺少 AI 依赖：${(runtime.missing || []).join(", ")}。网页能开，但启动 AI / 重训会失败。`);
  if (control.macro_enabled && !macro.ready) warnings.push("宏观开关已打开，但宏观模型不可用。");
  if (control.macro_enabled && !control.macro_shop_enabled) warnings.push("商店保护已开启：AI 不会买东西，也不会自动离开商店。");
  setPill("modelBadge", ready ? (needsRestart ? "需重启" : "模型齐") : "缺模型", ready ? (needsRestart ? "warn" : "on") : "off");

  const restartNotice = needsRestart
    ? `<div class="notice warn"><b>需要重启 AI。</b>当前 AI 进程仍可能在跑旧代码，点击左侧“重启 AI”后宏观模型才会进入运行时。</div>`
    : "";
  const warningHtml = warnings.length
    ? `<div class="warning-list">${warnings.map(w => `<div><span class="pill warn">注意</span> ${w}</div>`).join("")}</div>`
    : `<div class="notice good">战斗模型、候选动作模型和宏观模型都已就绪。</div>`;
  const registry = models.registry || {};
  const packages = registry.packages || [];
  const activeModelId = registry.active_model_id || "local";
  const autoKeepLimit = registry.auto_keep_limit || 5;
  const packageOptions = packages.map(pkg =>
    `<option value="${escapeHtml(pkg.id)}" ${pkg.id === activeModelId ? "selected" : ""}>${escapeHtml(pkg.label || pkg.id)}</option>`
  ).join("");
  const packageRows = packages.map(pkg => {
    const summary = pkg.summary || {};
    const sources = summary.accepted_sources || {};
    const combatSources = sources.combat || {};
    const macroSources = sources.macro || {};
    const isPinned = !!pkg.pinned || pkg.retention === "manual";
    const rowId = `modelLabel_${pkg.id}`;
    return `<tr>
      <td><code>${escapeHtml(pkg.id)}</code><br><span class="fine">${escapeHtml(pkg.created_at || "")}</span></td>
      <td>
        <input id="${escapeHtml(rowId)}" type="text" value="${escapeHtml(pkg.label || pkg.id)}" style="min-width:180px;width:100%;max-width:260px">
        <div class="fine">${escapeHtml(pkg.description || "")}</div>
      </td>
      <td>战斗 ${summary.combat_samples || 0} / 候选 ${summary.candidate_rows || 0}<br><span class="fine">宏观 ${summary.macro_samples || 0}，AI ${summary.include_ai ? "已纳入" : "未纳入"}</span></td>
      <td><span class="fine">C: H${combatSources.human || 0}/AI${combatSources.ai || 0}<br>M: H${macroSources.human || 0}/AI${macroSources.ai || 0}</span></td>
      <td>
        <span class="pill ${pkg.complete ? "on" : "off"}">${pkg.complete ? "完整" : "缺文件"}</span><br>
        <span class="pill ${isPinned ? "on" : "info"}">${isPinned ? "永久保留" : "自动保留"}</span><br>
        <span class="fine">${formatBytes(pkg.size || 0)}</span>
      </td>
      <td>
        <button onclick="renameModelPackage(${jsString(pkg.id)})">保存名称</button>
        <button onclick="pinModelPackage(${jsString(pkg.id)})" ${isPinned ? "disabled" : ""}>永久保留</button>
      </td>
    </tr>`;
  }).join("");
  const modelSwitchHtml = `
    <div class="notice">
      <b>训练模型切换</b>
      <div class="field" style="margin-top:8px">
        <span>模型包</span>
        <select id="modelPackageSelect" ${packages.length ? "" : "disabled"}>
          ${packageOptions || '<option value="">暂无模型包</option>'}
        </select>
      </div>
      <div class="button-row" style="margin-top:8px">
        <button class="primary" onclick="activateModelPackage()" ${packages.length ? "" : "disabled"}>切换到选中模型</button>
        <button onclick="restartAI()">重启 AI</button>
      </div>
      <div class="field" style="margin-top:10px">
        <span>永久保存当前模型</span>
        <input id="manualModelName" type="text" placeholder="例如：第一关稳定版">
      </div>
      <div class="button-row" style="margin-top:8px">
        <button onclick="saveCurrentModelSnapshot()">永久保存</button>
      </div>
      <div id="modelSwitchResult" class="fine" style="margin-top:6px">当前启用：${escapeHtml(registry.active_label || activeModelId)}。切换后需要重启 AI 进程。</div>
      <div class="fine" style="margin-top:6px">训练完成会自动保存模型快照，只保留最近 ${autoKeepLimit} 个；点“永久保留”的模型不会被自动清理。</div>
    </div>
    <div class="table-wrap" style="margin-top:10px">
      <table>
        <thead><tr><th>ID</th><th>名称</th><th>样本</th><th>来源</th><th>状态</th><th>操作</th></tr></thead>
        <tbody>${packageRows || '<tr><td colspan=6>暂无可切换模型包</td></tr>'}</tbody>
      </table>
    </div>`;

  document.getElementById("modelHealth").innerHTML = `
    <div class="kv"><span>战斗模型</span><span><span class="pill ${combat.ready ? "on" : "off"}">${combat.ready ? "可用" : "缺失"}</span> 样本 ${combatMeta.samples || "-"}，特征 ${combatMeta.features || "旧版"} ${combat.model && combat.model.mtime ? combat.model.mtime : ""}</span></div>
    <div class="kv"><span>候选动作模型</span><span><span class="pill ${candidate.ready ? "on" : "warn"}">${candidate.ready ? "线上优先" : "回退旧 BC"}</span> 行 ${candidateMeta.samples || 0}，正例 ${candidateMeta.positives || 0}，组 ${candidateMeta.groups || 0}，Top1 ${candidateMeta.best_group_top1 ? Number(candidateMeta.best_group_top1).toFixed(1) + "%" : "-"}</span></div>
    <div class="kv"><span>宏观模型</span><span><span class="pill ${macro.ready ? "on" : "off"}">${macro.ready ? "可用" : "缺失"}</span> 样本 ${macroSummary.samples || macroMeta.samples || 0}，动作 ${macroSummary.actions || macroMeta.classes || 0}</span></div>
    <div class="kv"><span>怪物画像</span><span><span class="pill ${monster.ready ? "on" : "warn"}">${monster.ready ? "可用" : "待生成"}</span> 怪物 ${monsterSummary.monsters || 0}，战斗 ${monsterSummary.encounters || 0}，回合样本 ${monsterSummary.monster_turn_rows || 0}</span></div>
    <div class="kv"><span>Python</span><span><span class="pill ${runtime && runtime.agent_ready ? "on" : "warn"}">${runtime && runtime.agent_ready ? "依赖可用" : "缺依赖"}</span> ${escapeHtml((runtime && runtime.executable) || "-")} ${runtime && runtime.version ? `(${runtime.version})` : ""}</span></div>
    <div class="kv"><span>AI 进程</span><span>${aiProcess.pid ? `PID ${aiProcess.pid}` : "未启动"}${aiProcess.started_at ? `，启动 ${aiProcess.started_at}` : ""}</span></div>
    <div class="kv"><span>宏观执行</span><span><span class="pill ${control.macro_enabled ? "warn" : "info"}">${control.macro_enabled ? "开启" : "关闭"}</span> ${control.macro_enabled ? "会自动点地图/奖励/选卡" : "只显示战斗托管"}</span></div>
    <div class="kv"><span>商店保护</span><span><span class="pill ${control.macro_shop_enabled ? "warn" : "on"}">${control.macro_shop_enabled ? "允许购买" : "保护中"}</span> ${control.macro_shop_enabled ? "AI 可买明确商品；不自动删牌" : "AI 不碰商店，避免抢操作"}</span></div>
    ${modelSwitchHtml}
    ${restartNotice}
    ${warningHtml}`;
}
function renderModelHealthV2(models, aiProcess, control, runtime, monsterProfiles) {
  const modelHealthDiv = document.getElementById("modelHealth");
  const forceRefresh = Date.now() < forceModelHealthRefreshUntil;
  if (!forceRefresh && modelHealthDiv && (modelHealthDiv.contains(document.activeElement) || modelHealthDiv.matches(':hover') || modelHealthDiv.querySelector('input:focus, select:focus, [data-editing="1"]'))) {
    return;
  }
  const combat = models.combat || {};
  const candidate = models.candidate || {};
  const macro = models.macro || {};
  const monster = monsterProfiles || {};
  const monsterSummary = monster.summary || {};
  const combatMeta = combat.metadata || {};
  const candidateMeta = candidate.metadata || {};
  const macroSummary = macro.summary || {};
  const macroMeta = macro.metadata || {};
  const ready = !!combat.ready && !!candidate.ready && !!macro.ready;
  const needsRestart = !!aiProcess.needs_restart;
  const warnings = [];
  if (!combat.ready) warnings.push("战斗 BC 模型缺失，需要重新训练。");
  if (!candidate.ready) warnings.push("候选动作评分模型缺失，AI 会回退到旧的战斗 BC。");
  if (!macro.ready) warnings.push("宏观 BC 模型缺失，需要先训练宏观模型。");
  if (needsRestart) warnings.push("AI 进程早于当前 ai_agent.py，需要重启 AI 后新逻辑才会生效。");
  if (runtime && runtime.agent_ready === false) warnings.push(`当前 Python 缺少 AI 依赖：${(runtime.missing || []).join(", ")}。网页能开，但启动 AI 或训练会失败。`);
  if (control.macro_enabled && !macro.ready) warnings.push("宏观开关已打开，但宏观模型当前不可用。");
  if (control.macro_enabled && !control.macro_shop_enabled) warnings.push("商店保护已开启：AI 不会买东西，也不会自动离开商店。");
  setPill("modelBadge", ready ? (needsRestart ? "需重启" : "模型齐") : "缺模型", ready ? (needsRestart ? "warn" : "on") : "off");

  const restartNotice = needsRestart
    ? `<div class="notice warn"><b>需要重启 AI。</b> 当前 AI 进程可能还在运行旧代码，重启后新的模型和宏观逻辑才会进入运行时。</div>`
    : "";
  const warningHtml = warnings.length
    ? `<div class="warning-list">${warnings.map(w => `<div><span class="pill warn">注意</span> ${escapeHtml(w)}</div>`).join("")}</div>`
    : `<div class="notice good">战斗模型、候选动作模型和宏观模型都已就绪。</div>`;

  const registry = models.registry || {};
  const packages = registry.packages || [];
  const activeModelId = registry.active_model_id || "local";
  const autoKeepLimit = registry.auto_keep_limit || 5;
  const packageOptions = packages.map(pkg =>
    `<option value="${escapeHtml(pkg.id)}" ${pkg.id === activeModelId ? "selected" : ""}>${escapeHtml(pkg.label || pkg.id)}</option>`
  ).join("");
  const packageCount = packages.length;
  const importHint = packageCount
    ? `已找到 ${packageCount} 个模型包，导入后会立刻出现在列表里。`
    : "当前还没有模型包，可以先导入别人分享的 zip 模型包。";

  const packageCards = packages.map(pkg => {
    const summary = pkg.summary || {};
    const sources = summary.accepted_sources || {};
    const combatSources = sources.combat || {};
    const macroSources = sources.macro || {};
    const isPinned = !!pkg.pinned || pkg.retention === "manual";
    const isActive = pkg.id === activeModelId;
    const rowId = `modelLabel_${pkg.id}`;
    const createdDate = (pkg.created_at || "").replace("T", " ");
    return `<div class="model-pkg-card ${isActive ? 'is-active' : ''}">
      <div class="model-pkg-card-head">
        <span class="pkg-label">${escapeHtml(pkg.label || pkg.id)}</span>
        ${isActive ? '<span class="pill on">当前启用</span>' : ''}
        <span class="pill ${pkg.complete ? "on" : "off"}">${pkg.complete ? "完整" : "缺文件"}</span>
        <span class="pill ${isPinned ? "on" : "info"}">${isPinned ? "永久保留" : "自动清理"}</span>
      </div>
      <div class="model-pkg-card-meta">
        <span>战斗 ${summary.combat_samples || 0}</span>
        <span>候选 ${summary.candidate_rows || 0}</span>
        <span>宏观 ${summary.macro_samples || 0}</span>
        <span>${formatBytes(pkg.size || 0)}</span>
        <span>C:H${combatSources.human||0}/A${combatSources.ai||0}</span>
        <span>M:H${macroSources.human||0}/A${macroSources.ai||0}</span>
        <span>${summary.include_ai ? "AI已纳入" : ""}</span>
      </div>
      ${pkg.description ? `<div class="model-pkg-desc">${escapeHtml(pkg.description)}</div>` : ''}
      <div class="model-pkg-card-meta">
        <span>ID: ${escapeHtml(pkg.id)}</span>
        <span>${escapeHtml(createdDate)}</span>
      </div>
      <div class="model-pkg-card-actions">
        <input id="${escapeAttr(rowId)}" type="text" value="${escapeAttr(pkg.label || pkg.id)}" onfocus="this.dataset.editing='1'" onblur="this.dataset.editing=''">
        <button onclick="renameModelPackage(${jsString(pkg.id)})">保存名称</button>
        <button onclick="pinModelPackage(${jsString(pkg.id)})" ${isPinned ? "disabled" : ""}>永久保留</button>
        <button onclick="activateModelById(${jsString(pkg.id)})" ${isActive ? "disabled" : ""}>切换启用</button>
        <button class="off" onclick="deleteModelPackage(${jsString(pkg.id)}, ${isActive ? "true" : "false"})">删除</button>
      </div>
    </div>`;
  }).join("");

  const modelSwitchHtml = `
    <div class="model-panel">
      <div class="model-summary-grid">
        <div class="compact-card">
          <div class="compact-card-title">当前启用</div>
          <div class="row"><span class="pill info">${escapeHtml(registry.active_label || activeModelId)}</span></div>
          <div class="fine">ID: ${escapeHtml(activeModelId)}</div>
        </div>
        <div class="compact-card">
          <div class="compact-card-title">可管理模型包</div>
          <div class="row"><span class="pill on">${packageCount}</span></div>
          <div class="fine">自动保留最多 ${autoKeepLimit} 个快照，永久保留不会被清理。</div>
        </div>
      </div>
      <div class="model-manager-grid">
        <div class="model-actions-grid">
          <div class="model-panel-block">
            <h3>训练模型切换</h3>
            <div class="field">
              <span>模型包</span>
              <select id="modelPackageSelect" ${packages.length ? "" : "disabled"}>
                ${packageOptions || '<option value="">暂无模型包</option>'}
              </select>
            </div>
            <div class="button-row">
              <button class="primary" onclick="activateModelPackage()" ${packages.length ? "" : "disabled"}>切换到选中模型</button>
              <button onclick="restartAI()">重启 AI</button>
            </div>
            <div class="model-panel-copy">切换后重启 AI，运行中的 agent 才会重新加载新模型。</div>
          </div>
          <div class="model-panel-block">
            <h3>当前模型快照</h3>
            <div class="field">
              <span>永久保存当前模型</span>
              <input id="manualModelName" type="text" placeholder="例如：第一关稳定版">
            </div>
            <div class="button-row">
              <button onclick="saveCurrentModelSnapshot()">永久保存</button>
            </div>
            <div class="model-panel-copy">把当前运行目录里的模型整理成一个可切换模型包。</div>
          </div>
          <div class="model-panel-block">
            <h3>导入别人的模型包</h3>
            <div class="model-import-row">
              <input id="modelImportFile" type="file" accept=".zip,application/zip">
              <button onclick="importModelPackage()">导入 zip 模型包</button>
            </div>
            <div class="model-panel-copy">${escapeHtml(importHint)}</div>
          </div>
          <div id="modelSwitchResult" class="notice">当前启用：${escapeHtml(registry.active_label || activeModelId)}。切换后需要重启 AI 进程。</div>
        </div>
        <div class="model-panel-block">
          <h3>模型包列表（${packageCount} 个）</h3>
          <div class="model-table-wrap">
            ${packageCards || '<div class="model-table-empty">暂无可切换模型包</div>'}
          </div>
        </div>
      </div>
    </div>`;

  document.getElementById("modelHealth").innerHTML = `
    <div class="model-panel">
      <div class="kv"><span>战斗模型</span><span><span class="pill ${combat.ready ? "on" : "off"}">${combat.ready ? "可用" : "缺失"}</span> 样本 ${combatMeta.samples || "-"}，特征 ${combatMeta.features || "旧版"} ${combat.model && combat.model.mtime ? combat.model.mtime : ""}</span></div>
      <div class="kv"><span>候选动作模型</span><span><span class="pill ${candidate.ready ? "on" : "warn"}">${candidate.ready ? "线上优先" : "回退旧 BC"}</span> 行 ${candidateMeta.samples || 0}，正例 ${candidateMeta.positives || 0}，组 ${candidateMeta.groups || 0}，Top1 ${candidateMeta.best_group_top1 ? Number(candidateMeta.best_group_top1).toFixed(1) + "%" : "-"}</span></div>
      <div class="kv"><span>宏观模型</span><span><span class="pill ${macro.ready ? "on" : "off"}">${macro.ready ? "可用" : "缺失"}</span> 样本 ${macroSummary.samples || macroMeta.samples || 0}，动作 ${macroSummary.actions || macroMeta.classes || 0}</span></div>
      <div class="kv"><span>怪物画像</span><span><span class="pill ${monster.ready ? "on" : "warn"}">${monster.ready ? "可用" : "待生成"}</span> 怪物 ${monsterSummary.monsters || 0}，战斗 ${monsterSummary.encounters || 0}，回合样本 ${monsterSummary.monster_turn_rows || 0}</span></div>
      <div class="kv"><span>Python</span><span><span class="pill ${runtime && runtime.agent_ready ? "on" : "warn"}">${runtime && runtime.agent_ready ? "依赖可用" : "缺依赖"}</span> ${escapeHtml((runtime && runtime.executable) || "-")} ${runtime && runtime.version ? `(${runtime.version})` : ""}</span></div>
      <div class="kv"><span>AI 进程</span><span>${aiProcess.pid ? `PID ${aiProcess.pid}` : "未启动"}${aiProcess.started_at ? `，启动于 ${aiProcess.started_at}` : ""}</span></div>
      <div class="kv"><span>宏观执行</span><span><span class="pill ${control.macro_enabled ? "warn" : "info"}">${control.macro_enabled ? "开启" : "关闭"}</span> ${control.macro_enabled ? "会自动点地图、奖励和选卡" : "只显示战斗托管"}</span></div>
      <div class="kv"><span>商店保护</span><span><span class="pill ${control.macro_shop_enabled ? "warn" : "on"}">${control.macro_shop_enabled ? "允许购买" : "保护中"}</span> ${control.macro_shop_enabled ? "AI 可以购买明确商品，但不会自动删牌" : "AI 不碰商店，避免抢操作"}</span></div>
      ${modelSwitchHtml}
    </div>
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
    const deck = logic.deck_profile || {};
    const baseline = logic.reward_baseline || {};
    const baselineReason = (baseline.chosen_reason || []).join(" / ") || "-";
    document.getElementById("aiLogic").innerHTML = `
      <div class="kv"><span>类型</span><span class="strong">宏观决策</span></div>
      <div class="kv"><span>时间</span><span>${logic.time || "-"}</span></div>
      <div class="kv"><span>场景</span><span>${logic.state_type || "-"}</span></div>
      <div class="kv"><span>执行</span><span class="strong">${logic.chosen_action || "-"}</span></div>
      <div class="kv"><span>Payload</span><code>${payload}</code></div>
      <div class="kv"><span>原因</span><span>${logic.reason || "-"}</span></div>
      ${baseline.mode ? `<div class="kv"><span>选卡基准</span><span>${baseline.mode}，权重 ${baseline.weight ?? "-"}，分 ${baseline.chosen_score ?? "-"}</span></div>` : ""}
      ${baseline.mode ? `<div class="kv"><span>基准理由</span><span>${escapeHtml(baselineReason)}</span></div>` : ""}
      ${deck.total ? `<div class="kv"><span>牌组结构</span><span>${deck.total} 张；攻 ${deck.attack || 0} / 技 ${deck.skill || 0} / 能 ${deck.power || 0}；抽 ${deck.draw || 0} / AOE ${deck.aoe || 0}</span></div>` : ""}`;
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
async function applyGameSpeed() {
  const enabled = document.getElementById("game_speed_enabled").checked;
  const multiplier = Number(document.getElementById("game_speed_multiplier").value || 2);
  const speedDetail = document.getElementById("speedDetail");
  const result = await api("/api/game-speed", {enabled, multiplier});
  if (speedDetail) {
    if (result.status === "ok") {
      speedDetail.textContent = enabled ? `已应用 ${multiplier}x：游戏时间加速，AI 等待同步缩短` : "已恢复正常速度";
    } else {
      speedDetail.textContent = `加速配置已保存，但游戏暂未应用：${result.error || result.message || "游戏 API 离线"}`;
    }
  }
  return result;
}
function currentSpeedControlKey() {
  return `${document.getElementById("game_speed_enabled").checked}:${Number(document.getElementById("game_speed_multiplier").value || 2)}`;
}
async function saveControl(applySpeed = false) {
  const nextSpeedKey = currentSpeedControlKey();
  await api("/api/control", {
    ai_enabled: document.getElementById("ai_enabled").checked,
    macro_enabled: document.getElementById("macro_enabled").checked,
    macro_shop_enabled: document.getElementById("macro_shop_enabled").checked,
    collection_enabled: document.getElementById("collection_enabled").checked,
    record_ai_actions: document.getElementById("record_ai_actions").checked,
    include_ai_in_training: document.getElementById("include_ai_in_training").checked,
    game_speed_enabled: document.getElementById("game_speed_enabled").checked,
    game_speed_multiplier: Number(document.getElementById("game_speed_multiplier").value || 2),
    min_training_quality: document.getElementById("min_training_quality").value
  });
  if (applySpeed || nextSpeedKey !== lastSpeedControlKey) {
    await applyGameSpeed();
    lastSpeedControlKey = nextSpeedKey;
  }
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
  updateModalLock();
}
function closeLLMProfileEditor(event) {
  if (event && event.target && event.target.id !== "llmProfileEditor") return;
  document.getElementById("llmProfileEditor").classList.remove("open");
  updateModalLock();
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
async function trainPPO(){ await api("/api/ppo/train", {}); refresh(); }
async function startSelfPlay(){
  setPill("selfPlayBadge", "启动中", "warn");
  const summary = document.getElementById("selfPlaySummary");
  if (summary) summary.textContent = "正在启动自训练 runner...";
  await saveSelfPlayConfig();
  await api("/api/self-play/start", {});
  refresh();
}
async function stopSelfPlay(){
  setPill("selfPlayBadge", "停止中", "warn");
  await api("/api/self-play/stop", {});
  refresh();
}
async function saveSelfPlayConfig(){
  ensureSelfPlaySeedField();
  ensurePPOFields();
  const body = {
    self_play_character: document.getElementById("self_play_character").value,
    self_play_seed: document.getElementById("self_play_seed")?.value || "",
    policy_mode: document.getElementById("policy_mode")?.value || "current_rl",
    ppo_seed_mode: document.getElementById("ppo_seed_mode")?.value || "fixed",
    ppo_fixed_seed: document.getElementById("ppo_fixed_seed")?.value || "101",
    self_play_target_runs: Number(document.getElementById("self_play_target_runs").value || 0),
    self_play_train_every_admitted_runs: Number(document.getElementById("self_play_train_every_admitted_runs").value || 5),
    self_play_max_run_minutes: Number(document.getElementById("self_play_max_run_minutes").value || 75),
    self_play_stall_seconds: Number(document.getElementById("self_play_stall_seconds").value || 120),
    self_play_game_speed_multiplier: Number(document.getElementById("self_play_game_speed_multiplier").value || 3),
    exploration_enabled: !!document.getElementById("exploration_enabled").checked,
    exploration_mode: "aggressive",
    self_play_constraint_mode: document.getElementById("self_play_constraint_mode")?.value || "explore",
    combat_exploration_epsilon: Number(document.getElementById("combat_exploration_epsilon").value || 0.35),
    macro_exploration_epsilon: Number(document.getElementById("macro_exploration_epsilon").value || 0.25),
    exploration_top_k: Number(document.getElementById("exploration_top_k").value || 5),
    exploration_temperature: Number(document.getElementById("exploration_temperature").value || 1.35),
  };
  await api("/api/self-play/config", body);
}
async function runWorkspaceUpdate(){
  setPill("updateBadge", "更新中", "warn");
  document.getElementById("updateOutput").textContent = "正在执行仓库更新...";
  await api("/api/update", {});
  refresh();
}
async function activateModelPackage(){
  const select = document.getElementById("modelPackageSelect");
  const resultEl = document.getElementById("modelSwitchResult");
  const model_id = select ? select.value : "";
  if (!model_id) return;
  if (resultEl) resultEl.textContent = "正在切换模型包...";
  const result = await api("/api/model/activate", {model_id});
  if (result.status === "ok") {
    if (resultEl) resultEl.textContent = `已切换到 ${result.active_label || result.active_model_id}。${result.needs_restart ? "请重启 AI 后生效。" : "下次启动 AI 时生效。"}`;
  } else if (resultEl) {
    resultEl.textContent = result.error || "切换失败";
  }
  refresh();
}
async function renameModelPackage(model_id){
  const resultEl = document.getElementById("modelSwitchResult");
  const input = document.getElementById(`modelLabel_${model_id}`);
  const label = input ? input.value.trim() : "";
  if (!model_id || !label) {
    if (resultEl) resultEl.textContent = "模型名称不能为空。";
    return;
  }
  const result = await api("/api/model/update", {model_id, label});
  if (resultEl) resultEl.textContent = result.status === "ok" ? "模型名称已保存。" : (result.error || "保存失败");
  refresh();
}
async function pinModelPackage(model_id){
  const resultEl = document.getElementById("modelSwitchResult");
  const input = document.getElementById(`modelLabel_${model_id}`);
  const label = input ? input.value.trim() : "";
  const body = {model_id, pin:true};
  if (label) body.label = label;
  const result = await api("/api/model/update", body);
  if (resultEl) resultEl.textContent = result.status === "ok" ? "该模型已设为永久保留。" : (result.error || "保留失败");
  refresh();
}
async function saveCurrentModelSnapshot(){
  const resultEl = document.getElementById("modelSwitchResult");
  const input = document.getElementById("manualModelName");
  const label = input ? input.value.trim() : "";
  const result = await api("/api/model/snapshot", {label});
  if (result.status === "ok") {
    if (input) input.value = "";
    if (resultEl) resultEl.textContent = `已永久保存当前模型：${result.label || result.model_id}`;
  } else if (resultEl) {
    resultEl.textContent = result.error || "保存失败";
  }
  refresh();
}
async function importModelPackage(){
  const resultEl = document.getElementById("modelSwitchResult");
  const input = document.getElementById("modelImportFile");
  const file = input && input.files && input.files[0];
  if (!file) {
    if (resultEl) resultEl.textContent = "请先选择一个 zip 模型包。";
    return;
  }
  if (!/\.zip$/i.test(file.name || "")) {
    if (resultEl) resultEl.textContent = "目前只支持导入 .zip 模型包。";
    return;
  }
  if (resultEl) resultEl.textContent = `正在导入 ${file.name} ...`;
  const bytes = await file.arrayBuffer();
  const uint8 = new Uint8Array(bytes);
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < uint8.length; i += chunkSize) {
    binary += String.fromCharCode(...uint8.subarray(i, i + chunkSize));
  }
  const content_base64 = btoa(binary);
  const result = await api("/api/model/import", {filename:file.name, content_base64});
  if (result.status === "ok") {
    if (input) input.value = "";
    if (resultEl) resultEl.textContent = `导入成功：${(result.package && result.package.label) || result.model_id}`;
  } else if (resultEl) {
    resultEl.textContent = result.error || "导入失败";
  }
  refresh();
}
async function deleteModelPackage(model_id, isActive=false){
  const activeNote = isActive ? "\n\n这是当前启用的模型包。删除后会先切回本地当前模型，已经加载到 ProcessedParams 的模型文件不会被删除。" : "";
  if (!confirm(`确定要彻底删除模型包 ${model_id} 吗？此操作不可恢复。${activeNote}`)) return;
  const resultEl = document.getElementById("modelSwitchResult");
  if (resultEl) resultEl.textContent = "正在删除...";
  forceModelHealthRefreshUntil = Date.now() + 3000;
  try {
    const result = await api("/api/model/delete", {model_id});
    if (result.status === "ok") {
      if (resultEl) resultEl.textContent = result.active_reset ? "模型包已删除，当前启用已切回本地当前模型。" : "模型包已删除。";
    } else if (resultEl) {
      resultEl.textContent = result.error || "删除失败";
    }
  } catch (err) {
    if (resultEl) resultEl.textContent = `删除请求失败：${err.message || err}`;
  }
  await refresh();
}
async function activateModelById(model_id){
  const resultEl = document.getElementById("modelSwitchResult");
  if (!model_id) return;
  if (resultEl) resultEl.textContent = "正在切换模型包...";
  const result = await api("/api/model/activate", {model_id});
  if (result.status === "ok") {
    if (resultEl) resultEl.textContent = `已切换到 ${result.active_label || result.active_model_id}。${result.needs_restart ? "请重启 AI 后生效。" : "下次启动 AI 时生效。"}`;
  } else if (resultEl) {
    resultEl.textContent = result.error || "切换失败";
  }
  refresh();
}
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
document.addEventListener("selectstart", event => {
  const target = event.target;
  const noSelectTarget = target && target.closest && target.closest(".module-card > .section-head, .module-item, .module-actions");
  if (draggingCardId || draggingModuleId || noSelectTarget) {
    event.preventDefault();
  }
});
document.addEventListener("dragstart", clearDragSelection, true);
document.addEventListener("dragend", clearDragUi, true);
let parallaxFrame = 0;
function updateFormulaParallax() {
  parallaxFrame = 0;
  const y = window.scrollY || document.documentElement.scrollTop || 0;
  const t = smoothstep01(Math.min(y / 900, 1));
  document.documentElement.style.setProperty("--parallax-back", `${Math.round(t * 18)}px`);
  document.documentElement.style.setProperty("--parallax-soft", `${Math.round(t * 10)}px`);
}
function requestFormulaParallax() {
  if (parallaxFrame) return;
  parallaxFrame = window.requestAnimationFrame(updateFormulaParallax);
}
syncModuleUI();
refresh();
updateFormulaParallax();
setInterval(refresh, 5000);
window.addEventListener("resize", positionGuide);
window.addEventListener("scroll", positionGuide, true);
window.addEventListener("scroll", requestFormulaParallax, {passive:true});
window.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    closeGuide();
    closeProjectGuide();
  }
});
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        try:
            self.end_headers()
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
        try:
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

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
        elif self.path == "/api/self-play/status":
            self._json(200, self_play_status_payload())
        elif self.path == "/api/self-play/config":
            self._json(200, {"control": read_control(), "self_play": self_play_status_payload()})
        elif self.path.startswith("/assets/"):
            name = Path(self.path.split("?", 1)[0].split("/assets/", 1)[1]).name
            path = ASSETS_DIR / name
            content_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}
            if path.exists() and path.is_file() and path.suffix.lower() in content_types:
                self._file(path, content_types[path.suffix.lower()], download=False)
            else:
                self._json(404, {"error": "asset not found"})
        elif self.path.startswith("/docs/"):
            name = Path(self.path.split("?", 1)[0].split("/docs/", 1)[1]).name
            path = DOCS_DIR / name
            if path.exists() and path.is_file() and path.suffix.lower() == ".md":
                self._file(path, "text/markdown; charset=utf-8", download=False)
            else:
                self._json(404, {"error": "doc not found"})
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
            elif self.path == "/api/game-speed":
                self._json(200, apply_game_speed(body))
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
            elif self.path == "/api/self-play/start":
                self._json(200, start_self_play())
            elif self.path == "/api/self-play/stop":
                self._json(200, stop_self_play())
            elif self.path == "/api/self-play/config":
                self._json(200, {"control": update_control(body), "self_play": self_play_status_payload()})
            elif self.path == "/api/train":
                self._json(200, run_training_background())
            elif self.path == "/api/ppo/train":
                self._json(200, run_ppo_training_background())
            elif self.path == "/api/update":
                self._json(200, run_workspace_update_background())
            elif self.path == "/api/model/activate":
                self._json(200, activate_model_package(body.get("model_id")))
            elif self.path == "/api/model/snapshot":
                self._json(200, create_model_snapshot(
                    label=body.get("label", ""),
                    retention="manual",
                    description=body.get("description", ""),
                    activate=bool(body.get("activate", False)),
                ))
            elif self.path == "/api/model/update":
                self._json(200, update_model_package(
                    body.get("model_id"),
                    label=body.get("label") if "label" in body else None,
                    description=body.get("description") if "description" in body else None,
                    pin=bool(body.get("pin", False)),
                ))
            elif self.path == "/api/model/import":
                self._json(200, import_model_package_archive(
                    body.get("filename", ""),
                    body.get("content_base64", ""),
                ))
            elif self.path == "/api/model/delete":
                self._json(200, delete_model_package(body.get("model_id")))
            elif self.path == "/api/export":
                self._json(200, export_database_package())
            elif self.path == "/api/run":
                result = set_run_discarded(body["run_id"], bool(body.get("discarded")))
                invalidate_dashboard_data_cache()
                self._json(200, result)
            elif self.path == "/api/quality":
                result = set_run_label(body["run_id"], body.get("quality", "unknown"), body.get("note", ""))
                invalidate_dashboard_data_cache()
                self._json(200, result)
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
    if not SELF_PLAY_STATE_PATH.exists():
        write_json(SELF_PLAY_STATE_PATH, default_self_play_state())
    ensure_llm_profiles_initialized()
    ensure_active_model_available()
    if not DISCARDED_PATH.exists():
        write_json(DISCARDED_PATH, {"discarded": []})
    threading.Thread(target=_refresh_python_runtime_cache, kwargs={"blocking": False}, daemon=True).start()
    threading.Thread(target=_refresh_dashboard_data_cache, kwargs={"blocking": False}, daemon=True).start()
    panel_port = resolve_panel_port()
    server = ThreadingHTTPServer(("127.0.0.1", panel_port), Handler)
    print(f"STS2 AI control panel: http://127.0.0.1:{panel_port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
