import json
import os
import signal
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

WORKSPACE = Path(__file__).resolve().parents[1]
AI_DIR = WORKSPACE / "AI_Training"
DATA_DIR = WORKSPACE / "RL_Datasets"
EXPORT_DIR = WORKSPACE / "Data_Packages"
CONTROL_PATH = AI_DIR / "control_state.json"
AI_LOGIC_PATH = AI_DIR / "ai_logic_state.json"
LLM_CONFIG_PATH = AI_DIR / "model_config.json"
LLM_LOGIC_PATH = AI_DIR / "llm_logic_state.json"
DISCARDED_PATH = DATA_DIR / "discarded_runs.json"
RUN_LABELS_PATH = DATA_DIR / "run_labels.json"
SERVER_STATE_PATH = AI_DIR / "control_panel_state.json"
DEFAULT_PYTHON_EXE = WORKSPACE / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = Path(os.environ.get("STS2_AI_PYTHON") or (DEFAULT_PYTHON_EXE if DEFAULT_PYTHON_EXE.exists() else sys.executable))
AGENT_PATH = AI_DIR / "ai_agent.py"
LLM_AGENT_PATH = AI_DIR / "llm_agent.py"
API_URL = "http://localhost:15526/api/v1/singleplayer"

DEFAULT_CONTROL = {
    "ai_enabled": False,
    "record_ai_actions": True,
    "include_ai_in_training": False,
    "next_run_mode": "new",
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
    "profiles": [],
    "active_profile_id": "",
}
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

TRAIN_LOCK = threading.Lock()
LAST_TRAIN = {"running": False, "started": None, "finished": None, "output": ""}
LAST_EXPORT = {"path": None, "filename": None, "created": None, "size": 0, "file_count": 0}


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
                data[key] = patch[key] if patch[key] in ("new", "continue") else "new"
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


def ensure_llm_profiles_initialized():
    data = DEFAULT_LLM_CONFIG.copy()
    data.update(read_json(LLM_CONFIG_PATH, {}))
    profiles = data.get("profiles") if isinstance(data.get("profiles"), list) else []
    if data.get("api_key") and not profiles:
        write_json(LLM_CONFIG_PATH, _upsert_current_llm_profile(data))


def read_run_labels():
    data = read_json(RUN_LABELS_PATH, {"labels": {}})
    if "labels" not in data:
        data = {"labels": data if isinstance(data, dict) else {}}
    return data


def infer_quality(item):
    if item.get("losses", 0) > 0:
        return "failed_run"
    if item.get("run_victory") or int(item.get("max_act") or 0) >= 4:
        return "perfect_run"
    if int(item.get("max_act") or 0) >= 3:
        return "partial_act2"
    if int(item.get("max_act") or 0) >= 2:
        return "partial_act1"
    if int(item.get("max_floor") or 0) > 0:
        return "before_act1_boss"
    return "unknown"


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


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def iter_record_states(rec):
    for key in ("state", "state_before", "state_after"):
        state = rec.get(key)
        if isinstance(state, dict):
            yield state


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
        data.pop("compact_state", None)
    return data


def latest_runs(limit=12):
    discarded = set(read_json(DISCARDED_PATH, {"discarded": []}).get("discarded", []))
    labels = read_run_labels().get("labels", {})
    files = []
    for sub in ["Combat", "Human/Combat", "AI_Combat", "Macro", "Human/Macro"]:
        root = DATA_DIR / sub
        if root.exists():
            files.extend(root.glob("*.jsonl"))

    runs = {}
    for path in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            run_id = rec.get("run_id")
            if not run_id:
                continue
            item = runs.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "records": 0,
                    "combat": 0,
                    "macro": 0,
                    "ai": 0,
                    "human": 0,
                    "wins": 0,
                    "losses": 0,
                    "play_card": 0,
                    "end_turn": 0,
                    "use_potion": 0,
                    "max_act": 0,
                    "max_floor": 0,
                    "run_victory": False,
                    "last_ts": 0,
                    "files": set(),
                },
            )
            item["records"] += 1
            item["files"].add(str(path.relative_to(DATA_DIR)))
            if "Combat" in str(path):
                item["combat"] += 1
            else:
                item["macro"] += 1
            if rec.get("source") == "ai":
                item["ai"] += 1
            if rec.get("source") == "human":
                item["human"] += 1
            if rec.get("type") == "battle_end":
                if rec.get("result") == "win":
                    item["wins"] += 1
                if rec.get("result") == "lose":
                    item["losses"] += 1
            if rec.get("type") in ("run_end", "game_end", "victory") and rec.get("result") in ("win", "victory", "complete", True):
                item["run_victory"] = True
            action_type = rec.get("action_type")
            if action_type in ("play_card", "end_turn", "use_potion"):
                item[action_type] += 1
            for state in iter_record_states(rec):
                item["max_act"] = max(item["max_act"], safe_int(state.get("act")))
                item["max_floor"] = max(item["max_floor"], safe_int(state.get("floor")))
            item["last_ts"] = max(item["last_ts"], int(rec.get("timestamp") or 0))

    result = []
    for item in runs.values():
        item["discarded"] = item["run_id"] in discarded
        inferred_quality = infer_quality(item)
        inferred_note = f"auto: max_act={item.get('max_act', 0)}, max_floor={item.get('max_floor', 0)}"
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
        item["inferred_quality"] = inferred_quality
        item["inferred_quality_label"] = QUALITY_LABELS.get(inferred_quality, inferred_quality)
        item["files"] = sorted(item["files"])
        item["last_time"] = (
            datetime.fromtimestamp(item["last_ts"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
            if item["last_ts"]
            else ""
        )
        result.append(item)
    result.sort(key=lambda x: x["last_ts"], reverse=True)
    return result[:limit]


def iter_recent_records(max_files=8):
    roots = ["Human/Combat", "Human/Macro", "Combat", "Macro", "AI_Combat", "LLM_Actions"]
    files = []
    for sub in roots:
        root = DATA_DIR / sub
        if root.exists():
            files.extend(root.glob("*.jsonl"))
    for path in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:max_files]:
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    rec["_file"] = str(path.relative_to(DATA_DIR))
                    yield rec
        except Exception:
            continue


def recent_records(limit=30):
    records = sorted(iter_recent_records(), key=lambda r: int(r.get("timestamp") or 0), reverse=True)[:limit]
    out = []
    for rec in records:
        ts = int(rec.get("timestamp") or 0)
        out.append({
            "time": datetime.fromtimestamp(ts / 1000).strftime("%H:%M:%S") if ts else "",
            "run_id": rec.get("run_id"),
            "type": rec.get("type"),
            "source": rec.get("source"),
            "action_type": rec.get("action_type") or (rec.get("decision") or {}).get("action"),
            "result": rec.get("result"),
            "round": rec.get("round") or (rec.get("state") or {}).get("round"),
            "file": rec.get("_file"),
        })
    return out


def current_data_summary():
    runs = latest_runs(limit=1)
    if not runs:
        return {"active_run": None, "warning": "暂无 run 数据"}
    run = runs[0].copy()
    warnings = []
    if run.get("combat", 0) > 0 and run.get("play_card", 0) == 0:
        warnings.append("当前/最近 run 没采到 play_card，只采到 end_turn/宏观动作。C# 出牌 Hook 需要修。")
    if run.get("losses", 0) > 0:
        warnings.append("这个 run 有失败记录，BC 训练前建议确认是否保留。")
    if run.get("wins", 0) == 0 and run.get("losses", 0) == 0:
        warnings.append("暂未看到 battle_end 胜负记录；如果已经通关，可能是胜利/终局 Hook 没捕获。")
    if run.get("quality") == "perfect_run" and run.get("play_card", 0) == 0:
        warnings.append("这个 run 已标为完美，但没有 play_card 标签；修好 Hook 后建议再采一局高质量数据。")
    return {"active_run": run, "warnings": warnings}


def ai_logic_snapshot():
    data = read_json(AI_LOGIC_PATH, {})
    ts = int(data.get("timestamp") or 0)
    if ts:
        data["time"] = datetime.fromtimestamp(ts / 1000).strftime("%H:%M:%S")
    return data


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


def run_training_background():
    def worker():
        with TRAIN_LOCK:
            LAST_TRAIN.update({"running": True, "started": datetime.now().isoformat(timespec="seconds"), "finished": None, "output": ""})
            output = []
            try:
                for cmd in [
                    [str(PYTHON_EXE), str(AI_DIR / "data_pipeline.py")],
                    [str(PYTHON_EXE), str(AI_DIR / "train_bc.py")],
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

    include_roots = ["Human", "AI", "Combat", "Macro"]
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


def status_payload():
    game = get_game_state()
    return {
        "control": read_control(),
        "ai_pid": get_ai_pid(),
        "game": {
            "online": "error" not in game,
            "error": game.get("error"),
            "state_type": game.get("state_type"),
            "character": game.get("player", {}).get("character"),
            "hp": game.get("player", {}).get("hp"),
            "max_hp": game.get("player", {}).get("max_hp"),
            "energy": game.get("player", {}).get("energy"),
        },
        "runs": latest_runs(),
        "current_data": current_data_summary(),
        "recent_records": recent_records(),
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
  <style>
    :root {
      color-scheme: light;
      --ink:#182230;
      --muted:#667085;
      --soft:#98a2b3;
      --line:#d9dee7;
      --bg:#f4f6f8;
      --panel:#ffffff;
      --panel-2:#f9fafb;
      --good:#087443;
      --good-bg:#ecfdf3;
      --warn:#a15c07;
      --warn-bg:#fffaeb;
      --bad:#b42318;
      --bad-bg:#fff1f0;
      --blue:#175cd3;
      --blue-bg:#eff4ff;
      --shadow:0 1px 2px rgba(16,24,40,.06);
    }
    * { box-sizing:border-box; }
    body {
      margin:0;
      font-family:"Segoe UI", system-ui, sans-serif;
      color:var(--ink);
      background:var(--bg);
      font-size:14px;
      line-height:1.45;
    }
    header {
      background:var(--panel);
      border-bottom:1px solid var(--line);
      padding:18px 24px;
    }
    .topbar { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
    h1 { margin:0; font-size:22px; letter-spacing:0; }
    h2 { margin:0 0 12px; font-size:15px; }
    h3 { margin:0; font-size:13px; color:var(--muted); font-weight:600; }
    .subtitle { margin-top:3px; color:var(--muted); }
    .status-grid {
      display:grid;
      grid-template-columns:repeat(4, minmax(160px, 1fr));
      gap:12px;
      margin-top:16px;
    }
    .status-card {
      border:1px solid var(--line);
      background:#fff;
      border-radius:8px;
      padding:14px;
      min-height:88px;
    }
    .status-title { color:var(--muted); font-size:12px; margin-bottom:6px; }
    .status-main { font-size:20px; font-weight:700; }
    .status-main.on, .status-main.off, .status-main.warn, .status-main.info {
      background:transparent;
      border-color:transparent;
      padding:0;
      display:block;
    }
    .status-sub { margin-top:4px; color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
    main {
      padding:20px 24px 28px;
      display:grid;
      grid-template-columns:minmax(320px, 400px) minmax(0, 1fr);
      gap:18px;
      align-items:start;
    }
    .stack { display:grid; gap:16px; }
    section {
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:8px;
      padding:16px;
      box-shadow:var(--shadow);
    }
    .section-head {
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:12px;
      margin-bottom:12px;
    }
    .sidebar { position:static; }
    .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    .field { display:grid; gap:6px; margin-top:12px; }
    .field span { color:var(--muted); font-size:12px; }
    .kv {
      display:grid;
      grid-template-columns:104px minmax(0, 1fr);
      gap:10px;
      padding:8px 0;
      border-bottom:1px solid #eef1f5;
    }
    .kv:last-child { border-bottom:0; }
    .kv span:first-child { color:var(--muted); }
    .muted { color:var(--muted); }
    .fine { color:var(--muted); font-size:12px; }
    .strong { font-weight:700; }
    button, select, input[type=text], input[type=password], input[type=number] {
      border:1px solid var(--line);
      background:#fff;
      border-radius:6px;
      padding:8px 11px;
      min-height:36px;
      font:inherit;
    }
    input[type=text], input[type=password], input[type=number] { width:100%; }
    button { cursor:pointer; font-weight:600; }
    button:hover { border-color:#b8c0cc; background:#f9fafb; }
    button.primary { background:var(--blue); color:#fff; border-color:var(--blue); }
    button.good { background:var(--good); color:#fff; border-color:var(--good); }
    button.bad { background:var(--bad); color:#fff; border-color:var(--bad); }
    .button-row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .button-row button, .field select { width:100%; }
    .segmented { display:grid; grid-template-columns:1fr 1fr; gap:6px; background:#eef1f5; padding:4px; border-radius:8px; }
    .segmented button { border:0; background:transparent; }
    .segmented button.active { background:#fff; color:var(--blue); box-shadow:var(--shadow); }
    .switch {
      display:grid;
      grid-template-columns:1fr auto;
      gap:10px;
      align-items:center;
      padding:10px 0;
      border-top:1px solid #eef1f5;
    }
    .switch:first-of-type { border-top:0; }
    .switch-title { font-weight:600; }
    .switch-note { color:var(--muted); font-size:12px; margin-top:2px; }
    input[type=checkbox] { width:20px; height:20px; accent-color:var(--blue); }
    .pill {
      display:inline-flex;
      align-items:center;
      gap:6px;
      padding:3px 8px;
      border-radius:999px;
      font-size:12px;
      border:1px solid var(--line);
      background:#fff;
      white-space:nowrap;
    }
    .on { color:var(--good); border-color:#9ad6b8; background:var(--good-bg); }
    .off { color:var(--bad); border-color:#f4b5ad; background:var(--bad-bg); }
    .warn { color:var(--warn); border-color:#fedf89; background:var(--warn-bg); }
    .info { color:var(--blue); border-color:#b2ccff; background:var(--blue-bg); }
    .metric-grid {
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(120px, 1fr));
      gap:10px;
      margin-top:12px;
    }
    .metric {
      background:var(--panel-2);
      border:1px solid #eef1f5;
      border-radius:8px;
      padding:10px;
    }
    .metric-value { font-size:20px; font-weight:700; }
    .metric-label { color:var(--muted); font-size:12px; }
    .warning-list { display:grid; gap:8px; margin-top:12px; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { border-bottom:1px solid #eef1f5; text-align:left; padding:9px 8px; vertical-align:top; }
    th { color:var(--muted); font-weight:600; background:#fbfcfe; }
    tr:hover td { background:#fcfcfd; }
    code, pre { font-family:Consolas, "Cascadia Mono", monospace; }
    code { overflow-wrap:anywhere; }
    pre {
      white-space:pre-wrap;
      max-height:260px;
      overflow:auto;
      background:#111827;
      color:#e5e7eb;
      padding:12px;
      border-radius:8px;
      margin:10px 0 0;
      font-size:12px;
    }
    .table-wrap { overflow:auto; border:1px solid #eef1f5; border-radius:8px; }
    .table-wrap table th, .table-wrap table td { white-space:nowrap; }
    .run-id { max-width:260px; white-space:normal; }
    .modal-backdrop {
      display:none;
      position:fixed;
      inset:0;
      z-index:20;
      background:rgba(15,23,42,.36);
      padding:20px;
      align-items:center;
      justify-content:center;
    }
    .modal-backdrop.open { display:flex; }
    .modal {
      width:min(560px, 100%);
      background:#fff;
      border:1px solid var(--line);
      border-radius:8px;
      padding:16px;
      box-shadow:0 18px 48px rgba(16,24,40,.18);
    }
    @media (max-width: 1120px) {
      .status-grid { grid-template-columns:repeat(2, minmax(160px, 1fr)); }
      main { grid-template-columns:1fr; }
      .sidebar { position:static; }
    }
    @media (max-width: 720px) {
      header { padding:14px; }
      main { padding:14px; }
      .topbar { display:block; }
      .status-grid, .metric-grid { grid-template-columns:1fr; }
      .button-row { grid-template-columns:1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <h1>STS2 AI 控制台</h1>
        <div class="subtitle">战斗托管、数据采集、BC 重训和 run 质量管理</div>
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
        <div class="status-title">采集入池</div>
        <div id="collectStatus" class="status-main">读取中</div>
        <div id="collectDetail" class="status-sub">-</div>
      </div>
      <div class="status-card">
        <div class="status-title">当前 Run</div>
        <div id="runQuality" class="status-main">读取中</div>
        <div id="runDetail" class="status-sub">-</div>
      </div>
    </div>
  </header>

  <main>
    <div class="stack sidebar">
      <section>
        <div class="section-head">
          <h2>战斗 AI</h2>
          <span id="aiProcessBadge" class="pill">-</span>
        </div>
        <div class="button-row">
          <button class="good" onclick="startAI()">启动 AI</button>
          <button class="bad" onclick="stopAI()">停止 AI</button>
        </div>
        <div class="switch">
          <div><div class="switch-title">允许 AI 出牌</div><div class="switch-note">只影响战斗自动打牌</div></div>
          <input id="ai_enabled" type="checkbox" onchange="saveControl()">
        </div>
        <div class="switch">
          <div><div class="switch-title">记录 AI 战斗动作</div><div class="switch-note">写入 AI_Combat，方便复盘</div></div>
          <input id="record_ai_actions" type="checkbox" onchange="saveControl()">
        </div>
        <div class="switch">
          <div><div class="switch-title">AI 数据进入 BC</div><div class="switch-note">默认关闭，避免自举污染</div></div>
          <input id="include_ai_in_training" type="checkbox" onchange="saveControl()">
        </div>
      </section>

      <section id="llmConfigSection">
        <div class="section-head">
          <h2>LLM 模型接入</h2>
          <span id="llmProcessBadge" class="pill">-</span>
        </div>
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
          <button onclick="testLLM()">测试连接</button>
        </div>
        <div id="llmTestResult" class="fine" style="margin-top:8px">未测试</div>
        <div class="fine" style="margin-top:10px">API Key 只保存在本机 <code>AI_Training/model_config.json</code>，不会进入 Git。</div>
        <div class="table-wrap" style="margin-top:10px">
          <table>
            <thead><tr><th>名称</th><th>Base URL</th><th>Model</th><th>Key</th><th>操作</th></tr></thead>
            <tbody id="llmProfiles"></tbody>
          </table>
        </div>
      </section>

      <section>
        <div class="section-head">
          <h2>采集与训练</h2>
          <span id="collectBadge" class="pill">-</span>
        </div>
        <div class="switch">
          <div><div class="switch-title">采集进入训练池</div><div class="switch-note">关闭后原始日志仍可保留，但训练会跳过该时间段</div></div>
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
          <button class="primary" onclick="train()">重建数据 + 重训 BC</button>
          <button onclick="proceed()">Proceed</button>
        </div>
      </section>

      <section>
        <div class="section-head">
          <h2>数据打包</h2>
          <span id="exportBadge" class="pill info">未导出</span>
        </div>
        <button class="primary" onclick="exportData()">一键打包数据库</button>
        <div id="exportInfo" class="fine" style="margin-top:10px">生成 zip 后，直接把这个文件发给你。</div>
      </section>

      <section>
        <div class="section-head">
          <h2>主菜单标记</h2>
          <span id="nextRunBadge" class="pill info">-</span>
        </div>
        <div class="segmented">
          <button id="modeNew" onclick="setRunMode('new')">新游戏</button>
          <button id="modeContinue" onclick="setRunMode('continue')">继续游戏</button>
        </div>
        <div class="fine" style="margin-top:10px">这里标记接下来的数据意图，便于区分新 episode 和续接片段。</div>
      </section>
    </div>

    <div class="stack">
      <section>
        <div class="section-head">
          <h2>当前数据</h2>
          <span id="currentDataBadge" class="pill">读取中</span>
        </div>
        <div id="currentData">读取中</div>
      </section>

      <section>
        <div class="section-head">
          <h2>AI 出牌逻辑</h2>
          <span id="aiDecisionBadge" class="pill">-</span>
        </div>
        <div id="aiLogic" class="muted">暂无 AI 决策</div>
        <div class="table-wrap" style="margin-top:10px">
          <table>
            <thead><tr><th>候选动作</th><th>概率</th><th>状态</th></tr></thead>
            <tbody id="aiTopActions"></tbody>
          </table>
        </div>
      </section>

      <section>
        <div class="section-head">
          <h2>LLM 决策</h2>
          <span id="llmDecisionBadge" class="pill">-</span>
        </div>
        <div id="llmLogic" class="muted">暂无 LLM 决策</div>
      </section>

      <section>
        <div class="section-head">
          <h2>最近 Run</h2>
          <span class="fine">手动质量不会被自动分类覆盖</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Run</th><th>进度</th><th>动作</th><th>来源</th><th>结果</th><th>质量</th><th>保留</th><th></th></tr></thead>
            <tbody id="runs"></tbody>
          </table>
        </div>
      </section>

      <section>
        <div class="section-head">
          <h2>最近采集记录</h2>
          <span class="fine">最近 12 条</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>时间</th><th>类型</th><th>来源</th><th>动作</th><th>文件</th></tr></thead>
            <tbody id="recentRecords"></tbody>
          </table>
        </div>
      </section>

      <section>
        <div class="section-head">
          <h2>训练输出</h2>
          <span id="trainStatus" class="pill">-</span>
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
  if (lower.includes("shop") || lower.includes("merchant")) return {label:"商店中", cls:"warn", detail:raw};
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
let llmFormDirty = false;
let llmProfilesCache = [];
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
  document.getElementById("llm_execute_combat").checked = !!llmCfg.execute_combat;
  document.getElementById("llm_base_url").value = llmCfg.base_url || "";
  document.getElementById("llm_model").value = llmCfg.model || "";
  document.getElementById("llm_api_key").placeholder = llmCfg.has_api_key ? "已保存，留空不修改" : "未配置 API Key";
}
async function refresh() {
  const s = await api("/api/status");
  const active = s.current_data && s.current_data.active_run;
  const phase = phaseInfo(s.game);

  document.getElementById("ai_enabled").checked = !!s.control.ai_enabled;
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
  document.getElementById("aiStatus").textContent = s.control.ai_enabled ? "允许出牌" : "手动模式";
  document.getElementById("aiStatus").className = `status-main ${s.control.ai_enabled ? "on" : "warn"}`;
  document.getElementById("aiDetail").textContent = s.ai_pid ? `托管进程 PID ${s.ai_pid}` : "AI 进程未由控制台托管";
  document.getElementById("collectStatus").textContent = s.control.collection_enabled ? "正在入池" : "暂停入池";
  document.getElementById("collectStatus").className = `status-main ${s.control.collection_enabled ? "on" : "off"}`;
  document.getElementById("collectDetail").textContent = s.control.include_ai_in_training ? "Human + AI 数据会进入训练" : "仅 Human 数据进入训练";
  document.getElementById("runQuality").textContent = active ? (active.quality_label || active.quality || "-") : "无 run";
  document.getElementById("runQuality").className = `status-main ${active && active.discarded ? "off" : "info"}`;
  document.getElementById("runDetail").textContent = active ? `Act ${active.max_act || 0} / Floor ${active.max_floor || 0}，${active.records || 0} 条` : "尚未读取到采集数据";

  setPill("lastRefresh", `已刷新 ${new Date().toLocaleTimeString()}`, "info");
  setPill("aiProcessBadge", s.ai_pid ? "运行中" : "未启动", s.ai_pid ? "on" : "warn");
  setPill("llmProcessBadge", s.llm && s.llm.pid ? "运行中" : "未启动", s.llm && s.llm.pid ? "on" : "warn");
  setPill("collectBadge", s.control.collection_enabled ? "启用" : "暂停", s.control.collection_enabled ? "on" : "off");
  setPill("nextRunBadge", s.control.next_run_mode === "continue" ? "继续游戏" : "新游戏", "info");
  document.getElementById("modeNew").className = s.control.next_run_mode === "new" ? "active" : "";
  document.getElementById("modeContinue").className = s.control.next_run_mode === "continue" ? "active" : "";
  renderExport(s.export);

  renderCurrentData(s.current_data);
  renderRuns(s.runs || []);
  renderRecentRecords(s.recent_records || []);
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
  const warnings = (data.warnings || []).map(w => `<div><span class="pill off">注意</span> ${w}</div>`).join("");
  setPill("currentDataBadge", run.play_card > 0 ? "有出牌样本" : "缺出牌样本", run.play_card > 0 ? "on" : "warn");
  document.getElementById("currentData").innerHTML = `
    <div class="kv"><span>Run</span><code>${run.run_id}</code></div>
    <div class="kv"><span>时间</span><span>${run.last_time || "-"}</span></div>
    <div class="metric-grid">
      <div class="metric"><div class="metric-value">${run.records || 0}</div><div class="metric-label">总记录</div></div>
      <div class="metric"><div class="metric-value">${run.combat || 0}</div><div class="metric-label">战斗记录</div></div>
      <div class="metric"><div class="metric-value">${run.play_card || 0}</div><div class="metric-label">出牌样本</div></div>
      <div class="metric"><div class="metric-value">${run.end_turn || 0}</div><div class="metric-label">结束回合</div></div>
      <div class="metric"><div class="metric-value">A${run.max_act || 0}/F${run.max_floor || 0}</div><div class="metric-label">${run.quality_label || "未知"}</div></div>
    </div>
    <div class="warning-list">${warnings || '<div><span class="pill on">正常</span> 最近 run 有数据写入</div>'}</div>`;
}
function renderRuns(runs) {
  const rows = runs.map(run => `
    <tr>
      <td class="run-id"><code>${run.run_id}</code><br><span class="fine">${run.last_time || ""}</span></td>
      <td>Act ${run.max_act || 0} / Floor ${run.max_floor || 0}<br><span class="fine">${run.records || 0} 条，C ${run.combat || 0} / M ${run.macro || 0}</span></td>
      <td>出牌 ${run.play_card || 0}<br><span class="fine">回合 ${run.end_turn || 0}</span></td>
      <td>Human ${run.human || 0}<br><span class="fine">AI ${run.ai || 0}</span></td>
      <td>胜 ${run.wins || 0}<br><span class="fine">败 ${run.losses || 0}</span></td>
      <td>
        <select onchange="setQuality('${run.run_id}', this.value)">${qualityOptions(run.quality)}</select>
        <br><span class="fine">${run.quality_manual ? "手动标签" : "自动推断"}</span>
      </td>
      <td>${run.discarded ? '<span class="pill off">丢弃</span>' : '<span class="pill on">保留</span>'}</td>
      <td><button onclick="markRun('${run.run_id}', ${!run.discarded})">${run.discarded ? "保留" : "丢弃"}</button></td>
    </tr>`).join("");
  document.getElementById("runs").innerHTML = rows || "<tr><td colspan=8>暂无数据</td></tr>";
}
function renderRecentRecords(records) {
  document.getElementById("recentRecords").innerHTML = records.slice(0, 12).map(r => `
    <tr><td>${r.time || ""}</td><td>${r.type || ""}</td><td>${r.source || ""}</td><td>${r.action_type || r.result || ""}</td><td>${r.file || ""}</td></tr>
  `).join("") || "<tr><td colspan=5>暂无记录</td></tr>";
}
function renderAiLogic(logic) {
  if (!logic || !logic.timestamp) {
    document.getElementById("aiLogic").innerHTML = '<div class="muted">暂无 AI 决策。启动 AI 并进入战斗后，这里会显示候选动作、概率和为什么没打出去。</div>';
    document.getElementById("aiTopActions").innerHTML = "<tr><td colspan=3>暂无概率</td></tr>";
    setPill("aiDecisionBadge", "无决策", "warn");
    return;
  }
  setPill("aiDecisionBadge", logic.chosen_action || "有决策", "info");
  document.getElementById("aiLogic").innerHTML = `
    <div class="kv"><span>时间</span><span>${logic.time || "-"}</span></div>
    <div class="kv"><span>执行</span><span class="strong">${logic.chosen_action || "-"}</span></div>
    <div class="kv"><span>手牌</span><span>${(logic.hand || []).join(", ") || "-"}</span></div>
    <div class="kv"><span>原因</span><span>${logic.reason || "-"}</span></div>`;
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
  const d = logic.decision || {};
  const payload = logic.payload ? JSON.stringify(logic.payload) : "-";
  setPill("llmDecisionBadge", logic.executed ? "已执行" : "建议", logic.executed ? "on" : "info");
  document.getElementById("llmLogic").innerHTML = `
    <div class="kv"><span>时间</span><span>${logic.time || "-"}</span></div>
    <div class="kv"><span>模式</span><span>${logic.mode || "-"}</span></div>
    <div class="kv"><span>模型</span><span>${logic.model || "-"}</span></div>
    <div class="kv"><span>场景</span><span>${logic.state_type || "-"}</span></div>
    <div class="kv"><span>建议</span><span class="strong">${d.action || "-"}</span></div>
    <div class="kv"><span>参数</span><code>${JSON.stringify(d.args || {})}</code></div>
    <div class="kv"><span>校验</span><span>${logic.validation || "-"}</span></div>
    <div class="kv"><span>执行</span><span>${logic.executed ? "是" : "否"} ${logic.ok === false ? "(失败)" : ""}</span></div>
    <div class="kv"><span>Payload</span><code>${payload}</code></div>
    <div class="kv"><span>手牌</span><span>${(logic.hand_summary || []).join(", ") || "-"}</span></div>
    <div class="kv"><span>理由</span><span>${d.reason || "-"}</span></div>`;
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
  await saveLLMConfig();
  document.getElementById("llmTestResult").textContent = "正在测试...";
  const result = await api("/api/llm/test", {});
  document.getElementById("llmTestResult").textContent = result.message || result.status || "测试完成";
}
async function setRunMode(mode) { await api("/api/control", {next_run_mode: mode}); refresh(); }
async function startAI(){ await api("/api/ai/start", {}); refresh(); }
async function stopAI(){ await api("/api/ai/stop", {}); refresh(); }
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
refresh();
setInterval(refresh, 3000);
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
        self.wfile.write(data)

    def _file(self, path, content_type="application/octet-stream"):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
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
        elif self.path == "/api/status":
            self._json(200, status_payload())
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
