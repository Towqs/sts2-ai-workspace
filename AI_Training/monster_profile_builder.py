import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "RL_Datasets"
MONSTER_DIR = DATA_DIR / "Monster"
PROCESSED_DIR = DATA_DIR / "Processed"

SCHEMA_TURN = "monster_turn_v1"
SCHEMA_ENCOUNTER = "encounter_v1"
SCHEMA_PROFILE = "monster_profiles_v1"
SCHEMA_ENCOUNTER_PROFILE = "encounter_profiles_v1"
SCHEMA_DAMAGE_ANOMALIES = "monster_damage_anomalies_v1"
SCHEMA_VOCAB = "monster_vocab_v1"

NORMAL_DAMAGE_MIN_SAMPLES = 4
NORMAL_DAMAGE_MIN_EXCESS = 8.0
NORMAL_DAMAGE_MIN_ABSOLUTE = 12.0
BOSS_FLOORS = {17, 34, 51}

INPUT_GLOBS = [
    "action_logs_*.jsonl",
    "Combat/*.jsonl",
    "Human/Combat/*.jsonl",
    "AI/Combat/*.jsonl",
    "AI_Combat/*.jsonl",
    "LLM_Actions/*.jsonl",
]

COMBAT_TYPES = {"monster", "elite", "boss"}


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_json_records(path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            record, next_index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            next_newline = text.find("\n", index)
            if next_newline < 0:
                break
            line = text[index:next_newline].strip()
            if line:
                try:
                    record = json.loads(line)
                    next_index = next_newline + 1
                except Exception:
                    index = next_newline + 1
                    continue
            else:
                index = next_newline + 1
                continue
        if isinstance(record, dict):
            yield record
        index = max(next_index, index + 1)


def iter_input_files(data_dir, limit_files=0):
    paths = []
    for pattern in INPUT_GLOBS:
        paths.extend(data_dir.glob(pattern))
    unique = sorted({p.resolve() for p in paths if p.is_file()}, key=lambda p: str(p).lower())
    if limit_files:
        unique = unique[:limit_files]
    return unique


def timestamp_ms(record, path):
    value = safe_int(record.get("timestamp"), 0)
    if value:
        return value
    try:
        return int(path.stat().st_mtime * 1000)
    except OSError:
        return 0


def date_key(timestamp):
    if not timestamp:
        return "unknown"
    try:
        seconds = timestamp / 1000 if timestamp > 10_000_000_000 else timestamp
        return datetime.fromtimestamp(seconds).strftime("%Y-%m-%d")
    except Exception:
        return "unknown"


def generated_at():
    return datetime.now().isoformat(timespec="seconds")


def clean_key(value, fallback="UNKNOWN"):
    raw = str(value or fallback).strip()
    raw = re.sub(r"_\d+$", "", raw)
    raw = re.sub(r"[^0-9A-Za-z_]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return (raw or fallback).upper()


def monster_key(enemy, slot=0):
    raw = (
        enemy.get("id")
        or enemy.get("monster_id")
        or enemy.get("entity_id")
        or enemy.get("combat_id")
        or enemy.get("name")
        or f"MONSTER_{slot}"
    )
    return clean_key(raw, f"MONSTER_{slot}")


def entity_id(enemy, slot=0):
    return str(enemy.get("entity_id") or enemy.get("id") or enemy.get("name") or f"enemy_{slot}")


def parse_numbers(text):
    return [int(n) for n in re.findall(r"\d+", str(text or ""))]


def parse_intent_damage(intent):
    if isinstance(intent, str):
        text = intent
    elif isinstance(intent, dict):
        if intent.get("damage") is not None:
            damage = safe_int(intent.get("damage"), 0)
            hits = max(1, safe_int(intent.get("hits"), 1))
            return damage * hits, hits
        text = " ".join(
            str(intent.get(key) or "")
            for key in ("label", "title", "description", "type")
        )
    else:
        return 0, 0
    numbers = parse_numbers(text)
    if not numbers:
        return 0, 0
    lowered = text.lower()
    if ("x" in lowered or "times" in lowered or "脳" in text) and len(numbers) >= 2:
        return numbers[0] * numbers[1], numbers[1]
    return numbers[0], 1


def normalize_intent(intent):
    if isinstance(intent, str):
        damage, hits = parse_intent_damage(intent)
        return {
            "type": clean_key(intent, "UNKNOWN"),
            "label": intent,
            "damage": damage,
            "hits": hits,
        }
    if not isinstance(intent, dict):
        return {"type": "UNKNOWN", "label": "", "damage": 0, "hits": 0}
    damage, hits = parse_intent_damage(intent)
    intent_type = intent.get("type") or intent.get("id") or intent.get("label") or "UNKNOWN"
    return {
        "type": clean_key(intent_type, "UNKNOWN"),
        "label": str(intent.get("label") or ""),
        "title": str(intent.get("title") or ""),
        "description": str(intent.get("description") or "")[:240],
        "damage": damage,
        "hits": hits,
    }


def normalize_state(state):
    if not isinstance(state, dict):
        return {}
    if isinstance(state.get("player"), dict) and isinstance(state.get("battle"), dict):
        player = dict(state.get("player") or {})
        battle = dict(state.get("battle") or {})
        run = dict(state.get("run") or {})
        state_type = str(state.get("state_type") or "").lower()
    else:
        player = {
            "character": state.get("character"),
            "hp": state.get("hp", 0),
            "max_hp": state.get("max_hp", 1),
            "block": state.get("block", 0),
            "energy": state.get("energy", 0),
            "max_energy": state.get("max_energy", 1),
            "hand": state.get("hand", []),
            "draw_pile_count": state.get("draw_pile_count", 0),
            "discard_pile_count": state.get("discard_pile_count", 0),
            "exhaust_pile_count": state.get("exhaust_pile_count", 0),
            "relics": state.get("relics", []),
            "potions": state.get("potions", []),
            "status": state.get("status", []),
            "gold": state.get("gold", 0),
        }
        battle = {
            "round": state.get("round", 0),
            "turn": state.get("turn"),
            "is_play_phase": state.get("is_play_phase"),
            "enemies": state.get("enemies", []),
        }
        run = dict(state.get("run") or {})
        run.setdefault("act", state.get("act"))
        run.setdefault("floor", state.get("floor"))
        run.setdefault("ascension", state.get("ascension"))
        state_type = str(state.get("state_type") or "").lower()
    enemies = []
    raw_enemies = battle.get("enemies") or []
    for slot, enemy in enumerate(raw_enemies):
        if not isinstance(enemy, dict):
            continue
        intents = [normalize_intent(item) for item in (enemy.get("intents") or [])]
        enemies.append({
            "slot": slot,
            "entity_id": entity_id(enemy, slot),
            "monster_key": monster_key(enemy, slot),
            "id": enemy.get("id") or enemy.get("monster_id") or enemy.get("entity_id"),
            "name": enemy.get("name") or enemy.get("id") or enemy.get("entity_id") or f"enemy_{slot}",
            "hp": safe_int(enemy.get("hp", enemy.get("current_hp", 0)), 0),
            "max_hp": safe_int(enemy.get("max_hp"), 0),
            "block": safe_int(enemy.get("block"), 0),
            "status": enemy.get("status") or [],
            "intents": intents,
        })
    if not state_type and enemies:
        state_type = "monster"
    battle["enemies"] = enemies
    battle["turn"] = str(battle.get("turn") or "").lower()
    return {
        "state_type": state_type,
        "player": player,
        "battle": battle,
        "run": run,
    }


def iter_record_states(record):
    for key in ("state_before", "state", "state_after", "compact_state"):
        state = record.get(key)
        if isinstance(state, dict):
            yield key, normalize_state(state)


def primary_state_for_record(record):
    if record.get("type") == "action" and isinstance(record.get("state_before"), dict):
        return "state_before", normalize_state(record.get("state_before"))
    for key in ("state", "compact_state", "state_before", "state_after"):
        if isinstance(record.get(key), dict):
            return key, normalize_state(record.get(key))
    return "", {}


def state_has_combat(state):
    if not state:
        return False
    enemies = (state.get("battle") or {}).get("enemies") or []
    state_type = str(state.get("state_type") or "").lower()
    return bool(enemies) and (not state_type or state_type in COMBAT_TYPES)


def intent_types(enemies):
    types = []
    for enemy in enemies:
        for intent in enemy.get("intents") or []:
            types.append(intent.get("type") or "UNKNOWN")
    return types or ["UNKNOWN"]


def incoming_damage(enemies):
    return sum(safe_int(intent.get("damage"), 0) for enemy in enemies for intent in enemy.get("intents") or [])


def card_ids(cards):
    result = []
    for card in cards or []:
        if isinstance(card, dict):
            result.append(str(card.get("id") or card.get("name") or "UNKNOWN"))
    return result


def potion_ids(potions):
    result = []
    for potion in potions or []:
        if isinstance(potion, dict):
            result.append(str(potion.get("id") or potion.get("name") or "UNKNOWN"))
    return result


def threat_tags_for_state(state, enemy):
    player = state.get("player") or {}
    battle = state.get("battle") or {}
    enemies = battle.get("enemies") or []
    hp = safe_int(player.get("hp"), 0)
    max_hp = max(1, safe_int(player.get("max_hp"), hp or 1))
    block = safe_int(player.get("block"), 0)
    enemy_damage = sum(safe_int(intent.get("damage"), 0) for intent in enemy.get("intents") or [])
    total_incoming = incoming_damage(enemies)
    text = " ".join(
        " ".join(str(intent.get(key) or "") for key in ("type", "label", "title", "description"))
        for intent in enemy.get("intents") or []
    ).lower()
    tags = set()
    if enemy_damage:
        tags.add("attack")
    if enemy_damage >= max(15, max_hp * 0.25):
        tags.add("high_damage")
    if hp - max(0, total_incoming - block) <= 0:
        tags.add("lethal_risk")
    if len(enemies) > 1:
        tags.add("multi_enemy")
    if safe_int(enemy.get("hp"), 0) <= max(1, safe_int(enemy.get("max_hp"), 1) * 0.35):
        tags.add("low_enemy_hp")
    if any(word in text for word in ("buff", "strength", "power", "ritual", "scaling")):
        tags.add("scaling")
    if any(word in text for word in ("weak", "vulnerable", "frail", "debuff")):
        tags.add("debuff")
    if any(word in text for word in ("wound", "dazed", "burn", "slime", "status", "curse")):
        tags.add("status_cards")
    if any(word in text for word in ("block", "defend", "shield", "armor")):
        tags.add("block")
    if any(word in text for word in ("summon", "minion", "spawn")):
        tags.add("summon")
    if not tags:
        tags.add("unknown_pattern")
    return sorted(tags)


def encounter_key(enemies):
    counts = Counter(enemy.get("monster_key") or "UNKNOWN" for enemy in enemies)
    return "+".join(f"{key}x{count}" if count > 1 else key for key, count in sorted(counts.items())) or "UNKNOWN"


def action_payload(record):
    action = record.get("action_data")
    if not isinstance(action, dict):
        action = record.get("action")
    if not isinstance(action, dict):
        action = {}
    action_type = action.get("action") or action.get("action_type") or record.get("action_type") or record.get("type")
    payload = {
        "action": action_type,
        "card_id": action.get("card_id") or action.get("card") or action.get("id"),
        "card_name": action.get("card_name") or action.get("card_title"),
        "potion_id": action.get("potion_id"),
        "slot": action.get("slot"),
        "target": action.get("target") or action.get("target_id"),
        "card_index": action.get("card_index"),
        "policy_name": action.get("policy_name") or record.get("policy_name"),
        "model_version": action.get("model_version") or record.get("model_version"),
    }
    return {key: value for key, value in payload.items() if value is not None and value != ""}


def combat_id_for(run_id, state, encounter):
    run = state.get("run") or {}
    act = safe_int(run.get("act"), 0)
    floor = safe_int(run.get("floor"), 0)
    if floor:
        return f"{run_id}:act{act}:floor{floor}:{encounter}"
    battle = state.get("battle") or {}
    round_no = safe_int(battle.get("round"), 0)
    return f"{run_id}:round{round_no}:{encounter}"


def source_label(record, path):
    if record.get("source"):
        return str(record.get("source"))
    text = str(path).replace("\\", "/").lower()
    if "/human/" in text:
        return "human"
    if "/ai/" in text or "/ai_combat/" in text:
        return "ai"
    if "/llm_actions/" in text:
        return "llm"
    return "unknown"


def update_counter(counter, values):
    for value in values:
        if value is not None and value != "":
            counter[str(value)] += 1


def top(counter, limit=12):
    return [{"key": key, "count": count} for key, count in counter.most_common(limit)]


def percentile(values, ratio):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def mean_value(values):
    return sum(values) / len(values) if values else 0.0


def stdev_value(values):
    if len(values) <= 1:
        return 0.0
    avg = mean_value(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return variance ** 0.5


def damage_severity(excess, ratio):
    if excess >= 20 or ratio >= 2.0:
        return "high"
    if excess >= 12 or ratio >= 1.6:
        return "medium"
    return "low"


def is_normal_monster_encounter(encounter):
    if str(encounter.get("state_type") or "").lower() != "monster":
        return False
    act = safe_int(encounter.get("act"), 0)
    floor = safe_int(encounter.get("floor"), 0)
    if act <= 0 or floor <= 0:
        return False
    if floor in BOSS_FLOORS:
        return False
    key = str(encounter.get("encounter_key") or "").upper()
    if "BOSS" in key or "ELITE" in key:
        return False
    return True


def normal_monster_damage_doc(finalized_encounters):
    samples = defaultdict(list)
    details = defaultdict(list)
    for encounter in finalized_encounters:
        if not is_normal_monster_encounter(encounter):
            continue
        hp_start = safe_int(encounter.get("player_hp_start"), 0)
        if hp_start <= 0:
            continue
        hp_lost = max(0, safe_int(encounter.get("hp_lost"), 0))
        detail = {
            "run_id": encounter.get("run_id"),
            "combat_id": encounter.get("combat_id"),
            "source": encounter.get("source"),
            "act": safe_int(encounter.get("act"), 0),
            "floor": safe_int(encounter.get("floor"), 0),
            "encounter_key": encounter.get("encounter_key"),
            "hp_lost": hp_lost,
            "player_hp_start": hp_start,
            "player_hp_end": safe_int(encounter.get("player_hp_end"), 0),
            "turns": safe_int(encounter.get("turns"), 0),
            "result": encounter.get("result"),
            "monsters": encounter.get("monsters") or [],
            "files": encounter.get("files") or [],
            "actions": encounter.get("actions") or {},
            "cards_played": encounter.get("cards_played") or {},
            "potions_used": encounter.get("potions_used") or {},
        }
        for monster in encounter.get("monsters") or []:
            samples[monster].append(float(hp_lost))
            details[monster].append(detail)

    baselines = {}
    anomalies = []
    for monster, values in sorted(samples.items()):
        count = len(values)
        avg = mean_value(values)
        stdev = stdev_value(values)
        p50 = percentile(values, 0.50)
        p75 = percentile(values, 0.75)
        p90 = percentile(values, 0.90)
        max_loss = max(values) if values else 0.0
        reliable = count >= NORMAL_DAMAGE_MIN_SAMPLES
        if reliable:
            threshold = max(avg + max(NORMAL_DAMAGE_MIN_EXCESS, stdev * 1.25), p75 + NORMAL_DAMAGE_MIN_EXCESS)
        else:
            threshold = max(avg + max(12.0, avg * 0.8), p90 + 6.0)
        threshold = round(threshold, 2)
        baselines[monster] = {
            "samples": count,
            "reliable": reliable,
            "avg_hp_lost": round(avg, 2),
            "median_hp_lost": round(p50, 2),
            "p75_hp_lost": round(p75, 2),
            "p90_hp_lost": round(p90, 2),
            "max_hp_lost": round(max_loss, 2),
            "stdev_hp_lost": round(stdev, 2),
            "anomaly_threshold": threshold,
            "anomaly_count": 0,
        }
        if not reliable:
            continue
        for detail in details[monster]:
            hp_lost = float(detail["hp_lost"])
            if hp_lost < threshold or hp_lost < NORMAL_DAMAGE_MIN_ABSOLUTE:
                continue
            excess = hp_lost - avg
            ratio = hp_lost / max(avg, 1.0)
            baselines[monster]["anomaly_count"] += 1
            anomalies.append({
                "monster_key": monster,
                "severity": damage_severity(excess, ratio),
                "hp_lost": int(hp_lost),
                "avg_hp_lost": round(avg, 2),
                "threshold": threshold,
                "excess_vs_avg": round(excess, 2),
                "ratio_vs_avg": round(ratio, 2),
                **detail,
            })

    anomalies.sort(key=lambda item: (item["severity"] != "high", -item["excess_vs_avg"], -item["hp_lost"], item["monster_key"]))
    return {
        "schema_version": SCHEMA_DAMAGE_ANOMALIES,
        "generated_at": generated_at(),
        "rules": {
            "scope": "normal monster fights only (state_type == monster, valid act/floor, excluding boss floors and explicit boss/elite keys)",
            "excluded_boss_floors": sorted(BOSS_FLOORS),
            "min_samples_for_threshold": NORMAL_DAMAGE_MIN_SAMPLES,
            "min_excess_hp": NORMAL_DAMAGE_MIN_EXCESS,
            "min_absolute_hp_lost": NORMAL_DAMAGE_MIN_ABSOLUTE,
            "threshold": "max(avg + max(min_excess_hp, stdev * 1.25), p75 + min_excess_hp) when sample count is reliable",
        },
        "summary": {
            "monster_baselines": len(baselines),
            "reliable_monsters": sum(1 for item in baselines.values() if item["reliable"]),
            "anomalies": len(anomalies),
            "high_severity": sum(1 for item in anomalies if item["severity"] == "high"),
            "medium_severity": sum(1 for item in anomalies if item["severity"] == "medium"),
        },
        "monsters": baselines,
        "anomalies": anomalies[:200],
    }


def init_profile():
    return {
        "names": Counter(),
        "ids": Counter(),
        "turn_observations": 0,
        "encounters_seen": 0,
        "wins": 0,
        "hp_lost_sum": 0,
        "turns_sum": 0,
        "intent_types": Counter(),
        "threat_tags": Counter(),
        "actions": Counter(),
        "cards": Counter(),
        "potions": Counter(),
        "encounter_keys": Counter(),
        "incoming_damage_sum": 0,
        "max_incoming_damage": 0,
    }


def init_encounter(combat_id, run_id, source, state, key, timestamp, path):
    run = state.get("run") or {}
    player = state.get("player") or {}
    return {
        "schema_version": SCHEMA_ENCOUNTER,
        "combat_id": combat_id,
        "run_id": run_id,
        "source": source,
        "act": safe_int(run.get("act"), 0),
        "floor": safe_int(run.get("floor"), 0),
        "state_type": state.get("state_type") or "",
        "encounter_key": key,
        "monsters": [],
        "monster_counts": {},
        "turns_seen": set(),
        "turns": 0,
        "result": "unknown",
        "player_hp_start": safe_int(player.get("hp"), 0),
        "player_hp_end": safe_int(player.get("hp"), 0),
        "player_max_hp": safe_int(player.get("max_hp"), 0),
        "hp_lost": 0,
        "potions_used": Counter(),
        "cards_played": Counter(),
        "actions": Counter(),
        "first_timestamp": timestamp,
        "last_timestamp": timestamp,
        "files": Counter({path.name: 1}),
    }


def finalize_encounter(encounter):
    encounter["turns"] = len(encounter.pop("turns_seen", set()))
    encounter["monster_counts"] = dict(sorted(encounter["monster_counts"].items()))
    encounter["monsters"] = sorted(encounter["monster_counts"])
    encounter["hp_lost"] = max(0, safe_int(encounter.get("player_hp_start"), 0) - safe_int(encounter.get("player_hp_end"), 0))
    encounter["potions_used"] = dict(encounter["potions_used"])
    encounter["cards_played"] = dict(encounter["cards_played"])
    encounter["actions"] = dict(encounter["actions"])
    encounter["files"] = [item["key"] for item in top(encounter["files"], 6)]
    return encounter


def build_profiles(data_dir=DATA_DIR, monster_dir=MONSTER_DIR, processed_dir=PROCESSED_DIR, limit_files=0):
    files = iter_input_files(data_dir, limit_files=limit_files)
    turns_by_date = defaultdict(list)
    encounters = {}
    profiles = defaultdict(init_profile)
    encounter_profiles = defaultdict(lambda: {
        "seen": 0,
        "wins": 0,
        "hp_lost_sum": 0,
        "turns_sum": 0,
        "actions": Counter(),
        "cards": Counter(),
        "potions": Counter(),
        "threat_tags": Counter(),
        "monsters": Counter(),
    })
    seen_turn_rows = set()
    records_scanned = 0
    states_seen = 0

    for path in files:
        for record in read_json_records(path):
            records_scanned += 1
            timestamp = timestamp_ms(record, path)
            source = source_label(record, path)
            run_id = str(record.get("run_id") or f"file:{path.stem}")
            state_key, state = primary_state_for_record(record)
            if not state_has_combat(state):
                continue
            states_seen += 1
            battle = state.get("battle") or {}
            player = state.get("player") or {}
            enemies = battle.get("enemies") or []
            key = encounter_key(enemies)
            combat_id = combat_id_for(run_id, state, key)
            date = date_key(timestamp)
            action = action_payload(record) if record.get("type") == "action" or record.get("action_type") else {}

            if combat_id not in encounters:
                encounters[combat_id] = init_encounter(combat_id, run_id, source, state, key, timestamp, path)
            encounter = encounters[combat_id]
            encounter["last_timestamp"] = max(safe_int(encounter.get("last_timestamp"), 0), timestamp)
            encounter["player_hp_end"] = safe_int(player.get("hp"), encounter.get("player_hp_end", 0))
            encounter["turns_seen"].add(safe_int(battle.get("round"), 0))
            encounter["files"][path.name] += 1
            state_monster_counts = Counter(enemy["monster_key"] for enemy in enemies)
            for monster, count in state_monster_counts.items():
                encounter["monster_counts"][monster] = max(count, encounter["monster_counts"].get(monster, 0))
            if record.get("type") in ("battle_end", "game_end", "victory"):
                result = record.get("result")
                if result:
                    encounter["result"] = str(result)
            if action:
                act = action.get("action") or "UNKNOWN"
                encounter["actions"][act] += 1
                if action.get("card_id"):
                    encounter["cards_played"][action["card_id"]] += 1
                if act == "use_potion":
                    encounter["potions_used"][action.get("potion_id") or action.get("slot") or "UNKNOWN"] += 1

            row_base = {
                "schema_version": SCHEMA_TURN,
                "run_id": run_id,
                "combat_id": combat_id,
                "source": source,
                "timestamp": timestamp,
                "date": date,
                "file": path.name,
                "observation": record.get("type") or state_key or "state",
                "act": safe_int((state.get("run") or {}).get("act"), 0),
                "floor": safe_int((state.get("run") or {}).get("floor"), 0),
                "state_type": state.get("state_type") or "",
                "round": safe_int(battle.get("round"), 0),
                "turn": battle.get("turn") or "",
                "is_play_phase": bool(battle.get("is_play_phase")),
                "encounter_key": key,
                "enemy_count": len(enemies),
                "incoming_damage": incoming_damage(enemies),
                "player_hp": safe_int(player.get("hp"), 0),
                "player_max_hp": safe_int(player.get("max_hp"), 0),
                "player_block": safe_int(player.get("block"), 0),
                "player_energy": safe_int(player.get("energy"), 0),
                "hand_ids": card_ids(player.get("hand") or []),
                "potions": potion_ids(player.get("potions") or []),
                "action_taken": action or None,
            }
            for enemy in enemies:
                tags = threat_tags_for_state(state, enemy)
                row_key = (
                    combat_id,
                    timestamp,
                    row_base["observation"],
                    enemy["slot"],
                    json.dumps(action or {}, sort_keys=True, ensure_ascii=False),
                )
                if row_key in seen_turn_rows:
                    continue
                seen_turn_rows.add(row_key)
                row = dict(row_base)
                row.update({
                    "monster_slot": enemy["slot"],
                    "monster_key": enemy["monster_key"],
                    "monster_id": enemy.get("id") or "",
                    "monster_name": enemy.get("name") or "",
                    "entity_id": enemy.get("entity_id") or "",
                    "hp": enemy.get("hp", 0),
                    "max_hp": enemy.get("max_hp", 0),
                    "block": enemy.get("block", 0),
                    "intents": enemy.get("intents") or [],
                    "status": enemy.get("status") or [],
                    "threat_tags": tags,
                })
                turns_by_date[date].append(row)

                profile = profiles[enemy["monster_key"]]
                profile["turn_observations"] += 1
                profile["names"][enemy.get("name") or enemy["monster_key"]] += 1
                profile["ids"][enemy.get("id") or enemy.get("entity_id") or enemy["monster_key"]] += 1
                update_counter(profile["intent_types"], intent_types([enemy]))
                update_counter(profile["threat_tags"], tags)
                profile["incoming_damage_sum"] += row["incoming_damage"]
                profile["max_incoming_damage"] = max(profile["max_incoming_damage"], row["incoming_damage"])
                if action:
                    profile["actions"][action.get("action") or "UNKNOWN"] += 1
                    if action.get("card_id"):
                        profile["cards"][action["card_id"]] += 1
                    if action.get("potion_id"):
                        profile["potions"][action["potion_id"]] += 1

    finalized_encounters = [finalize_encounter(item) for item in encounters.values()]
    for encounter in finalized_encounters:
        eprof = encounter_profiles[encounter["encounter_key"]]
        eprof["seen"] += 1
        if str(encounter.get("result")).lower() in ("win", "victory", "complete", "true"):
            eprof["wins"] += 1
        eprof["hp_lost_sum"] += safe_int(encounter.get("hp_lost"), 0)
        eprof["turns_sum"] += safe_int(encounter.get("turns"), 0)
        update_counter(eprof["monsters"], encounter.get("monsters") or [])
        for action_name, count in (encounter.get("actions") or {}).items():
            eprof["actions"][action_name] += safe_int(count, 0)
        for card, count in (encounter.get("cards_played") or {}).items():
            eprof["cards"][card] += safe_int(count, 0)
        for potion, count in (encounter.get("potions_used") or {}).items():
            eprof["potions"][potion] += safe_int(count, 0)
        for monster in encounter.get("monsters") or []:
            profile = profiles[monster]
            profile["encounters_seen"] += 1
            profile["encounter_keys"][encounter["encounter_key"]] += 1
            profile["hp_lost_sum"] += safe_int(encounter.get("hp_lost"), 0)
            profile["turns_sum"] += safe_int(encounter.get("turns"), 0)
            if str(encounter.get("result")).lower() in ("win", "victory", "complete", "true"):
                profile["wins"] += 1

    monster_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for date, rows in sorted(turns_by_date.items()):
        write_jsonl(monster_dir / f"monster_turns_{date}.jsonl", rows)

    encounters_by_date = defaultdict(list)
    for encounter in finalized_encounters:
        encounters_by_date[date_key(safe_int(encounter.get("first_timestamp"), 0))].append(encounter)
    for date, rows in sorted(encounters_by_date.items()):
        write_jsonl(monster_dir / f"encounters_{date}.jsonl", rows)

    damage_anomaly_doc = normal_monster_damage_doc(finalized_encounters)
    damage_baselines = damage_anomaly_doc.get("monsters", {})
    monster_profiles = {
        "schema_version": SCHEMA_PROFILE,
        "generated_at": generated_at(),
        "summary": {
            "files_scanned": len(files),
            "records_scanned": records_scanned,
            "combat_states_seen": states_seen,
            "monster_turn_rows": sum(len(rows) for rows in turns_by_date.values()),
            "encounters": len(finalized_encounters),
            "monsters": len(profiles),
            "normal_monster_damage_baselines": damage_anomaly_doc["summary"]["monster_baselines"],
            "damage_anomalies": damage_anomaly_doc["summary"]["anomalies"],
        },
        "monsters": {},
    }
    for key, profile in sorted(profiles.items()):
        seen = max(1, profile["turn_observations"])
        encounters_seen = max(1, profile["encounters_seen"])
        damage_stats = damage_baselines.get(key, {})
        monster_profiles["monsters"][key] = {
            "display_name": profile["names"].most_common(1)[0][0] if profile["names"] else key,
            "ids": top(profile["ids"], 8),
            "turn_observations": profile["turn_observations"],
            "encounters_seen": profile["encounters_seen"],
            "win_rate": round(profile["wins"] / encounters_seen, 4) if profile["encounters_seen"] else None,
            "avg_hp_lost": round(profile["hp_lost_sum"] / encounters_seen, 2) if profile["encounters_seen"] else None,
            "normal_damage": damage_stats or None,
            "avg_turns": round(profile["turns_sum"] / encounters_seen, 2) if profile["encounters_seen"] else None,
            "avg_incoming_damage": round(profile["incoming_damage_sum"] / seen, 2),
            "max_incoming_damage": profile["max_incoming_damage"],
            "common_intents": top(profile["intent_types"], 10),
            "threat_tags": top(profile["threat_tags"], 10),
            "common_actions": top(profile["actions"], 10),
            "common_cards": top(profile["cards"], 10),
            "common_potions": top(profile["potions"], 8),
            "encounter_keys": top(profile["encounter_keys"], 8),
        }

    encounter_profile_doc = {
        "schema_version": SCHEMA_ENCOUNTER_PROFILE,
        "generated_at": generated_at(),
        "summary": {
            "encounter_keys": len(encounter_profiles),
            "encounters": len(finalized_encounters),
        },
        "encounters": {},
    }
    for key, profile in sorted(encounter_profiles.items()):
        seen = max(1, profile["seen"])
        encounter_profile_doc["encounters"][key] = {
            "seen": profile["seen"],
            "win_rate": round(profile["wins"] / seen, 4),
            "avg_hp_lost": round(profile["hp_lost_sum"] / seen, 2),
            "avg_turns": round(profile["turns_sum"] / seen, 2),
            "monsters": top(profile["monsters"], 10),
            "common_actions": top(profile["actions"], 10),
            "common_cards": top(profile["cards"], 10),
            "common_potions": top(profile["potions"], 8),
        }

    write_json(monster_dir / "monster_profiles.json", monster_profiles)
    write_json(monster_dir / "encounter_profiles.json", encounter_profile_doc)
    write_json(monster_dir / "monster_damage_anomalies.json", damage_anomaly_doc)
    write_json(monster_dir / "monster_build_summary.json", monster_profiles["summary"])
    write_text(monster_dir / "README_MONSTER_DATA.md", monster_readme())

    vocab = build_vocab(monster_profiles, encounter_profile_doc)
    write_json(processed_dir / "monster_vocab.json", vocab)
    return monster_profiles["summary"]


def build_vocab(monster_profiles, encounter_profiles):
    monsters = {"PAD": 0, "UNKNOWN": 1}
    intents = {"PAD": 0, "UNKNOWN": 1}
    threat_tags = {"PAD": 0, "UNKNOWN": 1}
    encounters = {"PAD": 0, "UNKNOWN": 1}
    for key, profile in monster_profiles.get("monsters", {}).items():
        monsters.setdefault(key, len(monsters))
        for item in profile.get("common_intents", []):
            intents.setdefault(item["key"], len(intents))
        for item in profile.get("threat_tags", []):
            threat_tags.setdefault(item["key"], len(threat_tags))
        for item in profile.get("encounter_keys", []):
            encounters.setdefault(item["key"], len(encounters))
    for key in encounter_profiles.get("encounters", {}):
        encounters.setdefault(key, len(encounters))
    return {
        "schema_version": SCHEMA_VOCAB,
        "generated_at": generated_at(),
        "monsters": monsters,
        "intent_types": intents,
        "threat_tags": threat_tags,
        "encounters": encounters,
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def monster_readme():
    return """# Monster Dataset

This directory is generated from raw combat logs. It is safe to rebuild.

Files:
- monster_turns_YYYY-MM-DD.jsonl: one row per observed monster state/action context.
- encounters_YYYY-MM-DD.jsonl: one row per combat encounter summary.
- monster_profiles.json: aggregated monster-level play patterns.
- encounter_profiles.json: aggregated encounter-composition patterns.
- monster_damage_anomalies.json: normal-monster HP-loss baselines and outlier encounters.
- monster_build_summary.json: generation counts for dashboard and audit.

The raw logs remain the source of truth. Generated files should be rebuilt after new runs are collected.
"""


def main():
    parser = argparse.ArgumentParser(description="Build derived monster profile data from STS2 combat logs.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(MONSTER_DIR))
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--limit-files", type=int, default=0)
    args = parser.parse_args()
    summary = build_profiles(
        data_dir=Path(args.data_dir),
        monster_dir=Path(args.output_dir),
        processed_dir=Path(args.processed_dir),
        limit_files=args.limit_files,
    )
    print(json.dumps({"status": "ok", **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
