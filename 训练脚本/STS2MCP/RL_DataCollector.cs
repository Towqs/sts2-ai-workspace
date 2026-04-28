using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;

namespace STS2_MCP
{
    // ============================================================
    //  RL_DataCollector v3 — 训练数据采集器
    //
    //  采集策略:
    //  1. 每回合开始/结束记录完整状态快照
    //  2. 每个动作记录 action + state_diff (状态变化)
    //  3. 战斗开始/结束记录端点数据
    //  4. 统一数据格式，便于训练使用
    //
    //  v3 改进:
    //  - 数据来源分离: human (手动) vs ai (Claude Code/MCP)
    //  - 写入不同子目录，防止数据混淆
    //  - MCP 调用标记防重复记录
    // ============================================================

    internal static class RLDataPaths
    {
        private const string DefaultBaseDir = @"D:\2024 fa fan\XJ12615\STS2_AI_Workspace\RL_Datasets";

        public static string BaseDir { get; } = ResolveBaseDir();
        public static string LogPath => Path.Combine(BaseDir, "rl_monitor.log");

        private static string ResolveBaseDir()
        {
            var envDir = System.Environment.GetEnvironmentVariable("STS2_RL_DATA_DIR");
            return string.IsNullOrWhiteSpace(envDir) ? DefaultBaseDir : envDir;
        }
    }

    // Debug logging helper shared between RL_DataCollector and its Harmony hook classes
    internal static class Debug
    {
        private static readonly string _logPath = RLDataPaths.LogPath;
        private const int MAX_LINES = 5000;
        private static readonly object _lock = new();

        public static void Log(string msg)
        {
            lock (_lock)
            {
                try
                {
                    var line = $"{DateTime.Now:HH:mm:ss} {msg}";
                    var lines = File.Exists(_logPath)
                        ? File.ReadAllLines(_logPath).ToList()
                        : new List<string>();
                    lines.Add(line);
                    if (lines.Count > MAX_LINES)
                        lines = lines.Skip(lines.Count - MAX_LINES).ToList();
                    File.WriteAllLines(_logPath, lines);
                }
                catch { }
            }
        }

        public static void Log(string ctx, string msg) { Log($"[{ctx}] {msg}"); }
    }

    public static class RL_DataCollector
    {
        // ========== 数据来源枚举 ==========
        public enum DataSource
        {
            Human,   // 手动操作（玩家自己玩）
            AI       // AI 自动操作（Claude Code / MCP 工具调用）
        }

        // ========== 配置 ==========
        private static readonly System.Text.Json.JsonSerializerOptions _compactJson = new()
        {
            WriteIndented = false,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        };
        private static string _baseDir = RLDataPaths.BaseDir;
        private static string _combatDir => Path.Combine(_baseDir, "Combat");
        private static string _macroDir => Path.Combine(_baseDir, "Macro");
        // v3: 分离的目录
        private static string _humanCombatDir => Path.Combine(_baseDir, "Human", "Combat");
        private static string _humanMacroDir => Path.Combine(_baseDir, "Human", "Macro");
        private static string _aiCombatDir => Path.Combine(_baseDir, "AI", "Combat");
        private static string _aiMacroDir => Path.Combine(_baseDir, "AI", "Macro");
        private static string _debugLogPath => Path.Combine(_baseDir, "rl_monitor.log");
        private static string _sessionPath => Path.Combine(_baseDir, "active_run_session.json");
        private static string _controlPath => Path.Combine(Directory.GetParent(_baseDir)?.FullName ?? _baseDir, "AI_Training", "control_state.json");
        private static readonly List<Dictionary<string, object>> _lastShopSnapshot = new();

        // 文件句柄（保持打开避免频繁创建）
        private static StreamWriter? _combatWriter;
        private static StreamWriter? _macroWriter;
        private static string? _currentRunId;
        private static bool _initialized;
        private static bool _wasInRun;
        private static bool _wasInCombat;

        // 回合状态缓存（用于计算 diff）
        private static Dictionary<string, object>? _lastCombatState;
        private static Dictionary<string, object>? _lastMacroState;
        private static int _lastRoundNumber;
        private static bool _inCombat;

        // v3: 数据来源追踪
        private static DataSource _currentDataSource = DataSource.Human;
        // v3: MCP 防重复记录（记录上一个动作的哈希）
        private static string? _lastActionHash;
        private static long _lastActionTimestamp = 0;

        // ========== 初始化 ==========
        public static void Initialize()
        {
            if (_initialized) return;
            _initialized = true;

            // 确保所有目录存在
            Directory.CreateDirectory(_combatDir);
            Directory.CreateDirectory(_macroDir);
            Directory.CreateDirectory(_humanCombatDir);
            Directory.CreateDirectory(_humanMacroDir);
            Directory.CreateDirectory(_aiCombatDir);
            Directory.CreateDirectory(_aiMacroDir);

            Debug.Log("========================================");
            Debug.Log($"[INIT] RL_DataCollector v3 started");
            Debug.Log($"[INIT] Base dir: {_baseDir}");
            Debug.Log($"[INIT] Human dir: {_baseDir}\\Human\\");
            Debug.Log($"[INIT] AI dir: {_baseDir}\\AI\\");
            Debug.Log("========================================");
        }

        // v3: 切换数据来源（被 McpMod.Actions.cs 调用）
        public static void SetDataSource(DataSource source)
        {
            _currentDataSource = source;
            Debug.Log($"[SOURCE] Data source set to: {source}");
        }

        // v3: 获取当前数据来源
        public static DataSource GetDataSource() => _currentDataSource;

        public static void StartNewRun()
        {
            // 生成新的 run_id（带来源前缀）
            string sourcePrefix = _currentDataSource == DataSource.AI ? "ai" : "hum";
            _currentRunId = $"{sourcePrefix}_{DateTime.Now:yyyyMMdd_HHmmss}_{Guid.NewGuid().ToString("N")[..8]}";
            _inCombat = false;
            _lastRoundNumber = 0;
            _lastCombatState = null;
            _lastMacroState = null;
            _lastActionHash = null;
            _lastActionTimestamp = 0;

            Debug.Log($"[NEW RUN] Run ID: {_currentRunId}, Source: {_currentDataSource}");
            PersistRunSession(false);
        }

        public static bool StartOrResumeRun()
        {
            Initialize();
            GetCurrentProgress(out int act, out int floor);

            if (ConsumeNewRunRequest())
            {
                Debug.Log("RunState", "[NEW RUN REQUEST] control_state requested a fresh run");
                StartNewRun();
                return false;
            }

            if (TryResumeActiveRun(act, floor))
            {
                _inCombat = false;
                _lastRoundNumber = 0;
                _lastCombatState = null;
                _lastMacroState = null;
                _lastActionHash = null;
                _lastActionTimestamp = 0;
                Debug.Log($"[RESUME RUN] Run ID: {_currentRunId}, Source: {_currentDataSource}, act={act}, floor={floor}");
                PersistRunSession(false);
                return true;
            }

            StartNewRun();
            return false;
        }

        private static void GetCurrentProgress(out int act, out int floor)
        {
            act = 1;
            floor = 0;
            try
            {
                if (!RunManager.Instance.IsInProgress) return;
                var rs = RunManager.Instance.DebugOnlyGetState();
                if (rs == null) return;
                act = rs.CurrentActIndex + 1;
                floor = rs.TotalFloor;
            }
            catch { }
        }

        private static bool TryResumeActiveRun(int currentAct, int currentFloor)
        {
            try
            {
                if (!File.Exists(_sessionPath)) return false;
                using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(_sessionPath));
                var root = doc.RootElement;
                if (root.TryGetProperty("ended", out var endedElem) && endedElem.GetBoolean())
                    return false;
                if (!root.TryGetProperty("run_id", out var runElem))
                    return false;

                int lastAct = root.TryGetProperty("act", out var actElem) && actElem.TryGetInt32(out var a) ? a : 1;
                int lastFloor = root.TryGetProperty("floor", out var floorElem) && floorElem.TryGetInt32(out var f) ? f : 0;

                Debug.Log("RunState", $"[RESUME] Candidate run={runElem.GetString()}, last={lastAct}/{lastFloor}, current={currentAct}/{currentFloor}");

                if (lastFloor > 1 && currentFloor <= 1)
                {
                    Debug.Log("RunState", "[RESUME] Rejected: current floor looks like a fresh run");
                    return false;
                }
                if (currentAct < lastAct)
                {
                    Debug.Log("RunState", "[RESUME] Rejected: current act is before saved act");
                    return false;
                }

                _currentRunId = runElem.GetString();
                return !string.IsNullOrWhiteSpace(_currentRunId);
            }
            catch (Exception ex)
            {
                Debug.Log("RunState", $"[RESUME] Failed to read session: {ex.Message}");
                return false;
            }
        }

        private static void PersistRunSession(bool ended)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(_currentRunId)) return;
                GetCurrentProgress(out int act, out int floor);
                var session = new Dictionary<string, object?>
                {
                    ["run_id"] = _currentRunId,
                    ["source"] = _currentRunId.StartsWith("ai_", StringComparison.OrdinalIgnoreCase) ? "ai" : "human",
                    ["act"] = act,
                    ["floor"] = floor,
                    ["ended"] = ended,
                    ["updated_at"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
                };
                File.WriteAllText(_sessionPath, System.Text.Json.JsonSerializer.Serialize(session, _compactJson));
            }
            catch (Exception ex)
            {
                Debug.Log("RunState", $"[SESSION] Persist failed: {ex.Message}");
            }
        }

        private static bool ConsumeNewRunRequest()
        {
            try
            {
                if (!File.Exists(_controlPath)) return false;
                string json = File.ReadAllText(_controlPath);
                using var doc = System.Text.Json.JsonDocument.Parse(json);
                var root = doc.RootElement;
                if (!root.TryGetProperty("next_run_mode", out var modeElem)) return false;
                if (!string.Equals(modeElem.GetString(), "new", StringComparison.OrdinalIgnoreCase)) return false;

                string updated = System.Text.RegularExpressions.Regex.Replace(
                    json,
                    "\"next_run_mode\"\\s*:\\s*\"new\"",
                    "\"next_run_mode\": \"continue\""
                );
                File.WriteAllText(_controlPath, updated);
                return true;
            }
            catch (Exception ex)
            {
                Debug.Log("RunState", $"[NEW RUN REQUEST] Failed to read control_state: {ex.Message}");
                return false;
            }
        }

        public static void MarkRunEnded()
        {
            // IsInProgress also flips to false when the player closes the game. Keep the
            // session resumable so Continue Game appends to the same run_id.
            PersistRunSession(false);
            _currentRunId = null;
        }

        public static void OnModInit()
        {
            Initialize();
            Debug.Log("[INIT] Mod hook applied successfully");
        }

        public static void RecordGameStart()
        {
            try
            {
                Debug.Log("Data", "[RecordGameStart] Called");
                var state = BuildMinimalMacroState();

                var record = new Dictionary<string, object>
                {
                    ["type"] = "game_start",
                    ["run_id"] = _currentRunId,
                    ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    ["source"] = _currentDataSource == DataSource.AI ? "ai" : "human",
                    ["state"] = state,
                    ["screen_state"] = BuildCurrentScreenState()
                };

                WriteMacroRecord(record);
                Debug.Log("Data", $"[RecordGameStart] game_start recorded, run_id={_currentRunId}");
            }
            catch (Exception ex)
            {
                Debug.Log("Data", $"[RecordGameStart] ERROR: {ex.Message}");
            }
        }

        public static void RecordGameResume()
        {
            try
            {
                Debug.Log("Data", "[RecordGameResume] Called");
                var state = BuildMinimalMacroState();

                var record = new Dictionary<string, object>
                {
                    ["type"] = "game_resume",
                    ["run_id"] = _currentRunId,
                    ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    ["source"] = _currentDataSource == DataSource.AI ? "ai" : "human",
                    ["state"] = state,
                    ["screen_state"] = BuildCurrentScreenState()
                };

                WriteMacroRecord(record);
                Debug.Log("Data", $"[RecordGameResume] game_resume recorded, run_id={_currentRunId}");
            }
            catch (Exception ex)
            {
                Debug.Log("Data", $"[RecordGameResume] ERROR: {ex.Message}");
            }
        }

        private static string GetCombatLogPath()
        {
            string dateStr = DateTime.Now.ToString("yyyy-MM-dd");
            // v3: 根据数据来源选择目录
            string dir = _currentDataSource switch
            {
                DataSource.AI => _aiCombatDir,
                DataSource.Human => _humanCombatDir,
                _ => _combatDir
            };
            string path = Path.Combine(dir, $"combat_run_{dateStr}.jsonl");
            return path;
        }

        private static string GetMacroLogPath()
        {
            string dateStr = DateTime.Now.ToString("yyyy-MM-dd");
            // v3: 根据数据来源选择目录
            string dir = _currentDataSource switch
            {
                DataSource.AI => _aiMacroDir,
                DataSource.Human => _humanMacroDir,
                _ => _macroDir
            };
            string path = Path.Combine(dir, $"macro_run_{dateStr}.jsonl");
            return path;
        }

        private static StreamWriter GetCombatWriter()
        {
            if (_combatWriter == null)
            {
                string path = GetCombatLogPath();
                // v3: 确保目标目录存在
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                _combatWriter = new StreamWriter(path, append: true) { AutoFlush = false };
                Debug.Log($"[NEW FILE] Combat log: {path}");
            }
            return _combatWriter;
        }

        private static StreamWriter GetMacroWriter()
        {
            if (_macroWriter == null)
            {
                string path = GetMacroLogPath();
                // v3: 确保目标目录存在
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                _macroWriter = new StreamWriter(path, append: true) { AutoFlush = false };
                Debug.Log($"[NEW FILE] Macro log: {path}");
            }
            return _macroWriter;
        }

        private static void FlushAll()
        {
            try { _combatWriter?.Flush(); } catch { }
            try { _macroWriter?.Flush(); } catch { }
        }

        // ========== 状态构建（简化版，仅采集训练必需数据）==========
        private static Dictionary<string, object> BuildMinimalCombatState()
        {
            var state = new Dictionary<string, object>();

            try
            {
                if (!RunManager.Instance.IsInProgress)
                    return state;

                var runState = RunManager.Instance.DebugOnlyGetState();
                if (runState == null)
                    return state;

                var player = MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
                if (player == null)
                    return state;

                var creature = player.Creature;
                var combatState = player.PlayerCombatState;

                // 运行进度。训练管线会把这些字段转成 act/floor/round 特征。
                state["act"] = runState.CurrentActIndex + 1;
                state["floor"] = runState.TotalFloor;
                state["ascension"] = runState.AscensionLevel;

                // 玩家基础信息
                state["character"] = McpMod.SafeGetText(() => player.Character.Title) ?? "Unknown";
                state["hp"] = creature.CurrentHp;
                state["max_hp"] = creature.MaxHp;
                state["block"] = creature.Block;

                // 能量信息
                if (combatState != null && CombatManager.Instance.IsInProgress)
                {
                    state["energy"] = combatState.Energy;
                    state["max_energy"] = combatState.MaxEnergy;

                    // 手牌
                    var hand = new List<Dictionary<string, object>>();
                    int idx = 0;
                    foreach (var card in combatState.Hand.Cards)
                    {
                        hand.Add(new Dictionary<string, object>
                        {
                            ["index"] = idx,
                            ["id"] = card.Id.Entry,
                            ["name"] = card.Title.ToString(),
                            ["type"] = card.Type.ToString(),
                            ["cost"] = card.EnergyCost.CostsX ? "X" : card.EnergyCost.GetAmountToSpend().ToString(),
                            ["target_type"] = card.TargetType.ToString(),
                            ["can_play"] = card.CanPlay(out _, out _)
                        });
                        idx++;
                    }
                    state["hand"] = hand;
                    state["hand_count"] = hand.Count;

                    // 牌堆数量
                    state["draw_pile_count"] = combatState.DrawPile.Cards.Count;
                    state["discard_pile_count"] = combatState.DiscardPile.Cards.Count;
                    state["exhaust_pile_count"] = combatState.ExhaustPile.Cards.Count;
                }

                // 战斗回合信息
                var combat = CombatManager.Instance.DebugOnlyGetState();
                if (combat != null)
                {
                    state["round"] = combat.RoundNumber;
                    state["turn"] = combat.CurrentSide.ToString();
                    state["is_play_phase"] = CombatManager.Instance.IsPlayPhase;
                }

                // 敌人信息（简化）
                if (combat != null)
                {
                    var enemies = new List<Dictionary<string, object>>();
                    foreach (var enemy in combat.Enemies)
                    {
                        if (!enemy.IsAlive) continue;

                        var enemyData = new Dictionary<string, object>
                        {
                            ["id"] = enemy.Monster?.Id.Entry ?? "unknown",
                            ["hp"] = enemy.CurrentHp,
                            ["max_hp"] = enemy.MaxHp,
                            ["block"] = enemy.Block
                        };

                        // 意图（简化）
                        if (enemy.Monster?.NextMove is MoveState move)
                        {
                            var intents = new List<string>();
                            foreach (var intent in move.Intents)
                            {
                                intents.Add(intent.IntentType.ToString());
                            }
                            enemyData["intents"] = intents;
                        }

                        enemies.Add(enemyData);
                    }
                    state["enemies"] = enemies;
                    state["enemy_count"] = enemies.Count;
                }

                // 遗物效果（仅采集被动遗物状态）
                var relics = new List<Dictionary<string, object>>();
                foreach (var relic in player.Relics)
                {
                    relics.Add(new Dictionary<string, object>
                    {
                        ["id"] = relic.Id.Entry,
                        ["name"] = relic.Title.ToString(),
                        ["counter"] = relic.ShowCounter ? relic.DisplayAmount : null
                    });
                }
                state["relics"] = relics;

                // 药水
                var potions = new List<Dictionary<string, object>>();
                int slot = 0;
                foreach (var p in player.PotionSlots)
                {
                    if (p != null)
                    {
                        potions.Add(new Dictionary<string, object>
                        {
                            ["slot"] = slot,
                            ["id"] = p.Id.Entry,
                            ["name"] = p.Title.ToString()
                        });
                    }
                    slot++;
                }
                state["potions"] = potions;
            }
            catch (Exception ex)
            {
                Debug.Log("Data", $"[ERROR] BuildMinimalCombatState: {ex.Message}");
            }

            return state;
        }

        private static Dictionary<string, object> BuildMinimalMacroState()
        {
            var state = new Dictionary<string, object>();

            try
            {
                if (!RunManager.Instance.IsInProgress)
                    return state;

                var runState = RunManager.Instance.DebugOnlyGetState();
                if (runState == null)
                    return state;

                var player = MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState);
                if (player == null)
                    return state;

                // 运行信息
                state["act"] = runState.CurrentActIndex + 1;
                state["floor"] = runState.TotalFloor;
                state["ascension"] = runState.AscensionLevel;
                state["character"] = McpMod.SafeGetText(() => player.Character.Title) ?? "Unknown";

                // 玩家状态
                state["hp"] = player.Creature.CurrentHp;
                state["max_hp"] = player.Creature.MaxHp;
                state["gold"] = player.Gold;

                // 牌组统计
                state["deck_size"] = 0; // 需要遍历牌组计算
                state["relic_count"] = player.Relics.Count;
                state["potion_slots_filled"] = 0;
                int potionCount = 0;
                foreach (var p in player.PotionSlots)
                    if (p != null) potionCount++;
                state["potion_slots_filled"] = potionCount;

                // 遗物
                var relics = new List<Dictionary<string, object?>>();
                foreach (var relic in player.Relics)
                {
                    relics.Add(new Dictionary<string, object?>
                    {
                        ["id"] = relic.Id.Entry,
                        ["name"] = McpMod.SafeGetText(() => relic.Title),
                        ["counter"] = relic.ShowCounter ? relic.DisplayAmount : null
                    });
                }
                state["relics"] = relics;

                // 药水
                var potions = new List<Dictionary<string, object?>>();
                int slot = 0;
                foreach (var p in player.PotionSlots)
                {
                    if (p != null)
                    {
                        potions.Add(new Dictionary<string, object?>
                        {
                            ["slot"] = slot,
                            ["id"] = p.Id.Entry,
                            ["name"] = McpMod.SafeGetText(() => p.Title)
                        });
                    }
                    slot++;
                }
                state["potions"] = potions;

                // 当前房间类型
                var room = runState.CurrentRoom;
                if (room != null)
                {
                    state["room_type"] = room.GetType().Name;
                }
            }
            catch (Exception ex)
            {
                Debug.Log($"[ERROR] BuildMinimalMacroState: {ex.Message}");
            }

            return state;
        }

        // ========== 状态差异计算 ==========
        private static Dictionary<string, object?> BuildCurrentScreenState()
        {
            try
            {
                return McpMod.BuildGameState();
            }
            catch (Exception ex)
            {
                Debug.Log("Data", $"[ERROR] BuildCurrentScreenState: {ex.Message}");
                return new Dictionary<string, object?>
                {
                    ["error"] = ex.Message
                };
            }
        }

        private static Dictionary<string, object> ComputeStateDiff(
            Dictionary<string, object> oldState,
            Dictionary<string, object> newState)
        {
            var diff = new Dictionary<string, object>();

            // 简化版：只记录变化的字段
            foreach (var key in newState.Keys)
            {
                if (!oldState.ContainsKey(key))
                {
                    diff[key] = newState[key];
                }
                else
                {
                    var oldVal = oldState[key]?.ToString() ?? "";
                    var newVal = newState[key]?.ToString() ?? "";
                    if (oldVal != newVal)
                    {
                        diff[key] = newState[key];
                    }
                }
            }

            return diff;
        }

        // ========== 数据记录 ==========
        private static void WriteCombatRecord(Dictionary<string, object> record)
        {
            try
            {
                if (!IsCollectionEnabled())
                {
                    Debug.Log("Data", "[Collection] disabled by control_state; skipped combat record");
                    return;
                }
                var writer = GetCombatWriter();
                lock (writer)
                {
                    writer.WriteLine(System.Text.Json.JsonSerializer.Serialize(record, _compactJson));
                    writer.Flush();
                }
                PersistRunSession(false);
            }
            catch (Exception ex)
            {
                Debug.Log($"[ERROR] WriteCombatRecord: {ex.Message}");
            }
        }

        private static void WriteMacroRecord(Dictionary<string, object> record)
        {
            try
            {
                if (!IsCollectionEnabled())
                {
                    Debug.Log("Data", "[Collection] disabled by control_state; skipped macro record");
                    return;
                }
                var writer = GetMacroWriter();
                lock (writer)
                {
                    writer.WriteLine(System.Text.Json.JsonSerializer.Serialize(record, _compactJson));
                    writer.Flush();
                }
                PersistRunSession(false);
            }
            catch (Exception ex)
            {
                Debug.Log($"[ERROR] WriteMacroRecord: {ex.Message}");
            }
        }

        private static bool IsCollectionEnabled()
        {
            try
            {
                if (!File.Exists(_controlPath)) return true;
                using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(_controlPath));
                var root = doc.RootElement;
                if (root.TryGetProperty("collection_enabled", out var elem) && elem.ValueKind == System.Text.Json.JsonValueKind.False)
                    return false;
            }
            catch (Exception ex)
            {
                Debug.Log("Data", $"[Collection] failed to read control_state, defaulting enabled: {ex.Message}");
            }
            return true;
        }

        // ========== 回合快照 ==========
        public static void RecordTurnStart()
        {
            if (!RunManager.Instance.IsInProgress)
            {
                Debug.Log("Data", "[RecordTurnStart] Skipped: no run in progress");
                return;
            }
            if (McpMod.IsMultiplayerRun())
            {
                Debug.Log("Data", "[RecordTurnStart] Skipped: multiplayer run");
                return;
            }

            try
            {
                var combat = CombatManager.Instance.DebugOnlyGetState();
                if (combat == null)
                {
                    Debug.Log("Data", "[RecordTurnStart] Skipped: combat null");
                    return;
                }

                var currentRound = combat.RoundNumber;
                if (currentRound == _lastRoundNumber)
                {
                    Debug.Log("Data", $"[RecordTurnStart] Skipped: duplicate (round {currentRound})");
                    return;
                }

                Debug.Log("Data", $"[RecordTurnStart] Recording round={currentRound}");
                _lastRoundNumber = currentRound;
                var state = BuildMinimalCombatState();

                var record = new Dictionary<string, object>
                {
                    ["type"] = "turn_start",
                    ["run_id"] = _currentRunId,
                    ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    ["source"] = _currentDataSource == DataSource.AI ? "ai" : "human",
                    ["round"] = currentRound,
                    ["state"] = state
                };

                _lastCombatState = state;
                WriteCombatRecord(record);
                Debug.Log("Data", $"[RecordTurnStart] Done round={currentRound}");
            }
            catch (Exception ex)
            {
                Debug.Log("Data", $"[RecordTurnStart] ERROR: {ex.Message}");
            }
        }

        public static void RecordTurnEnd(int cardsLeft, int energyLeft)
        {
            if (!RunManager.Instance.IsInProgress)
            {
                Debug.Log("Data", "[RecordTurnEnd] Skipped: no run in progress");
                return;
            }
            if (McpMod.IsMultiplayerRun())
            {
                Debug.Log("Data", "[RecordTurnEnd] Skipped: multiplayer run");
                return;
            }

            try
            {
                var combat = CombatManager.Instance.DebugOnlyGetState();
                if (combat == null)
                {
                    Debug.Log("Data", "[RecordTurnEnd] Skipped: combat null");
                    return;
                }

                Debug.Log("Data", $"[RecordTurnEnd] Recording round={combat.RoundNumber} cards={cardsLeft} energy={energyLeft}");
                var state = BuildMinimalCombatState();
                var diff = ComputeStateDiff(_lastCombatState ?? new Dictionary<string, object>(), state);

                var record = new Dictionary<string, object>
                {
                    ["type"] = "turn_end",
                    ["run_id"] = _currentRunId,
                    ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    ["source"] = _currentDataSource == DataSource.AI ? "ai" : "human",
                    ["round"] = combat.RoundNumber,
                    ["state"] = state,
                    ["state_diff"] = diff,
                    ["cards_left"] = cardsLeft,
                    ["energy_left"] = energyLeft
                };

                WriteCombatRecord(record);
                Debug.Log("Data", $"[RecordTurnEnd] Done round={combat.RoundNumber}");
            }
            catch (Exception ex)
            {
                Debug.Log("Data", $"[RecordTurnEnd] ERROR: {ex.Message}");
            }
        }

        // ========== 动作记录 ==========
        public static void RecordAction(string actionType, Dictionary<string, object> actionData)
        {
            if (!RunManager.Instance.IsInProgress)
            {
                Debug.Log("Data", $"[RecordAction:{actionType}] Skipped: no run in progress");
                return;
            }
            if (McpMod.IsMultiplayerRun())
            {
                Debug.Log("Data", $"[RecordAction:{actionType}] Skipped: multiplayer run");
                return;
            }

            try
            {
                bool inCombat = IsInCombatContext();
                Debug.Log("Data", $"[RecordAction:{actionType}] inCombat={inCombat}");

                // v3: MCP 防重复记录检查
                // 如果同一来源在 500ms 内记录了相同的动作哈希，跳过
                string actionHash = ComputeActionHash(actionType, actionData);
                long now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                if (_lastActionHash == actionHash && (now - _lastActionTimestamp) < 500)
                {
                    Debug.Log("Data", $"[RecordAction:{actionType}] DEDUP: skipped duplicate (hash={actionHash})");
                    return;
                }
                _lastActionHash = actionHash;
                _lastActionTimestamp = now;

                // v3: source 字段加入所有记录
                string sourceTag = _currentDataSource == DataSource.AI ? "ai" : "human";

                // 战斗动作
                if (inCombat)
                {
                    Debug.Log("Data", $"[RecordAction:{actionType}] Writing COMBAT record (source={sourceTag})");

                    var beforeState = _lastCombatState ?? BuildMinimalCombatState();
                    var afterState = BuildMinimalCombatState();
                    var diff = ComputeStateDiff(beforeState, afterState);

                    var record = new Dictionary<string, object>
                    {
                        ["type"] = "action",
                        ["run_id"] = _currentRunId,
                        ["timestamp"] = now,
                        ["source"] = sourceTag,  // v3: 来源标记
                        ["action_type"] = actionType,
                        ["action_data"] = actionData,
                        ["state_before"] = beforeState,
                        ["state_after"] = afterState,
                        ["state_diff"] = diff
                    };

                    _lastCombatState = afterState;
                    IncrementStateSeq();
                    WriteCombatRecord(record);
                    Debug.Log("Data", $"[RecordAction:{actionType}] Combat record written");
                }
                // 宏观动作
                else
                {
                    Debug.Log("Data", $"[RecordAction:{actionType}] Writing MACRO record (source={sourceTag})");

                    var state = BuildMinimalMacroState();

                    var record = new Dictionary<string, object>
                    {
                        ["type"] = "macro_action",
                        ["run_id"] = _currentRunId,
                        ["timestamp"] = now,
                        ["source"] = sourceTag,  // v3: 来源标记
                        ["action_type"] = actionType,
                        ["action_data"] = actionData,
                        ["state"] = state,
                        ["screen_state"] = BuildCurrentScreenState()
                    };

                    WriteMacroRecord(record);
                    IncrementStateSeq();
                    Debug.Log("Data", $"[RecordAction:{actionType}] Macro record written");
                }
            }
            catch (Exception ex)
            {
                Debug.Log("Data", $"[RecordAction:{actionType}] ERROR: {ex.Message}");
            }
        }

        // v3: 计算动作哈希（用于去重）
        // 包含: 动作类型 + 动作数据 + 状态序列号
        private static string ComputeActionHash(string actionType, Dictionary<string, object> actionData)
        {
            // 状态序列号：每次状态变化（出牌/抽牌/能量变化）时递增
            // 这确保了同名牌连续打出不会被误判为重复
            int seq = GetStateSeq();
            var sb = new System.Text.StringBuilder();
            sb.Append(actionType);
            sb.Append('|');
            sb.Append(seq);
            sb.Append('|');
            foreach (var kv in actionData.OrderBy(k => k.Key))
            {
                sb.Append(kv.Key);
                sb.Append('=');
                sb.Append(kv.Value?.ToString() ?? "null");
                sb.Append('|');
            }
            return sb.ToString().GetHashCode().ToString("X8");
        }

        // v3: 状态序列号（用于区分同名牌连续打出）
        private static int _stateSeq = 0;
        private static int GetStateSeq() { return _stateSeq; }
        public static void IncrementStateSeq() { _stateSeq++; }

        private static bool IsInCombatContext()
        {
            try
            {
                return RunManager.Instance.IsInProgress
                    && CombatManager.Instance.IsInProgress
                    && CombatManager.Instance.IsPlayPhase;
            }
            catch
            {
                return false;
            }
        }

        // ========== 战斗端点 ==========
        public static void RecordBattleStart()
        {
            if (!RunManager.Instance.IsInProgress)
            {
                Debug.Log("Data", "[RecordBattleStart] Skipped: no run in progress");
                return;
            }
            if (McpMod.IsMultiplayerRun())
            {
                Debug.Log("Data", "[RecordBattleStart] Skipped: multiplayer run");
                return;
            }

            try
            {
                if (_inCombat)
                {
                    Debug.Log("Data", "[RecordBattleStart] Skipped: already in combat");
                    return;
                }
                _inCombat = true;
                _lastRoundNumber = 0;

                Debug.Log("Data", "[RecordBattleStart] Recording battle_start");
                var state = BuildMinimalCombatState();

                var record = new Dictionary<string, object>
                {
                    ["type"] = "battle_start",
                    ["run_id"] = _currentRunId,
                    ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    ["source"] = _currentDataSource == DataSource.AI ? "ai" : "human",
                    ["state"] = state
                };

                _lastCombatState = state;
                WriteCombatRecord(record);
                Debug.Log("Data", "[RecordBattleStart] Done");
            }
            catch (Exception ex)
            {
                Debug.Log("Data", $"[RecordBattleStart] ERROR: {ex.Message}");
            }
        }

        public static void RecordBattleEnd(string result, int hp, int maxHp, int floor, int round)
        {
            if (!RunManager.Instance.IsInProgress)
            {
                Debug.Log("Data", $"[RecordBattleEnd:{result}] Skipped: no run in progress");
                return;
            }
            if (McpMod.IsMultiplayerRun())
            {
                Debug.Log("Data", $"[RecordBattleEnd:{result}] Skipped: multiplayer run");
                return;
            }

            try
            {
                _inCombat = false;
                Debug.Log("Data", $"[RecordBattleEnd:{result}] Recording hp={hp}/{maxHp} floor={floor} rounds={round}");
                var state = BuildMinimalCombatState();

                var record = new Dictionary<string, object>
                {
                    ["type"] = "battle_end",
                    ["run_id"] = _currentRunId,
                    ["timestamp"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    ["source"] = _currentDataSource == DataSource.AI ? "ai" : "human",
                    ["result"] = result,
                    ["remaining_hp"] = hp,
                    ["max_hp"] = maxHp,
                    ["floor"] = floor,
                    ["rounds"] = round,
                    ["state"] = state
                };

                WriteCombatRecord(record);
                Debug.Log("Data", $"[RecordBattleEnd:{result}] Done");
            }
            catch (Exception ex)
            {
                Debug.Log("Data", $"[RecordBattleEnd:{result}] ERROR: {ex.Message}");
            }
        }

        // ============================================================
        //  Harmony Hooks — 拦截游戏事件
        // ============================================================

        // Hook_RunStateChange Postfix (called manually via TryApplyHarmonyPatches)
        public static void Hook_RunStateChange_Postfix(RunManager __instance, ref bool __result)
        {
            bool nowInRun = __result;
            if (!nowInRun && _wasInRun)
            {
                _wasInRun = false;
                Debug.Log("RunState", "[HOOK] Run ended");
                RL_DataCollector.MarkRunEnded();
                return;
            }

            if (nowInRun && !_wasInRun)
            {
                _wasInRun = true;
                Debug.Log("RunState", "[HOOK] Run started");
                RL_DataCollector.Initialize();
                // v3: 手动开始新游戏，切换回 Human 数据源
                RL_DataCollector.SetDataSource(RL_DataCollector.DataSource.Human);
                bool resumed = RL_DataCollector.StartOrResumeRun();
                if (resumed)
                    RL_DataCollector.RecordGameResume();
                else
                    RL_DataCollector.RecordGameStart();
            }
        }

        // 战斗状态变化检测
        public static void Hook_CombatStateChange_Postfix(CombatManager __instance, ref bool __result)
        {
            bool nowInCombat = __result;
            if (nowInCombat && !_wasInCombat && RunManager.Instance.IsInProgress)
            {
                _wasInCombat = nowInCombat; // Set flag FIRST
                Debug.Log("Combat", "[HOOK] Combat started");
                RL_DataCollector.RecordBattleStart();
            }
            else if (!nowInCombat)
            {
                _wasInCombat = false;
            }
        }

        // 出牌
        public static void Hook_PlayCard_Postfix(object[] __args)
        {
            Debug.Log("PlayCard", "[HOOK] PlayCard fired, args.Length=" + (__args?.Length.ToString() ?? "null"));
            if (__args == null || __args.Length < 2) return;
            string cardId = "?", cardName = "?", targetId = "none";
            try
            {
                dynamic card = __args[0];
                cardId = ((CardModel)card).Id.Entry;
                cardName = ((CardModel)card).Title.ToString();
            }
            catch { }
            try
            {
                if (__args[1] != null)
                {
                    dynamic target = __args[1];
                    targetId = ((Creature)target).Monster?.Id.Entry ?? "player";
                }
            }
            catch { }
            var actionData = new Dictionary<string, object>
            {
                ["action"] = "play_card",
                ["card_id"] = cardId,
                ["card_name"] = cardName,
                ["target_id"] = targetId
            };
            RL_DataCollector.RecordAction("play_card", actionData);
        }

        [HarmonyPatch(typeof(PlayCardAction), MethodType.Constructor,
            new Type[] { typeof(CardModel), typeof(Creature) })]
        public static class Hook_PlayCard_ActionCtor
        {
            public static void Prefix(CardModel cardModel, Creature? target)
            {
                Debug.Log("PlayCard", $"[HOOK] PlayCard fired: {cardModel.Title}");

                try
                {
                    string cardId = cardModel.Id.Entry;
                    string cardName = cardModel.Title.ToString();
                    string targetId = target?.Monster?.Id.Entry ?? "none";

                    var actionData = new Dictionary<string, object>
                    {
                        ["action"] = "play_card",
                        ["card_id"] = cardId,
                        ["card_name"] = cardName,
                        ["target_id"] = targetId
                    };

                    RL_DataCollector.RecordAction("play_card", actionData);
                    Debug.Log("PlayCard", "[HOOK] RecordAction done");
                }
                catch (Exception ex)
                {
                    Debug.Log("PlayCard", $"[HOOK] ERROR: {ex.Message}");
                }
            }
        }

        // 结束回合
        public static void Hook_EndTurn_Prefix()
        {
            Debug.Log("EndTurn", "[HOOK] EndTurn fired");
            int cardsLeft = 0, energyLeft = 0;
            try
            {
                var rs = RunManager.Instance.DebugOnlyGetState();
                var p = MegaCrit.Sts2.Core.Context.LocalContext.GetMe(rs);
                cardsLeft = p?.PlayerCombatState?.Hand?.Cards?.Count ?? 0;
                energyLeft = p?.PlayerCombatState?.Energy ?? 0;
            }
            catch { }
            RL_DataCollector.RecordTurnEnd(cardsLeft, energyLeft);
            var actionData = new Dictionary<string, object>
            {
                ["action"] = "end_turn",
                ["cards_left"] = cardsLeft,
                ["energy_left"] = energyLeft
            };
            RL_DataCollector.RecordAction("end_turn", actionData);
        }

        // 使用药水
        public static void Hook_UsePotion_Prefix(object __instance)
        {
            Debug.Log("Potion", "[HOOK] UsePotion fired");
            string potionId = "?", potionName = "?";
            try
            {
                dynamic p = __instance;
                potionId = ((PotionModel)p).Id.Entry;
                potionName = ((PotionModel)p).Title.ToString();
            }
            catch { }
            var actionData = new Dictionary<string, object>
            {
                ["action"] = "use_potion",
                ["potion_id"] = potionId,
                ["potion_name"] = potionName
            };
            RL_DataCollector.RecordAction("use_potion", actionData);
        }

        // 地图选路
        public static void Hook_MapNode_Postfix(object[] __args)
        {
            Debug.Log("MapNode", "[HOOK] MapNode fired, args.Length=" + (__args?.Length.ToString() ?? "null"));
            if (__args == null || __args.Length == 0 || __args[0] == null) return;
            string nodeType = "?";
            int col = -1, row = -1;
            try
            {
                dynamic pt = __args[0];
                nodeType = ((MapPoint)pt.Point).PointType.ToString();
                col = pt.Point.coord.col;
                row = pt.Point.coord.row;
            }
            catch { }
            var actionData = new Dictionary<string, object>
            {
                ["action"] = "select_map_node",
                ["node_type"] = nodeType,
                ["col"] = col,
                ["row"] = row
            };
            RL_DataCollector.RecordAction("select_map_node", actionData);
        }

        // 战斗胜利
        [HarmonyPatch(typeof(MegaCrit.Sts2.Core.GameActions.UsePotionAction), MethodType.Constructor,
            new Type[] { typeof(PotionModel), typeof(Creature), typeof(bool) })]
        public static class Hook_UsePotionActionCtor
        {
            public static void Prefix(PotionModel potion, Creature? target, bool isCombatInProgress)
            {
                Debug.Log("Potion", $"[HOOK] UsePotionAction fired: {potion?.Title}");
                try
                {
                    var actionData = new Dictionary<string, object>
                    {
                        ["action"] = "use_potion",
                        ["potion_id"] = potion?.Id.Entry ?? "unknown",
                        ["potion_name"] = potion?.Title?.ToString() ?? "unknown",
                        ["target_id"] = target?.Monster?.Id.Entry ?? "self",
                        ["in_combat"] = isCombatInProgress
                    };
                    RL_DataCollector.RecordAction("use_potion", actionData);
                }
                catch (Exception ex)
                {
                    Debug.Log("Potion", $"[HOOK] ERROR: {ex.Message}");
                }
            }
        }

        [HarmonyPatch(typeof(MegaCrit.Sts2.Core.GameActions.MoveToMapCoordAction), MethodType.Constructor,
            new Type[] { typeof(Player), typeof(MapCoord) })]
        public static class Hook_MoveToMapCoordActionCtor
        {
            public static void Prefix(Player player, MapCoord destination)
            {
                Debug.Log("Map", $"[HOOK] MoveToMapCoord fired: ({destination.col}, {destination.row})");
                try
                {
                    var rs = RunManager.Instance.DebugOnlyGetState();
                    var pt = rs?.Map?.GetPoint(destination);
                    if ((pt == null || pt.PointType.ToString() == "Unknown") && NMapScreen.Instance != null)
                    {
                        var uiPoint = McpMod.FindAll<NMapPoint>(NMapScreen.Instance)
                            .FirstOrDefault(mp => mp.Point != null
                                && mp.Point.coord.col == destination.col
                                && mp.Point.coord.row == destination.row);
                        if (uiPoint?.Point != null)
                            pt = uiPoint.Point;
                    }
                string nodeType = pt?.PointType.ToString() ?? "Unknown";
                var actionData = new Dictionary<string, object>
                {
                    ["action"] = "move_to_map_coord",
                    ["node_type"] = nodeType,
                        ["col"] = destination.col,
                        ["row"] = destination.row
                };
                if (nodeType == "Unknown")
                {
                    actionData["node_type"] = "UnknownEvent";
                    actionData["raw_node_type"] = "Unknown";
                    actionData["is_hidden_room"] = true;
                    actionData["node_type_source"] = "unknown_after_map_and_ui_lookup";
                }
                RL_DataCollector.RecordAction("select_map_node", actionData);
                }
                catch (Exception ex)
                {
                    Debug.Log("Map", $"[HOOK] ERROR: {ex.Message}");
                }
            }
        }

        [HarmonyPatch(typeof(NMapScreen), "OnMapPointSelectedLocally")]
        public static class Hook_MapPointSelectedLocally
        {
            public static void Prefix(NMapPoint point)
            {
                Debug.Log("MapNode", "[HOOK] OnMapPointSelectedLocally fired");
                try
                {
                    if (point?.Point == null) return;
                    var pt = point.Point;
                    var actionData = new Dictionary<string, object>
                    {
                        ["action"] = "select_map_node",
                        ["node_type"] = pt.PointType.ToString(),
                        ["col"] = pt.coord.col,
                        ["row"] = pt.coord.row
                    };
                    RL_DataCollector.RecordAction("select_map_node", actionData);
                }
                catch (Exception ex)
                {
                    Debug.Log("MapNode", $"[HOOK] ERROR: {ex.Message}");
                }
            }
        }

        [HarmonyPatch]
        public static class Hook_CardRewardChoiceDynamic
        {
            private static string? _lastChoiceKey;
            private static long _lastChoiceMs;

            public static IEnumerable<MethodBase> TargetMethods()
            {
                var wantedNames = new HashSet<string>
                {
                    "OnHolderPressed",
                    "OnCardSelected",
                    "OnLocalCardSelected",
                    "SelectCard",
                    "SetSelectedCard"
                };

                foreach (var method in typeof(NCardRewardSelectionScreen).GetMethods(
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly))
                {
                    if (!wantedNames.Contains(method.Name))
                        continue;

                    var parameters = method.GetParameters();
                    bool hasUsefulArg = parameters.Any(p =>
                        p.ParameterType == typeof(NCardHolder)
                        || p.ParameterType == typeof(CardModel)
                        || p.ParameterType.Name.Contains("CardHolder")
                        || p.ParameterType.Name.Contains("CardModel"));
                    if (!hasUsefulArg)
                        continue;

                    Debug.Log("Harmony", $"[PATCH] CardRewardChoice target: {method.Name}({string.Join(",", parameters.Select(p => p.ParameterType.Name))})");
                    yield return method;
                }
            }

            public static void Prefix(MethodBase __originalMethod, object[] __args)
            {
                try
                {
                    CardModel? card = null;
                    foreach (var arg in __args)
                    {
                        if (arg is NCardHolder holder && holder.CardModel != null)
                        {
                            card = holder.CardModel;
                            break;
                        }
                        if (arg is CardModel model)
                        {
                            card = model;
                            break;
                        }
                    }
                    if (card == null)
                    {
                        Debug.Log("Reward", $"[HOOK] CardRewardChoice {__originalMethod.Name}: no card argument");
                        return;
                    }

                    long now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                    string choiceKey = $"{card.Id.Entry}|{__originalMethod.Name}";
                    if (_lastChoiceKey == choiceKey && now - _lastChoiceMs < 1000)
                    {
                        Debug.Log("Reward", $"[HOOK] CardRewardChoice duplicate skipped: {choiceKey}");
                        return;
                    }
                    _lastChoiceKey = choiceKey;
                    _lastChoiceMs = now;

                    Debug.Log("Reward", $"[HOOK] Card reward selected via {__originalMethod.Name}: {card.Title}");
                    var actionData = new Dictionary<string, object>
                    {
                        ["action"] = "choose_card",
                        ["card_id"] = card.Id.Entry,
                        ["card_title"] = card.Title.ToString(),
                        ["card_name"] = card.Title.ToString(),
                        ["hook_method"] = __originalMethod.Name
                    };
                    RL_DataCollector.RecordAction("choose_card", actionData);
                }
                catch (Exception ex)
                {
                    Debug.Log("Reward", $"[HOOK] Card reward choice ERROR: {ex.Message}");
                }
            }
        }

        [HarmonyPatch(typeof(CombatManager), "EndCombatInternal")]
        public static class Hook_BattleWin_EndCombatInternal
        {
            public static void Postfix(CombatManager __instance)
            {
                Debug.Log("BattleWin", "[HOOK] EndCombatInternal fired");
                try
                {
                    if (!RunManager.Instance.IsInProgress) return;
                    var combat = __instance.DebugOnlyGetState();
                    if (combat == null) return;

                    bool allEnemiesDead = true;
                    foreach (var enemy in combat.Enemies)
                    {
                        if (enemy.IsAlive)
                        {
                            allEnemiesDead = false;
                            break;
                        }
                    }
                    if (!allEnemiesDead) return;

                    var rs = RunManager.Instance.DebugOnlyGetState();
                    var p = MegaCrit.Sts2.Core.Context.LocalContext.GetMe(rs);
                    RL_DataCollector.RecordBattleEnd("win",
                        p?.Creature?.CurrentHp ?? 0,
                        p?.Creature?.MaxHp ?? 0,
                        rs?.TotalFloor ?? 0,
                        combat.RoundNumber);
                }
                catch (Exception ex)
                {
                    Debug.Log("BattleWin", $"[HOOK] ERROR: {ex.Message}");
                }
            }
        }

        public static void Hook_BattleWin_Postfix(CombatManager __instance)
        {
            Debug.Log("BattleWin", "[HOOK] BattleWin fired (ResolveCombatEnd)");
            try
            {
                if (!RunManager.Instance.IsInProgress) return;
                var combat = __instance.DebugOnlyGetState();
                if (combat == null) return;
                bool allEnemiesDead = true;
                foreach (var enemy in combat.Enemies)
                {
                    if (enemy.IsAlive) { allEnemiesDead = false; break; }
                }
                if (allEnemiesDead)
                {
                    var rs = RunManager.Instance.DebugOnlyGetState();
                    var p = MegaCrit.Sts2.Core.Context.LocalContext.GetMe(rs);
                    RL_DataCollector.RecordBattleEnd("win",
                        p?.Creature?.CurrentHp ?? 0,
                        p?.Creature?.MaxHp ?? 0,
                        rs?.TotalFloor ?? 0,
                        combat.RoundNumber);
                }
            }
            catch (Exception ex)
            {
                Debug.Log("BattleWin", $"[HOOK] BattleWin exception: {ex.Message}");
            }
        }

        // 战斗失败 (LoseCombat — 非 async，100% 能 Hook)
        public static void Hook_BattleLose_Prefix()
        {
            Debug.Log("BattleLose", "[HOOK] BattleLose fired (LoseCombat)");
            try
            {
                if (!RunManager.Instance.IsInProgress) return;
                var rs = RunManager.Instance.DebugOnlyGetState();
                var p = MegaCrit.Sts2.Core.Context.LocalContext.GetMe(rs);
                var combat = CombatManager.Instance.DebugOnlyGetState();
                RL_DataCollector.RecordBattleEnd("lose",
                    0,
                    p?.Creature?.MaxHp ?? 0,
                    rs?.TotalFloor ?? 0,
                    combat?.RoundNumber ?? 0);
            }
            catch (Exception ex)
            {
                Debug.Log("BattleLose", $"[HOOK] BattleLose exception: {ex.Message}");
            }
        }

        // ===================== 宏观决策 Hooks =====================

        // 领取奖励 (Reward.OnSelectWrapper — 所有奖励类型的共用入口)
        public static void Hook_RewardClaimed_Prefix(object __instance)
        {
            Debug.Log("Reward", "[HOOK] RewardClaimed fired");
            try
            {
                if (!RunManager.Instance.IsInProgress) return;
                string rewardType = __instance?.GetType().Name ?? "?";
                string rewardName = "?";
                try { dynamic r = __instance; rewardName = r.Description?.ToString() ?? "?"; } catch { }
                var actionData = new Dictionary<string, object>
                {
                    ["action"] = "claim_reward",
                    ["reward_type"] = rewardType,
                    ["reward_name"] = rewardName
                };
                RL_DataCollector.RecordAction("claim_reward", actionData);
            }
            catch (Exception ex)
            {
                Debug.Log("Reward", $"[HOOK] RewardClaimed exception: {ex.Message}");
            }
        }

        // 跳过奖励（CardReward.OnSkipped / PotionReward.OnSkipped）
        public static void Hook_RewardSkipped_Prefix(object __instance)
        {
            Debug.Log("Reward", "[HOOK] RewardSkipped fired");
            try
            {
                if (!RunManager.Instance.IsInProgress) return;
                string rewardType = __instance?.GetType().Name ?? "?";
                var actionData = new Dictionary<string, object>
                {
                    ["action"] = "skip_reward",
                    ["reward_type"] = rewardType
                };
                RL_DataCollector.RecordAction("skip_reward", actionData);
            }
            catch (Exception ex)
            {
                Debug.Log("Reward", $"[HOOK] RewardSkipped exception: {ex.Message}");
            }
        }

        // 篝火选择 (NRestSiteButton.OnPress — 虚方法，NButton子类)
        public static void Hook_RestSite_Prefix(object __instance)
        {
            Debug.Log("RestSite", "[HOOK] RestSiteButton.OnPress fired");
            try
            {
                if (!RunManager.Instance.IsInProgress) return;
                string optionId = "?", optionTitle = "?";
                try { dynamic btn = __instance; optionId = btn.Option?.OptionId ?? "?"; optionTitle = btn.Option?.Title?.ToString() ?? "?"; } catch { }
                var actionData = new Dictionary<string, object>
                {
                    ["action"] = "choose_rest_option",
                    ["option_id"] = optionId,
                    ["option_title"] = optionTitle
                };
                RL_DataCollector.RecordAction("choose_rest_option", actionData);
            }
            catch (Exception ex)
            {
                Debug.Log("RestSite", $"[HOOK] RestSite exception: {ex.Message}");
            }
        }

        // 事件选项 (NEventOptionButton.OnPress — 虚方法，NButton子类)
        public static void Hook_EventOption_Prefix(object __instance)
        {
            Debug.Log("Event", "[HOOK] EventOptionButton.OnPress fired");
            try
            {
                if (!RunManager.Instance.IsInProgress) return;
                string optionTitle = "?", eventName = "?";
                try { dynamic btn = __instance; optionTitle = btn.Option?.Title?.ToString() ?? "?"; eventName = btn.Event?.Title?.ToString() ?? "?"; } catch { }
                var actionData = new Dictionary<string, object>
                {
                    ["action"] = "choose_event_option",
                    ["event_name"] = eventName,
                    ["option_title"] = optionTitle
                };
                RL_DataCollector.RecordAction("choose_event_option", actionData);
            }
            catch (Exception ex)
            {
                Debug.Log("Event", $"[HOOK] EventOption exception: {ex.Message}");
            }
        }

        // 商店购买完成 (NMerchantInventory.OnPurchaseCompleted)
        public static void Hook_ShopPurchase_Postfix(object[] __args)
        {
            Debug.Log("Shop", "[HOOK] OnPurchaseCompleted fired");
            try
            {
                if (!RunManager.Instance.IsInProgress) return;
                string status = "?", itemName = "?", itemType = "?"; int cost = 0;
                string itemId = "?";
                try
                {
                    if (__args?.Length >= 2)
                    {
                        dynamic s = __args[0]; status = s.ToString();
                        var entryObj = __args[1];
                        dynamic entry = entryObj;
                        itemType = entryObj?.GetType().Name ?? "?";
                        cost = (int)entry.Cost;
                        if (entryObj is MerchantCardEntry cardEntry && cardEntry.CreationResult?.Card != null)
                        {
                            itemId = cardEntry.CreationResult.Card.Id.Entry;
                            itemName = cardEntry.CreationResult.Card.Title.ToString();
                        }
                        else if (entryObj is MerchantRelicEntry relicEntry && relicEntry.Model != null)
                        {
                            itemId = relicEntry.Model.Id.Entry;
                            itemName = relicEntry.Model.Title.ToString();
                        }
                        else if (entryObj is MerchantPotionEntry potionEntry && potionEntry.Model != null)
                        {
                            itemId = potionEntry.Model.Id.Entry;
                            itemName = potionEntry.Model.Title.ToString();
                        }
                        else if (entryObj is MerchantCardRemovalEntry)
                        {
                            itemId = "card_removal";
                            itemName = "Card Removal";
                        }
                        else
                        {
                            try { itemName = entry.Title?.ToString() ?? entry.Name?.ToString() ?? "?"; } catch { }
                            try { itemId = entry.Id?.Entry?.ToString() ?? "?"; } catch { }
                        }
                    }
                } catch { }
                var snapshotMatch = FindShopSnapshotMatch(itemType, cost);
                if ((itemId == "?" || itemName == "?") && snapshotMatch != null)
                {
                    itemId = snapshotMatch.TryGetValue("item_id", out var sid) ? sid?.ToString() ?? itemId : itemId;
                    itemName = snapshotMatch.TryGetValue("item_name", out var sn) ? sn?.ToString() ?? itemName : itemName;
                }
                // 只记录成功购买
                if (status.Contains("Success") || status.Contains("success") || status == "Purchased")
                {
                    var actionData = new Dictionary<string, object>
                    {
                        ["action"] = "buy_item",
                        ["item_type"] = itemType,
                        ["item_id"] = itemId,
                        ["item_name"] = itemName,
                        ["category"] = snapshotMatch?.GetValueOrDefault("category")?.ToString() ?? "?",
                        ["cost"] = cost,
                        ["purchase_status"] = status
                    };
                    RL_DataCollector.RecordAction("buy_item", actionData);
                }
            }
            catch (Exception ex)
            {
                Debug.Log("Shop", $"[HOOK] ShopPurchase exception: {ex.Message}");
            }
        }

        [HarmonyPatch(typeof(MegaCrit.Sts2.Core.Nodes.Screens.Shops.NMerchantInventory), "Open")]
        public static class Hook_ShopOpenSnapshot
        {
            public static void Postfix(object __instance)
            {
                try
                {
                    _lastShopSnapshot.Clear();
                    dynamic invNode = __instance;
                    var inventory = invNode.Inventory;
                    if (inventory == null) return;

                    foreach (MerchantCardEntry entry in inventory.CardEntries)
                    {
                        var item = BuildShopSnapshotItem("MerchantCardEntry", entry.Cost, "card");
                        if (entry.CreationResult?.Card is { } card)
                        {
                            item["item_id"] = card.Id.Entry;
                            item["item_name"] = McpMod.SafeGetText(() => card.Title) ?? card.Title.ToString();
                        }
                        _lastShopSnapshot.Add(item);
                    }
                    foreach (MerchantRelicEntry entry in inventory.RelicEntries)
                    {
                        var item = BuildShopSnapshotItem("MerchantRelicEntry", entry.Cost, "relic");
                        if (entry.Model != null)
                        {
                            item["item_id"] = entry.Model.Id.Entry;
                            item["item_name"] = McpMod.SafeGetText(() => entry.Model.Title) ?? entry.Model.Title.ToString();
                        }
                        _lastShopSnapshot.Add(item);
                    }
                    foreach (MerchantPotionEntry entry in inventory.PotionEntries)
                    {
                        var item = BuildShopSnapshotItem("MerchantPotionEntry", entry.Cost, "potion");
                        if (entry.Model != null)
                        {
                            item["item_id"] = entry.Model.Id.Entry;
                            item["item_name"] = McpMod.SafeGetText(() => entry.Model.Title) ?? entry.Model.Title.ToString();
                        }
                        _lastShopSnapshot.Add(item);
                    }
                    if (inventory.CardRemovalEntry is MerchantCardRemovalEntry removal)
                    {
                        var item = BuildShopSnapshotItem("MerchantCardRemovalEntry", removal.Cost, "card_removal");
                        item["item_id"] = "card_removal";
                        item["item_name"] = "Card Removal";
                        _lastShopSnapshot.Add(item);
                    }
                    Debug.Log("Shop", $"[HOOK] Shop snapshot captured: {_lastShopSnapshot.Count} items");
                }
                catch (Exception ex)
                {
                    Debug.Log("Shop", $"[HOOK] Shop snapshot exception: {ex.Message}");
                }
            }
        }

        private static Dictionary<string, object> BuildShopSnapshotItem(string itemType, int cost, string category)
        {
            return new Dictionary<string, object>
            {
                ["item_type"] = itemType,
                ["item_id"] = "?",
                ["item_name"] = "?",
                ["category"] = category,
                ["cost"] = cost
            };
        }

        private static Dictionary<string, object>? FindShopSnapshotMatch(string itemType, int cost)
        {
            Dictionary<string, object>? fallback = null;
            foreach (var item in _lastShopSnapshot)
            {
                if (!Equals(item.GetValueOrDefault("item_type"), itemType))
                    continue;
                fallback ??= item;
                if (item.TryGetValue("cost", out var c) && c is int itemCost && itemCost == cost)
                    return item;
            }
            return fallback;
        }
    }
}
