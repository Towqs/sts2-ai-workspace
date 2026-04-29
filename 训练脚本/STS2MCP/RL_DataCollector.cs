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

    public static partial class RL_DataCollector
    {
        private const int SchemaVersion = 4;

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
        private static string? _currentPolicyName;
        private static string? _currentModelVersion;
        private static string? _pendingMenuRunIntent;
        private static string? _pendingMenuRunSeed;
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
            if (_currentDataSource == source) return;
            _currentDataSource = source;
            if (source == DataSource.Human)
            {
                _currentPolicyName = null;
                _currentModelVersion = null;
            }
            // 关闭旧 writer，下一次写入时会按新来源目录重建
            try { _combatWriter?.Close(); } catch { }
            try { _macroWriter?.Close(); } catch { }
            _combatWriter = null;
            _macroWriter = null;
            Debug.Log($"[SOURCE] Data source set to: {source}, writers reset");
        }

        // v3: 获取当前数据来源
        public static DataSource GetDataSource() => _currentDataSource;

        public static void SetPolicyContext(string? policyName, string? modelVersion)
        {
            _currentPolicyName = string.IsNullOrWhiteSpace(policyName) ? null : policyName;
            _currentModelVersion = string.IsNullOrWhiteSpace(modelVersion) ? null : modelVersion;
        }

        public static void MarkMenuRunIntent(string intent, string? seed = null)
        {
            if (!string.Equals(intent, "new", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(intent, "continue", StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            _pendingMenuRunIntent = intent.ToLowerInvariant();
            _pendingMenuRunSeed = string.IsNullOrWhiteSpace(seed) ? null : seed;
            Debug.Log("RunState", $"[MENU INTENT] {_pendingMenuRunIntent}"
                + (_pendingMenuRunSeed == null ? "" : $" seed={_pendingMenuRunSeed}"));
        }

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
            string controlMode = ReadNextRunMode();

            if (string.Equals(controlMode, "new", StringComparison.OrdinalIgnoreCase))
            {
                Debug.Log("RunState", "[NEW RUN REQUEST] control_state requested a fresh run");
                ResetNextRunModeToAuto();
                StartNewRun();
                return false;
            }

            string? menuIntent = ConsumePendingMenuRunIntent();
            if (string.Equals(menuIntent, "new", StringComparison.OrdinalIgnoreCase))
            {
                Debug.Log("RunState", "[MENU INTENT] Starting a fresh run from character select");
                StartNewRun();
                return false;
            }

            bool forceContinue = string.Equals(controlMode, "continue", StringComparison.OrdinalIgnoreCase)
                || string.Equals(menuIntent, "continue", StringComparison.OrdinalIgnoreCase);
            if (forceContinue)
            {
                Debug.Log("RunState", "[CONTINUE REQUEST] Trying to resume active run");
                ResetNextRunModeToAuto();
            }

            if (TryResumeActiveRun(act, floor, forceContinue))
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

        private static bool TryResumeActiveRun(int currentAct, int currentFloor, bool forceContinue = false)
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

                if (!forceContinue && lastFloor > 1 && currentFloor <= 1)
                {
                    Debug.Log("RunState", "[RESUME] Rejected: current floor looks like a fresh run");
                    return false;
                }
                if (!forceContinue && currentAct < lastAct)
                {
                    Debug.Log("RunState", "[RESUME] Rejected: current act is before saved act");
                    return false;
                }
                if (!forceContinue && lastAct <= 1 && lastFloor <= 1 && currentAct <= 1 && currentFloor <= 1)
                {
                    Debug.Log("RunState", "[RESUME] Rejected: ambiguous early-floor run without Continue intent");
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

        private static string ConsumePendingMenuRunIntent()
        {
            string intent = _pendingMenuRunIntent ?? "";
            _pendingMenuRunIntent = null;
            _pendingMenuRunSeed = null;
            return intent;
        }

        private static string ReadNextRunMode()
        {
            try
            {
                if (!File.Exists(_controlPath)) return "auto";
                string json = File.ReadAllText(_controlPath);
                using var doc = System.Text.Json.JsonDocument.Parse(json);
                var root = doc.RootElement;
                if (!root.TryGetProperty("next_run_mode", out var modeElem)) return "auto";
                string mode = modeElem.GetString() ?? "auto";
                return mode.Equals("new", StringComparison.OrdinalIgnoreCase)
                    || mode.Equals("continue", StringComparison.OrdinalIgnoreCase)
                    || mode.Equals("auto", StringComparison.OrdinalIgnoreCase)
                    ? mode.ToLowerInvariant()
                    : "auto";
            }
            catch (Exception ex)
            {
                Debug.Log("RunState", $"[RUN MODE] Failed to read control_state: {ex.Message}");
                return "auto";
            }
        }

        private static void ResetNextRunModeToAuto()
        {
            try
            {
                if (!File.Exists(_controlPath)) return;
                string json = File.ReadAllText(_controlPath);
                string updated = System.Text.RegularExpressions.Regex.Replace(
                    json,
                    "\"next_run_mode\"\\s*:\\s*\"(?:new|continue)\"",
                    "\"next_run_mode\": \"auto\""
                );
                File.WriteAllText(_controlPath, updated);
            }
            catch (Exception ex)
            {
                Debug.Log("RunState", $"[RUN MODE] Failed to reset control_state: {ex.Message}");
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

    }
}
