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
    public static partial class RL_DataCollector
    {
        // ========== 数据记录 ==========
        private static void StampSchemaVersion(Dictionary<string, object> record)
        {
            if (!record.ContainsKey("schema_version"))
                record["schema_version"] = SchemaVersion;
        }

        private static void StampPolicyContext(Dictionary<string, object> record)
        {
            if (_currentDataSource != DataSource.AI)
                return;
            if (!string.IsNullOrWhiteSpace(_currentPolicyName) && !record.ContainsKey("policy_name"))
                record["policy_name"] = _currentPolicyName;
            if (!string.IsNullOrWhiteSpace(_currentModelVersion) && !record.ContainsKey("model_version"))
                record["model_version"] = _currentModelVersion;
        }

        private static void WriteCombatRecord(Dictionary<string, object> record)
        {
            try
            {
                if (!IsCollectionEnabled())
                {
                    Debug.Log("Data", "[Collection] disabled by control_state; skipped combat record");
                    return;
                }
                StampSchemaVersion(record);
                StampPolicyContext(record);
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
                StampSchemaVersion(record);
                StampPolicyContext(record);
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

    }
}
