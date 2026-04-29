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

    }
}
