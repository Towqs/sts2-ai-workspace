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
        // ============================================================
        //  Harmony Hooks — 拦截游戏事件
        // ============================================================

        public static void Hook_MainMenuContinue_Prefix()
        {
            RL_DataCollector.MarkMenuRunIntent("continue");
        }

        public static void Hook_CharacterSelectBeginRun_Prefix(string seed)
        {
            RL_DataCollector.MarkMenuRunIntent("new", seed);
        }

        public static void Hook_TurnStarted(CombatState state)
        {
            try
            {
                if (state == null)
                {
                    Debug.Log("TurnStart", "[HOOK] skipped: state null");
                    return;
                }

                if (!string.Equals(state.CurrentSide.ToString(), "Player", StringComparison.OrdinalIgnoreCase))
                {
                    Debug.Log("TurnStart", $"[HOOK] skipped: side={state.CurrentSide}");
                    return;
                }

                if (!CombatManager.Instance.IsPlayPhase)
                {
                    Debug.Log("TurnStart", "[HOOK] skipped: not play phase");
                    return;
                }

                Debug.Log("TurnStart", $"[HOOK] TurnStarted fired round={state.RoundNumber}");
                RL_DataCollector.RecordTurnStart();
            }
            catch (Exception ex)
            {
                Debug.Log("TurnStart", $"[HOOK] ERROR: {ex.Message}");
            }
        }

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
