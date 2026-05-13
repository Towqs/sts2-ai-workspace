using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;

namespace STS2_MCP;

public static partial class McpMod
{
    private static string? OptionalString(Dictionary<string, JsonElement> data, string key)
    {
        if (!data.TryGetValue(key, out var elem) || elem.ValueKind == JsonValueKind.Null)
            return null;
        return elem.ValueKind == JsonValueKind.String ? elem.GetString() : elem.ToString();
    }

    private static Dictionary<string, object?> ExecuteAction(string action, Dictionary<string, JsonElement> data)
    {
        if (action == "set_game_speed")
            return ExecuteSetGameSpeed(data);
        if (action == "set_data_source")
            return ExecuteSetDataSource(data);

        if (action == "start_new_run")
            return ExecuteStartNewRun(data);
        if (action == "continue_run")
            return ExecuteContinueRun();
        if (action == "debug_menu_buttons")
            return ExecuteDebugMenuButtons();
        if (action == "debug_crystal_sphere")
            return ExecuteDebugCrystalSphere();
        if (action == "return_to_menu" || action == "abandon_run")
            return ExecuteReturnToMenu(action);

        if (!RunManager.Instance.IsInProgress)
            return Error("No run in progress");

        // v3: MCP 调用全部标记为 AI 来源（防止与手动操作数据混淆）
        RL_DataCollector.SetDataSource(RL_DataCollector.DataSource.AI);
        RL_DataCollector.SetPolicyContext(OptionalString(data, "policy_name"), OptionalString(data, "model_version"));

        var runState = RunManager.Instance.DebugOnlyGetState()!;
        var player = LocalContext.GetMe(runState);
        if (player == null)
            return Error("Could not find local player");

        return action switch
        {
            "play_card" => ExecutePlayCard(player, data),
            "use_potion" => ExecuteUsePotion(player, data),
            "discard_potion" => ExecuteDiscardPotion(player, data),
            "end_turn" => ExecuteEndTurn(player),
            "choose_map_node" => ExecuteChooseMapNode(data),
            "choose_event_option" => ExecuteChooseEventOption(data),
            "advance_dialogue" => ExecuteAdvanceDialogue(),
            "choose_rest_option" => ExecuteChooseRestOption(data),
            "shop_purchase" => ExecuteShopPurchase(player, data),
            "claim_reward" => ExecuteClaimReward(data),
            "select_card_reward" => ExecuteSelectCardReward(data),
            "skip_card_reward" => ExecuteSkipCardReward(),
            "proceed" => ExecuteProceed(),
            "select_card" => ExecuteSelectCard(data),
            "confirm_selection" => ExecuteConfirmSelection(),
            "cancel_selection" => ExecuteCancelSelection(),
            "select_bundle" => ExecuteSelectBundle(data),
            "confirm_bundle_selection" => ExecuteConfirmBundleSelection(),
            "cancel_bundle_selection" => ExecuteCancelBundleSelection(),
            "combat_select_card" => ExecuteCombatSelectCard(data),
            "combat_confirm_selection" => ExecuteCombatConfirmSelection(),
            "select_relic" => ExecuteSelectRelic(data),
            "skip_relic_selection" => ExecuteSkipRelicSelection(),
            "claim_treasure_relic" => ExecuteClaimTreasureRelic(data),
            "crystal_sphere_set_tool" => ExecuteCrystalSphereSetTool(data),
            "crystal_sphere_click_cell" => ExecuteCrystalSphereClickCell(data),
            "crystal_sphere_proceed" => ExecuteCrystalSphereProceed(),
            _ => Error($"Unknown action: {action}")
        };
    }

    private static double OptionalDouble(Dictionary<string, JsonElement> data, string key, double fallback)
    {
        if (!data.TryGetValue(key, out var elem) || elem.ValueKind == JsonValueKind.Null)
            return fallback;
        if (elem.ValueKind == JsonValueKind.Number && elem.TryGetDouble(out var number))
            return number;
        if (elem.ValueKind == JsonValueKind.String && double.TryParse(elem.GetString(), out number))
            return number;
        return fallback;
    }

    private static bool OptionalBool(Dictionary<string, JsonElement> data, string key, bool fallback)
    {
        if (!data.TryGetValue(key, out var elem) || elem.ValueKind == JsonValueKind.Null)
            return fallback;
        if (elem.ValueKind == JsonValueKind.True) return true;
        if (elem.ValueKind == JsonValueKind.False) return false;
        if (elem.ValueKind == JsonValueKind.String && bool.TryParse(elem.GetString(), out var parsed))
            return parsed;
        return fallback;
    }

    private static Dictionary<string, object?> ExecuteSetGameSpeed(Dictionary<string, JsonElement> data)
    {
        var enabled = OptionalBool(data, "enabled", true);
        var requestedSpeed = OptionalDouble(data, "speed", OptionalDouble(data, "multiplier", 2.0));
        var speed = enabled ? requestedSpeed : 1.0;
        if (speed < 1.0) speed = 1.0;
        if (speed > 6.0) speed = 6.0;

        Engine.TimeScale = speed;

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["enabled"] = enabled,
            ["speed"] = speed,
            ["message"] = enabled ? $"Game speed set to {speed:0.##}x" : "Game speed restored to 1x"
        };
    }

    private static Dictionary<string, object?> ExecuteSetDataSource(Dictionary<string, JsonElement> data)
    {
        var source = (OptionalString(data, "source") ?? "human").Trim().ToLowerInvariant();
        var dataSource = source == "ai" ? RL_DataCollector.DataSource.AI : RL_DataCollector.DataSource.Human;
        RL_DataCollector.SetDataSource(dataSource);
        RL_DataCollector.SetPolicyContext(OptionalString(data, "policy_name"), OptionalString(data, "model_version"));
        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["source"] = dataSource == RL_DataCollector.DataSource.AI ? "ai" : "human",
            ["message"] = $"Data source set to {source}"
        };
    }

    private static Dictionary<string, object?> ExecuteStartNewRun(Dictionary<string, JsonElement> data)
    {
        var character = (OptionalString(data, "character") ?? "IRONCLAD").Trim().ToUpperInvariant();
        var ascension = (int)OptionalDouble(data, "ascension", 0);
        var seed = OptionalString(data, "seed");
        if (!string.IsNullOrWhiteSpace(seed))
            seed = seed.Trim().ToUpperInvariant();

        if (character != "IRONCLAD")
            return Error("start_new_run currently supports character=IRONCLAD only");
        if (ascension != 0)
            return Error("start_new_run currently supports ascension=0 only");
        if (RunManager.Instance.IsInProgress)
            return Error("A run is already in progress");

        RL_DataCollector.SetDataSource(RL_DataCollector.DataSource.AI);
        RL_DataCollector.MarkMenuRunIntent("new", seed, RL_DataCollector.DataSource.AI);

        var root = ((SceneTree)Engine.GetMainLoop()).Root;
        var characterSelect = TryHandleCharacterSelectStart(root, seed);
        if (characterSelect != null)
            return characterSelect;

        var direct = TryInvokeMenuStartMethod(root, seed);
        if (direct != null)
            return direct;

        if (!string.IsNullOrWhiteSpace(seed))
        {
            var opened = TryClickMenuStartButton(root);
            if (opened != null)
            {
                opened["message"] = $"{opened["message"]}; opened character select before applying seed {seed}";
                return opened;
            }

            opened = TryInvokeMenuStartMethod(root);
            if (opened != null)
            {
                opened["message"] = $"{opened["message"]}; opened character select before applying seed {seed}";
                return opened;
            }

            return Error("start_new_run with seed could not reach the character select screen");
        }

        var clicked = TryClickMenuStartButton(root);
        if (clicked != null)
            return clicked;

        direct = TryInvokeMenuStartMethod(root);
        if (direct != null)
            return direct;

        return Error("Could not find a main menu / character select control to start an Ironclad run");
    }

    private static Dictionary<string, object?> ExecuteReturnToMenu(string action)
    {
        var root = ((SceneTree)Engine.GetMainLoop()).Root;

        var direct = TryInvokeMenuReturnMethod(root);
        if (direct != null)
            return direct;

        var clicked = TryClickReturnToMenuButton(root);
        if (clicked != null)
            return clicked;

        if (!RunManager.Instance.IsInProgress)
        {
            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = "Already at menu or no run in progress"
            };
        }

        return Error($"Could not execute {action}; no return-to-menu or abandon control was found");
    }

    private static Dictionary<string, object?> ExecuteContinueRun()
    {
        if (RunManager.Instance.IsInProgress)
        {
            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = "Run is already in progress"
            };
        }

        RL_DataCollector.SetDataSource(RL_DataCollector.DataSource.AI);
        RL_DataCollector.MarkMenuRunIntent("continue", null, RL_DataCollector.DataSource.AI);

        var root = ((SceneTree)Engine.GetMainLoop()).Root;
        var buttons = FindAll<NButton>(root)
            .Where(button => button.IsVisibleInTree() && button.IsEnabled)
            .Select(button => new { Button = button, Text = NodeSearchText(button).ToLowerInvariant() })
            .ToList();

        foreach (var row in buttons)
        {
            if (!ContainsAny(row.Text, "continue", "继续游戏"))
                continue;

            row.Button.ForceClick();
            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = $"Clicked continue button {row.Button.GetType().Name}: {row.Text}"
            };
        }

        return Error("Could not find an enabled continue button");
    }

    private static Dictionary<string, object?> ExecuteDebugMenuButtons()
    {
        var root = ((SceneTree)Engine.GetMainLoop()).Root;
        var buttons = FindAll<NButton>(root)
            .Select(button => new Dictionary<string, object?>
            {
                ["type"] = button.GetType().FullName ?? button.GetType().Name,
                ["name"] = button.Name.ToString(),
                ["visible"] = button.IsVisibleInTree(),
                ["enabled"] = button.IsEnabled,
                ["text"] = NodeSearchText(button),
            })
            .Cast<object?>()
            .ToList();
        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["count"] = buttons.Count,
            ["buttons"] = buttons,
        };
    }

    private static Dictionary<string, object?> ExecuteDebugCrystalSphere()
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NCrystalSphereScreen screen)
            return Error("Crystal Sphere screen is not open");

        var screenType = screen.GetType();
        var fields = screenType
            .GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static)
            .Select(field =>
            {
                object? value = null;
                string? valueType = null;
                string? valueText = null;
                try
                {
                    value = field.GetValue(field.IsStatic ? null : screen);
                    valueType = value?.GetType().FullName;
                    valueText = value?.ToString();
                }
                catch { }
                return new Dictionary<string, object?>
                {
                    ["name"] = field.Name,
                    ["type"] = field.FieldType.FullName,
                    ["value_type"] = valueType,
                    ["value"] = valueText
                };
            })
            .Cast<object?>()
            .ToList();
        var properties = screenType
            .GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static)
            .Select(prop => new Dictionary<string, object?>
            {
                ["name"] = prop.Name,
                ["type"] = prop.PropertyType.FullName,
                ["can_read"] = prop.CanRead,
                ["can_write"] = prop.CanWrite
            })
            .Cast<object?>()
            .ToList();
        var methods = screenType
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static)
            .Where(method => !method.IsSpecialName)
            .Select(method => new Dictionary<string, object?>
            {
                ["name"] = method.Name,
                ["return_type"] = method.ReturnType.FullName,
                ["params"] = method.GetParameters()
                    .Select(param => $"{param.ParameterType.FullName} {param.Name}")
                    .ToList()
            })
            .Cast<object?>()
            .ToList();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["screen_type"] = screenType.FullName,
            ["base_type"] = screenType.BaseType?.FullName,
            ["fields"] = fields,
            ["properties"] = properties,
            ["methods"] = methods
        };
    }

    private static Dictionary<string, object?>? TryHandleCharacterSelectStart(Node root, string? seed = null)
    {
        var characterButtons = FindAll<NCharacterSelectButton>(root)
            .Where(button => button.IsVisibleInTree() && button.IsEnabled)
            .ToList();

        if (characterButtons.Count == 0)
            return null;

        var ironcladButton = characterButtons.FirstOrDefault(button =>
        {
            var text = NodeSearchText(button).ToLowerInvariant();
            return ContainsAny(text, "ironclad", "铁甲", "铁甲战士");
        });

        if (ironcladButton != null)
        {
            ironcladButton.ForceClick();

            if (!string.IsNullOrWhiteSpace(seed))
            {
                var seeded = TryStartCharacterSelectWithSeed(root, seed);
                if (seeded != null)
                    return seeded;
                return Error("Could not set seed and embark on the character select screen");
            }

            var confirmButton = FindAll<NConfirmButton>(root)
                .FirstOrDefault(button => button.IsVisibleInTree() && button.IsEnabled);
            if (confirmButton != null)
            {
                confirmButton.ForceClick();
                return new Dictionary<string, object?>
                {
                    ["status"] = "ok",
                    ["message"] = $"Selected {NodeSearchText(ironcladButton)} and clicked confirm"
                };
            }

            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = $"Selected {NodeSearchText(ironcladButton)}"
            };
        }

        var confirmOnly = FindAll<NConfirmButton>(root)
            .FirstOrDefault(button => button.IsVisibleInTree() && button.IsEnabled);
        if (confirmOnly != null)
        {
            if (!string.IsNullOrWhiteSpace(seed))
            {
                var seeded = TryStartCharacterSelectWithSeed(root, seed);
                if (seeded != null)
                    return seeded;
                return Error("Could not set seed and embark on the character select screen");
            }

            confirmOnly.ForceClick();
            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = $"Clicked character select confirm button {confirmOnly.Name}"
            };
        }

        return null;
    }

    private static Dictionary<string, object?>? TryStartCharacterSelectWithSeed(Node root, string seed)
    {
        foreach (var screen in FindAll<NCharacterSelectScreen>(root))
        {
            if (!screen.IsVisibleInTree())
                continue;

            try
            {
                var lobby = screen.Lobby;
                var begin = lobby.GetType().GetMethod("BeginRunLocally", BindingFlags.NonPublic | BindingFlags.Instance);
                if (begin == null)
                    return Error("Could not find StartRunLobby.BeginRunLocally(seed, modifiers)");

                var parameters = begin.GetParameters();
                if (parameters.Length != 2 || parameters[0].ParameterType != typeof(string))
                    return Error("StartRunLobby.BeginRunLocally signature is not supported");

                var emptyModifiers = System.Activator.CreateInstance(parameters[1].ParameterType);
                begin.Invoke(lobby, new object?[] { seed, emptyModifiers });
                return new Dictionary<string, object?>
                {
                    ["status"] = "ok",
                    ["message"] = $"Invoked StartRunLobby.BeginRunLocally with seed {seed}"
                };
            }
            catch (System.Exception ex)
            {
                var message = ex is TargetInvocationException && ex.InnerException != null
                    ? ex.InnerException.Message
                    : ex.Message;
                return Error($"Failed to start seeded character select run: {message}");
            }
        }

        return null;
    }

    private static Dictionary<string, object?>? TryInvokeMenuStartMethod(Node root, string? seed = null)
    {
        foreach (var node in FindAll<Node>(root))
        {
            string typeName = node.GetType().FullName ?? "";
            if (!typeName.Contains("MainMenu") && !typeName.Contains("Singleplayer") && !typeName.Contains("CharacterSelect"))
                continue;

            foreach (var methodName in string.IsNullOrWhiteSpace(seed)
                     ? new[]
                     {
                         "StartNewSingleplayerRun",
                         "StartNewRun",
                         "BeginRun",
                         "OnEmbarkPressed",
                         "OnEmbarkButtonPressed",
                         "OnNewRunButtonPressed",
                         "OnSingleplayerButtonPressed",
                     }
                     : new[]
                     {
                         "BeginRun",
                         "StartNewRun",
                         "StartNewSingleplayerRun",
                         "OnNewRunButtonPressed",
                         "OnEmbarkPressed",
                         "OnEmbarkButtonPressed",
                         "OnSingleplayerButtonPressed",
                     })
            {
                var result = TryInvokeMenuMethod(node, methodName, seed);
                if (result)
                {
                    return new Dictionary<string, object?>
                    {
                        ["status"] = "ok",
                        ["message"] = $"Invoked {node.GetType().Name}.{methodName}()"
                    };
                }
            }
        }
        return null;
    }

    private static Dictionary<string, object?>? TryClickMenuStartButton(Node root)
    {
        var buttons = FindAll<NButton>(root)
            .Where(button => button.IsVisibleInTree() && button.IsEnabled)
            .Select(button => new { Button = button, Text = NodeSearchText(button).ToLowerInvariant() })
            .ToList();

        foreach (var priority in new[]
                 {
                     new[] { "embark", "begin", "启程", "出发", "开始游戏", "开始冒险" },
                     new[] { "ironclad", "铁甲", "铁甲战士" },
                     new[] { "singleplayer", "single player", "单人", "单人模式" },
                     new[] { "new run", "new game", "start", "开始", "新游戏" },
                 })
        {
            foreach (var row in buttons)
            {
                if (!ContainsAny(row.Text, priority))
                    continue;

                row.Button.ForceClick();
                return new Dictionary<string, object?>
                {
                    ["status"] = "ok",
                    ["message"] = $"Clicked menu button {row.Button.GetType().Name}: {row.Text}"
                };
            }
        }
        return null;
    }

    private static Dictionary<string, object?>? TryInvokeMenuReturnMethod(Node root)
    {
        foreach (var node in FindAll<Node>(root))
        {
            string typeName = node.GetType().FullName ?? "";
            if (!typeName.Contains("GameOver") && !typeName.Contains("MainMenu") && !typeName.Contains("Run") && !typeName.Contains("Menu"))
                continue;

            foreach (var methodName in new[]
                     {
                         "ReturnToMenu",
                         "OnReturnToMenuPressed",
                         "OnMainMenuButtonPressed",
                         "OnAbandonRunPressed",
                         "AbandonRun",
                         "OnQuitToMenuPressed",
                     })
            {
                var result = TryInvokeMenuMethod(node, methodName);
                if (result)
                {
                    return new Dictionary<string, object?>
                    {
                        ["status"] = "ok",
                        ["message"] = $"Invoked {node.GetType().Name}.{methodName}()"
                    };
                }
            }
        }
        return null;
    }

    private static Dictionary<string, object?>? TryClickReturnToMenuButton(Node root)
    {
        foreach (var button in FindAll<NButton>(root))
        {
            if (!button.IsVisibleInTree() || !button.IsEnabled)
                continue;

            string text = NodeSearchText(button).ToLowerInvariant();
            if (!ContainsAny(text, "return", "main menu", "abandon", "quit", "leave", "menu", "返回", "主菜单", "放弃", "退出", "离开"))
                continue;

            button.ForceClick();
            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = $"Clicked menu/abandon button {button.GetType().Name}"
            };
        }
        return null;
    }

    private static bool TryInvokeMenuMethod(object target, string methodName, string? seed = null)
    {
        try
        {
            var methods = target.GetType()
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static)
                .Where(method => method.Name == methodName);
            if (!string.IsNullOrWhiteSpace(seed))
            {
                methods = methods
                    .OrderByDescending(method => method.GetParameters().Length == 1
                        && method.GetParameters()[0].ParameterType == typeof(string))
                    .ThenBy(method => method.GetParameters().Length);
            }
            else
            {
                methods = methods.OrderBy(method => method.GetParameters().Length);
            }

            foreach (var method in methods)
            {
                var parameters = method.GetParameters();
                object?[] args;
                if (parameters.Length == 0)
                {
                    if (!string.IsNullOrWhiteSpace(seed))
                        continue;
                    args = System.Array.Empty<object?>();
                }
                else if (parameters.Length == 1 && parameters[0].ParameterType == typeof(string))
                {
                    args = new object?[] { seed ?? "" };
                }
                else if (parameters.Length == 1 && !parameters[0].ParameterType.IsValueType)
                {
                    if (!string.IsNullOrWhiteSpace(seed))
                        continue;
                    args = new object?[] { null };
                }
                else
                {
                    continue;
                }

                method.Invoke(method.IsStatic ? null : target, args);
                return true;
            }
        }
        catch
        {
        }
        return false;
    }

    private static bool ContainsAny(string haystack, params string[] needles)
    {
        foreach (var needle in needles)
        {
            if (haystack.Contains(needle))
                return true;
        }
        return false;
    }

    private static string NodeSearchText(Node node)
    {
        var parts = new List<string>
        {
            node.Name.ToString(),
            node.GetType().Name
        };
        foreach (var propName in new[] { "Text", "Title", "Label", "Description" })
        {
            try
            {
                var prop = node.GetType().GetProperty(propName, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                var value = prop?.GetValue(node);
                if (value != null)
                    parts.Add(SafeGetText(() => value) ?? value.ToString() ?? "");
            }
            catch { }
        }
        foreach (var child in node.GetChildren())
        {
            if (child is Label label)
                parts.Add(label.Text);
            else if (child is RichTextLabel rich)
                parts.Add(rich.Text);
            else if (child is Button button)
                parts.Add(button.Text);
        }
        return string.Join(" ", parts.Where(part => !string.IsNullOrWhiteSpace(part)));
    }

    private static Dictionary<string, object?> ExecutePlayCard(Player player, Dictionary<string, JsonElement> data)
    {
        if (!CombatManager.Instance.IsInProgress)
            return Error("Not in combat");
        if (!CombatManager.Instance.IsPlayPhase)
            return Error("Not in play phase - cannot act during enemy turn");
        if (CombatManager.Instance.PlayerActionsDisabled)
            return Error("Player actions are currently disabled");
        if (!player.Creature.IsAlive)
            return Error("Player creature is dead - cannot play cards");

        var combatState = player.Creature.CombatState;
        if (combatState == null)
            return Error("No combat state");

        // Get card by index in hand
        if (!data.TryGetValue("card_index", out var indexElem))
            return Error("Missing 'card_index'");

        int cardIndex = indexElem.GetInt32();
        var hand = player.PlayerCombatState?.Hand;
        if (hand == null)
            return Error("No hand available");

        if (cardIndex < 0 || cardIndex >= hand.Cards.Count)
            return Error($"card_index {cardIndex} out of range (hand has {hand.Cards.Count} cards)");

        var card = hand.Cards[cardIndex];

        if (!card.CanPlay(out var reason, out _))
            return Error($"Card '{card.Title}' cannot be played: {reason}");

        // Resolve target
        Creature? target = null;
        if (card.TargetType == TargetType.AnyEnemy)
        {
            if (!data.TryGetValue("target", out var targetElem))
                return Error("Card requires a target. Provide 'target' with an entity_id.");

            string targetId = targetElem.GetString() ?? "";
            target = ResolveTarget(combatState, targetId);
            if (target == null)
                return Error($"Target '{targetId}' not found among alive enemies");
        }

        // Play the card via the action queue (same path as the game UI)
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new PlayCardAction(card, target));

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Playing '{card.Title}'" + (target != null ? $" targeting {SafeGetText(() => target.Monster?.Title) ?? "target"}" : "")
        };
    }

    private static Dictionary<string, object?> ExecuteEndTurn(Player player)
    {
        if (!CombatManager.Instance.IsInProgress)
            return Error("Not in combat");
        if (!CombatManager.Instance.IsPlayPhase)
            return Error("Not in play phase - cannot act during enemy turn");
        if (CombatManager.Instance.PlayerActionsDisabled)
            return Error("Player actions are currently disabled (turn may already be ending)");

        // Match the game's own CanTurnBeEnded guard (NEndTurnButton.cs:114-123)
        var hand = NCombatRoom.Instance?.Ui?.Hand;
        if (hand != null && (hand.InCardPlay || hand.CurrentMode != NPlayerHand.Mode.Play))
            return Error("Cannot end turn while a card is being played or hand is in selection mode");

        PlayerCmd.EndTurn(player, canBackOut: false);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Ending turn"
        };
    }

    private static Dictionary<string, object?> ExecuteUsePotion(Player player, Dictionary<string, JsonElement> data)
    {
        if (!data.TryGetValue("slot", out var slotElem))
            return Error("Missing 'slot' (potion slot index)");

        int slot = slotElem.GetInt32();
        if (slot < 0 || slot >= player.PotionSlots.Count)
            return Error($"Potion slot {slot} out of range (player has {player.PotionSlots.Count} slots)");

        var potion = player.GetPotionAtSlotIndex(slot);
        if (potion == null)
            return Error($"No potion in slot {slot}");
        if (potion.IsQueued)
            return Error($"Potion '{SafeGetText(() => potion.Title)}' is already queued for use");
        if (potion.Owner.Creature.IsDead)
            return Error("Cannot use potion - player creature is dead");
        if (!potion.PassesCustomUsabilityCheck)
            return Error($"Potion '{SafeGetText(() => potion.Title)}' cannot be used right now");

        bool inCombat = CombatManager.Instance.IsInProgress;
        if (potion.Usage == PotionUsage.CombatOnly)
        {
            if (!inCombat)
                return Error($"Potion '{SafeGetText(() => potion.Title)}' can only be used in combat");
            if (!CombatManager.Instance.IsPlayPhase)
                return Error("Cannot use potions outside of play phase");
        }
        else if (potion.Usage == PotionUsage.Automatic)
            return Error($"Potion '{SafeGetText(() => potion.Title)}' is automatic and cannot be manually used");

        if (inCombat && CombatManager.Instance.PlayerActionsDisabled)
            return Error("Player actions are currently disabled");

        // Resolve target
        Creature? target = null;
        var combatState = player.Creature.CombatState;

        switch (potion.TargetType)
        {
            case TargetType.AnyEnemy:
                if (!data.TryGetValue("target", out var targetElem))
                    return Error("Potion requires a target enemy. Provide 'target' with an entity_id.");
                string targetId = targetElem.GetString() ?? "";
                if (combatState == null)
                    return Error("No combat state for target resolution");
                target = ResolveTarget(combatState, targetId);
                if (target == null)
                    return Error($"Target '{targetId}' not found among alive enemies");
                break;
            case TargetType.Self:
            case TargetType.AnyAlly:
            case TargetType.AnyPlayer:
                target = player.Creature;
                break;
            default:
                target = null;
                break;
        }

        potion.EnqueueManualUse(target);

        string targetMsg = potion.TargetType switch
        {
            TargetType.AnyEnemy => $" targeting {SafeGetText(() => target?.Monster?.Title) ?? "enemy"}",
            TargetType.Self or TargetType.AnyPlayer or TargetType.AnyAlly => " on self",
            _ => ""
        };

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Using potion '{SafeGetText(() => potion.Title)}' from slot {slot}{targetMsg}"
        };
    }

    private static Dictionary<string, object?> ExecuteDiscardPotion(Player player, Dictionary<string, JsonElement> data)
    {
        if (!data.TryGetValue("slot", out var slotElem))
            return Error("Missing 'slot' (potion slot index)");

        int slot = slotElem.GetInt32();
        if (slot < 0 || slot >= player.PotionSlots.Count)
            return Error($"Potion slot {slot} out of range (player has {player.PotionSlots.Count} slots)");

        var potion = player.GetPotionAtSlotIndex(slot);
        if (potion == null)
            return Error($"No potion in slot {slot}");

        string potionName = SafeGetText(() => potion.Title) ?? "unknown";
        _ = PotionCmd.Discard(potion);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Discarded potion '{potionName}' from slot {slot}"
        };
    }

    private static Dictionary<string, object?> ExecuteChooseEventOption(Dictionary<string, JsonElement> data)
    {
        var uiRoom = NEventRoom.Instance;
        if (uiRoom == null)
            return Error("Event room is not open");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (event option index)");

        int index = indexElem.GetInt32();

        var buttons = FindAll<NEventOptionButton>(uiRoom);

        if (buttons.Count == 0)
            return Error("No event options available");
        if (index < 0 || index >= buttons.Count)
            return Error($"Event option index {index} out of range ({buttons.Count} options)");

        var button = buttons[index];
        if (button.Option.IsLocked)
            return Error($"Event option {index} is locked");
        string title = SafeGetText(() => button.Option.Title) ?? "option";
            RL_DataCollector.RecordAction("choose_event_option", new Dictionary<string, object> {
                ["action"] = "choose_event_option",
                ["title"]  = title
            });
        button.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Choosing event option: {title}"
        };
    }

    private static Dictionary<string, object?> ExecuteAdvanceDialogue()
    {
        var uiRoom = NEventRoom.Instance;
        if (uiRoom == null)
            return Error("Event room is not open");

        var ancientLayout = FindFirst<NAncientEventLayout>(uiRoom);
        if (ancientLayout == null)
            return Error("No ancient dialogue active");

        var hitbox = ancientLayout.GetNodeOrNull<NClickableControl>("%DialogueHitbox");
        if (hitbox == null || !hitbox.Visible || !hitbox.IsEnabled)
            return Error("Dialogue hitbox not available - dialogue may have ended");

        hitbox.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Advancing dialogue"
        };
    }

    private static Dictionary<string, object?> ExecuteChooseRestOption(Dictionary<string, JsonElement> data)
    {
        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (rest site option index)");

        int index = indexElem.GetInt32();

        var restRoom = NRestSiteRoom.Instance;
        if (restRoom == null)
            return Error("Rest site room is not open");

        var buttons = FindAll<NRestSiteButton>(restRoom);

        if (buttons.Count == 0)
            return Error("No rest site options available");
        if (index < 0 || index >= buttons.Count)
            return Error($"Rest option index {index} out of range ({buttons.Count} options)");

        var button = buttons[index];
        if (!button.Option.IsEnabled)
            return Error($"Rest option {index} ({button.Option.OptionId}) is disabled");
        string optionName = SafeGetText(() => button.Option.Title) ?? button.Option.OptionId;
            RL_DataCollector.RecordAction("choose_rest_option", new Dictionary<string, object> {
                ["action"]    = "choose_rest_option",
                ["option_id"] = optionName
            });
        button.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Selecting rest site option: {optionName}"
        };
    }

    private static Dictionary<string, object?> ExecuteShopPurchase(Player player, Dictionary<string, JsonElement> data)
    {
        MerchantInventory? inventory = null;

        if (player.RunState.CurrentRoom is MerchantRoom merchantRoom)
        {
            // Regular merchant - auto-open inventory if needed
            var merchUI = NMerchantRoom.Instance;
            if (merchUI?.Inventory != null && !merchUI.Inventory.IsOpen)
                merchUI.OpenInventory();
            inventory = merchantRoom.Inventory;
        }
        else if (player.RunState.CurrentRoom is EventRoom eventRoom
                 && eventRoom.CanonicalEvent is FakeMerchant
                 && (eventRoom.LocalMutableEvent ?? eventRoom.CanonicalEvent) is FakeMerchant fakeMerchant)
        {
            // Fake merchant event - auto-open via button if needed
            if (!fakeMerchant.StartedFight)
            {
                var uiRoom = NEventRoom.Instance;
                if (uiRoom != null)
                {
                    var fmNode = FindFirst<NFakeMerchant>(uiRoom);
                    if (fmNode != null)
                    {
                        var inventoryUI = FindFirst<NMerchantInventory>(fmNode);
                        if (inventoryUI != null && !inventoryUI.IsOpen)
                        {
                            var btn = fmNode.MerchantButton;
                            if (btn != null && btn.Visible && btn.IsEnabled)
                                btn.ForceClick();
                        }
                    }
                }
            }
            inventory = fakeMerchant.Inventory;
        }
        else
        {
            return Error("Not in a shop");
        }

        if (inventory == null)
            return Error("Shop inventory not ready yet; wait a moment and retry");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (shop item index)");

        int index = indexElem.GetInt32();

        var allEntries = inventory.AllEntries.ToList();
        if (index < 0 || index >= allEntries.Count)
            return Error($"Shop item index {index} out of range ({allEntries.Count} items)");

        var entry = allEntries[index];
        if (!entry.IsStocked)
            return Error("Item is sold out");
        if (!entry.EnoughGold)
            return Error($"Not enough gold (need {entry.Cost}, have {player.Gold})");

        // 记录购买决策
        string itemName = $"item_cost{entry.Cost}";
        try { dynamic de = entry; itemName = de.Description ?? itemName; } catch { }
        RL_DataCollector.RecordAction("buy_item", new Dictionary<string, object> {
            ["action"]  = "buy_item",
            ["item_id"] = itemName,
            ["cost"]    = entry.Cost
        });
        // Fire-and-forget purchase (same path as AutoSlay)
        _ = entry.OnTryPurchaseWrapper(inventory);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Purchasing item for {entry.Cost} gold"
        };
    }

    private static Dictionary<string, object?> ExecuteChooseMapNode(Dictionary<string, JsonElement> data)
    {
        var mapScreen = NMapScreen.Instance;
        if (mapScreen == null || !mapScreen.IsOpen)
            return Error("Map screen is not open");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (map node index from next_options)");

        int index = indexElem.GetInt32();

        var travelable = FindAll<NMapPoint>(mapScreen)
            .Where(mp => mp.State == MapPointState.Travelable && mp.Point != null)
            .OrderBy(mp => mp.Point!.coord.col)
            .ToList();

        if (travelable.Count == 0)
            return Error("No travelable map nodes available");
        if (index < 0 || index >= travelable.Count)
            return Error($"Map node index {index} out of range ({travelable.Count} options available)");

        var target = travelable[index];
        var pt = target.Point!;

        mapScreen.OnMapPointSelectedLocally(target);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Traveling to {pt.PointType} at ({pt.coord.col},{pt.coord.row})"
        };
    }

    private static Dictionary<string, object?> ExecuteClaimReward(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NRewardsScreen rewardsScreen)
            return Error("Rewards screen is not open");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (reward index)");

        int index = indexElem.GetInt32();

        var enabledButtons = FindAll<NRewardButton>(rewardsScreen)
            .Where(b => b.IsEnabled && b.Reward != null)
            .ToList();

        if (index < 0 || index >= enabledButtons.Count)
            return Error($"Reward index {index} out of range (screen has {enabledButtons.Count} claimable rewards)");

        var button = enabledButtons[index];
        var reward = button.Reward!;
        string rewardDesc = GetRewardTypeName(reward);
        if (reward is GoldReward g)
            rewardDesc = $"gold ({g.Amount})";
        else if (reward is PotionReward p)
            rewardDesc = $"potion ({SafeGetText(() => p.Potion?.Title)})";
        else if (reward is CardReward)
            rewardDesc = "card (opens card selection)";

        RL_DataCollector.RecordAction("claim_reward", new Dictionary<string, object> {
            ["action"]      = "claim_reward",
            ["reward_type"] = rewardDesc
        });
        button.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Claiming reward: {rewardDesc}"
        };
    }

    private static Dictionary<string, object?> ExecuteSelectCardReward(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NCardRewardSelectionScreen cardScreen)
            return Error("Card reward selection screen is not open");

        if (!data.TryGetValue("card_index", out var indexElem))
            return Error("Missing 'card_index'");

        int cardIndex = indexElem.GetInt32();

        var cardHolders = FindAllSortedByPosition<NCardHolder>(cardScreen);
        if (cardIndex < 0 || cardIndex >= cardHolders.Count)
            return Error($"Card index {cardIndex} out of range (screen has {cardHolders.Count} cards)");

        var holder = cardHolders[cardIndex];
        string cardName = SafeGetText(() => holder.CardModel?.Title) ?? "unknown";
        RL_DataCollector.RecordAction("choose_card", new Dictionary<string, object> {
            ["action"]     = "choose_card",
            ["card_title"] = cardName
        });
        holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Selecting card: {cardName}"
        };
    }

    private static Dictionary<string, object?> ExecuteSkipCardReward()
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NCardRewardSelectionScreen cardScreen)
            return Error("Card reward selection screen is not open");

        var altButtons = FindAll<NCardRewardAlternativeButton>(cardScreen);
        if (altButtons.Count == 0)
            return Error("No skip option available on this card reward");

        RL_DataCollector.RecordAction("skip_card_reward", new Dictionary<string, object> {
            ["action"] = "skip_card_reward"
        });
        altButtons[0].ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Skipping card reward"
        };
    }

    private static Dictionary<string, object?> ExecuteProceed()
    {
        // Try rewards overlay
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is NRewardsScreen rewardsScreen)
        {
            var btn = FindFirst<NProceedButton>(rewardsScreen);
            if (btn is { IsEnabled: true })
            {
                btn.ForceClick();
                return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Proceeding from rewards" };
            }
        }

        // Try rest site
        if (NRestSiteRoom.Instance is { } restRoom && restRoom.ProceedButton.IsEnabled)
        {
            restRoom.ProceedButton.ForceClick();
            return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Proceeding from rest site" };
        }

        // Try merchant - close inventory first if open, then proceed
        if (NMerchantRoom.Instance is { } merchRoom)
        {
            if (merchRoom.Inventory?.IsOpen == true)
            {
                var backBtn = FindFirst<NBackButton>(merchRoom);
                if (backBtn is { IsEnabled: true })
                {
                    backBtn.ForceClick();
                    return new Dictionary<string, object?>
                    {
                        ["status"] = "ok",
                        ["message"] = "Closing shop inventory before proceeding"
                    };
                }
            }
            if (merchRoom.ProceedButton.IsEnabled)
            {
                merchRoom.ProceedButton.ForceClick();
                return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Proceeding from shop" };
            }
        }

        // Try fake merchant - close inventory first if open, then proceed
        if (NEventRoom.Instance is { } evtRoom)
        {
            var fmNode = FindFirst<NFakeMerchant>(evtRoom);
            if (fmNode != null)
            {
                var fmInventory = FindFirst<NMerchantInventory>(fmNode);
                if (fmInventory is { IsOpen: true })
                {
                    var backBtn = FindFirst<NBackButton>(fmNode);
                    if (backBtn is { IsEnabled: true })
                    {
                        backBtn.ForceClick();
                        return new Dictionary<string, object?>
                        {
                            ["status"] = "ok",
                            ["message"] = "Closing fake merchant inventory before proceeding"
                        };
                    }
                }
                var proceedBtn = FindFirst<NProceedButton>(fmNode);
                if (proceedBtn is { IsEnabled: true })
                {
                    proceedBtn.ForceClick();
                    return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Proceeding from fake merchant" };
                }
            }
        }

        // Try treasure room
        var treasureUI = FindFirst<NTreasureRoom>(
            ((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root);
        if (treasureUI != null && treasureUI.ProceedButton.IsEnabled)
        {
            treasureUI.ProceedButton.ForceClick();
            return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Proceeding from treasure room" };
        }

        return Error("No proceed button available or enabled");
    }

    private static Dictionary<string, object?> ExecuteSelectCard(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (card index in the grid)");

        int index = indexElem.GetInt32();

        if (overlay is NCardGridSelectionScreen gridScreen)
        {
            var grid = FindFirst<NCardGrid>(gridScreen);
            if (grid == null)
                return Error("Card grid not found in selection screen");

            var holders = FindAllSortedByPosition<NGridCardHolder>(gridScreen);
            if (index < 0 || index >= holders.Count)
                return Error($"Card index {index} out of range ({holders.Count} cards available)");

            var holder = holders[index];
            string cardName = SafeGetText(() => holder.CardModel?.Title) ?? "unknown";
            grid.EmitSignal(NCardGrid.SignalName.HolderPressed, holder);

            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = $"Toggling card selection: {cardName}"
            };
        }
        else if (overlay is NChooseACardSelectionScreen chooseScreen)
        {
            var holders = FindAllSortedByPosition<NGridCardHolder>(chooseScreen);
            if (index < 0 || index >= holders.Count)
                return Error($"Card index {index} out of range ({holders.Count} cards available)");

            var holder = holders[index];
            string cardName = SafeGetText(() => holder.CardModel?.Title) ?? "unknown";
            holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);

            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = $"Choosing card: {cardName}"
            };
        }

        return Error("No card selection screen is open");
    }

    private static Dictionary<string, object?> ExecuteConfirmSelection()
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is NChooseACardSelectionScreen)
            return Error("Choose-a-card screen requires no confirmation - use select_card(index) to pick directly");
        if (overlay is not NCardGridSelectionScreen screen)
            return Error("No card selection screen is open");

        // Check all preview containers (upgrade uses UpgradeSinglePreviewContainer / UpgradeMultiPreviewContainer,
        // NDeckCardSelectScreen uses PreviewContainer with %PreviewConfirm)
        foreach (var containerName in new[] { "%UpgradeSinglePreviewContainer", "%UpgradeMultiPreviewContainer", "%PreviewContainer" })
        {
            var container = screen.GetNodeOrNull<Godot.Control>(containerName);
            if (container?.Visible == true)
            {
                var confirm = container.GetNodeOrNull<NConfirmButton>("Confirm")
                              ?? container.GetNodeOrNull<NConfirmButton>("%PreviewConfirm");
                if (confirm is { IsEnabled: true })
                {
                    confirm.ForceClick();
                    return new Dictionary<string, object?>
                    {
                        ["status"] = "ok",
                        ["message"] = "Confirming selection from preview"
                    };
                }
            }
        }

        // Try main confirm button
        var mainConfirm = screen.GetNodeOrNull<NConfirmButton>("Confirm")
                          ?? screen.GetNodeOrNull<NConfirmButton>("%Confirm");
        if (mainConfirm is { IsEnabled: true })
        {
            mainConfirm.ForceClick();
            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = "Confirming selection"
            };
        }

        // Fallback: find ANY enabled NConfirmButton in the screen tree.
        // Covers NCardGridSelectionScreen subclasses (like NDeckEnchantSelectScreen)
        // whose confirm button isn't in any of the known container paths above.
        var allConfirmButtons = FindAll<NConfirmButton>(screen);
        foreach (var btn in allConfirmButtons)
        {
            if (btn.IsEnabled && btn.IsVisibleInTree())
            {
                btn.ForceClick();
                return new Dictionary<string, object?>
                {
                    ["status"] = "ok",
                    ["message"] = "Confirming selection"
                };
            }
        }

        return Error("No confirm button is currently enabled - select more cards first");
    }

    private static Dictionary<string, object?> ExecuteCancelSelection()
    {
        var overlay = NOverlayStack.Instance?.Peek();

        // Handle choose-a-card screen (skip button)
        if (overlay is NChooseACardSelectionScreen chooseScreen)
        {
            var skipButton = chooseScreen.GetNodeOrNull<NClickableControl>("SkipButton");
            if (skipButton is { IsEnabled: true })
            {
                skipButton.ForceClick();
                return new Dictionary<string, object?>
                {
                    ["status"] = "ok",
                    ["message"] = "Skipping card choice"
                };
            }
            return Error("No skip option available - a card must be chosen");
        }

        if (overlay is not NCardGridSelectionScreen screen)
            return Error("No card selection screen is open");

        // If preview is showing, cancel back to selection
        foreach (var containerName in new[] { "%UpgradeSinglePreviewContainer", "%UpgradeMultiPreviewContainer", "%PreviewContainer" })
        {
            var container = screen.GetNodeOrNull<Godot.Control>(containerName);
            if (container?.Visible == true)
            {
                var cancelBtn = container.GetNodeOrNull<NBackButton>("Cancel")
                                ?? container.GetNodeOrNull<NBackButton>("%PreviewCancel");
                if (cancelBtn is { IsEnabled: true })
                {
                    cancelBtn.ForceClick();
                    return new Dictionary<string, object?>
                    {
                        ["status"] = "ok",
                        ["message"] = "Cancelling preview - returning to card selection"
                    };
                }
            }
        }

        // Close the screen entirely
        var closeButton = screen.GetNodeOrNull<NBackButton>("%Close");
        if (closeButton is { IsEnabled: true })
        {
            closeButton.ForceClick();
            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = "Closing card selection screen"
            };
        }

        return Error("No cancel/close button is currently enabled - selection may be mandatory");
    }

    private static Dictionary<string, object?> ExecuteSelectBundle(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NChooseABundleSelectionScreen screen)
            return Error("No bundle selection screen is open");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (bundle index)");

        int index = indexElem.GetInt32();
        var previewContainer = screen.GetNodeOrNull<Godot.Control>("%BundlePreviewContainer");
        if (previewContainer?.Visible == true)
            return Error("A bundle preview is already open - confirm or cancel it first");

        var bundles = FindAll<NCardBundle>(screen);
        if (index < 0 || index >= bundles.Count)
            return Error($"Bundle index {index} out of range ({bundles.Count} bundles available)");

        bundles[index].Hitbox.ForceClick();
        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Selecting bundle {index}"
        };
    }

    private static Dictionary<string, object?> ExecuteConfirmBundleSelection()
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NChooseABundleSelectionScreen screen)
            return Error("No bundle selection screen is open");

        var confirmButton = screen.GetNodeOrNull<NConfirmButton>("%Confirm");
        if (confirmButton is not { IsEnabled: true })
            return Error("Bundle confirm button is not enabled");

        confirmButton.ForceClick();
        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Confirming bundle selection"
        };
    }

    private static Dictionary<string, object?> ExecuteCancelBundleSelection()
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NChooseABundleSelectionScreen screen)
            return Error("No bundle selection screen is open");

        var cancelButton = screen.GetNodeOrNull<NBackButton>("%Cancel");
        if (cancelButton is not { IsEnabled: true })
            return Error("Bundle cancel button is not enabled");

        cancelButton.ForceClick();
        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Cancelling bundle selection"
        };
    }

    private static Dictionary<string, object?> ExecuteCombatSelectCard(Dictionary<string, JsonElement> data)
    {
        var hand = NPlayerHand.Instance;
        if (hand == null || !hand.IsInCardSelection)
            return Error("No in-combat card selection is active");

        if (!data.TryGetValue("card_index", out var indexElem))
            return Error("Missing 'card_index' (index of the card in hand)");

        int index = indexElem.GetInt32();
        var holders = hand.ActiveHolders;
        if (index < 0 || index >= holders.Count)
            return Error($"Card index {index} out of range ({holders.Count} selectable cards)");

        var holder = holders[index];
        string cardName = SafeGetText(() => holder.CardModel?.Title) ?? "unknown";

        // Emit the Pressed signal - same path the game UI uses
        holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Selecting card from hand: {cardName}"
        };
    }

    private static Dictionary<string, object?> ExecuteCombatConfirmSelection()
    {
        var hand = NPlayerHand.Instance;
        if (hand == null || !hand.IsInCardSelection)
            return Error("No in-combat card selection is active");

        var confirmBtn = hand.GetNodeOrNull<NConfirmButton>("%SelectModeConfirmButton");
        if (confirmBtn == null || !confirmBtn.IsEnabled)
            return Error("Confirm button is not enabled - select more cards first");

        confirmBtn.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Confirming combat card selection"
        };
    }

    private static Dictionary<string, object?> ExecuteSelectRelic(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NChooseARelicSelection screen)
            return Error("No relic selection screen is open");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (relic index)");

        int index = indexElem.GetInt32();

        var holders = FindAll<NRelicBasicHolder>(screen);
        if (index < 0 || index >= holders.Count)
            return Error($"Relic index {index} out of range ({holders.Count} relics available)");

        var holder = holders[index];
        string relicName = SafeGetText(() => holder.Relic?.Model?.Title) ?? "unknown";
        holder.ForceClick();

        RL_DataCollector.RecordAction("select_relic", new Dictionary<string, object>
        {
            ["action"] = "select_relic",
            ["relic_name"] = relicName,
            ["screen"] = screen.GetType().Name
        });

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Selecting relic: {relicName}"
        };
    }

    private static Dictionary<string, object?> ExecuteSkipRelicSelection()
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NChooseARelicSelection screen)
            return Error("No relic selection screen is open");

        var skipButton = screen.GetNodeOrNull<NClickableControl>("SkipButton");
        if (skipButton is not { IsEnabled: true })
            return Error("No skip option available");

        skipButton.ForceClick();

        RL_DataCollector.RecordAction("skip_relic_selection", new Dictionary<string, object>
        {
            ["action"] = "skip_relic_selection",
            ["screen"] = screen.GetType().Name
        });

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Skipping relic selection"
        };
    }

    private static Dictionary<string, object?> ExecuteClaimTreasureRelic(Dictionary<string, JsonElement> data)
    {
        var treasureUI = FindFirst<NTreasureRoom>(
            ((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root);
        if (treasureUI == null)
            return Error("Treasure room is not open");

        var relicCollection = treasureUI.GetNodeOrNull<NTreasureRoomRelicCollection>("%RelicCollection");
        if (relicCollection?.Visible != true)
            return Error("Relic collection is not visible - chest may not be opened yet");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (relic index)");

        int index = indexElem.GetInt32();

        var holders = FindAll<NTreasureRoomRelicHolder>(relicCollection)
            .Where(h => h.IsEnabled && h.Visible)
            .ToList();

        if (index < 0 || index >= holders.Count)
            return Error($"Relic index {index} out of range ({holders.Count} relics available)");

        var holder = holders[index];
        string relicName = SafeGetText(() => holder.Relic?.Model?.Title) ?? "unknown";
        holder.ForceClick();

        RL_DataCollector.RecordAction("claim_treasure_relic", new Dictionary<string, object>
        {
            ["action"] = "claim_treasure_relic",
            ["relic_name"] = relicName
        });

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Claiming treasure relic: {relicName}"
        };
    }

    private static Dictionary<string, object?> ExecuteCrystalSphereSetTool(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NCrystalSphereScreen screen)
            return Error("Crystal Sphere screen is not open");

        if (!data.TryGetValue("tool", out var toolElem))
            return Error("Missing 'tool' (expected 'big' or 'small')");

        string tool = toolElem.GetString() ?? "";
        var button = tool switch
        {
            "big" => screen.GetNodeOrNull<NClickableControl>("%BigDivinationButton"),
            "small" => screen.GetNodeOrNull<NClickableControl>("%SmallDivinationButton"),
            _ => null
        };

        if (button == null)
            return Error($"Unknown Crystal Sphere tool: {tool}");
        if (!button.Visible || !button.IsEnabled)
            return Error($"Crystal Sphere tool '{tool}' is not available");

        button.ForceClick();
        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Setting Crystal Sphere tool to {tool}"
        };
    }

    private static Dictionary<string, object?> ExecuteCrystalSphereClickCell(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NCrystalSphereScreen screen)
            return Error("Crystal Sphere screen is not open");

        if (!data.TryGetValue("x", out var xElem))
            return Error("Missing 'x' (cell x-coordinate)");
        if (!data.TryGetValue("y", out var yElem))
            return Error("Missing 'y' (cell y-coordinate)");

        int x = xElem.GetInt32();
        int y = yElem.GetInt32();

        var cell = FindAll<NCrystalSphereCell>(screen)
            .FirstOrDefault(c => c.Entity.X == x && c.Entity.Y == y);
        if (cell == null)
            return Error($"Crystal Sphere cell ({x}, {y}) was not found");
        if (!cell.Entity.IsHidden || !cell.Visible)
            return Error($"Crystal Sphere cell ({x}, {y}) is not clickable");

        cell.ForceClick();
        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Clicking Crystal Sphere cell ({x}, {y})"
        };
    }

    private static Dictionary<string, object?> ExecuteCrystalSphereProceed()
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NCrystalSphereScreen screen)
            return Error("Crystal Sphere screen is not open");

        var proceedButton = screen.GetNodeOrNull<NProceedButton>("%ProceedButton");
        proceedButton ??= screen.GetType()
            .GetField("_proceedButton", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            ?.GetValue(screen) as NProceedButton;
        var minigameFinishedMethod = screen.GetType().GetMethod(
            "OnMinigameFinished",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        if (minigameFinishedMethod != null)
        {
            minigameFinishedMethod.Invoke(screen, System.Array.Empty<object?>());
        }

        var entity = screen.GetType()
            .GetField("_entity", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            ?.GetValue(screen);
        var completeMethod = entity?.GetType().GetMethod(
            "CompleteMinigame",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        if (completeMethod != null && completeMethod.GetParameters().Length == 0)
        {
            completeMethod.Invoke(entity, System.Array.Empty<object?>());
        }

        var directMethod = screen.GetType().GetMethod(
            "OnProceedButtonPressed",
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        if (directMethod != null && proceedButton != null)
        {
            directMethod.Invoke(screen, new object?[] { proceedButton });
            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = "Proceeding from Crystal Sphere via OnProceedButtonPressed"
            };
        }

        if (proceedButton is { IsEnabled: true })
        {
            proceedButton.ForceClick();
            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = "Proceeding from Crystal Sphere"
            };
        }

        var eventProceedButton = NEventRoom.Instance == null
            ? null
            : FindAll<NEventOptionButton>(NEventRoom.Instance)
                .FirstOrDefault(button =>
                    button.IsEnabled
                    && button.IsVisibleInTree()
                    && !button.Option.IsLocked
                    && (button.Option.IsProceed
                        || ContainsAny(NodeSearchText(button).ToLowerInvariant(), "continue", "继续")));
        if (eventProceedButton == null)
            return Error("Crystal Sphere proceed button is not enabled");

        string title = SafeGetText(() => eventProceedButton.Option.Title) ?? "continue";
        eventProceedButton.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Proceeding from Crystal Sphere via event option: {title}"
        };
    }

    private static Creature? ResolveTarget(CombatState combatState, string entityId)
    {
        // Try to match by entity_id pattern: "model_entry_N"
        // First try matching by combat_id if it's a pure number
        if (uint.TryParse(entityId, out uint combatId))
            return combatState.GetCreature(combatId);

        // Match by entity_id pattern (e.g., "jaw_worm_0")
        // We rebuild the entity IDs the same way as BuildEnemyState
        var entityCounts = new Dictionary<string, int>();
        foreach (var creature in combatState.Enemies)
        {
            if (!creature.IsAlive) continue;
            string baseId = creature.Monster?.Id.Entry ?? "unknown";
            if (!entityCounts.TryGetValue(baseId, out int count))
                count = 0;
            entityCounts[baseId] = count + 1;
            string generatedId = $"{baseId}_{count}";

            if (generatedId == entityId)
                return creature;
        }

        return null;
    }
}
