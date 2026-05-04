// 临时方法扫描器 — 用完就删
// 在 RL_Logger.OnInit() 被调用时，扫描目标类的所有方法并写入 scan_report.txt

using System;
using System.IO;
using System.Linq;
using System.Reflection;

namespace STS2_MCP
{
    public static class MethodScanner
    {
        private static readonly string _reportPath =
            Path.Combine(RLDataPaths.BaseDir, "scan_report.txt");

        public static void ScanAll()
        {
            try
            {
                var sb = new System.Text.StringBuilder();
                sb.AppendLine($"=== Method Scan Report {DateTime.Now} ===\n");

                // 要扫描的类（宏观决策相关）
                string[] targetClasses = new[]
                {
                    "MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen",
                    "MegaCrit.Sts2.Core.Nodes.Rewards.NRewardsScreen",
                    "MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NCardRewardSelectionScreen",
                    "MegaCrit.Sts2.Core.Nodes.RestSite.NRestSiteRoom",
                    "MegaCrit.Sts2.Core.Nodes.RestSite.NRestSiteButton",
                    "MegaCrit.Sts2.Core.Nodes.Events.NEventRoom",
                    "MegaCrit.Sts2.Core.Nodes.Events.NEventOptionButton",
                    "MegaCrit.Sts2.Core.Entities.Merchant.MerchantInventoryEntry",
                    "MegaCrit.Sts2.Core.Nodes.Screens.Shops.NMerchantRoom",
                    "MegaCrit.Sts2.Core.Nodes.Screens.Overlays.NChooseARelicSelection",
                    "MegaCrit.Sts2.Core.Nodes.Screens.MainMenu.NMainMenu",
                    "MegaCrit.Sts2.Core.Nodes.Screens.MainMenu.NSingleplayerSubmenu",
                    "MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect.NCharacterSelectScreen",
                    "MegaCrit.Sts2.Core.Nodes.Screens.GameOver.NGameOverScreen",
                    "MegaCrit.Sts2.Core.Nodes.Rewards.NRewardButton",
                    "MegaCrit.Sts2.Core.Rewards.Reward",
                    "MegaCrit.Sts2.Core.Rewards.GoldReward",
                    "MegaCrit.Sts2.Core.Rewards.CardReward",
                    "MegaCrit.Sts2.Core.Rewards.PotionReward",
                    "MegaCrit.Sts2.Core.Entities.Potions.Potion",
                    "MegaCrit.Sts2.Core.Commands.PlayerCmd",
                    "MegaCrit.Sts2.Core.Combat.CombatManager",
                    "MegaCrit.Sts2.Core.Nodes.Screens.Shops.NMerchantInventory",
                };

                foreach (var className in targetClasses)
                {
                    sb.AppendLine($"▶ {className}");
                    try
                    {
                        var type = Type.GetType(className + ", sts2");
                        if (type == null)
                        {
                            // 尝试从所有已加载 assembly 搜索
                            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
                            {
                                type = asm.GetType(className);
                                if (type != null) break;
                            }
                        }

                        if (type == null)
                        {
                            sb.AppendLine("  ❌ CLASS NOT FOUND");
                            sb.AppendLine();
                            continue;
                        }

                        sb.AppendLine($"  ✅ Found in {type.Assembly.GetName().Name}");
                        sb.AppendLine($"  Base: {type.BaseType?.FullName ?? "none"}");

                        var methods = type.GetMethods(
                            BindingFlags.Public | BindingFlags.NonPublic |
                            BindingFlags.Instance | BindingFlags.Static |
                            BindingFlags.DeclaredOnly);

                        foreach (var m in methods.OrderBy(m => m.Name))
                        {
                            string accessMod = m.IsPublic ? "public" : m.IsPrivate ? "private" : "protected";
                            string staticMod = m.IsStatic ? " static" : "";
                            string virtualMod = m.IsVirtual ? " virtual" : "";
                            string asyncMod = m.ReturnType.Name.Contains("Task") ? " [ASYNC]" : "";
                            string paramStr = string.Join(", ",
                                m.GetParameters().Select(p => $"{p.ParameterType.Name} {p.Name}"));
                            sb.AppendLine($"  {accessMod}{staticMod}{virtualMod}{asyncMod} {m.ReturnType.Name} {m.Name}({paramStr})");
                        }
                    }
                    catch (Exception ex)
                    {
                        sb.AppendLine($"  ❌ SCAN ERROR: {ex.Message}");
                    }
                    sb.AppendLine();
                }

                File.WriteAllText(_reportPath, sb.ToString());
                Godot.GD.Print($"[RL] Method scan → {_reportPath}");
            }
            catch (Exception ex)
            {
                Godot.GD.PrintErr($"[RL] Scan failed: {ex.Message}");
            }
        }
    }
}
