using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Reflection;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Multiplayer.Game;

namespace STS2_MCP;

[ModInitializer("Initialize")]
public static partial class McpMod
{
    public const string Version = "0.3.4";
    public const int DefaultPort = 15526;
    private const string ConfigFileName = "STS2_MCP.conf";

    private static HttpListener? _listener;
    private static Thread? _serverThread;
    private static readonly ConcurrentQueue<Action> _mainThreadQueue = new();
    internal static readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    private static int LoadPort()
    {
        try
        {
            string? modDir = Path.GetDirectoryName(
                System.Reflection.Assembly.GetExecutingAssembly().Location);
            if (modDir == null) return DefaultPort;

            string configPath = Path.Combine(modDir, ConfigFileName);
            if (!File.Exists(configPath))
            {
                // Create default config so the user knows it's configurable
                var defaultConfig = new Dictionary<string, object> { ["port"] = DefaultPort };
                string json = JsonSerializer.Serialize(defaultConfig, _jsonOptions);
                File.WriteAllText(configPath, json);
                GD.Print($"[STS2 MCP] Created default config at {configPath}");
                return DefaultPort;
            }

            string content = File.ReadAllText(configPath);
            using var doc = JsonDocument.Parse(content);
            if (doc.RootElement.TryGetProperty("port", out var portElem)
                && portElem.TryGetInt32(out int port)
                && port is > 0 and <= 65535)
            {
                return port;
            }

            GD.PrintErr($"[STS2 MCP] Invalid or missing 'port' in {configPath}, using default {DefaultPort}");
            return DefaultPort;
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[STS2 MCP] Failed to load config: {ex.Message}, using default port {DefaultPort}");
            return DefaultPort;
        }
    }

    public static void Initialize()
    {
        try
        {
            // Optional settings UI patches should not block the HTTP bridge itself.
            TryApplyHarmonyPatches();

            // 扫描游戏类方法（临时诊断，找到正确方法名后删除）
            try { MethodScanner.ScanAll(); } catch { }

            // Connect to main thread process frame for action execution
            var tree = (SceneTree)Engine.GetMainLoop();
            tree.Connect(SceneTree.SignalName.ProcessFrame, Callable.From(ProcessMainThreadQueue));

            int port = LoadPort();

            _listener = new HttpListener();
            _listener.Prefixes.Add($"http://localhost:{port}/");
            _listener.Prefixes.Add($"http://127.0.0.1:{port}/");
            _listener.Start();

            _serverThread = new Thread(ServerLoop)
            {
                IsBackground = true,
                Name = "STS2_MCP_Server"
            };
            _serverThread.Start();

            GD.Print($"[STS2 MCP] v{Version} server started on http://localhost:{port}/");
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[STS2 MCP] Failed to start: {ex}");
        }
    }

        private static void TryApplyHarmonyPatches()
        {
            var harmony = new Harmony("com.sts2mcp");
            var failed = new List<string>();

            // 1. RunStateChange
            try {
                harmony.Patch(
                    AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.Runs.RunManager), "IsInProgress"),
                    postfix: new HarmonyMethod(typeof(RL_DataCollector).GetMethod("Hook_RunStateChange_Postfix", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static))
                );
                Debug.Log("Harmony", "[PATCH] RunStateChange OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] RunStateChange FAILED: " + ex.Message);
                failed.Add("RunStateChange");
            }

            try {
                harmony.Patch(
                    AccessTools.Method(typeof(MegaCrit.Sts2.Core.Nodes.Screens.MainMenu.NMainMenu), "OnContinueButtonPressed"),
                    prefix: typeof(RL_DataCollector).GetMethod("Hook_MainMenuContinue_Prefix", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                );
                Debug.Log("Harmony", "[PATCH] MainMenuContinueIntent OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] MainMenuContinueIntent FAILED: " + ex.Message);
                failed.Add("MainMenuContinueIntent");
            }

            try {
                harmony.Patch(
                    AccessTools.Method(typeof(MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect.NCharacterSelectScreen), "BeginRun"),
                    prefix: typeof(RL_DataCollector).GetMethod("Hook_CharacterSelectBeginRun_Prefix", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                );
                Debug.Log("Harmony", "[PATCH] CharacterSelectNewRunIntent OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] CharacterSelectNewRunIntent FAILED: " + ex.Message);
                failed.Add("CharacterSelectNewRunIntent");
            }

            // 2. CombatStateChange
            try {
                harmony.Patch(
                    AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.Combat.CombatManager), "IsInProgress"),
                    postfix: typeof(RL_DataCollector).GetMethod("Hook_CombatStateChange_Postfix", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                );
                Debug.Log("Harmony", "[PATCH] CombatStateChange OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] CombatStateChange FAILED: " + ex.Message);
                failed.Add("CombatStateChange");
            }

            // 3. EndTurn
            try {
                harmony.Patch(
                    AccessTools.Method(typeof(MegaCrit.Sts2.Core.Commands.PlayerCmd), "EndTurn"),
                    prefix: typeof(RL_DataCollector).GetMethod("Hook_EndTurn_Prefix", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                );
                Debug.Log("Harmony", "[PATCH] EndTurn OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] EndTurn FAILED: " + ex.Message);
                failed.Add("EndTurn");
            }

            // 4. PlayCard - hook the action constructor; PlayerCmd.PlayCard is not stable across builds.
            try {
                harmony.CreateClassProcessor(typeof(RL_DataCollector.Hook_PlayCard_ActionCtor)).Patch();
                Debug.Log("Harmony", "[PATCH] PlayCard OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] PlayCard FAILED: " + ex.Message);
                failed.Add("PlayCard");
            }

            // 5. UsePotion — 截获药水使用
            try {
                harmony.CreateClassProcessor(typeof(RL_DataCollector.Hook_UsePotionActionCtor)).Patch();
                Debug.Log("Harmony", "[PATCH] UsePotion OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] UsePotion FAILED: " + ex.Message);
                failed.Add("UsePotion");
            }

            // 6. MapNode — 截获地图选路
            try {
                harmony.CreateClassProcessor(typeof(RL_DataCollector.Hook_MoveToMapCoordActionCtor)).Patch();
                Debug.Log("Harmony", "[PATCH] MapNode OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] MapNode FAILED: " + ex.Message);
                failed.Add("MapNode");
            }

            // NOTE: BattleWin (ResolveCombatEnd) and BattleLose (OnPlayerDied) methods not found —
            // will use CombatStateChange + timer fallback to detect battle end instead.

            // 7. BattleLose — LoseCombat (非 async，扫描报告确认存在)
            try {
                harmony.CreateClassProcessor(typeof(RL_DataCollector.Hook_BattleWin_EndCombatInternal)).Patch();
                Debug.Log("Harmony", "[PATCH] BattleWin OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] BattleWin FAILED: " + ex.Message);
                failed.Add("BattleWin");
            }

            try {
                harmony.CreateClassProcessor(typeof(RL_DataCollector.Hook_CardRewardChoiceDynamic)).Patch();
                Debug.Log("Harmony", "[PATCH] CardRewardChoice OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] CardRewardChoice FAILED: " + ex.Message);
                failed.Add("CardRewardChoice");
            }

            try {
                harmony.Patch(
                    AccessTools.Method(typeof(MegaCrit.Sts2.Core.Combat.CombatManager), "LoseCombat"),
                    prefix: typeof(RL_DataCollector).GetMethod("Hook_BattleLose_Prefix", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                );
                Debug.Log("Harmony", "[PATCH] BattleLose (LoseCombat) OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] BattleLose FAILED: " + ex.Message);
                failed.Add("BattleLose");
            }

            // 8. RewardClaimed — Reward.OnSelectWrapper (领奖励，所有类型共用)
            try {
                harmony.Patch(
                    AccessTools.Method(typeof(MegaCrit.Sts2.Core.Rewards.Reward), "OnSelectWrapper"),
                    prefix: typeof(RL_DataCollector).GetMethod("Hook_RewardClaimed_Prefix", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                );
                Debug.Log("Harmony", "[PATCH] RewardClaimed OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] RewardClaimed FAILED: " + ex.Message);
                failed.Add("RewardClaimed");
            }

            // 9. RewardSkipped — Reward.OnSkipped (跳过奖励)
            try {
                harmony.Patch(
                    AccessTools.Method(typeof(MegaCrit.Sts2.Core.Rewards.Reward), "OnSkipped"),
                    prefix: typeof(RL_DataCollector).GetMethod("Hook_RewardSkipped_Prefix", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                );
                Debug.Log("Harmony", "[PATCH] RewardSkipped OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] RewardSkipped FAILED: " + ex.Message);
                failed.Add("RewardSkipped");
            }

            // 10. RestSite — NRestSiteButton.OnPress (篝火选择)
            try {
                harmony.Patch(
                    AccessTools.Method(typeof(MegaCrit.Sts2.Core.Nodes.RestSite.NRestSiteButton), "OnPress"),
                    prefix: typeof(RL_DataCollector).GetMethod("Hook_RestSite_Prefix", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                );
                Debug.Log("Harmony", "[PATCH] RestSite OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] RestSite FAILED: " + ex.Message);
                failed.Add("RestSite");
            }

            // 11. EventOption — NEventOptionButton.OnPress (事件选项)
            try {
                harmony.Patch(
                    AccessTools.Method(typeof(MegaCrit.Sts2.Core.Nodes.Events.NEventOptionButton), "OnPress"),
                    prefix: typeof(RL_DataCollector).GetMethod("Hook_EventOption_Prefix", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                );
                Debug.Log("Harmony", "[PATCH] EventOption OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] EventOption FAILED: " + ex.Message);
                failed.Add("EventOption");
            }

            // 12. ShopPurchase — NMerchantInventory.OnPurchaseCompleted (商店购买)
            try {
                harmony.CreateClassProcessor(typeof(RL_DataCollector.Hook_ShopOpenSnapshot)).Patch();
                harmony.Patch(
                    AccessTools.Method(typeof(MegaCrit.Sts2.Core.Nodes.Screens.Shops.NMerchantInventory), "OnPurchaseCompleted"),
                    postfix: typeof(RL_DataCollector).GetMethod("Hook_ShopPurchase_Postfix", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                );
                Debug.Log("Harmony", "[PATCH] ShopPurchase OK");
            } catch (Exception ex) {
                Debug.Log("Harmony", "[PATCH] ShopPurchase FAILED: " + ex.Message);
                failed.Add("ShopPurchase");
            }

            if (failed.Count > 0)
                Debug.Log("Harmony", $"[INIT] Failed patches: {string.Join(", ", failed)}");
            else
                Debug.Log("Harmony", "[INIT] All patches applied successfully");

            try { RL_DataCollector.Initialize(); } catch { }
            Debug.Log("Harmony", "[INIT] RL_DataCollector ready");
        }

    private static void ProcessMainThreadQueue()
    {
        int processed = 0;
        while (_mainThreadQueue.TryDequeue(out var action) && processed < 10)
        {
            try { action(); }
            catch (Exception ex) { GD.PrintErr($"[STS2 MCP] Main thread action error: {ex}"); }
            processed++;
        }
    }

    internal static Task<T> RunOnMainThread<T>(Func<T> func)
    {
        var tcs = new TaskCompletionSource<T>();
        _mainThreadQueue.Enqueue(() =>
        {
            try { tcs.SetResult(func()); }
            catch (Exception ex) { tcs.SetException(ex); }
        });
        return tcs.Task;
    }

    internal static Task RunOnMainThread(Action action)
    {
        var tcs = new TaskCompletionSource<bool>();
        _mainThreadQueue.Enqueue(() =>
        {
            try { action(); tcs.SetResult(true); }
            catch (Exception ex) { tcs.SetException(ex); }
        });
        return tcs.Task;
    }

    private static void ServerLoop()
    {
        while (_listener?.IsListening == true)
        {
            try
            {
                var context = _listener.GetContext();
                // Handle each request asynchronously so we don't block the listener
                ThreadPool.QueueUserWorkItem(_ => HandleRequest(context));
            }
            catch (HttpListenerException) { break; }
            catch (ObjectDisposedException) { break; }
        }
    }

    private static void HandleRequest(HttpListenerContext context)
    {
        try
        {
            var request = context.Request;
            var response = context.Response;
            response.Headers.Add("Access-Control-Allow-Origin", "*");
            response.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            response.Headers.Add("Access-Control-Allow-Headers", "Content-Type");

            if (request.HttpMethod == "OPTIONS")
            {
                response.StatusCode = 204;
                response.Close();
                return;
            }

            string path = request.Url?.AbsolutePath ?? "/";

            if (path == "/")
            {
                SendJson(response, new { message = $"Hello from STS2 MCP v{Version}", status = "ok" });
            }
            else if (path == "/api/v1/singleplayer")
            {
                // Hard-block singleplayer endpoint during multiplayer runs
                // to prevent calling the non-sync-safe end_turn path
                if (IsMultiplayerRun())
                {
                    SendError(response, 409,
                        "Multiplayer run is active. Use /api/v1/multiplayer instead.");
                    return;
                }

                if (request.HttpMethod == "GET")
                    HandleGetState(request, response);
                else if (request.HttpMethod == "POST")
                    HandlePostAction(request, response);
                else
                    SendError(response, 405, "Method not allowed");
            }
            else if (path == "/api/v1/multiplayer")
            {
                // Guard: reject multiplayer endpoint during singleplayer runs
                if (!IsMultiplayerRun())
                {
                    SendError(response, 409,
                        "Not in a multiplayer run. Use /api/v1/singleplayer instead.");
                    return;
                }

                if (request.HttpMethod == "GET")
                    HandleGetMultiplayerState(request, response);
                else if (request.HttpMethod == "POST")
                    HandlePostMultiplayerAction(request, response);
                else
                    SendError(response, 405, "Method not allowed");
            }
            else
            {
                SendError(response, 404, "Not found");
            }
        }
        catch (Exception ex)
        {
            try
            {
                SendError(context.Response, 500, $"Internal error: {ex.Message}");
            }
            catch { /* response may already be closed */ }
        }
    }

    // Called on HTTP thread (not main thread) as a best-effort guard.
    // The try/catch handles race conditions during run transitions.
    // Authoritative checks happen inside RunOnMainThread lambdas.
    internal static bool IsMultiplayerRun()
    {
        try
        {
            return MegaCrit.Sts2.Core.Runs.RunManager.Instance.IsInProgress
                && MegaCrit.Sts2.Core.Runs.RunManager.Instance.NetService.Type.IsMultiplayer();
        }
        catch { return false; }
    }

    private static void HandleGetMultiplayerState(HttpListenerRequest request, HttpListenerResponse response)
    {
        string format = request.QueryString["format"] ?? "json";

        try
        {
            var stateTask = RunOnMainThread(() => BuildMultiplayerGameState());
            var state = stateTask.GetAwaiter().GetResult();

            if (format == "markdown")
            {
                string md = FormatAsMarkdown(state);
                SendText(response, md, "text/markdown");
            }
            else
            {
                SendJson(response, state);
            }
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[STS2 MCP] HandleGetMultiplayerState: {ex}");
            try
            {
                response.StatusCode = 500;
                SendJson(response, new Dictionary<string, object?>
                {
                    ["error"] = $"Failed to read multiplayer game state: {ex.Message}",
                    ["exception_type"] = ex.GetType().FullName,
                    ["stack_trace"] = ex.StackTrace
                });
            }
            catch { /* response may be unusable */ }
        }
    }

    private static void HandlePostMultiplayerAction(HttpListenerRequest request, HttpListenerResponse response)
    {
        string body;
        using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
            body = reader.ReadToEnd();

        Dictionary<string, JsonElement>? parsed;
        try
        {
            parsed = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(body);
        }
        catch
        {
            SendError(response, 400, "Invalid JSON");
            return;
        }

        if (parsed == null || !parsed.TryGetValue("action", out var actionElem))
        {
            SendError(response, 400, "Missing 'action' field");
            return;
        }

        string action = actionElem.GetString() ?? "";

        try
        {
            var resultTask = RunOnMainThread(() => ExecuteMultiplayerAction(action, parsed));
            var result = resultTask.GetAwaiter().GetResult();
            SendJson(response, result);
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Multiplayer action failed: {ex.Message}");
        }
    }

    private static void HandleGetState(HttpListenerRequest request, HttpListenerResponse response)
    {
        string format = request.QueryString["format"] ?? "json";

        try
        {
            var stateTask = RunOnMainThread(() => BuildGameState());
            var state = stateTask.GetAwaiter().GetResult();

            if (format == "markdown")
            {
                try
                {
                    SendText(response, FormatAsMarkdown(state), "text/markdown");
                }
                catch (Exception ex)
                {
                    GD.PrintErr($"[STS2 MCP] FormatAsMarkdown failed, returning JSON: {ex}");
                    SendJson(response, state);
                }
            }
            else
            {
                SendJson(response, state);
            }
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[STS2 MCP] HandleGetState: {ex}");
            try
            {
                response.StatusCode = 500;
                SendJson(response, new Dictionary<string, object?>
                {
                    ["error"] = $"Failed to read game state: {ex.Message}",
                    ["exception_type"] = ex.GetType().FullName,
                    ["stack_trace"] = ex.StackTrace
                });
            }
            catch { /* response may be unusable */ }
        }
    }

    private static void HandlePostAction(HttpListenerRequest request, HttpListenerResponse response)
    {
        string body;
        using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
            body = reader.ReadToEnd();

        Dictionary<string, JsonElement>? parsed;
        try
        {
            parsed = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(body);
        }
        catch
        {
            SendError(response, 400, "Invalid JSON");
            return;
        }

        if (parsed == null || !parsed.TryGetValue("action", out var actionElem))
        {
            SendError(response, 400, "Missing 'action' field");
            return;
        }

        string action = actionElem.GetString() ?? "";

        try
        {
            var resultTask = RunOnMainThread(() => ExecuteAction(action, parsed));
            var result = resultTask.GetAwaiter().GetResult();
            SendJson(response, result);
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Action failed: {ex.Message}");
        }
    }
}
