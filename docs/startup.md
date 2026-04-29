# 一键启动说明

推荐双击仓库根目录的 `start_all.bat`。

`一键启动全部.bat` 仍然保留，但它只是包装 `start_all.bat`。如果 Windows、快捷方式或压缩包对中文文件名处理不稳定，直接用 ASCII 文件名的 `start_all.bat`。

它会做四件事：

1. 启动 AI 控制台服务，并打开 `http://127.0.0.1:8765/`。
2. 打开一个控制台窗口显示控制台和 Agent 后台输出，日志同时写入 `RuntimeLogs/control_panel.log`。
3. 打开一个 RL 日志窗口，实时查看 `RL_Datasets/rl_monitor.log`。
4. 通过控制台 API 启动 BC AI；如果 LLM 已配置模型名和 API Key，也会一起启动 LLM。

注意：

- 游戏和 STS2_MCP Mod 仍然需要先启动；否则网页会显示游戏未连接。
- 启动器会先测试 `.venv\Scripts\python.exe` 是否真的可运行；如果本地 venv 损坏，会自动跳过并尝试 Codex bundled Python 或 PATH 里的 Python。
- 如果 LLM 没有配置 API Key 或模型名，总启动器会跳过 LLM，避免空配置循环报错。
- 如果战斗模型不存在，总启动器会跳过 BC AI，需要先在控制台里训练或导入模型。

常用手动参数：

```powershell
.\tools\start_all.ps1 -NoLlm
.\tools\start_all.ps1 -NoAi
.\tools\start_all.ps1 -NoMonitor
.\tools\start_all.ps1 -NoBrowser
.\tools\start_all.ps1 -ForceLlm
```
