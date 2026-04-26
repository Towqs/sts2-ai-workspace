# STS2 AI Workspace

这个仓库用于 Slay the Spire 2 的半自动数据采集、控制台监控和战斗出牌模型训练。

当前重点是三块：

- `训练脚本/STS2MCP/`：游戏 Mod 源码，负责暴露 MCP/HTTP 接口，并采集战斗、奖励、地图、商店等 run 数据。
- `AI_Training/`：网页控制台、数据管线、BC 战斗模型训练脚本和自动出牌代理。
- `sts2_mcp_player/`：MCP 客户端侧辅助脚本。

本仓库不提交本机虚拟环境、原始采集数据、打包数据、训练输出模型和编译产物。这些内容由 `.gitignore` 排除。

## 本地使用

### 1. 构建 Mod

可选：如果数据目录不在默认位置，先设置采集目录：

```powershell
$env:STS2_RL_DATA_DIR = "D:\path\to\RL_Datasets"
```

```powershell
powershell -ExecutionPolicy Bypass -File .\训练脚本\STS2MCP\build.ps1 -GameDir "<Slay the Spire 2 安装目录>"
```

构建完成后，把 `训练脚本/STS2MCP/out/STS2_MCP/STS2_MCP.dll` 和 `mod_manifest.json` 放到游戏 `mods` 目录。

### 2. 启动控制台

```powershell
python .\AI_Training\control_panel.py
```

然后打开：

```text
http://127.0.0.1:8765/
```

控制台用于查看采集状态、run 分类、样本数量、AI 出牌日志、采集开关、丢弃/保留数据和一键打包数据。

### 3. 生成训练数据

```powershell
python .\AI_Training\data_pipeline.py
```

输出会写到 `AI_Training/ProcessedParams/`。该目录是训练产物，不提交到 Git。

### 4. 训练战斗 BC 模型

```powershell
python .\AI_Training\train_bc.py
```

模型会写到 `AI_Training/ProcessedParams/bc_model_best.pth`。

## 当前状态

- 战斗出牌 BC 数据管线可以运行。
- 当前本机数据已经可以生成战斗样本并训练出模型。
- 地图选择、奖励候选、选卡、继续游戏续写、run 分类已经进入采集系统。
- 宏观数据已采集，但宏观决策模型训练还没有接入。

## 注意

`训练脚本/STS2MCP/RL_DataCollector.cs` 支持通过 `STS2_RL_DATA_DIR` 指定采集目录。没有设置时会回退到当前本机默认目录，用于不破坏现有采集流程。
