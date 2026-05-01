# STS2 AI Workspace

<p align="center">
  <img src="AI_Training/assets/sts2_ai_logo.png" alt="STS2 AI Logo" width="180">
</p>

STS2 AI Workspace 是一个面向 **Slay the Spire 2** 的本地 AI 工作区。它把游戏状态读取、数据采集、网页控制台、行为克隆训练、模型切换和 OpenAI-compatible LLM 接入放在同一个工作流里。

项目当前是半成品，但已经可以跑通一条完整闭环：

```text
游戏 Mod 读取状态 -> 控制台展示和写日志 -> 数据管线生成样本 -> 训练 BC 模型 -> AI/LLM 决策 -> 继续回收数据
```

## 当前能做什么

| 能力 | 当前状态 |
| --- | --- |
| 游戏状态读取 | 已接入本地 Mod API，可读取战斗、奖励、地图、事件、营火、商店等状态。 |
| 网页控制台 | 已可用，地址为 `http://127.0.0.1:8765/`。 |
| 战斗 AI | 初版可用，支持战斗出牌和候选动作评分。 |
| 宏观 AI | 初版可用，支持地图、奖励、选卡、事件和营火；商店购买默认保护。 |
| 训练数据采集 | 支持 Human / AI / LLM 来源，按 run_id 汇总和体检。 |
| 数据重构与训练 | 支持战斗 BC、候选动作 BC、宏观 BC。 |
| LLM 接入 | 支持 OpenAI-compatible Chat Completions；推荐只从合法候选动作中选择。 |
| 模型包切换 | 已支持 `AI_Training/ModelZoo/` 下的可切换模型包。 |

## 当前限制

这个项目不是稳定成品 AI。现在的模型主要用于演示、采样和迭代，不应宣传为稳定通关工具。

主要短板：

- 高质量人类数据仍然太少。
- 宏观选择还需要更多路线、奖励、事件、营火和商店样本。
- AI 数据已经可以筛选入训，但仍需要人工复核失败原因。
- LLM 更适合作为“大脑”和解释器，自动执行必须继续受合法动作校验约束。

## 快速开始

### 1. 克隆仓库

```powershell
git clone https://github.com/Towqs/sts2-ai-workspace.git
cd sts2-ai-workspace
```

### 2. 一键安装环境和游戏 Mod

新人请优先运行：

```text
一键安装环境与Mod.bat
```

它会做三件事：

- 创建/更新 Python 虚拟环境并安装依赖。
- 自动定位《Slay the Spire 2》安装目录；找不到时会要求手动输入。
- 编译并安装游戏内 **STS2 MCP** Mod 到游戏 `mods` 目录。

如果自动定位失败，也可以手动指定游戏目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\Setup_Environment.ps1 -GameDir "D:\SteamLibrary\steamapps\common\Slay the Spire 2"
```

安装完成后，必须启动游戏并在 **Settings -> Mods** 里启用 `STS2 MCP`。  
启用后可以打开下面地址验证 Mod API 是否在线：

```text
http://localhost:15526/
```

看到 `Hello from STS2 MCP` 或 `status: ok`，说明游戏内 MCP 已经装好。

### 3. 手动安装方式（可选）

如果不使用一键安装，可以手动安装 Python 依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果 PyTorch 安装失败，请按自己的 CUDA / 显卡环境从 PyTorch 官方命令安装 `torch`，再安装：

```powershell
.\.venv\Scripts\python.exe -m pip install numpy requests colorama
```

然后构建并安装 Mod：

```powershell
powershell -ExecutionPolicy Bypass -File .\训练脚本\STS2MCP\build.ps1 -GameDir "<Slay the Spire 2 安装目录>"
```

把构建产物放进游戏 `mods` 根目录，并把 manifest 改名为 `STS2_MCP.json`：

- `训练脚本/STS2MCP/out/STS2_MCP/STS2_MCP.dll` -> `<game_install>/mods/STS2_MCP.dll`
- `训练脚本/STS2MCP/mod_manifest.json` -> `<game_install>/mods/STS2_MCP.json`

注意：不要只启动 AI 控制台。游戏里没有启用 `STS2 MCP` 时，控制台会显示游戏未连接，AI 也无法读状态或执行动作。

### 4. 启动控制台和 AI

推荐直接双击：

```text
start_all.bat
```

它会启动控制台、日志窗口和本地 AI 流程。中文文件名的 `一键启动全部.bat` 也保留，但如果 Windows 对中文路径处理不稳定，优先使用 `start_all.bat`。

手动启动：

```powershell
.\.venv\Scripts\python.exe .\AI_Training\control_panel.py
```

控制台地址：

```text
http://127.0.0.1:8765/
```

游戏 Mod API 默认地址：

```text
http://localhost:15526/api/v1/singleplayer
```

## 试用模型包

仓库现在包含一个可直接试用的演示模型包：

```text
AI_Training/ModelZoo/demo_local_20260430/
```

模型包内容：

| 模型 | 样本规模 |
| --- | --- |
| 战斗 BC | 642 条样本 |
| 候选动作评分 | 2948 行候选动作 |
| 宏观 BC | 528 条样本 |

数据来源摘要：

| 数据集 | Human | AI |
| --- | ---: | ---: |
| 战斗 | 448 | 194 |
| 宏观 | 359 | 169 |

控制台启动时，如果当前运行目录没有模型，会自动从完整模型包恢复。也可以在控制台的 **AI 模型状态 -> 训练模型切换** 中手动选择模型包并切换。切换后需要重启 AI 进程，运行时才会重新加载模型。

## 推荐演示方式

1. 启动游戏并确认 Mod 已加载。
2. 打开控制台，看顶部“游戏连接”是否在线。
3. 在 **AI 模型状态** 中确认 `demo_local_20260430` 完整可用。
4. 只演示战斗 AI 时，打开“允许 AI 出牌”，先关闭“允许 AI 宏观操作”和“允许 AI 商店购买”。
5. 演示宏观 AI 时，再打开“允许 AI 宏观操作”。商店购买建议保持关闭，避免抢玩家操作。
6. 演示 LLM 时，先配置 Base URL / API Key / Model，再使用“只从合法候选动作里选”的推荐模式。

## 控制台主要开关

| 开关 | 含义 |
| --- | --- |
| 采集总开关 | 开启后写入新数据；关闭后暂停采集。 |
| 启动 AI / 停止 AI | 启停本地 BC Agent 进程。 |
| 允许 AI 出牌 | 只影响战斗自动出牌。 |
| 允许 AI 宏观操作 | 地图、奖励、选卡、事件、营火等战斗外行为。 |
| 允许 AI 商店购买 | 默认关闭；打开后 AI 才能买明确商品。 |
| 记录 AI 战斗动作 | 写入 AI 战斗日志，便于复盘和后续入训。 |
| AI 数据进入 BC | 默认关闭；打开后只允许合格 AI 样本进入训练数据。 |
| 最低训练质量 | 过滤低质量 run，例如只使用一关 Boss 后或更高质量数据。 |

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| `训练脚本/STS2MCP/` | 游戏 Mod 源码，提供本地 API、执行动作并写入 run 数据。 |
| `AI_Training/control_panel.py` | 本地网页控制台。 |
| `AI_Training/ai_agent.py` | BC Agent，负责战斗和宏观动作执行。 |
| `AI_Training/llm_agent.py` | LLM Agent，负责模型建议和受约束执行。 |
| `AI_Training/data_pipeline.py` | 战斗数据管线。 |
| `AI_Training/macro_data_pipeline.py` | 宏观数据管线。 |
| `AI_Training/ModelZoo/` | 可提交到 GitHub 的演示模型包。 |
| `RL_Datasets/` | 本机原始采集数据，默认不提交 Git。 |
| `Data_Packages/` | 控制台一键打包出来的数据包，默认不提交 Git。 |
| `docs/` | 公开说明和使用文档。 |

## 文档入口

- [`docs/project_guide.md`](docs/project_guide.md)：项目说明和控制台读法。
- [`docs/startup.md`](docs/startup.md)：一键启动脚本说明。
- [`docs/data_contribution.md`](docs/data_contribution.md)：如何打包数据并提交给维护者。
- [`docs/public_roadmap.md`](docs/public_roadmap.md)：公开路线图。
- [`docs/monster_data.md`](docs/monster_data.md)：怪物数据采集和后续用途。
- [`AI_Training/ModelZoo/README.md`](AI_Training/ModelZoo/README.md)：模型包结构和切换说明。

## 训练

控制台可以点“一键训练”。命令行方式：

```powershell
.\.venv\Scripts\python.exe .\AI_Training\data_pipeline.py
.\.venv\Scripts\python.exe .\AI_Training\train_bc.py
.\.venv\Scripts\python.exe .\AI_Training\train_candidate_bc.py
.\.venv\Scripts\python.exe .\AI_Training\macro_data_pipeline.py
.\.venv\Scripts\python.exe .\AI_Training\train_macro_bc.py
```

训练输出默认在：

- `AI_Training/ProcessedParams/`
- `AI_Training/ProcessedMacroParams/`

这两个目录默认不提交 Git。如果要分享可试用模型，请整理成 `AI_Training/ModelZoo/<model_id>/` 结构，并附上 `manifest.json`。

## 数据贡献

最需要的贡献仍然是高质量真实数据。推荐通过控制台点击“一键打包数据库”，把 `Data_Packages/` 下生成的 zip 发给维护者，或在 GitHub Issue 中提供下载链接。

如果不方便走 GitHub，也可以通过 QQ 联系维护者：`2775089081`。

提交数据时请说明：

```text
游戏版本：
Mod 版本或提交号：
采集日期：
大概 run 数：
主要角色：
是否包含 AI/LLM 自动操作：
是否手动丢弃过坏 run：
备注：
```

不要提交 API Key、本地账号信息、游戏本体文件、`.venv/`、无关日志或私人路径。

## Git 忽略策略

默认不提交：

- `.venv/`
- `RL_Datasets/`
- `Data_Packages/`
- `AI_Training/ProcessedParams/`
- `AI_Training/ProcessedMacroParams/`
- `AI_Training/model_config.json`
- `AI_Training/control_state.json`
- Mod 编译产物，如 `out/`、`bin/`、`obj/`、`*.dll`

允许提交：

- 公开说明文档。
- 控制台和训练代码。
- `AI_Training/assets/` 下的项目图标。
- `AI_Training/ModelZoo/` 下明确整理过的演示模型包。

## 远程仓库

```text
https://github.com/Towqs/sts2-ai-workspace.git
```
