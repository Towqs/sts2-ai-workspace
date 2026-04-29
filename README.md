# STS2 AI Workspace

这是一个 Slay the Spire 2 半自动 AI 工作区，用于本地数据采集、网页控制台、BC 模型训练，以及 OpenAI-compatible 大模型接入。

当前工作流是：玩家负责宏观判断，AI 可以托管战斗；也可以打开宏观 AI，让模型自动处理地图、奖励、选卡、事件和营火。商店自动购买默认关闭，避免和玩家抢操作。

## 项目记忆入口

后续 AI 训练大纲、阶段路线、当前模型状态和近期执行清单记录在：

- [`docs/ai_training_roadmap.md`](docs/ai_training_roadmap.md)
- [`docs/startup.md`](docs/startup.md)：一键启动控制台、日志窗口、BC AI 和 LLM。

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| `训练脚本/STS2MCP/` | 游戏 Mod 源码，提供 HTTP/MCP 接口，并写入 run 数据。 |
| `AI_Training/` | 控制台、战斗 BC、宏观 BC、LLM Agent、训练数据管线。 |
| `sts2_mcp_player/` | MCP 客户端辅助脚本和玩法说明。 |
| `RL_Datasets/` | 本机采集的原始数据，默认不提交 Git。 |
| `Data_Packages/` | 一键打包出来的数据包，默认不提交 Git。 |

## 不提交到 Git 的内容

`.gitignore` 已排除这些本机文件：

- `.venv/`
- `RL_Datasets/`
- `Data_Packages/`
- `AI_Training/ProcessedParams/`
- `AI_Training/ProcessedMacroParams/`
- `AI_Training/model_config.json`
- `AI_Training/control_state.json`
- Mod 编译产物，如 `out/`、`bin/`、`obj/`、`*.dll`

也就是说，GitHub 仓库只放代码和说明；数据、模型、API Key 都留在本机。别人拉仓库后需要自己安装依赖、构建 Mod、采集数据或导入你打包的数据。

## 本地启动

已经装好依赖和 Mod 后，日常使用可以直接双击：

```text
start_all.bat
```

它会打开网页控制台、后台输出窗口、RL 日志窗口，并在模型配置可用时启动 BC AI 和 LLM。中文名的 `一键启动全部.bat` 也保留，但如果路径或快捷方式有中文兼容问题，优先用 `start_all.bat`。详细说明见 [`docs/startup.md`](docs/startup.md)。

### 1. 安装 Python 依赖

推荐使用仓库根目录下的虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果 PyTorch 安装失败，按自己的显卡/CUDA 环境从 PyTorch 官方命令单独安装 `torch`，再安装 `numpy requests colorama`。

### 2. 构建并安装 Mod

如果采集目录不使用默认位置，可以先设置：

```powershell
$env:STS2_RL_DATA_DIR = "D:\path\to\RL_Datasets"
```

构建：

```powershell
powershell -ExecutionPolicy Bypass -File .\训练脚本\STS2MCP\build.ps1 -GameDir "<Slay the Spire 2 安装目录>"
```

然后把以下文件放到游戏 `mods` 目录：

- `训练脚本/STS2MCP/out/STS2_MCP/STS2_MCP.dll`
- `训练脚本/STS2MCP/out/STS2_MCP/mod_manifest.json`

### 3. 启动控制台

双击：

```text
启动AI控制台.bat
```

或手动运行：

```powershell
.\.venv\Scripts\python.exe .\AI_Training\control_panel.py
```

浏览器打开：

```text
http://127.0.0.1:8765/
```

游戏 Mod 的本地 API 默认是：

```text
http://localhost:15526/api/v1/singleplayer
```

控制台显示“未连接”时，通常是游戏没开、Mod 没加载、或本地 API 超时。

## 控制台开关

| 开关 | 含义 |
| --- | --- |
| 采集总开关 | 开启时写入新数据；关闭时暂停采集，并记录关闭时间段。 |
| 启动 AI / 停止 AI | 启停 BC Agent 进程。 |
| 允许 AI 出牌 | 只影响战斗自动出牌。 |
| 允许 AI 宏观操作 | 地图、奖励、选卡、事件、营火等宏观操作。 |
| 允许商店购买 | 默认关闭；打开后 AI 才能在商店购买明确商品。 |
| 记录 AI 战斗动作 | 控制台镜像写入 `RL_Datasets/AI_Combat`，方便复盘。Mod 的正式 AI 战斗记录写入 `RL_Datasets/AI/Combat`。 |
| AI 数据进入 BC | 默认关闭；打开后训练会使用 `source=ai` 的样本。 |
| 最低训练质量 | 过滤低质量 run，例如只训练一关 Boss 后或更高质量数据。 |

## 数据目录说明

| 路径 | 数据来源 |
| --- | --- |
| `RL_Datasets/Human/Combat` | 玩家手动战斗动作。 |
| `RL_Datasets/Human/Macro` | 玩家手动宏观动作。 |
| `RL_Datasets/AI/Combat` | Mod 记录的 AI 战斗动作，和当前 run_id 对齐。 |
| `RL_Datasets/AI/Macro` | Mod 记录的 AI 宏观动作。 |
| `RL_Datasets/AI_Combat` | 控制台额外镜像的 AI 战斗动作，主要用于复盘。 |
| `RL_Datasets/LLM_Actions` | 大模型建议、校验和执行日志。 |

如果你是“人选宏观、AI 打战斗”，完整 run 会由 `Human/Macro` 和 `AI/Combat` 合并统计。控制台的 Run 体检会按 `run_id` 汇总这些文件。

## 训练

控制台里可以点“一键训练”，也可以用命令行：

```powershell
.\.venv\Scripts\python.exe .\AI_Training\data_pipeline.py
.\.venv\Scripts\python.exe .\AI_Training\train_bc.py
.\.venv\Scripts\python.exe .\AI_Training\macro_data_pipeline.py
.\.venv\Scripts\python.exe .\AI_Training\train_macro_bc.py
```

输出位置：

- 战斗 BC：`AI_Training/ProcessedParams/`
- 宏观 BC：`AI_Training/ProcessedMacroParams/`

这些输出不会提交 Git。换机器后需要重新训练，或用数据包重新生成。

## LLM 接入

控制台支持 OpenAI-compatible Chat Completions 接口：

- `Advisor`：只给建议，不执行动作。
- `Combat Auto`：只允许在玩家战斗出牌阶段执行 `play_card` / `end_turn`。
- API Key 只保存到本机 `AI_Training/model_config.json`，不会进 Git。
- 已保存的 API 配置会显示在控制台表格里，可以切换、修改、删除。
- 连接测试有冷却，避免连续点击烧请求。

当前 LLM 的优势适合做高层判断和复杂解释；战斗自动执行仍受动作校验限制，不会越过可用动作列表强行操作。

## 数据打包

控制台提供“一键打包数据库”。打包文件会放在：

```text
Data_Packages/
```

数据包包含原始 jsonl、run 标签、丢弃列表和摘要；不包含训练后的 numpy 矩阵和模型权重。别人收到 zip 后可以用于复盘或重新训练。

## 当前已知限制

- GitHub 不包含本机模型和数据，拉库后不能直接启动可用 AI，需要先训练或导入数据。
- 宏观 BC 样本量还偏小，地图选择目前加入了规则评分来降低“一直走同一边”的数据偏差。
- 商店自动购买默认保护；如果不打开“允许商店购买”，AI 不会买东西，也不会主动离开商店。
- 战斗数据里 `battle_start` / `turn_start` 仍依赖 Mod Hook 是否触发；即使缺少这些快照，`play_card`、`end_turn`、`battle_end` 仍可用于 BC。
- 修改 Python 或前端代码后，需要重启控制台；修改 Mod 后需要重新构建并重启游戏。

## GitHub

远程仓库：

```text
https://github.com/Towqs/sts2-ai-workspace.git
```

常用提交流程：

```powershell
git status
git add README.md requirements.txt AI_Training/data_pipeline.py AI_Training/control_panel.py "训练脚本/STS2MCP/RL_DataCollector.cs"
git commit -m "Update docs"
git push origin main
```

实际提交时不要照抄上面的 `git add`，应按本次改动选择文件。
