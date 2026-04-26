# 🦞 OpenClaw — 杀戮尖塔2 全自动 AI 智能体

> 让龙虾替你打牌。从第一层到 BOSS，全程自动。

## 项目简介

OpenClaw (代号: 龙虾) 是一个基于 [STS2MCP](https://github.com/Gennadiyev/STS2MCP) Mod 的《杀戮尖塔2》全自动 AI 玩家。它通过 REST API 实时读取游戏状态，做出战斗、选卡、选路等决策，并操控游戏执行。

**当前版本:** `v0.3.0` (角色适配版)

## 架构

```
┌─────────────────┐     HTTP REST      ┌──────────────────┐
│  杀戮尖塔2 游戏  │ ◄──────────────── │  agent_brain.py  │
│  + STS2MCP Mod  │  localhost:15526   │  (龙虾大脑)       │
└─────────────────┘                    └────────┬─────────┘
                                                │
                              ┌─────────────────┼─────────────────┐
                              │                 │                 │
                    ┌─────────▼──┐    ┌────────▼────────┐  ┌────▼──────────────┐
                    │ mcp_       │    │ combat_         │  │ character_        │
                    │ connector  │    │ analyzer        │  │ strategy          │
                    │ .py        │    │ .py             │  │ .py               │
                    │ (通讯层)    │    │ (战场分析)       │  │ (角色策略)         │
                    └────────────┘    └─────────────────┘  └───────────────────┘
```

## 快速开始

### 前提条件
- Python 3.8+
- 《杀戮尖塔2》已安装
- `requests` 库 (可选, 没有则自动回退到 `urllib`)

### 安装步骤

1. **安装 STS2MCP Mod**
   将 `STS2_MCP.dll` 和 `STS2_MCP.json` 放入游戏的 `mods/` 文件夹

2. **启动游戏**
   - 打开《杀戮尖塔2》
   - 进入 设置 → Mods → 启用 `STS2 MCP`
   - 手动选择角色，开始一局新游戏

3. **启动龙虾**
   ```bash
   python scripts/agent_brain.py
   ```

4. **看龙虾表演**
   龙虾会自动识别角色、分析局面、出牌、选路、选卡，直到通关或阵亡。

## 支持角色

| 角色 | 英文名 | 策略风格 | 适配状态 |
|------|--------|---------|---------|
| 铁甲战士 | Ironclad | 力量叠加 + 重型攻击 | ✅ |
| 猎人 | Silent | 毒液 + 碎片流 | ✅ |
| 缺陷体 | Defect | 球位管理 | ✅ |
| 观察者 | Watcher | 姿态切换 | ✅ |
| 死灵绑定者 | Necrobinder | 宠物协同 | ✅ |
| 储君 | The Regent | 星星资源 | ✅ |

## 文件结构

```
sts2_mcp_player/
├── CHANGELOG.md           # 开发日志 (你在这里)
├── README.md              # 项目说明
├── SKILL.md               # AI 策略指南
└── scripts/
    ├── agent_brain.py      # 核心决策引擎 (主入口)
    ├── mcp_connector.py    # STS2MCP REST API 通讯层
    ├── combat_analyzer.py  # 战场分析 + 威胁评估
    └── character_strategy.py  # 角色策略配置
```

## 开发日志

详见 [CHANGELOG.md](CHANGELOG.md)

## License

MIT
