# Changelog

All notable changes to the OpenClaw (龙虾) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.4.0] - 2026-04-18

### Added
- **牌组需求分析** (`analyze_deck_needs()`): 分析牌组中攻击/防御/能力/技能的比例，判断缺哪类牌
- **智能选卡**: 选卡时根据牌组当前构成加分——缺攻击加攻击分、缺防御加防御分、缺能力加能力分
- **牌组构成日志**: 选卡时打印 "牌组分析: 12张 (攻5 防3 能1 技3)" 及缺口提示
- **精英/Boss开局增益药水**: 在精英/Boss战回合1自动使用力量/敏捷类增益药水

### Changed
- **地图选路**: 完全重写！改为**评分系统**（不再总走最左边）
  - 血少(<40%) → 篝火100分
  - 血多(>阈值) → 精英80分（拿遗物）
  - 正常血量 → 怪物50分 > 事件45分 > 商店35分 > 宝箱55分
  - **前瞻加分**: 看下一层节点类型额外打分
- **药水策略**: 从"有就用"改为**保守使用**
  - 小怪: 只有致命威胁才用药
  - 精英/Boss: HP<60% 用回血药，回合1用增益药
- **选卡评分**: 新增通用好牌加分（抽牌+1.5, AOE+1.5, 升级+1.0）

### Fixed
- 🐛 地图永远走 `next_options[0]`（最左边）的问题
- 🐛 药水在小怪战也消耗的问题

---

## [0.3.1] - 2026-04-18

### Fixed
- 🐛 **[P0] 药水无法领取**: 重写 `handle_rewards` 为循环领取模式，每次领取后刷新状态重新获取索引，处理药水槽满时的错误（跳过而非崩溃）
- 🐛 **[P0] 药水无法使用**: 扩大战斗中药水使用的触发条件——致命威胁时使用任意可用药水（不再限制只用回血/格挡），HP < 50% 时主动使用回血药水
- 🐛 **卡牌奖励未领取**: 奖励列表中的 `card` 和 `special_card` 类型现在也会通过 `claim_reward` 正确领取，触发选卡界面

### Changed
- **奖励领取流程**: 改为 `金币 > 遗物 > 药水 > 卡牌` 优先级循环，每次操作后重新获取状态
- **药水使用时机**: 新增 HP < 50% 时主动喝回血药水的逻辑

---

## [0.3.0] - 2026-04-18

### Added
- **角色策略系统** (`character_strategy.py`): 新增6个角色策略档案 (Ironclad/Silent/Defect/Watcher/Necrobinder/Regent)，含被动技能描述、选卡偏好关键词、攻击倾向度、HP阈值等
- **中文角色名映射**: 支持游戏内显示的中文名自动匹配 (储君→Regent, 铁甲战士→Ironclad, 猎人→Silent 等)
- **智能选卡评分**: 根据角色策略给卡牌奖励打分，优先选适合当前角色被动的牌
- **牌组膨胀检测**: 当牌组超过20/25张时，低分卡牌会被自动跳过
- **角色自动识别**: 首次进入战斗时读取 `player.character`，自动加载匹配策略并打印角色信息
- **PRE_END_DELAY (2.5s)**: 新增结束回合前的缓冲延迟，解决"最后一张牌打完太快就结束"的问题

### Changed
- **出牌延迟**: ACTION_DELAY 1.5s → 2.0s
- **回合结束延迟**: END_TURN_DELAY 3.0s → 4.0s
- **UI切换延迟**: UI_DELAY 2.0s → 2.5s
- **篝火阈值**: 改为根据角色策略动态调整 (战士50%, 猎人55%, 缺陷体55%...)
- **地图选路**: 增加优先级排序 (低血→篝火, 高血→精英, 正常→怪物>事件>商店)
- **防御牌判断**: 移除"敌方不攻击时不出防御牌"的限制，有费就出

### Fixed
- 🐛 **[P0] 同名牌只出一张**: `used_card_names` 列表检查导致3张"打击"只出第1张。改为 `used_indices` (set) 按卡牌索引追踪
- 🐛 **[P0] "储君"角色未识别**: 角色映射表只有英文名，游戏返回中文名"储君"，命中 DEFAULT_PROFILE
- 🐛 **其他牌只出0费**: `other_cards` 循环中有 `if card["cost"] == 0` 限制，导致有费用的其他技能牌被浪费

---

## [0.2.0] - 2026-04-18

### Added
- **Debuff 牌分类** (`combat_analyzer.py`): 新增 `debuff_cards` 类别，自动识别含 Vulnerable/Weak/Poison 关键词的卡牌
- **药水分析** (`combat_analyzer.py`): 新增 `analyze_potions()` 函数，分类回血/格挡/伤害药水
- **致命威胁药水使用**: 当预期伤害 ≥ 当前HP时，自动使用回血/格挡药水
- **实时出牌匹配** (`play_card_realtime()`): 每出一张牌先刷新状态再按名字找牌，避免索引漂移
- **全界面处理函数**:
  - `handle_relic_select()`: BOSS 遗物三选一
  - `handle_bundle_select()`: 捆绑包选择
  - `handle_treasure()`: 宝箱遗物领取

### Changed
- **出牌优先级重排**: 能力牌 → Debuff牌(先上易伤) → 攻击牌 → 防御牌 → 其他
- **战场简报增强**: 新增药水显示、debuff牌/能力牌分类标注、敌方BUFF状态提示
- **奖励领取流程**: 每领取一项后刷新状态检查界面变化，避免流程断裂

### Fixed
- 🐛 选卡奖励领取后未正确等待界面切换
- 🐛 防御牌判断遗漏 (仅匹配format但未考虑实际格挡值)

---

## [0.1.0] - 2026-04-17

### Added
- **项目初始化**: 基于 STS2MCP (v0.3.5-rc1) 真实 API 规范搭建
- `mcp_connector.py`: HTTP REST 通讯层
  - 端口: 15526 (STS2MCP 默认)
  - GET `/api/v1/singleplayer?format=json` 获取状态
  - POST `/api/v1/singleplayer` 执行动作
  - 覆盖全部 20+ 个 API 工具 (play_card, end_turn, use_potion, choose_map_node, etc.)
  - 兼容 `requests` 和 `urllib` 双通道
- `combat_analyzer.py`: 战斗分析引擎
  - 敌人意图解析 (支持 "12" 单次和 "6x3" 多段格式)
  - 手牌攻防能力评估 (费用感知)
  - 斩杀线计算 (敌人HP+格挡 vs 最大输出)
  - 战场简报生成 (威胁等级: 安全/可控/高危/致命)
- `agent_brain.py`: 核心决策引擎
  - `state_type` 全界面状态路由 (15种界面)
  - 规则引擎兜底战斗决策
  - LLM 挂载点预留 (call_llm / parse_llm_actions)
  - 防卡死检测 (连续相同状态计数器)
  - 重试机制 (连续错误计数器)
- `SKILL.md`: AI 策略指南文档

### Infrastructure
- STS2MCP Mod 文件下载并安装到 `Slay the Spire 2/mods/`
- API 深度调研: 阅读 STS2MCP 源码 (server.py, Actions.cs, StateBuilder.cs)
- 代码审计报告 + 实施计划文档

---

## Version History Summary

| Version | Date | Codename | Milestone |
|---------|------|----------|-----------|
| 0.1.0 | 2026-04-17 | 破壳 | API 对接, 基础战斗循环 |
| 0.2.0 | 2026-04-18 | 初战 | Debuff优先, 药水使用, 出牌优化 |
| 0.3.0 | 2026-04-18 | 识人 | 角色适配, 同名牌修复, 智能选卡 |
| 0.3.1 | 2026-04-18 | 拾药 | 药水领取/使用修复, 奖励流程重写 |
| 0.4.0 | 2026-04-18 | 识路 | 智能选路, 药水保守使用, 牌组需求选卡 |

---

## Roadmap

- [ ] **v0.4.0 - 通关**: 第一章 BOSS 通关验证
- [ ] **v0.5.0 - 接脑**: 接入 LLM (DeepSeek/Ollama) 替代规则引擎
- [ ] **v0.6.0 - 全程**: 第二章、第三章适配
- [ ] **v1.0.0 - 龙虾成年**: 稳定通关率 > 50%
