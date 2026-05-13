# Ironclad 选牌 Shadow Mode

本文说明第一阶段 Ironclad 选牌系统的最小闭环。它的目标不是立刻替换宏观模型，也不是重写 PPO，而是先让系统在 `card_reward` 场景里稳定生成“每张牌 + skip”的可解释评分，并在不改变旧策略行为的前提下记录对照数据。

## 当前定位

当前默认模式是 `shadow`：

```text
旧选牌逻辑继续执行
新 card scorer 只评分、只记录、只做对照
```

这样可以在不影响现有 `current_rl`、macro BC、candidate scorer、ModelZoo 和 PPO v0 的情况下，观察新的 option-scoring 选牌系统是否稳定。

## 新增模块

| 文件 | 作用 |
| --- | --- |
| `AI_Training/state_encoder.py` | 提供轻量 state summary 和 `state_features_version`，第一阶段不替换旧 encoder |
| `AI_Training/options/base.py` | 定义 option schema、option feature version 和通用 `Option` / `OptionResult` |
| `AI_Training/options/combat.py` | 包装现有 `combat_actions.py`，不改变战斗策略 |
| `AI_Training/deck_summary.py` | 汇总战士卡组结构、费用曲线、力量/格挡/消耗等 archetype 信号 |
| `AI_Training/options/cards.py` | 对 card reward 中每张牌和 `skip` 打分 |
| `AI_Training/configs/archetype_templates.yaml` | 配置战士构筑模板和 card scorer 默认模式 |

## 模板配置

默认启用三套 Ironclad 模板：

```text
strength_multihit
barricade_block
exhaust_engine
```

`self_damage_rupture` 已写入配置，但默认关闭：

```yaml
self_damage_rupture:
  enabled: false
```

自伤流需要更稳定的血量、药水和路线风险判断，后续再启用。

## 开关模式

配置位置：

```text
AI_Training/configs/archetype_templates.yaml
```

当前支持三档：

```yaml
option_card_scorer:
  mode: shadow
```

| 模式 | 行为 |
| --- | --- |
| `off` | 完全关闭新 card scorer，只使用旧选牌逻辑 |
| `shadow` | 旧逻辑执行，新 card scorer 只记录评分和推荐动作 |
| `active` | 新 card scorer 接管 card reward 选择 |

第一阶段默认必须保持 `shadow`。只有在 shadow 日志验证稳定后，才建议手动切换到 `active`。

## Shadow 日志

当 AI 进入 `card_reward` 场景时，shadow mode 会写入：

```text
RL_Datasets/OptionShadow/card_scorer_YYYY-MM-DD.jsonl
```

每条记录包含：

```text
actual_payload
legacy_chosen_action
recommended_action
recommended_payload
skip_available
skip_recommended
legal_option_count
option_schema
state_features_version
option_features_version
template_id
archetype_consistency
deck_summary
score_distribution
options
```

这些字段用于回答三个问题：

```text
旧策略实际选了什么？
新 scorer 会推荐什么？
skip 是否合理参与竞争？
```

## Shadow 分析脚本

可以用下面的脚本汇总 shadow 日志：

```powershell
.\.venv\Scripts\python.exe .\AI_Training\analyze_card_shadow.py --date 2026-05-13 --report
```

默认读取：

```text
RL_Datasets/OptionShadow/card_scorer_YYYY-MM-DD.jsonl
```

并生成：

```text
RL_Datasets/OptionShadow/reports/shadow_report_YYYY-MM-DD.md
```

核心指标包括：

```text
total_card_reward_events
avg_candidate_count
old_vs_scorer_agreement_rate
scorer_disagreed_with_old_policy
scorer_recommended_skip_rate
old_policy_skip_rate
avg_confidence_gap
score_nan_count
score_inf_count
archetype_distribution
archetype_consistency
avg_deck_size
avg_deck_bloat_score
reward_term_distribution
reward_term_nan_count
reward_term_inf_count
```

其中 `confidence_gap = best_score - second_best_score`。后续判断能否打开 active 时，重点看分歧案例里 scorer 是否有足够高的 confidence gap，以及是否存在异常 skip、NaN、inf 或候选数量异常。

每条 shadow JSONL 还会记录可解释字段，便于人工审查分歧案例：

```text
old_policy_action / old_policy_card
scorer_action / scorer_card
template_scores
template_lock
confidence_gap
scorer_disagreed_with_old_policy
skip_score
options[].card_id / name / total_score / score_breakdown / reasons
```

模板选择默认启用 `locked_after_warmup`：

```yaml
template_selection:
  mode: locked_after_warmup
  warmup_card_rewards: 3
  switch_margin: 1.0
  switch_patience: 2
```

也就是说，前 3 次 card reward 允许自由判断构筑；之后锁定主模板。只有另一个模板连续 2 次超过当前模板且分差至少 1.0，才会切换。

## PPO / Rollout 预留字段

PPO v0 仍然保持原训练语义，但 rollout 记录已补齐后续 PPO v2 需要的字段：

```text
policy_version
option_schema
reward_schema
reward_terms
state_features_version
option_features_version
action_index
old_logprob
old_value
done
truncated
episode_id
run_id
screen_type
legal_option_count
```

这些字段现在主要用于兼容和审计。真正的 masked PPO + state-value critic 放到后续阶段。

## 验收方式

本阶段的基础验证命令：

```powershell
.\.venv\Scripts\python.exe -m py_compile AI_Training\state_encoder.py AI_Training\deck_summary.py AI_Training\options\base.py AI_Training\options\combat.py AI_Training\options\cards.py AI_Training\ai_agent.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

当前测试覆盖：

- 空 deck、缺字段 state、未知卡。
- 中文名 / 英文 id 混合。
- 三个默认模板稳定评分。
- `self_damage_rupture` 存在但默认关闭。
- card reward 候选包含所有卡和 `skip`。
- 卡组膨胀时 skip 分数上升。
- 多段攻击在 `strength_multihit` 下评分更高。

## 观察指标

后续启用 active 前，建议先观察：

| 指标 | 期望 |
| --- | --- |
| `legal_option_count` | 每次 card reward 都包含 N 张卡 + skip |
| `skip_recommended` | 大卡组或低价值候选时增加，但不应无脑跳牌 |
| `deck_summary.deck_size` | 长期应比旧策略更稳定 |
| `archetype_consistency` | 卡组应逐渐围绕同一模板成长 |
| `score_distribution` | 不应出现 NaN / inf 或异常极值 |

如果 shadow 日志显示推荐动作稳定、skip 逻辑合理、卡组没有被错误模板带偏，再考虑把 `mode` 从 `shadow` 改成 `active`。
