# 怪物数据与画像库

怪物数据不直接改写原始采集日志。原始日志仍然是事实来源，怪物画像由脚本从原始日志派生生成。

## 目录

```text
RL_Datasets/
  action_logs_*.jsonl
  Human/Combat/*.jsonl
  AI/Combat/*.jsonl
  AI_Combat/*.jsonl
  LLM_Actions/*.jsonl
  Monster/
    monster_turns_YYYY-MM-DD.jsonl
    encounters_YYYY-MM-DD.jsonl
    monster_profiles.json
    encounter_profiles.json
    monster_build_summary.json
    README_MONSTER_DATA.md
  Processed/
    monster_vocab.json
```

`RL_Datasets/Monster/` 是派生数据目录，可以重新生成。`RL_Datasets/Processed/monster_vocab.json` 给后续训练和候选动作打分使用。

## 生成方式

```powershell
python AI_Training\monster_profile_builder.py
```

测试少量文件时：

```powershell
python AI_Training\monster_profile_builder.py --limit-files 3
```

网页里的“重构数据加重练”流程会先生成怪物画像，再继续重构战斗/宏观训练数据。

## 文件含义

`monster_turns_YYYY-MM-DD.jsonl`：每一行表示一次“面对某个怪物当前状态时”的观察，包含当前怪物、意图、我方血量/格挡/能量、手牌、药水、当时采取的动作。

`encounters_YYYY-MM-DD.jsonl`：每一行表示一场战斗摘要，包含怪物组合、回合数、掉血、出牌、药水使用和结果。

`monster_profiles.json`：按怪物聚合后的画像，例如常见意图、威胁标签、平均掉血、常见应对动作。

`encounter_profiles.json`：按怪物组合聚合后的画像，例如多怪组合、精英组合、Boss 组合的平均回合数和掉血。

`monster_vocab.json`：怪物、意图、威胁标签、战斗组合的词表，后面给模型编码用。

## 设计原则

- 原始日志不变，画像可以随时重算。
- 怪物 ID 会做标准化，`NIBBIT_0`、`NIBBIT_1` 会归到 `NIBBIT`。
- 不记录未来抽牌顺序，只记录当前可见状态和派生统计。
- 画像先服务分析和 LLM 提示，后续再接入模型特征。
