# ModelZoo

这里放可以提交到 GitHub、并能在控制台里直接切换的推理模型包。

## 当前模型包

| 模型包 | 状态 | 说明 |
| --- | --- | --- |
| `demo_local_20260430` | 可用 | 早期演示模型：战斗 BC 642 样本、候选动作 2948 行、宏观 BC 528 样本。 |

这个模型包只是 demo baseline，不代表稳定通关能力。它的作用是让别人拉取仓库后能直接试用控制台和 AI 流程。

## 目录结构

每个模型包保持和运行目录一致的结构：

```text
ModelZoo/<model_id>/
  manifest.json
  ProcessedParams/
    bc_model_best.pth
    candidate_bc_model_best.pth
    vocab.json
    metadata.json
    candidate_metadata.json
  ProcessedMacroParams/
    macro_bc_model_best.pth
    vocab.json
    metadata.json
    training_summary.json
```

## 如何切换

1. 打开网页控制台。
2. 打开“AI 模型状态”。
3. 在“训练模型切换”里选择模型包。
4. 点击“切换到选中模型”。
5. 重启 AI。

切换会把模型包复制到当前运行目录：

- `AI_Training/ProcessedParams/`
- `AI_Training/ProcessedMacroParams/`

AI 进程只在启动时加载模型，所以切换后必须重启 AI。

## 提交新模型包

新模型包至少要附带：

- 模型文件。
- 对应 vocab。
- metadata / training_summary。
- `manifest.json`，说明训练日期、样本规模、数据来源和已知问题。

不要把原始数据、训练矩阵、API Key 或本机配置放进 ModelZoo。
