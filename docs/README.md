# Documentation Index

用途：给后续对话和人工查看提供按需加载入口。先读少量入口文档，再按问题加载证据文档，避免把历史、命令、指标和方法分析混在一起。

## 读取顺序

每次继续 HAZE4K/DEA-Net 工作时，先读：

1. `docs/CURRENT_CONTEXT.md`：当前状态、硬约束、正在处理的 run。
2. `docs/README.md`：本文档，决定下一步该加载哪个文件。

然后按任务选择：

| 需求 | 读取文档 | 不应放入这里的内容 |
| --- | --- | --- |
| 精确命令、GitHub/SSH/tmux/暂停/恢复模板 | `docs/WORKFLOW.md` | 长篇实验解释、完整指标历史 |
| 训练结果、checkpoint、停止原因、时间线 | `docs/EXPERIMENT_LOG.md` | 方法论推理、清理策略 |
| 远端 artifact 保留/删除/瘦身判断 | `docs/HAZE4K_RUN_MANIFEST.md` | 新路线设计、训练命令模板 |
| 主路线、论文实验设计、阶段计划 | `docs/DEA_NET_LFCR_HAZE4K_PLAN.md` | 每次运行的完整流水账 |
| 已失败路线的原因、架构层诊断 | `docs/HAZE4K_FAILURE_ANALYSIS_20260523.md` | 操作命令、artifact 清单 |
| 模型优化流程是否合理、gate 纪律 | `docs/HAZE4K_OPTIMIZATION_WORKFLOW_REVIEW_20260523.md` | 单个 run 的详细日志 |
| Conditional LF 立项理由、实现约束、实验卡 | `docs/HAZE4K_CONDITIONAL_LF_ROUTE_AUDIT_20260523.md` | 已发生 run 的最新状态 |
| 历史清理过程和删除确认记录 | `docs/HAZE4K_CLEANUP_PLAN_20260523.md` | 日常运行入口 |
| 服务器环境、数据集布局、运行环境恢复 | `docs/CORE_SERVER_RUNBOOK.md` | 当前实验结论 |
| 下载来源和复现基础说明 | `docs/DOWNLOADS.md`, `docs/REPRODUCTION.md` | 当前 HAZE4K 路线决策 |

## 模型优化闭环

后续新增或更新文档时，按这个闭环归档：

1. **路线立项**：写清 hypothesis、单一变量、风险和 stop gates。路线专属内容放到 dated audit；主路线长期计划同步到 `DEA_NET_LFCR_HAZE4K_PLAN.md`。
2. **实现和 smoke**：代码改动保持在 feature branch。命令模板写 `WORKFLOW.md`；smoke 只作为入口验证，不能进入候选指标表。
3. **公平训练**：正式 HAZE4K candidate 必须从启动时就是 `100000` total steps。`10k`、`20k`、`50k` 只是同一条 run 内部的 stop/continue gate。
4. **训练结果记录**：run id、commit、配置、checkpoint、metric、停止原因写 `EXPERIMENT_LOG.md`。
5. **评估和可视化**：per-image、fixed-sample、mask 统计、视觉分析产物写入对应 artifact 目录，并在 `HAZE4K_RUN_MANIFEST.md` 建索引。
6. **结果解释**：成功路线更新主计划；失败路线进入 failure analysis 或 route audit 的失败分叉，不继续堆补丁。
7. **清理整理**：是否保留/删除 checkpoint 和日志，先看 `HAZE4K_RUN_MANIFEST.md`，需要过程追溯时再看 `HAZE4K_CLEANUP_PLAN_20260523.md`。

## 写入规则

- `CURRENT_CONTEXT.md` 只保留当前可执行上下文，不放完整历史。
- `WORKFLOW.md` 只放可复用命令和操作边界，不解释大段方法论。
- `EXPERIMENT_LOG.md` 是指标和决策的时间账本；新增 run 必须有 run id、commit、配置和结论。
- `HAZE4K_RUN_MANIFEST.md` 是 artifact 索引；删除前必须能在这里找到保留/删除理由。
- `DEA_NET_LFCR_HAZE4K_PLAN.md` 保持主线计划，不把每次临时排查都塞进去。
- dated analysis docs 用于一次路线或一次流程审查，后续只在该主题仍然相关时追加。
- 本地只做编码、文档、Git 和轻量静态检查；dry-run、smoke、训练、benchmark、evaluation 属于云端 CUDA 环境。只有用户明确要求时才同步或操作服务器。
- 不为减少文件数量而删除有证据价值的文档；优先通过索引、职责边界和精炼交叉引用来降低上下文成本。
