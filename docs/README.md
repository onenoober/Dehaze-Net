# Documentation Index

用途：作为后续对话和人工查看的最小入口。默认只读少量入口文档，
再按任务加载一个或少数几个证据文档，避免把当前状态、命令、指标、
artifact 和方法分析混在一起。

本文档使用 UTF-8。Windows PowerShell 中建议用
`Get-Content -Encoding UTF8` 读取中文文档。

## Default Read Path

每次继续 HAZE4K/DEA-Net 工作时，默认只读：

1. `docs/CURRENT_CONTEXT.md`：当前可执行状态、硬约束、默认服务器、
   禁止事项和下一步入口。
2. `docs/README.md`：本文档，决定是否需要继续加载更深文档。

然后按任务选择：

| 任务 | 读取文档 | 不应放入这里的内容 |
| --- | --- | --- |
| 精确命令、GitHub/SSH/tmux/检查/暂停/恢复模板 | `docs/WORKFLOW.md` | 长篇实验解释、完整指标历史、服务器环境事实 |
| 评估、可视化、诊断脚本命令 | `docs/ANALYSIS_COMMANDS.md` | 训练启动模板、run 状态、路线结论 |
| 服务器硬件、Python 环境、数据集 symlink、恢复说明 | `docs/CORE_SERVER_RUNBOOK.md` | 方法结论、论文路线判断、单次 run 流水账 |
| 训练结果、checkpoint、停止原因、时间线 | `docs/EXPERIMENT_LOG.md` | 方法论推理、清理策略、命令模板 |
| artifact 保留/删除/瘦身判断 | `docs/HAZE4K_RUN_MANIFEST.md` | 新路线设计、训练命令模板、完整实验解释 |
| 新模型/损失/selector/mask/guard 修改协议和实验卡模板 | `docs/HAZE4K_MODEL_CHANGE_PROTOCOL.md` | 单个 run 的详细日志、artifact 清理 |
| 主路线、论文实验设计、阶段计划 | `docs/DEA_NET_LFCR_HAZE4K_PLAN.md` | 每次运行的完整流水账、单次故障排查 |
| 下载来源和复现基础说明 | `docs/DOWNLOADS.md`, `docs/REPRODUCTION.md` | 当前 HAZE4K 路线决策 |

## Research Evidence Map

这些文档保留为证据链，不需要每次都读。

| 主题 | 首选入口 | 说明 |
| --- | --- | --- |
| 多轮失败原因和早期流程复盘 | `docs/HAZE4K_FAILURE_ANALYSIS_20260523.md` | 解释 LF-v1 之前和初期失败路线的根因；其中“下一步”建议可能已被后续实验更新。 |
| 优化流程纪律和论文可靠性 | `docs/HAZE4K_OPTIMIZATION_WORKFLOW_REVIEW_20260523.md` | 流程审查、validation split、多 seed、复杂度指标等方法论建议。 |
| Conditional LF 路线 | `docs/HAZE4K_CONDITIONAL_LF_ROUTE_AUDIT_20260523.md` | 立项理由和实验卡；实际 run 已失败，当前状态以 `EXPERIMENT_LOG.md` 为准。 |
| LF-v2 Haze-Aware Mask | `docs/HAZE4K_LF_V2_HAZE_AWARE_MASK_PLAN_20260524.md` | 实验卡和 30k 失败结果。 |
| LF-v1 residual 根因 | `docs/HAZE4K_LF_RESIDUAL_DIRECTION_DIAGNOSIS_20260524.md` | LF-v1 低频 residual 方向/幅度诊断，是后续 ResidualCalib 与 loss 路线的根证据。 |
| ResidualCalib | `docs/HAZE4K_LF_RESIDUAL_CALIBRATION_PLAN_20260525.md` | 实验卡和结果；正向消融但不替代 LF-v1。 |
| 三模型输出分析 | `docs/HAZE4K_THREE_WAY_OUTPUT_ANALYSIS_20260525.md` | baseline/LF-v1/ResidualCalib 的 full-test pattern 分析。 |
| 下一路线审查和 selector 立项 | `docs/HAZE4K_NEXT_ROUTE_REVIEW_20260525.md`, `docs/HAZE4K_LF_RESIDUAL_SELECTOR_PLAN_20260525.md` | 历史路线依据；后续 selector 失败和 proxy audit 已关闭该方向。 |
| Selector 最终结论 | `docs/HAZE4K_SELECTOR_EVIDENCE_CLOSURE_20260526.md` | selector-v2 当前总入口；strict/rich/activation audit 是细节证据。 |
| Selector 细节证据 | `docs/HAZE4K_SELECTOR_PROXY_AUDIT_20260526.md`, `docs/HAZE4K_RICH_SELECTOR_PROXY_AUDIT_20260526.md`, `docs/HAZE4K_ACTIVATION_SELECTOR_PROXY_AUDIT_20260526.md` | 已由 closure 汇总；只在需要复查 proxy 特征、split、pass line 时加载。 |
| ResidualDirLoss | `docs/HAZE4K_RESIDUAL_DIRECTION_LOSS_SCALE_PLAN_20260526.md` | scale 诊断、实现验证和 30k 失败结果。 |
| CRPlus-v2 | `docs/HAZE4K_CRPLUS_V2_FREQ_CURRICULUM_PLAN_20260526.md` | 当前 CRPlus-v2 路线卡、scale 诊断、实现验证和 gates。 |
| LFCR-v1 | `docs/HAZE4K_LFCR_V1_COMBINATION_PLAN_20260527.md` | LF-v1 + CRPlus-v2 组合路线卡、权重选择规则、机制指标和 gates。 |
| 历史清理记录 | `docs/archive/HAZE4K_CLEANUP_PLAN_20260523.md` | 仅用于追溯 2026-05-23 清理过程；日常 artifact 判断看 manifest。 |

## Authority Rules

同一事实只能有一个权威来源；其他文档只做短摘要和引用。

| 事实类型 | 权威来源 | 允许的重复方式 |
| --- | --- | --- |
| 当前活跃路线、默认服务器、是否可同步 | `CURRENT_CONTEXT.md` | 其他文档只写“以 CURRENT_CONTEXT 为准”，或注明记录时状态。 |
| 服务器环境、硬件、数据布局、恢复步骤 | `CORE_SERVER_RUNBOOK.md` | `WORKFLOW.md` 可引用路径和健康检查命令，不重复完整环境说明。 |
| 可复制命令、tmux/SSH/恢复模板 | `WORKFLOW.md` | 其他文档只给脚本名、命令名或入口链接。 |
| 评估、可视化、诊断脚本命令 | `ANALYSIS_COMMANDS.md` | route card 可说明需要做哪类分析，但不内嵌长命令。 |
| 每个 run 的指标、commit、停止/恢复状态 | `EXPERIMENT_LOG.md` | 主计划只保留关键结论；route card 只保留与该路线有关的结果。 |
| artifact 保留/删除策略和路径索引 | `HAZE4K_RUN_MANIFEST.md` | 日志只记录产物路径，不重复清理规则。 |
| 方法假设、阶段路线、论文证据链 | `DEA_NET_LFCR_HAZE4K_PLAN.md` | 专题审查可保存更细论证，不能变成总入口。 |
| 新路线证据链和 gates | `HAZE4K_MODEL_CHANGE_PROTOCOL.md` 加对应 route card | `CURRENT_CONTEXT.md` 只保留是否可继续和下一步入口。 |

## Model-Change Loop

后续新增或更新文档时，按这个闭环归档：

1. **模型修改协议**：任何长程 HAZE4K model/loss/selector/mask/guard scout
   前，先读 `docs/HAZE4K_MODEL_CHANGE_PROTOCOL.md`，写清 failure mode、
   mechanism hypothesis、route-specific mechanism metrics 和 matched gate
   rules。PSNR/SSIM 是全局护栏，不是唯一决策证据。
2. **路线立项**：写清 hypothesis、单一变量、风险和 stop gates。路线专属内容
   放到 dated route card；长期主线只同步到
   `DEA_NET_LFCR_HAZE4K_PLAN.md` 的摘要和决策。
3. **实现和 smoke**：代码改动保持在 feature branch。远程/Git/tmux/训练启动模板写
   `WORKFLOW.md`；评估、可视化和诊断脚本命令写 `ANALYSIS_COMMANDS.md`；
   smoke 只作为入口验证，不能进入候选指标表。
4. **公平训练**：正式 HAZE4K candidate 必须从启动时就是 `100000`
   total steps。`10k`、`20k`、`30k`、`50k` 只是同一条 run 内部的
   stop/continue gate。
5. **训练结果记录**：run id、branch/commit、配置、checkpoint、metric、
   停止原因写 `EXPERIMENT_LOG.md`。
6. **评估和可视化**：per-image、fixed-sample、mask 统计、视觉分析产物写入
   对应 artifact 目录，并在 `HAZE4K_RUN_MANIFEST.md` 建索引。
7. **结果解释**：成功路线更新主计划；失败路线进入 failure analysis 或 route
   card 的失败分叉，不继续堆补丁。
8. **清理整理**：是否保留/删除 checkpoint 和日志，先看
   `HAZE4K_RUN_MANIFEST.md`；需要过程追溯时再看
   `docs/archive/HAZE4K_CLEANUP_PLAN_20260523.md`。

## Writing Rules

- 修改文档前先在本文件的分类表中选择目标文档。不能确定分类时，先更新
  `CURRENT_CONTEXT.md` 的最小入口或在当前对话中说明不确定点，不要把内容临时
  堆到最近打开的文档里。
- 新事实只写入一个权威来源。其他文档最多保留一句摘要、入口链接或
  “以 X 为准”的提示。
- 新增当前状态只改 `CURRENT_CONTEXT.md`；新增历史 run 事实只改
  `EXPERIMENT_LOG.md`；新增 artifact 路径/保留判断只改
  `HAZE4K_RUN_MANIFEST.md`。
- `CURRENT_CONTEXT.md` 只保留当前可执行上下文，不放完整历史。
- `WORKFLOW.md` 只放可复用命令和操作边界，不解释大段方法论。
- `ANALYSIS_COMMANDS.md` 只放评估、可视化、诊断脚本命令；不放训练启动、
  SSH/Git/tmux 模板或 run 结论。
- `CORE_SERVER_RUNBOOK.md` 只放服务器环境、数据布局和恢复事实，不记录实验结论。
- `EXPERIMENT_LOG.md` 是指标和决策的时间账本；新增 run 必须有 run id、
  branch/commit、配置、metric 和结论。
- `HAZE4K_RUN_MANIFEST.md` 是 artifact 索引；删除前必须能在这里找到保留/
  删除理由。
- `DEA_NET_LFCR_HAZE4K_PLAN.md` 保持主线计划、阶段路线、论文证据链和晋级规则；
  不要塞入完整 run 流水账。
- Dated analysis docs 用于一次路线或一次流程审查，后续只在该主题仍然相关时
  追加；不要把它变成新的总入口。若文档中的建议被后续结果替代，必须在顶部加
  `Status:`，说明它是历史证据、失败结果或当前路线卡。
- 本地只做编码、文档、Git 和轻量静态检查；dry-run、smoke、训练、
  benchmark、evaluation 属于云端 CUDA 环境。只有用户明确要求时才同步或操作服务器。
- 不为减少文件数量而删除有证据价值的文档；优先通过索引、职责边界和精炼
  交叉引用来降低上下文成本。
