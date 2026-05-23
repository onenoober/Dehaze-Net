# HAZE4K Model Optimization Workflow Review

日期：2026-05-23

范围：复核当前 DEA-Net HAZE4K 架构尝试、训练测试、结果分析和文档总结流程，判断它是否合理、是否符合主流可靠的模型优化流程，并给出下一轮实验约束。

## 1. 总体结论

当前流程的大方向是合理的，也接近主流可靠的 empirical ML / low-level vision 架构优化流程：

1. 先复现或建立强 baseline。
2. 再做小范围、单假设驱动的架构或 loss 改动。
3. 每个改动先 smoke，再 scout，再决定是否继续。
4. 用同预算曲线与 baseline / 当前最佳候选比较。
5. 对正向候选做 full-test、per-image、固定样本可视化和失败分组分析。
6. 用 manifest / experiment log / cleanup plan 固化证据链。

这套流程已经能避免“看到一个指标就盲目继续训练”的问题，也能把失败实验转化为可复查证据。它不是随机试模型，而是有 baseline、消融、早停、全量评估和 artifact 管理的研究流程。

但它还没有达到强论文结论所需的严格程度。主要短板是：

- 缺少多 seed 或至少 top candidate 重跑，无法估计训练随机性的影响。
- 目前频繁用 HAZE4K test 做路线选择，适合 scouting，但正式论文应锁定 final test，尽量从 train 中划 validation 子集做路线筛选。
- 失败变体中有些一次改动太多，例如 Conservative LF 同时改 channels、center、dropout、gate clamp、gate L2，因果解释不如单因素消融干净。
- 还缺复杂度指标：params、FLOPs、显存、速度、推理尺寸限制。
- 每次实验还缺统一 experiment card，虽然 `EXPERIMENT_LOG` 已经记录了很多事实。

因此，当前流程适合作为毕业论文/工程研究的主线，但下一轮应把流程升级为“实验卡 + 单因素假设 + 早停门槛 + full-test + 复杂度 + top candidate 重跑”。

## 2. 外部方法论依据

本判断参考以下主流规范和去雾研究趋势：

- NeurIPS Paper Checklist Guidelines：强调可复现路径、代码/数据/指令、训练细节、统计显著性和计算资源说明。当前 repo 的 runbook、manifest、experiment log 与这个方向一致，但多 seed / error bar 仍不足。链接：https://nips.cc/public/guides/PaperChecklist
- Pineau et al., JMLR 2021, Improving Reproducibility in Machine Learning Research：强调可复现性本身是验证研究可靠性的必要步骤。当前清理 manifest 和 exact run ID 记录正是必要基础。链接：https://www.jmlr.org/papers/v22/20-303.html
- Lindauer & Hutter, JMLR 2020, Best Practices for Scientific Research on Neural Architecture Search：虽然本项目不是自动 NAS，但其中关于强 baseline、公平预算、清楚消融和避免经验评估偏差的原则适用于手工架构优化。链接：https://jmlr.csail.mit.edu/papers/v21/20-056.html
- DEA-Net 原始工作：DEA-Net 的关键是 DEConv 与 CGA，且强调低复杂度和重参数化后无额外推理成本。后续改动必须保护这个轻量定位。链接：https://arxiv.org/abs/2301.04805
- DehazeFormer：低层视觉去雾中，直接照搬通用 Transformer 设计并不一定合适，需要针对去雾调整 normalization、activation 和空间聚合。它支持“围绕任务特性做小而明确的结构改动”的思路。链接：https://arxiv.org/abs/2204.03883
- CVPR 2025 DehazeXL：大图去雾继续强调 global context 和大范围一致性。它支持本项目探索低频/全局上下文的方向，但也提醒要同时评估大尺度一致性、局部 artifact 和推理成本。链接：https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Tokenize_Image_Patches_Global_Context_Fusion_for_Effective_Haze_Removal_CVPR_2025_paper.pdf

## 3. 当前流程中做得对的部分

| 流程环节 | 当前做法 | 评价 |
| --- | --- | --- |
| baseline 建立 | `DEA-Net-CR-H4K-Baseline-scout-20260520-101334` 100k，记录完整曲线 | 合理。后续所有 scout 都有参照物 |
| 资源基准 | batch-size benchmark 后保留 `bs=16` | 合理。保护训练预算 |
| smoke test | 多个 1-2 step smoke 用于启动验证，之后已清理 | 合理。smoke 不混入论文结果 |
| 100k scout gate | 10k/20k/50k/100k 作为同一条 100k-target run 内部的 stop/continue gate | 合理。适合有限 GPU 预算；禁止把单独短 horizon 当公平比较 |
| 早停 | CRPlus、LowFreqLoss、TeacherGuard、PostMix 按曲线止损 | 合理。避免失败路线消耗完整训练 |
| full-test | LF-v1 对 baseline 做 1000 图 per-image eval | 很关键。纠正了固定样本的负面直觉 |
| 固定样本分析 | visual compare + objective analysis + gate sweep | 合理。补充平均指标无法解释的视觉风险 |
| artifact 管理 | run manifest、cleanup plan、保留策略 | 合理。避免误删和结果混淆 |

这说明流程已经从“试模型”转向“假设驱动的实验系统”。这一步很重要。

## 4. 当前流程中需要修正的部分

### 4.1 不要让 test set 继续承担路线搜索

目前 full per-image evaluation 用的是 HAZE4K test。对当前阶段来说，这是为了理解 LF-v1 的真实收益和风险，可以接受。但如果继续用 test 来决定下一轮结构，最后论文里的 test 结果会有 selection bias。

建议：

- 从 HAZE4K train 中固定划出一个 validation subset，例如 300 张，命名并记录。
- 后续 `10k/20k/50k` stop/continue gate 必须来自同一条公平
  `100k`-target run，并尽量看 validation subset。
- HAZE4K test 只用于阶段性确认和最终正式结果。

如果短期不想改 loader，至少要在论文里诚实说明 HAZE4K test 被用于 scouting，并把最终结论降级为实验性改进而不是强泛化声明。

### 4.2 单因素消融要更严格

`PostMix` 是比较干净的结构消融，因为它只改变 `lf_prior_injection`。相比之下，`Conservative LF` 同时改了多个控制量，虽然结果有价值，但不能说明到底是 channels、dropout、gate clamp 还是 gate L2 导致失败。

建议：

- 下一轮每个 run 只改变一个主机制。
- 如果机制内部有多个超参，先固定为保守默认，只在成功后再调参。
- 不再把多个“看起来都合理”的限制叠在一个实验里。

### 4.3 失败路线应进入方法论，而不是继续堆补丁

当前失败路线共同指向一个结论：LF-v1 的问题不是“低频太强”这么简单，而是“使用低频的时机和位置不够条件化”。

因此不建议继续：

- 单纯减小 scalar gate。
- 继续加 lowfreq L1。
- 继续调当前 hard TeacherGuard。
- 在 `pre_mix/post_mix` 上继续平移而不引入条件选择。

这些已经被当前证据基本排除。

### 4.4 需要补复杂度和可部署性指标

DEA-Net 的原始卖点包括轻量、速度和低复杂度。LF-v1 如果带来小幅 PSNR 提升，但显著增加推理时间或破坏重参数化优势，论文价值会弱。

每个晋级候选应补：

- 参数量。
- FLOPs 或 MACs。
- 512x512 / 1024x1024 推理速度。
- 显存峰值。
- 是否影响 `.pk -> .pth` 导出或官方 `eval.py` 合约。

### 4.5 top candidate 需要至少重跑一次

当前 LF-v1 的 `+0.2030 dB` 是有价值的，但单次训练不足以证明稳定。正式阶段至少应：

- 对 LF-v1 或下一轮 Conditional LF 追加一个不同 seed 的 100k run。
- 如果资源有限，至少对 top candidate 做一次公平 `100k`-target run 的
  `50k` 中途复查，看是否同方向；不要单独启动 50k horizon。
- final claim 只使用经过复查的候选。

## 5. 对已尝试架构的流程审查

| 尝试 | 设计是否合理 | 实验是否干净 | 结论是否可靠 | 下一步 |
| --- | --- | --- | --- | --- |
| LF-v1 | 合理。瓶颈前低频先验符合全局雾幕建模直觉 | 较干净 | 可靠为当前正向候选，但需重跑/复杂度 | 保留，作为所有新 LF 变体参照 |
| Conservative LF | 想法合理，但一次约束过多 | 不够干净 | 可靠地说明这组强约束失败，但不说明哪个约束最坏 | 不继续 |
| CRPlus-P1 lowpass negative | 直觉可理解，但 ratio loss denominator 被扰动 | 较干净 | 10k 明显失败，结论可靠 | 不继续同设计 |
| LowFreqLoss | 直觉可理解，但与 L1 重叠 | 干净 | 20k 失败可信 | 不继续单独路线 |
| LF + LowFreqLoss | 检验“LF+低频损失”是否互补 | 干净 | 50k 失败可信 | 不继续 |
| TeacherGuard | 方向可理解，但当前设置太硬 | 中等；10k 已落后，不能全归因于 guard | 当前设置失败可靠，teacher 思路未完全排除 | 不继续当前参数 |
| PostMix | 非常干净的结构消融 | 干净 | 50k 失败可信 | 不继续 |

## 6. 推荐的下一轮标准流程

每个新模型尝试都应使用同一个 experiment card：

```text
Run ID:
Branch / commit:
Hypothesis:
Single changed mechanism:
Control run:
Dataset split:
Training budget:
Stop gate:
Metrics:
Artifacts:
Decision:
```

推荐 gate：

1. `dry_run`：检查参数、目录、checkpoint、teacher 路径。
2. `2-step smoke`：只验证 train/eval 能跑，不保留为结果。
3. `10k gate`：只能来自同一条公平 `100k`-target run；若比 baseline 10k 低 `>0.5 dB`，直接停止。
4. `20k gate`：只能来自同一条公平 `100k`-target run；若低于 baseline 与 LF-v1 同步曲线，且没有明显视觉/分组优势，停止。
5. `50k gate`：只能来自同一条公平 `100k`-target run；只有接近或超过 LF-v1 曲线才继续。
6. `100k`：只给 baseline、LF-v1、最终候选或强正向候选。
7. `full-test + per-image`：只给晋级候选，不给所有失败尝试。
8. `complexity report`：晋级候选必须补。
9. `rerun`：最终候选至少一次复查。

## 7. 下一轮推荐方向

推荐下一轮不是继续 teacher / lowfreq loss，而是 Conditional LF：

- 主假设：LF-v1 对 baseline 弱样本有帮助，但对强样本和局部区域过度介入；因此需要空间/样本条件化，而不是全局 scalar gate。
- 最小结构：保留 LF-v1 的 `pre_mix` 插入点，新增一个轻量 mask/gate head，输入可以是 bottleneck feature、低通 hazy 或二者拼接，输出 spatial mask。
- 初始化：scalar gate 仍保持 LF-v1 的 near-identity 风格；Conditional LF 的
  spatial mask 第一版应接近 LF-v1 行为，而不是接近 0，避免重复 Conservative
  LF 把收益也关掉的问题。
- 正则：第一版不加 sparsity / TV / lowfreq L1。只有当 mask 统计显示长期接近
  全 1 且指标没有改善时，才考虑轻量 regularization。
- 对照：baseline、LF-v1、Conditional LF。不要同时叠 teacher guard。
- 成功标准：至少 50k 不低于 LF-v1 同步曲线；full-test 平均 PSNR 高于 LF-v1 或回退样本显著减少；固定 20 样本不再扩大负面视觉风险。

这条路线比继续调全局 gate 更符合当前证据，也更接近主流架构优化里的“从失败诊断中提出可检验的单一机制”。

## 8. 论文/毕业设计角度判断

当前流程已经足够支撑毕业论文中的“系统性实验探索”：

- 有官方 checkpoint 评测。
- 有 HAZE4K baseline scout。
- 有正向 LF-v1。
- 有多条失败消融和原因解释。
- 有全量 per-image 结果和固定样本视觉风险分析。
- 有清理后的 artifact manifest。

但如果要形成更稳的论文贡献，最终还需要补：

1. final candidate 的 100k 或更完整训练。
2. 至少一次复现实验或 seed 检查。
3. 参数量 / FLOPs / 推理速度。
4. 定量表格 + 可视化 + 失败案例。
5. 明确声明哪些结果是 scout，哪些是 formal。

## 9. 最终建议

当前模型优化流程可以继续使用，但要升级为更严格的版本：

- 继续保留“baseline -> small hypothesis -> smoke -> scout -> gate -> full eval -> diagnosis -> manifest”的骨架。
- 停止堆叠多个补丁式约束。
- 下一轮只做 Conditional LF 这一条主线。
- 从现在开始，每个新实验先写 experiment card，再启动训练。
- 对最终候选补复杂度和复现，不只看单次 PSNR。

一句话：流程方向是主流、可靠、可行的；当前证据链比普通毕业项目已经扎实，但如果要让结论更硬，必须补上 validation split、多 seed/复查和复杂度报告。
