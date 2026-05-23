# HAZE4K Conditional LF Route Audit

日期：2026-05-23

目的：在继续训练前，对下一轮最稳路线 `Conditional LF` 做立项核查，尽量降低无效训练概率。本文只做路线审查、实现约束和实验卡，不启动新训练。

## 1. 结论

`Conditional LF` 是当前证据下最稳、最可行、最值得继续的一条路线，但不能保证必然正收益。它的可靠性来自三点：

1. `LF-v1` 已有同长度 baseline 上的全量正收益：1000 张 HAZE4K test 上 mean delta `+0.2030 dB`，说明低频先验方向不是无效方向。
2. 已失败路线共同排除了更粗暴的补救手段：减小/压硬 LF、加 low-frequency L1、加当前强 teacher guard、移动到 `post_mix` 都没有保留 LF-v1 收益。
3. 失败诊断指向同一个机制缺陷：当前 LF-v1 是全局 scalar gate，无法决定“何时、何地、对哪些区域使用低频 residual”。

因此，下一轮不应继续堆 loss 或继续调全局 gate，而应做最小条件化：保留 LF-v1 的 `pre_mix` 插入点，只给 LF residual 增加一个轻量空间 mask。

推荐立项，但必须加硬性止损：

- 先 `dry_run + 2-step smoke`。
- `10k` 只看异常和明显崩溃。
- `20k` 必须接近 baseline/LF-v1 同步曲线。
- `50k` 若低于 baseline 50k `31.2384`，或明显低于 LF-v1 50k `31.3419` 且没有回退样本改善，停止。
- 只有 50k 接近或超过 LF-v1，才继续 100k。

## 2. 为什么不是继续其他路线

| 路线 | 当前证据 | 是否继续 |
| --- | --- | --- |
| Conservative LF | 100k `32.1083 / 0.9843`，低于 baseline 和 LF-v1；固定样本更差 | 不继续 |
| CRPlus-P1 lowpass negative | 10k `24.9623 / 0.9504`，明显低于 baseline 10k | 不继续 |
| LowFreqLoss | 20k 低于 baseline；LF+LowFreqLoss 到 50k 仍低于 baseline 和 LF-v1 | 不继续 |
| TeacherGuard | 当前 `0.05 + warmup 20k + max 2.0` 20k 失败，且容易拉回 baseline 局部解 | 不继续当前设置 |
| PostMix | 20k 短暂好看，50k 掉到 `30.7103 / 0.9814` | 不继续 |
| 继续调 scalar gate | gate sweep 置零到原始 gate 都无法解决固定样本退化 | 不继续 |
| Conditional LF | 针对 LF-v1 的核心缺陷：全局 gate 不具备区域选择性 | 推荐 |

## 3. 现有代码插入点核查

当前 `code/model/backbone_train.py` 中 LF-v1 路径为：

```text
hazy input
  -> avg_pool2d / interpolate
  -> adapter
  -> scalar gate * prior
  -> x8 + residual
  -> mix1
```

具体位置：

```text
x8 = self.level3_block8(x7)
if self.lf_prior is not None and self.lf_prior_injection == 'pre_mix':
    x8 = self.lf_prior(hazy, x8)
x_level3_mix = self.mix1(x_down3, x8)
```

这个位置必须保留，因为：

- LF-v1 的正收益来自 `pre_mix`，不是 `post_mix`。
- `post_mix` 已经 50k 失败，说明把 LF 从 CGA 融合前移走会丢失有效作用点。
- 继续在浅层或多层加入 LF 会扩大变量数量，风险高于当前需要。

## 4. 最小实现方案

推荐只改 `LowFrequencyPrior`，不改训练损失。

### 4.1 结构

```text
low = avg_pool2d(hazy)
low = interpolate(low, target_size)
prior = adapter(low)
mask = sigmoid(mask_head(concat_or_low_input))
out = target + scalar_gate * mask * prior
```

第一版推荐：

- mask shape：`B,1,H,W`，不要做 `B,C,H,W`。
- mask 输入：优先只用 low-pass hazy 的 3 通道特征，或复用 adapter 中间特征；不要拼接 `x8`，避免 mask 与主干特征强共适配。
- mask head：`Conv3x3 -> ReLU -> Conv1x1 -> Sigmoid`。
- mask 初始化：接近常数 1 或接近温和初值，不要接近 0。

### 4.2 为什么 mask 不建议初始化为 0

上一份流程审查里提过“mask/gate 接近 0”是为了保护 baseline，但进一步核查后，Conditional LF 第一版更适合接近 LF-v1，而不是接近 baseline：

- LF-v1 已经是正向候选，下一轮目标是减少回退，不是重新证明 LF 是否有用。
- 若 mask 初始接近 0，会把模型推回 baseline 路线，可能重演 Conservative LF “收益也被关掉”的问题。
- 更稳的是保留 scalar gate 的 near-zero identity 初始化，同时让 mask 初始接近 1，使新结构早期近似 LF-v1；训练再学习局部抑制。

推荐初始化：

- `mask_head` 最后一层 weight 置 0。
- bias 置 `2.0`，使 sigmoid 约 `0.88`；或 bias 置 `1.5`，sigmoid 约 `0.82`。
- 不加 mask sparsity loss 的第一版，避免又变成硬约束。

### 4.3 参数和复杂度控制

如果复用 `adapter_channels=8`，额外参数约为：

- `Conv3x3 3->8`：约 `224` 参数。
- `Conv1x1 8->1`：约 `9` 参数。

这几乎不会改变 DEA-Net 的轻量定位。若复用 adapter 中间特征，额外只需要 `Conv1x1 8->1`。

第一版建议优先复用 adapter 中间特征，这样更轻，且 mask 与 prior 来自同一低频输入。

## 5. 可靠性判断

### 5.1 正向收益概率

相对其他候选，Conditional LF 的正向概率最高，因为它不是从零开始的新方向，而是在唯一正向候选 LF-v1 上修补已定位的缺陷。

有利证据：

- LF-v1 full-test mean delta 为正。
- LF-v1 对 baseline 最弱四分位提升 `+0.4921 dB`，次弱四分位提升 `+0.3946 dB`。
- 回退主要出现在 baseline 已强的样本，符合“过度介入”假设。
- 全局 gate sweep 无法解决，说明需要训练期条件选择。

不利风险：

- mask 可能学成全 1，退化为 LF-v1。
- mask 可能学成全低值，退化为 baseline / Conservative LF。
- 如果没有 validation split，继续用 test 做路线筛选会引入选择偏差。
- 如果 mask 只看 low-pass hazy，未必能判断 baseline 是否已经处理好。

综合判断：值得做，但必须把 50k 作为硬 gate，不直接承诺 100k。

### 5.2 与主流研究方向的一致性

该路线符合当前去雾架构趋势：

- DEA-Net 本身强调 detail-enhanced convolution 和 content-guided attention，说明局部细节和内容选择很重要。
- DehazeFormer 等去雾 Transformer 工作表明，去雾任务需要针对低层视觉特性定制结构，而不是盲目扩大通用模块。
- CVPR 2025 DehazeXL 强调大图去雾中的 global context，支持低频/全局上下文方向；但它也说明全局信息需要和局部结构融合，不能粗粒度全局注入。

Conditional LF 的设计正好是：保留低频全局趋势，但通过空间 mask 做局部选择。

## 6. 必须避免的实现陷阱

1. 第一版不要同时加 teacher guard、lowfreq loss、CRPlus 或新的 regularization。
2. 不要把 mask 做成多层大模块，否则复杂度和变量都增加。
3. 不要改变 `lf_prior_injection=pre_mix`。
4. 不要在 10k 结果稍差时立刻放弃，除非明显崩溃；LF-v1 和 baseline 曲线早期都有波动。
5. 不要用固定 20 张样本均值替代 full-test 或分层样本判断。
6. 不要把 HAZE4K test 继续当成无限调参集；若时间允许，应先从 train 划 validation subset。

## 7. 实验卡

```text
Run ID:
  DEA-Net-LF-ConditionalMask-H4K-scout-20260523

Branch:
  codex/haze4k-conditional-lf

Hypothesis:
  LF-v1 的正收益来自困难样本补偿，负收益来自全局 scalar gate 对强 baseline 样本和局部区域过度介入。
  空间条件 mask 可以保留 LF-v1 的困难样本收益，同时减少强样本/局部区域回退。

Single changed mechanism:
  在 LowFrequencyPrior 中增加 Bx1xHxW 的低频 mask。
  不改变 CR，不加 teacher，不加 lowfreq loss，不移动 pre_mix 插入点。

Control runs:
  baseline: DEA-Net-CR-H4K-Baseline-scout-20260520-101334
  LF-v1: DEA-Net-LF-H4K-scout-20260521-003100

Training budget:
  bs=16, patch_size=256, epochs=20, iters_per_epoch=5000, max 100k steps.
  checkpoint/eval every 10k.

Stop gates:
  2-step smoke must pass.
  10k: stop only if clearly broken or >0.8 dB below both baseline and LF-v1.
  20k: if below both baseline and LF-v1 by >0.5 dB and no visual/diagnostic gain, stop.
  50k: must be >= baseline 50k 31.2384 and close to LF-v1 50k 31.3419; otherwise stop.
  100k: run only if 50k gate passes.

Primary metrics:
  PSNR, SSIM, per-image delta, better/worse counts, delta bins.

Secondary diagnostics:
  fixed 20-sample compare, baseline weak/strong quartile, worst regressions, mask visualization.

Success:
  50k close to or above LF-v1, and fixed/strong-baseline regressions smaller than LF-v1.
  Full-test mean PSNR >= LF-v1 or slightly lower but with clearly reduced regression count and better visuals.

Failure:
  50k lower than baseline, or mask degenerates and no regression improvement.
```

## 8. Validation split recommendation

最可靠做法是在启动下一轮前先建立固定 validation subset：

- 从 HAZE4K train 中抽 300 张，不参与训练，固定 seed。
- 保持 haze/gt pairing。
- scout gate 主要看该 validation subset。
- HAZE4K test 保留给 full-test 和最终报告。

如果当前代码改动成本过高，可以先继续使用现有 test gate，但必须在文档中标注它是 scouting gate，不把这次 scout 当最终无偏 test 结果。

## 9. 是否“高度可靠可行”

结论分级：

- 工程可行性：高。只改一个小模块，不影响数据加载、训练入口、checkpoint 格式和现有评估工具。
- 训练稳定性：中高。初始化若接近 LF-v1，可避免从零破坏主干；但 mask 仍可能退化。
- 正向收益概率：中高。因为它基于唯一正向候选 LF-v1，而不是新开方向。
- 资源风险：可控。50k gate 足以止损；不应直接跑满 100k。
- 论文价值：高于继续调 loss。它能自然解释为“从低频先验到条件化低频融合”，有清晰消融逻辑。

所以可以继续，但必须先实现最小版本并执行 smoke/early gate。下一步不应启动长训练，而应：

1. 创建 `codex/haze4k-conditional-lf` 分支。
2. 实现 minimal Conditional LF mask。
3. 本地 `compileall`。
4. 同步远端。
5. 远端 `dry_run` 和 `2-step smoke`。
6. smoke 通过后再决定是否启动 10k/20k scout。
