# HAZE4K Conditional LF Route Audit

日期：2026-05-23

目的：在继续训练前，对下一轮 `Conditional LF` 做立项核查。这里的目标不是做最保守、最小幅度的改动，而是选择单位训练成本信息价值最高的尝试：成功时要有更明显收益，失败时也要能明确告诉下一步该优化什么。本文只做路线审查、实现约束和实验卡，不启动新训练。

## 1. 结论

`Conditional LF` 是当前证据下最有价值、最可控、最值得继续的一条路线，但不能保证必然正收益。它的价值来自三点：

1. `LF-v1` 已有同长度 baseline 上的全量正收益：1000 张 HAZE4K test 上 mean delta `+0.2030 dB`，说明低频先验方向不是无效方向。
2. 已失败路线共同排除了更粗暴的补救手段：减小/压硬 LF、加 low-frequency L1、当前 TeacherGuard run、移动到 `post_mix` 都没有保留 LF-v1 收益。
3. 失败诊断指向同一个机制缺陷：当前 LF-v1 是全局 scalar gate，无法决定“何时、何地、对哪些区域使用低频 residual”。

因此，下一轮不应继续堆 loss 或继续调全局 gate，而应做有诊断价值的条件化：保留 LF-v1 的 `pre_mix` 插入点，给 LF residual 增加一个内容感知的轻量空间 mask。这个 mask 不只是为了“少改一点”，而是为了直接回答核心问题：低频收益能不能通过区域/内容选择被保留并放大。

推荐立项，但必须加硬性止损：

- 先 `dry_run + 2-step smoke`。
- `10k` 只看异常和明显崩溃，且必须来自同一条公平 `100k`-target run。
- `20k` 必须接近 baseline/LF-v1 同步曲线，且必须来自同一条公平 `100k`-target run。
- `50k` 若低于 baseline 50k `31.2384`，或明显低于 LF-v1 50k `31.3419` 且没有回退样本改善，停止；不要单独启动 50k horizon。
- 只有 50k 接近或超过 LF-v1，才让同一条 `100k`-target run 继续跑到 100k。

## 1.1 策略校准：从“最稳”改为“价值-风险平衡”

用户指出，只追求最稳会变成过慢的保守策略。这一点是正确的。下一轮实验不应该只是“最小改动以避免失败”，而应该满足：

1. **收益空间足够大**：如果成功，应有机会超过 LF-v1，而不是只把 LF-v1 的退化稍微变小。
2. **变量仍然可解释**：如果失败，能判断是“条件 mask 机制无效”、还是“mask 输入不够”、还是“初始化/训练预算问题”。
3. **不把失败变成黑箱**：不要一次叠 teacher、CRPlus、lowfreq loss、post_mix、多层 LF 等多个机制。
4. **训练预算可止损**：可以接受公平 `100k`-target run 的 20k/50k gate 失败并提前停止，但不能单独开短 horizon 当正式对比。

因此推荐的是中等力度的高价值尝试，而不是最保守的 low-only mask。

## 2. 为什么不是继续其他路线

| 路线 | 当前证据 | 是否继续 |
| --- | --- | --- |
| Conservative LF | 100k `32.1083 / 0.9843`，低于 baseline 和 LF-v1；固定样本更差 | 不继续 |
| CRPlus-P1 lowpass negative | 10k `24.9623 / 0.9504`，明显低于 baseline 10k | 不继续 |
| LowFreqLoss | 20k 低于 baseline；LF+LowFreqLoss 到 50k 仍低于 baseline 和 LF-v1 | 不继续 |
| TeacherGuard | 当前 `0.05 + warmup 20k + max 2.0` run 在 20k 失败；10k 已落后，不能把早期劣化全归因于 guard loss | 不继续当前 run |
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

## 4. 推荐实现方案：内容感知 Conditional LF

推荐只改 `LowFrequencyPrior`，不改训练损失，但 mask 不应只看低通 hazy。更有价值的第一版应该让 mask 同时看到：

- 低频分支自己的中间特征，表示 haze 的大尺度亮度/颜色/雾幕趋势。
- `x8.detach()` 的轻量内容摘要，表示主干 bottleneck 已经提取到的结构和语义状态。

使用 `detach()` 的目的不是削弱模型，而是控制风险：mask 可以利用主干内容判断哪里需要 LF，但 mask 分支的梯度不反向推动主干为了迎合 mask 而重新共适配。

### 4.1 结构

```text
low = avg_pool2d(hazy)
low = interpolate(low, target_size)
low_feat = low_encoder(low)
target_hint = target_hint_proj(target.detach())
prior = prior_head(low_feat)
mask = sigmoid(mask_head(concat(low_feat, target_hint)))
out = target + scalar_gate * mask * prior
```

第一版推荐：

- mask shape：`B,1,H,W`，不要做 `B,C,H,W`。
- mask 输入：`low_feat + target_hint`，其中 `target_hint` 来自 `target.detach()` 的 `1x1` 投影。
- mask head：`Conv3x3 -> ReLU -> Conv1x1 -> Sigmoid`，输出单通道空间 mask。
- mask 初始化：接近常数 1 或接近温和初值，不要接近 0。
- 第一版不加新的 loss，但必须记录 mask mean/std/min/max，最好能保存少量 mask 可视化。

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

如果使用 `adapter_channels=8`、`target` 通道为 128，额外参数仍很小：

- `target_hint_proj 128->8`：约 `1032` 参数。
- `mask_head Conv3x3 16->8`：约 `1160` 参数。
- `mask_head Conv1x1 8->1`：约 `9` 参数。

总增量约 2k 参数量级，远小于 DEA-Net 主体，不会改变轻量定位。相比 low-only mask，这个版本稍微更大胆，但更有信息价值：如果失败，可以说明“低频 + bottleneck 内容提示”仍不足以稳定 LF；如果成功，则论文贡献也更清晰。

## 4.4 为什么不只做 low-only mask

low-only mask 更稳，但信息价值不够高：

- 它只能根据 hazy 的低频外观判断哪里使用 LF，未必知道主干是否已经处理好。
- LF-v1 的负收益更像“对强 baseline 样本过度介入”，这需要某种主干状态或内容提示。
- 如果 low-only mask 失败，很难判断是条件化方向错了，还是输入信息不够。

因此，第一轮高价值尝试应使用 `low_feat + target.detach()`。这仍然是单机制改动，因为只改 LF prior 的融合方式，没有额外损失和新训练目标。

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
- 如果 `target.detach()` 提示太强，mask 可能学成某种训练集特化的内容选择。

综合判断：值得做，但必须把同一条 `100k`-target run 的 50k checkpoint 作为硬 gate；不通过就停止，不能另开或修改短 horizon。

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
7. 不要只记录 PSNR/SSIM；必须记录 mask 统计，否则失败时无法判断 mask 是否退化。

## 6.1 失败后的信息价值分叉

这次实验即使失败，也必须能回答下一步问题：

| 观察 | 说明 | 下一步 |
| --- | --- | --- |
| mask mean 接近 1，std 很小，指标接近 LF-v1 | 条件化没有真正发生 | 加轻量 mask entropy/sparsity 或降低 mask bias |
| mask mean 接近 0，指标退回 baseline 或低于 baseline | LF 被关掉，重复 Conservative 问题 | 提高 mask bias，或从 LF-v1 gate/mask warm-start |
| mask 有空间变化，但 50k 低于 baseline | 条件输入可能和去雾质量不对齐 | 改 mask 输入，尝试只用 low_feat 或加入 simple confidence proxy |
| 弱 baseline 样本收益保留，强 baseline 回退减少 | 路线成立 | 继续 100k，并做 full per-image + mask 可视化 |
| PSNR 略低于 LF-v1，但回退样本显著减少且视觉更稳 | 可能有论文价值 | 继续到 100k 或 full-test 后决定是否作为稳健版 |

这样设计的价值在于：失败不会只告诉我们“又低了”，而会告诉我们是 mask 没学到、LF 被关掉、输入不够，还是条件化方向本身不成立。

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
  在 LowFrequencyPrior 中增加 Bx1xHxW 的内容感知 LF mask。
  mask 输入为 low-frequency feature + detached bottleneck target hint。
  不改变 CR，不加 teacher，不加 lowfreq loss，不移动 pre_mix 插入点。

Control runs:
  baseline: DEA-Net-CR-H4K-Baseline-scout-20260520-101334
  LF-v1: DEA-Net-LF-H4K-scout-20260521-003100

Training budget:
  bs=16, patch_size=256, epochs=20, iters_per_epoch=5000.
  The run must be launched as a 100k-target run, with checkpoint/eval every 10k.

Stop gates:
  2-step smoke must pass.
  10k: stop only if clearly broken or >0.8 dB below both baseline and LF-v1.
  20k: if below both baseline and LF-v1 by >0.5 dB and no visual/diagnostic gain, stop.
  50k: must be >= baseline 50k 31.2384 and close to LF-v1 50k 31.3419; otherwise stop.
  100k: continue the same 100k-target run to 100k only if 50k gate passes.

Primary metrics:
  PSNR, SSIM, per-image delta, better/worse counts, delta bins.

Secondary diagnostics:
  fixed 20-sample compare, baseline weak/strong quartile, worst regressions, mask visualization.
  mask mean/std/min/max per eval checkpoint.

Success:
  50k close to or above LF-v1, and fixed/strong-baseline regressions smaller than LF-v1.
  Full-test mean PSNR >= LF-v1 or slightly lower but with clearly reduced regression count and better visuals.

Failure:
  50k lower than baseline, or mask degenerates and no regression improvement.
  If mask is near all-1, next step should regularize or reduce target hint.
  If mask is near all-0, next step should increase LF gate/mask bias or warm-start from LF-v1.
  If mask is spatially diverse but metrics fall, condition mechanism is not aligned with dehazing quality.
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
- 训练稳定性：中等偏高。比 low-only mask 稍激进，但通过 `target.detach()` 控制共适配风险。
- 正向收益概率：中高。因为它基于唯一正向候选 LF-v1，并补足 LF-v1 最明显的结构缺陷。
- 资源风险：可控。50k gate 是同一条 `100k`-target run 的止损点；不过关即停止，不另开或修改短 horizon。
- 论文价值：高于继续调 loss，也高于 low-only mask。它能自然解释为“从低频先验到内容感知的条件化低频融合”，有清晰消融逻辑。

所以可以继续，但必须先实现最小版本并执行 smoke/early gate。本审计作为立项依据保留；当前执行状态以 `docs/CURRENT_CONTEXT.md` 和 `docs/EXPERIMENT_LOG.md` 为准。若重新开启同路线，应：

1. 创建 `codex/haze4k-conditional-lf` 分支。
2. 实现 content-aware Conditional LF mask。
3. 本地 `compileall`。
4. 在用户明确要求后再同步远端。
5. 远端 `dry_run` 和 `2-step smoke`。
6. smoke 通过后启动公平 `100k`-target scout，并只把 10k/20k/50k 作为中途 gate。
