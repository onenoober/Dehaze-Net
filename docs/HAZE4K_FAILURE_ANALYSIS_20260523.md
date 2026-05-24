# DEA-Net HAZE4K 多轮改动失败审查

日期：2026-05-23

范围：本报告审查 `DEA-Net-CR` baseline、`LF-v1`、Conservative LF、CRPlus-P1、LowFreqLoss、LF+LowFreqLoss、TeacherGuard、PostMix 等 HAZE4K scouting 结果。目标不是继续试新模型，而是解释为什么多数尝试失败，并为下一轮实验给出更稳的方向。

本次审查只做只读核验和代码分析，没有启动新的训练。

## 1. 结论先行

1. 不能把所有尝试都归为失败。`LF-v1` 是目前唯一正向候选：全量 1000 张 HAZE4K test 上相对同长度 baseline `best.pk` 提升 `+0.2030 dB` PSNR，SSIM 基本持平。
2. `LF-v1` 的核心问题不是均值无效，而是方差过大：PSNR 最差个案 `-6.2238 dB`，最好个案 `+6.4500 dB`，`>= +0.3 dB` 有 `453` 张，`<= -0.3 dB` 也有 `351` 张。
3. A/beta 雾参数不是主要解释变量。全量评估中 `corr(A, delta PSNR)=-0.0687`，`corr(beta, delta PSNR)=0.0224`。更有解释力的是 baseline 本身是否已经强：baseline 最弱四分位平均 `+0.4921 dB`，最强四分位平均 `-0.0520 dB`。
4. 后续失败变体大多试图压低 `LF-v1` 的回退风险，但采用了过硬或过粗的约束，导致有效自由度也被压掉。
5. 下一轮不应继续简单减小 LF gate、加 lowfreq L1、复用当前 TeacherGuard run 设置，或者移动到 `post_mix`。更值得做的是“条件化 LF”：让模型学习何时、何地、以多大幅度使用低频先验。

## 2. 已核验结果矩阵

| Run | 关键设置 | 最好/停止点 | 相对判断 |
| --- | --- | --- | --- |
| `DEA-Net-CR-H4K-Baseline-scout-20260520-101334` | `bs=16`, `patch=256`, `w_loss_CR=0.1`, 100k step | best step 90k: `32.2255 / 0.9844`; latest 100k: `32.0952 / 0.9844` | 同长度 scouting 基线 |
| `DEA-Net-LF-H4K-scout-20260521-003100` | `use_lf_prior`, channels 8, pool 8, `pre_mix`, scalar gate | best step 90k: `32.4281 / 0.9845`; latest 100k: `32.3857 / 0.9845` | 唯一全量正收益候选 |
| `DEA-Net-LF-Conservative-H4K-scout-20260522-145904` | channels 4, residual center, dropout 0.25, gate clamp 0.02, gate L2 0.01 | best step 100k: `32.1083 / 0.9843` | 低于 baseline best 和 LF-v1 |
| `DEA-Net-CRPlus-P1-w005-H4K-scout-20260523-011100` | CR weight 0.05, negative 加 `hazy_lowpass` | stopped 10k: `24.9623 / 0.9504` | 明显失败 |
| `DEA-Net-LowFreqLoss-w005-H4K-scout-20260523-015600` | baseline + `w_loss_lowfreq=0.05` | stopped 20k: `27.8852 / 0.9716` | 低于 baseline 20k `28.9030 / 0.9713` |
| `DEA-Net-LF-LowFreqLoss-w001-H4K-scout-20260523-031600` | LF-v1 + `w_loss_lowfreq=0.01` | stopped 50k: `31.0707 / 0.9811` | 低于 baseline 50k 和 LF-v1 50k |
| `DEA-Net-LF-TeacherGuard-H4K-scout-20260523-112658` | LF-v1 + frozen baseline guard, warmup 20k, weight 0.05, max 2.0 | stopped 20k: `27.7075 / 0.9704` | 失败，但 10k 已落后，不能完全归因于 guard 生效后 |
| `DEA-Net-LF-PostMix-H4K-scout-20260523-133020` | 与 LF-v1 相同，仅 `pre_mix -> post_mix` | stopped 50k: `30.7103 / 0.9814` | 20k 短暂好看，50k 明显掉队 |

官方 HAZE4K `.pth` 参考仍是 `PSNR3426_SSIM9885.pth` 的 full eval：`34.2556 / 0.9885`。所以当前 scouting 结果只适合同长度相互比较，不能直接声称已达到官方完整训练水平。

## 3. LF-v1 到底好在哪里，坏在哪里

`LF-v1` 的设计位置在 `code/model/backbone_train.py`：

- `LowFrequencyPrior` 从 hazy 输入做 `avg_pool2d -> interpolate -> adapter`。
- 低频 residual 通过一个全局 scalar gate 加到 bottleneck 特征。
- 默认插入点是 `x8 -> mix1` 之前，也就是 CGA 第一次融合前。

这个选择有合理性：DEA-Net 的 `DEConv` 强在局部细节，CGA 强在特征融合，低频雾幕信息确实应该在瓶颈或全局上下文附近补充。

但当前实现有三个结构性限制：

1. gate 是全局标量，不知道某个区域是否已经去雾充分。
2. 低频输入只来自 hazy RGB 的固定低通，缺少对当前主干置信度、局部纹理、天空/远景/背光区域的区分。
3. residual 在 `mix1` 前参与 CGA 融合，训练后主干与 LF 分支会共同适配，因此推理时简单缩放 gate 并不能恢复 baseline 行为。

全量 per-image 诊断支持这个解释：

| 分组 | baseline 平均 PSNR | LF-v1 delta PSNR | 解释 |
| --- | ---: | ---: | --- |
| baseline 最弱 25% | `25.6388` | `+0.4921` | LF 主要帮助 baseline 本来处理不好的样本 |
| baseline 次弱 25% | `30.6086` | `+0.3946` | 仍然明显正收益 |
| baseline 次强 25% | `34.0449` | `-0.0229` | 收益消失 |
| baseline 最强 25% | `38.6090` | `-0.0520` | 有轻微回退 |

这说明 `LF-v1` 像一个“困难样本补偿器”，而不是稳定全局增强器。它对弱 baseline 样本的提升足够真实，但对已经被 baseline 处理很好的样本会过度介入。

固定 20 张样本上的负均值 `-0.2793 dB` 不能否定 LF-v1，因为全量结果为正；但它暴露了很重要的视觉风险：部分样本 delta-E、亮度/饱和度、暗通道偏差变差，且 `384/479/952` 等样本随 gate 增大更差。这个风险需要进入下一轮设计，而不是被全量均值掩盖。

## 4. 为什么后续几轮多数失败

### 4.1 Conservative LF：把问题压小，也把收益压没了

Conservative LF 同时用了：

- `lf_prior_channels=4`
- `lf_prior_residual_center=true`
- `lf_prior_train_dropout=0.25`
- `lf_prior_gate_max=0.02`
- `w_loss_lf_gate=0.01`

结果 learned gate 只有 `-0.001107`，几乎把 LF 分支关掉。100k best `32.1083 / 0.9843` 低于 baseline best `32.2255 / 0.9844`，固定样本也比 LF-v1 更差：Conservative 固定样本均值 `-0.7850 dB`，LF-v1 是 `-0.2793 dB`。

根因：这个变体没有解决“何时使用 LF”的问题，只是降低 LF 的存在感。它削弱了正向困难样本补偿，却没有稳定修复回退样本。

### 4.2 CRPlus-P1：低通 hazy negative 破坏了 CR 的语义

当前 CR 在 `code/loss/cr.py` 中是 VGG feature ratio：

```text
d_ap = L1(VGG(output), VGG(clear))
d_an = L1(VGG(output), VGG(hazy))
loss = d_ap / (d_an + eps)
```

CRPlus-P1 又加入 `lowpass(hazy)` 作为第二个 negative，并与原 hazy negative 等权平均。这个设计看起来“更接近雾退化”，但实际会带来两个问题：

1. `lowpass(hazy)` 不是任务语义上的 hard negative，而是强平滑/低细节版本，可能和 clear 的结构距离、颜色距离都不稳定。
2. ratio loss 对 denominator 很敏感。加一个低通 negative 会改变 CR 的尺度和梯度方向，不一定是“更难”，可能是“更乱”。

结果 10k 只有 `24.9623 / 0.9504`，远低于 baseline 10k `27.1101 / 0.9615`。保留该实验作为失败消融即可，不建议继续同一 negative 设计。

### 4.3 LowFreqLoss：重复约束低频重建，和主任务/细节恢复冲突

`low_frequency_loss` 在 `code/train.py` 中对 `out` 和 `target` 做 avgpool 后 L1。这个约束有两个缺陷：

1. 原始 pixel L1 已经包含低频误差；额外 lowfreq L1 本质上重新加权亮度/颜色的大尺度误差。
2. DEA-Net 的优势来自 DEConv 细节增强和 CGA 融合。过强的低频重建会让优化更保守，降低对细节和局部结构的恢复速度。

结果单独 LowFreqLoss 20k 低于 baseline，LF+LowFreqLoss 到 50k 也低于 baseline 和 LF-v1。这说明“低频先验有用”不等于“低频 L1 有用”。

### 4.4 TeacherGuard：当前结果不能证明 teacher 思路错，但证明当前设置错

TeacherGuard 的代码逻辑是：

- warmup 前不启用；
- warmup 后，如果当前输出比 teacher 更差，则对 `out` 和 `teacher_out` 加权 L1；
- 权重按 batch/patch 归一化，并 clamp 到上限。

这个 run 在 10k 时已经只有 `24.8283 / 0.9425`，而 warmup 是 20k，所以 10k 的落后不能只归因于 teacher loss 生效。可能还有代码版本、随机训练扰动、LF 初期不稳定等因素。

但 20k 后仍只有 `27.7075 / 0.9704`，低于 baseline 20k `28.9030 / 0.9713` 和 LF-v1 20k `28.8563 / 0.9751`。因此当前 `w_loss_teacher_guard=0.05 + warmup=20000 + max_weight=2.0` 这组设置应判定为失败。

更深层原因：它把“不要比 baseline 差”实现成了“靠近 baseline 输出”。这会在困难样本上把 LF 的正向修正拉回 baseline 局部解，尤其当 teacher 本身不是最终最优模型时，约束会过早地削弱新分支探索。

### 4.5 PostMix：避开 CGA 共适配后，LF 也失去关键作用点

PostMix 只把 LF residual 从 `mix1` 前移到 `mix1` 后。它 20k 曾达到 `29.0681 / 0.9740`，略高于 baseline 和 LF-v1 20k，但 50k 掉到 `30.7103 / 0.9814`，低于 baseline 50k `31.2384 / 0.9817` 和 LF-v1 50k `31.3419 / 0.9817`。

这说明 `pre_mix` 的收益确实可能来自参与 CGA 融合，而不只是加一个低频 residual。把 LF 放到 `post_mix` 后虽然减少了对 CGA 的干扰，但也削弱了它帮助深层融合的能力。

## 5. 架构层面的根因

### 5.1 当前 LF 是“低频信息注入”，不是“低频退化建模”

`LF-v1` 的低通来自 hazy 输入的平均池化。它能提供全局色调、亮度、雾幕趋势，但没有显式区分：

- 哪些低频属于 haze；
- 哪些低频属于真实天空、远景、背光或自然光照；
- 哪些区域 baseline 已经足够好。

所以它能帮助一些 baseline 困难样本，却会在 strong-baseline 样本上改变原本已经正确的低频结构。

### 5.2 全局 scalar gate 不足以处理场景选择性

`LF-v1` learned gate 是 `0.033890`，看起来很小，但固定样本 gate sweep 显示：

| scale | effective gate | fixed subset delta PSNR |
| --- | ---: | ---: |
| 0.0 | `0.000000` | `-0.2694` |
| 0.25 | `0.008473` | `-0.2683` |
| 0.5 | `0.016945` | `-0.2695` |
| 0.75 | `0.025418` | `-0.2732` |
| 1.0 | `0.033890` | `-0.2793` |

即使 gate 置零，固定样本仍低于 baseline。这证明失败不只是 residual 幅度，而是训练时主干已经和 LF 分支发生共适配。

### 5.3 损失层面的新约束没有和 DEA-Net 的优势对齐

DEA-Net 原始优势是：

- DEConv 增强细节和泛化；
- CGA 对通道/空间/像素重要性做内容引导融合；
- CR 作为辅助正则已经存在。

后续失败 loss 大多没有利用这一点，而是在输出像素或 VGG ratio 上加粗粒度约束。它们不是告诉模型“哪些区域该用 LF”，而是整体改变优化目标。

## 6. 与近期研究的对照

外部研究给出的方向和当前失败结果基本一致：

1. DEA-Net 原论文强调 DEConv 与 CGA 的轻量特征学习，不是靠简单加深加宽。继续保持主干稳定是合理的。
   - arXiv: https://arxiv.org/abs/2301.04805
   - official repo: https://github.com/cecret3350/DEA-Net
2. TCL-Net (ACCV 2024) 也说明频域信息对轻量去雾有价值，但它同时处理高低频并做融合，而不是只把低频 residual 加到主干。
   - CVF: https://openaccess.thecvf.com/content/ACCV2024/papers/Tang_TCL-Net_A_Lightweight_and_Efficient_Dehazing_Network_with_Frequency-Domain_Fusion_ACCV_2024_paper.pdf
3. DehazeXL (CVPR 2025) 强调 haze removal 同时需要 global context 和 local detail，并提出针对全局上下文利用的分析思路。这支持下一步做区域/样本选择性，而不是全局同强度 LF。
   - arXiv: https://arxiv.org/abs/2504.09621
4. FrDiff (ICCV 2025) 明确从频域角度建模 haze，指出 haze 退化主要体现在 amplitude spectrum，同时也提示 contrastive learning 可能引入与 haze 无关的内容干扰。这与 CRPlus-P1 的失败高度相关：不要把任意低通图都当作有效 negative。
   - arXiv: https://arxiv.org/abs/2507.01275

这些工作不意味着要马上换大模型、扩散模型或 Transformer。对当前毕业路线来说，更稳的做法是借鉴它们的原则：频率建模要可分解、可选择、可解释，而不是再加一个全局低频项。

## 7. 下一轮实验建议

### 7.1 先暂停这些路线

不建议继续：

- Conservative LF 的 `channels=4 + dropout + gate clamp + gate L2` 组合；
- CRPlus-P1 的 `hazy_lowpass` equal-weight negative；
- 单独或组合的 avgpool LowFreqLoss；
- 当前 TeacherGuard run 设置；
- 单点 `post_mix` 结构放大训练预算。

这些路线已有足够失败证据，继续跑完整训练大概率只是消耗预算。

### 7.2 保留 LF-v1，但不要把它直接当最终模型

LF-v1 可以作为论文里的正向消融：

- 同长度 baseline 对比有 `+0.2030 dB`；
- 复杂度增量较小；
- 能证明“低频先验在 DEA-Net 上有价值”。

但它还不够稳，不能只凭均值提升作为最终主模型。必须补充：

- full per-image 表；
- 最好/最坏样本图；
- baseline 强弱分组；
- 固定样本 objective analysis；
- 视觉风险讨论。

### 7.3 下一轮最值得做：条件化 LF mask

推荐的下一个结构方向不是弱化 LF，而是让 LF 有局部选择能力：

```text
hazy low-pass
  + bottleneck feature
  + optional local contrast / dark-channel style cue
  -> tiny confidence adapter
  -> sigmoid mask, initialized near 0
  -> x8 + gate * mask * prior
  -> mix1
```

设计要点：

- 保持 `pre_mix`，因为它是当前唯一正向结构。
- gate 保持 identity-friendly 初始化，但不要只靠一个 scalar。
- mask 可以是 `B,1,H,W`，先不做 `B,C,H,W`，避免参数过多。
- mask 需要可视化保存，用于解释“哪些区域使用了 LF”。
- 第一轮只测结构，不加新的 loss。

通过标准：

- 50k gate 必须来自同一条公平 `100k`-target run；不低于 baseline 50k
  `31.2384`，且最好接近 LF-v1 50k `31.3419`；
- 100k 不低于 LF-v1 的 full-test delta；
- strong-baseline 四分位的负 delta 要收窄；
- 固定 20 样本不应继续大幅负于 baseline。

### 7.4 如果重试 TeacherGuard，必须改成晚期弱约束

Teacher 思路不是完全无效；当前 run 的 10k 劣化发生在 guard loss 启用前，
所以不能把早期失败完全归因于 teacher penalty。若以后重试：

- warmup 至少推到 `50000` step；
- `w_loss_teacher_guard` 从 `0.005` 或 `0.01` 起；
- 不直接拉近 teacher output，而是惩罚 `current_err - teacher_err - margin`；
- 只在 strong-baseline 风险样本或风险区域启用；
- 先做 20/50 张 hard-case 离线诊断，再启动训练。

### 7.5 CRPlus 应换成频域/幅度约束，而不是 lowpass negative

如果继续 CRPlus，更稳的方向是：

- 记录每层 `d_ap/d_an`，先看 ratio 是否尺度失控；
- 用 margin ranking 形式替代简单多 negative ratio；
- negative 用 `prediction.detach()` 的受控退化版本，而不是 `lowpass(hazy)`；
- 或者参考 FrDiff 思路，只在 amplitude residual 上做轻量一致性/排序约束，避免引入内容无关的 VGG negative。

第一轮建议只做离线评估和 loss scale 打印，不直接长训。

## 8. 推荐的近期工作顺序

1. 把本报告、`EXPERIMENT_LOG.md`、`CURRENT_CONTEXT.md` 保持同步。
2. 用已有 full per-image evaluator 扩展一个 `baseline-weak/strong` 固定样本集，不再只用均匀抽样 20 张。
3. 对 LF-v1 的最好/最坏/中性样本做可视化归因：低频 residual、mask 候选、dark-channel/luma/edge error。
4. 只在上述诊断完成后，设计 `LF-v2 ConditionalMask`。
5. 下一轮训练前设置硬 gate：10k/20k/50k 只能来自同一条公平
   `100k`-target run；10k 只看异常，50k 决定是否继续，100k 决定是否进入
   full training。

## 9. 当前可写入论文的经验

可以形成一个清晰故事：

- DEA-Net-CR 是强 baseline。
- 简单 LF-v1 确认低频先验有价值，但它具有场景选择性。
- 过度保守的 LF、简单低频损失、直接低通 negative、当前 TeacherGuard run 都失败，说明低频信息不能被粗粒度地全局加入或硬约束。
- 后续改进应从“是否加入低频”转向“何时何地加入低频”，即条件化、区域化、可解释的低频融合。

这条经验对毕业论文是有价值的：它不是单纯堆实验失败，而是把失败收束成下一步方法设计的依据。
