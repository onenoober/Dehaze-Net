# DEA-Net-LFCR HAZE4K 主线计划

日期：2026-05-25

Status: 主路线/论文计划文档，不是当前 run 状态。当前可执行状态以
`docs/CURRENT_CONTEXT.md` 为准；每次 run 的事实以 `docs/EXPERIMENT_LOG.md`
为准。

用途：记录 DEA-Net 在 HAZE4K 上的主路线、阶段计划、晋级规则和论文证据链。本文不保存完整运行流水账；具体 run 事实以 `docs/EXPERIMENT_LOG.md` 为准，artifact 保留策略以 `docs/HAZE4K_RUN_MANIFEST.md` 为准，当前可执行状态以 `docs/CURRENT_CONTEXT.md` 为准。

## 1. 文档边界

本文件只回答四类问题：

1. 为什么当前主线仍是 DEA-Net / LFCR。
2. 每个阶段要验证什么假设。
3. 什么结果可以晋级、停止或回退。
4. 最终论文需要哪些训练、评估、测试、结果分析和总结证据。

不要把以下内容写入本文件：

- 单次 run 的完整日志、tmux 名称、停止命令和长指标表；写入 `EXPERIMENT_LOG.md`。
- 远端目录清理、保留和删除判断；写入 `HAZE4K_RUN_MANIFEST.md`。
- 可复制 SSH / Git / tmux / resume 命令模板；写入 `WORKFLOW.md`。
- 单条失败路线的详细架构诊断；写入 dated analysis docs。

## 2. 总目标

以 HAZE4K 为主数据集，以当前已复现的 DEA-Net-CR 为强基线，围绕轻量、可复现、可消融、可写入毕业论文的原则推进改进模型。后续候选暂按 **DEA-Net-LFCR** 组织：

- **LF**：低频/频域退化先验，用于补充 DEA-Net 对全局雾幕、亮度和低频退化的建模。
- **CRPlus**：在现有 DEA-Net-CR 对比正则上的任务相关改进，保持为训练损失层面的独立变量。
- **LFCR**：仅当 LF 与 CRPlus 至少一个方向通过公平 scout 后，再考虑组合。

预期论文证据链：

- HAZE4K 上的 DEA-Net-CR baseline 可复现。
- 至少一个轻量改动带来可测量收益，或形成有充分证据的失败分析。
- 最终候选在 PSNR/SSIM 外补充 LPIPS、per-image、视觉对比和复杂度。
- 失败路线能解释为什么停止，而不是只作为零散试错记录。

## 3. 核心假设

DEA-Net 的主要优势来自 DEConv 的局部细节建模和 CGA 的内容引导融合。后续不优先重写主干，而是围绕以下位置做轻量改造：

1. bottleneck / `mix1` 附近的低频或频域先验。
2. 训练期对比正则的 task-aware negative 或频域约束。
3. 测试和分析阶段的 per-image、可视化和复杂度证据。

当前原则：

> 保持 DEConv 和 CGA 主体稳定，在 bottleneck、fusion 和 loss 附近做单假设、小步验证。

第一阶段不引入大型 Transformer、Mamba、多任务框架或深层双分支重写。

## 4. 权威运行规则

正式 HAZE4K candidate/scout 必须从启动时就是同一套 `100000` step 目标：

```text
epochs=20
iters_per_epoch=5000
total steps=100000
bs=16
patch_size=256
w_loss_L1=1.0
w_loss_CR=0.1
start_lr=0.0001
end_lr=0.000001
checkpoint_interval_steps=10000
eval_interval_steps=10000
save_epoch_checkpoints=false
```

`10k`、`20k`、`50k` 只是同一条 `100k` run 内部的 stop/continue gate。单独启动 `epochs=4`、`epochs=10`、`T=20000`、`T=50000`，或 resume 时改变 `epochs * iters_per_epoch`，只能作为 smoke/诊断/误启动证据，不能进入候选公平对比表。

Exact command templates belong in `docs/WORKFLOW.md`。本文件只保留上述规则和阶段判断。

## 5. 数据、入口和 checkpoint 合约

主数据集：

```text
dataset/HAZE4K/
  train/{hazy,clear}
  test/{hazy,clear}
```

服务器实际路径和 symlink 规则见 `docs/CORE_SERVER_RUNBOOK.md`。如果数据数量、命名或 pairing 异常，先修复 loader / symlink，再启动训练。

官方入口保持稳定：

- 训练：`code/train.py`
- 官方 `.pth` 推理评测：`code/eval.py`

当前仓库有两类权重：

- 官方 `*.pth`：重参数化后的推理模型权重，匹配 `eval.py` 中的 `Backbone()`。
- 训练产生的 `best.pk` / `latest.pk`：训练态 checkpoint，包含 `model`、`optimizer`、step 和日志，需用训练态评测入口或导出/重参数化入口。

不要把 `best.pk` 直接改名为 `.pth` 交给 `eval.py`。最终表格前必须明确每个结果使用的是官方 `.pth`、训练态 `.pk` 评测，还是导出后的推理权重。

## 6. 当前证据摘要

完整时间线见 `docs/EXPERIMENT_LOG.md`，保留/删除策略见 `docs/HAZE4K_RUN_MANIFEST.md`。

| 路线 | 当前结论 | 主证据 | 后续动作 |
| --- | --- | --- | --- |
| DEA-Net-CR baseline | 有效基线 | `DEA-Net-CR-H4K-Baseline-scout-20260520-101334` best 90k `32.2255 / 0.9844` | 作为所有 candidate 的同协议参照 |
| LF-v1 | 当前唯一正向单模块候选 | `DEA-Net-LF-H4K-scout-20260521-003100` best 90k `32.4281 / 0.9845`，full per-image mean delta `+0.2030 dB` | 保留为 LF 对照；需补稳定性、复杂度和视觉风险 |
| ResidualCalib | 正向消融，但不替代 LF-v1 | `DEA-Net-LF-ResidualCalib-H4K-scout100k-20260525-122654` best 90k `32.3936 / 0.9845`；full per-image vs baseline `+0.1682 dB`，vs LF-v1 `-0.0347 dB`；wrong-direction vs CR `163`、vs LF-v1 `211` | 保留为 residual calibration 消融；下一步只做能直接减少 wrong-direction 和强 baseline 样本回退的候选 |
| CRPlus-v2 | 正向 CR-only 组件候选，但不替代 LF-v1 | `DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540` best/final 100k `32.3633 / 0.9847`；full per-image vs CR `+0.1396 dB`，vs LF-v1 `-0.0633 dB`，vs ResidualCalib `-0.0286 dB` | 保留为无推理成本的正向 loss 消融；下一步若继续，应优先做 lite/组合并减少强样本回退 |
| Conservative LF | 失败 | 100k `32.1083 / 0.9843`，低于 baseline/LF-v1 | 不继续此强约束组合 |
| CRPlus-P1 lowpass negative | 失败 | 10k `24.9623 / 0.9504` | 不继续同一 negative 设计 |
| LowFreqLoss / LF+LowFreqLoss | 失败 | 20k/50k 均低于对应参照 | 不继续简单低频 L1 路线 |
| TeacherGuard 当前设置 | 失败 | 20k `27.7075 / 0.9704`；10k 已明显落后，而 guard loss 在 20k 前关闭 | 不继续此 run；不能把早期劣化完全归因于 teacher penalty |
| PostMix | 失败 | 50k `30.7103 / 0.9814` | 不继续单点移动插入位置 |
| Conditional LF | 失败，不晋级 | 公平 `100k` run 的 30k gate 为 `30.1830 / 0.9783`，明显低于 LF-v1 30k；mask 仍近似常数 | 不继续当前设置；若重启条件化 LF，必须先重写 experiment card |
| LF-v2 Haze-Aware Mask | 失败，不晋级 | 30k `30.1157 / 0.9770`，基本追平 baseline 30k 但明显低于 LF-v1 30k；mask 已激活但收益不足 | 不继续此简单 dark-channel/luma mask；保留为“选择性增强不等于方向修正”的负证据 |

## 7. 阶段计划

### 7.1 阶段一：基线复现

目标：确认数据、权重、训练入口、评测入口和 HAZE4K 指标链路可用。

必须产出：

- 官方 HAZE4K `.pth` full eval 结果。
- DEA-Net-CR 同协议 `100k` scout 曲线。
- baseline `best.pk` / `latest.pk` 的训练态评测记录。
- 固定视觉样例或 full per-image 对比所需的 baseline 输出。

当前状态：已完成。baseline scout 和官方 checkpoint 评测事实已记录在 `EXPERIMENT_LOG.md`。

### 7.2 阶段二：LF 单模块路线

目标：验证低频先验是否能补强 DEA-Net 对全局雾幕和低频退化的建模。

已确认：

- LF-v1 的 `pre_mix` bottleneck 插入点有全量正收益。
- LF-v1 的问题不是均值无效，而是 per-image 方差大；它更像困难样本补偿器。
- 固定样本和 gate sweep 暴露了部分样例的颜色、亮度、暗通道和过校正风险。

当前策略：

1. 保留 LF-v1 作为正向对照。
2. 不再继续“更保守 LF”或“推理时缩小 scalar gate”这种低信息量路线。
3. Conditional LF 的当前设置已在 30k hard gate 停止；它没有证明内容感知空间 mask 已有效激活。
4. LF-v2 Haze-Aware Mask 已在 30k hard gate 停止；它证明 mask 可以被激活，但简单低频 RGB + dark-channel + luma 线索没有把 LF-v1 的收益保住。
5. LF-v1 residual 方向/幅度诊断已完成；`delta PSNR` 与低频 residual cosine 的相关系数为 `0.8775`，说明下一步应优先校准 residual 方向和幅度，而不是继续堆纯 mask。
6. ResidualCalib 已完成公平 `100k` scout；它验证了 residual 方向/幅度路线有意义，但第一版仍低于 LF-v1，不能作为主替代路线。

Conditional LF 的立项理由、结构约束和实验卡见 `docs/HAZE4K_CONDITIONAL_LF_ROUTE_AUDIT_20260523.md`。已发生 run 的最新状态见 `CURRENT_CONTEXT.md` 和 `EXPERIMENT_LOG.md`。

Conditional LF 当前结论（2026-05-24）：

- 公平 `100k` run `DEA-Net-LF-ConditionalMask-H4K-scout100k-20260523-224315`
  恢复到 30k 后停止。
- 30k 指标为 `30.1830 / 0.9783`，相对 baseline 30k
  `30.1143 / 0.9776` 仅小幅领先，相对 LF-v1 30k `30.6253 / 0.9783`
  明显落后。
- `latest.pk` 中 mask mean 约 `0.878747`、std 约 `0.000085`，min/max 约
  `0.877925/0.878916`，scalar `lf_prior.gate` 约 `0.014778`。mask 仍近似常数，
  未形成有效空间选择。
- 结论：停止当前 Conditional LF 设置，不继续到 50k 或 100k。LF-v1 仍是当前
  唯一正向 LF 单模块候选。

LF-v2 Haze-Aware Mask 的实验卡见 `docs/HAZE4K_LF_V2_HAZE_AWARE_MASK_PLAN_20260524.md`。已发生 run 的停止事实见 `CURRENT_CONTEXT.md` 和 `EXPERIMENT_LOG.md`。

LF-v2 当前结论（2026-05-24）：

- 公平 `100k` run `DEA-Net-LF-HazeAwareMask-H4K-scout100k-20260524-152758`
  在 30k hard gate 停止。
- 曲线为 10k `27.5613 / 0.9571`、20k `28.6961 / 0.9724`、
  30k `30.1157 / 0.9770`。
- 30k 相对 baseline `30.1143 / 0.9776` 基本持平，但相对 LF-v1
  `30.6253 / 0.9783` 明显落后。
- `latest.pk` 中 mask std last 约 `0.001312`，tail-50 mean 约 `0.001059`，
  说明它不是 Conditional LF 的“近似常数 mask”问题。
- 结论：停止此设置，不继续到 50k 或 100k。根因判断需要前进一步：
  LF-v1 短板不是只缺空间选择，而是缺少能判断低频残差方向和幅度是否正确的约束或结构。

LF-v1 residual 方向诊断见 `docs/HAZE4K_LF_RESIDUAL_DIRECTION_DIAGNOSIS_20260524.md`。
核心事实：

- read-only full-test 诊断路径：
  `experiment/HAZE4K/residual_diagnostic/CR-vs-LF-v1-20260524/`。
- `LF-v1` 相对 baseline 的 mean delta 仍为 `+0.2030 dB`，但
  `corr(delta PSNR, residual cosine)=0.8775`。
- 最差回退样本多为 residual cosine 为负，且 low-frequency MSE 变差；
  最好收益样本 residual cosine 接近 `0.9`，且 low-frequency MSE 改善。
- 结论：下一轮 LF 路线应命名为 residual calibration / direction-magnitude
  control。纯空间 mask 只有在同时预测 residual 方向和幅度时才值得重启。

ResidualCalib 的实验卡见 `docs/HAZE4K_LF_RESIDUAL_CALIBRATION_PLAN_20260525.md`。当前结论（2026-05-25）：

- 公平 `100k` run `DEA-Net-LF-ResidualCalib-H4K-scout100k-20260525-122654`
  已自然完成。
- 最佳 checkpoint 为 90k：`32.3936 / 0.9845`；final 100k 为
  `32.3858 / 0.9846`。
- 相对 baseline full per-image mean delta 为 `+0.1682 dB`，说明不是无效结构；
  相对 LF-v1 mean delta 为 `-0.0347 dB`，说明不能替代当前正向 LF-v1。
- residual 诊断仍有 wrong-direction：vs CR 为 `163`，vs LF-v1 为 `211`；
  低频 MSE 改善/回退仍接近对半分，强 baseline 样本仍有回退。
- 结论：ResidualCalib 保留为“方向/幅度校准有效但不足”的正向消融。
  下一轮若继续 LF，只能围绕 wrong-direction count、强 baseline 保护或显式 residual
  方向约束推进；不再启动单纯空间 mask 或 mask 叠加路线。

已删除的晋级假设：当前 Conditional LF、LF-v2 Haze-Aware Mask 和第一版
ResidualCalib 都不再作为活动主候选排队长训。LF-v1 仍是当前主正向 LF 对照。

### 7.3 阶段三：CRPlus 独立损失路线

目标：在不增加推理参数的前提下，改进已有 CR 的 task-aware 约束。

当前结论：

- 现有 CR 已经是 DEA-Net-CR 的有效组成，不能把“加入 CR”作为创新点。
- CRPlus-P1 的 `hazy_lowpass` equal-weight negative 已失败，不继续同一设计。
- CRPlus-v2 频域/低频/VGG ratio curriculum 已完成第一条公平 `100k`
  scout。它相对 CR baseline 是正向结果：best/final `32.3633 / 0.9847`，
  full-test mean delta `+0.1396 dB`。
- CRPlus-v2 仍低于 LF-v1 和 ResidualCalib 的最佳 PSNR，且 per-image
  诊断显示它主要补偿弱样本、对 LF-family 强样本有回退。因此它应作为
  CR-only loss 组件候选和消融证据，而不是当前最佳 standalone 替代。

后续若重启 CRPlus，应先做离线 loss scale / feature distance 诊断，再考虑训练。候选方向：

- 记录 `d_ap/d_an`，确认 ratio loss 是否尺度失控。
- 用 margin ranking 形式替代简单多 negative ratio。
- negative 使用受控退化的 `prediction.detach()`，而不是直接把 `lowpass(hazy)` 当 hard negative。
- 或只在 amplitude / frequency residual 上做轻量一致性或排序约束，避免内容无关的 VGG negative。
- 优先设计 CRPlus-v2-lite 或 LF+CRPlus 组合，目标是保留弱样本收益，
  同时减少强 baseline / 强 LF-v1 样本回退和 VGG 约束重复。

CRPlus 必须保持为独立消融，不依赖 LF 输出，不新增推理参数。

### 7.4 阶段四：LFCR 组合

只有在单因素结果明确后再组合：

- LF 有提升、CRPlus 中性：采用 LF + 默认 CR。
- CRPlus 有提升、LF 中性：以 CRPlus 作为主模型。
- 二者都有提升：组合为 LFCR。
- 某个模块造成明显伪影或复杂度过高：保留为消融，不进入最终模型。
- LF 和 CRPlus 都没有收益：不强行训练 LFCR，回退到 baseline + 失败分析或真实域扩展。

组合模型必须沿用公平 `100k` scout，不能直接跳到长训。

### 7.5 可选阶段：真实域或 TTA

TTA / 真实域适配只作为扩展，不阻塞 HAZE4K 主结果。

要求：

- 冻结主模型或保持推理期可控。
- HAZE4K 配对测试不崩。
- 真实图像视觉质量更自然。
- 无参考指标只能作为补充，不能替代 HAZE4K PSNR/SSIM。

## 8. 评估与结果分析

每个晋级候选至少包含：

- HAZE4K PSNR / SSIM。
- 训练态 checkpoint 的 step 或 epoch。
- 与 baseline 和 LF-v1 的同步曲线对比。
- per-image delta、better/worse counts、delta bins。
- 固定视觉样例对比：input、baseline、candidate、GT。
- 典型正向、负向和中性样例。
- 参数量、FLOPs/MACs、推理速度、显存峰值。
- 是否影响 `.pk -> .pth` 导出或 `eval.py` 合约。

补充指标：

- LPIPS：用于补充 PSNR/SSIM 对纹理和感知质量的不敏感。
- MAE/MSE：可选。
- FADE/NIQE/PIQE/MUSIQ：仅用于真实或无配对图像的辅助分析。

视觉检查项目：

- 雾残留。
- 颜色偏移。
- 边缘 halo。
- 天空平滑度。
- 纹理恢复。
- 过锐化。
- 暗部噪声。

客观分析脚本和 compare-dir 操作模板见 `WORKFLOW.md` 与相关 artifact 目录。程序输出用于初筛和定位，最终论文展示样例仍需人工确认自然度、颜色和边缘观感。

## 9. 消融表骨架

毕业论文建议保留如下主表结构：

| ID | Model | LF prior | CRPlus | TTA | Params | FLOPs | PSNR | SSIM | LPIPS | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | DEA-Net-CR | No | No | No | TBD | TBD | TBD | TBD | TBD | 强基线 |
| A1 | DEA-Net-LF-v1 | Yes | No | No | TBD | TBD | TBD | TBD | TBD | 当前正向 LF 对照 |
| A2 | DEA-Net-Conditional-LF | Yes | No | No | TBD | TBD | TBD | TBD | TBD | 30k gate 失败，放入失败消融 |
| A3 | DEA-Net-LF-v2-HazeAwareMask | Yes | No | No | TBD | TBD | TBD | TBD | TBD | 30k gate 失败，mask 激活但指标不晋级 |
| A4 | DEA-Net-LF-ResidualCalib | Yes | No | No | TBD | TBD | `32.3936` | `0.9845` | TBD | 正向消融但未超过 LF-v1；保留 residual calibration 证据 |
| A5 | DEA-Net-CRPlus-v2 | No | Yes | No | TBD | TBD | `32.3633` | `0.9847` | TBD | 正向 CR-only loss 消融；未超过 LF-v1/ResidualCalib PSNR |
| A6 | DEA-Net-LFCR | Yes | Yes | No | TBD | TBD | TBD | TBD | TBD | 最终组合候选 |
| A7 | DEA-Net-LFCR-TTA | Yes | Yes | Yes | TBD | TBD | TBD | TBD | TBD | 可选扩展 |

如果某条路线没有通过公平 scout，不进入主结果表；可以进入失败消融表或讨论章节。

## 10. 停止、晋级与回退

停止当前路线：

- 公平 `100k` scout 的 gate 明显低于同协议 baseline 和 LF-v1。
- 当前 Conditional LF 已在 30k gate 停止：略高于 baseline 但明显低于 LF-v1，且 mask 仍近似常数，记录为“条件化未充分激活”。
- LF-v2 Haze-Aware Mask 已在 30k gate 停止：mask 已激活，但指标只追平 baseline、明显低于 LF-v1，记录为“选择性不足以修正方向错误”。
- ResidualCalib 已完成 100k：强于 baseline 但 full per-image 低于 LF-v1，wrong-direction 仍未消除，记录为“方向/幅度校准有效但不足”，不晋级为 LF-v1 替代主线。
- CRPlus-v2 已完成 100k：强于 baseline 且无推理成本，但 PSNR 低于 LF-v1
  和 ResidualCalib；记录为“正向 CR-only 组件候选”，不直接晋级为 standalone
  最终模型。
- loss 不稳定或训练异常无法解释。
- 输出出现系统性 halo、偏色、过锐化或大片雾残留。
- 复杂度增长明显，但指标或视觉收益很小。
- checkpoint 格式或评测链路无法复现。

晋级到 full per-image / 100k / formal reporting：

- 同一条 `100k` run 的 `50k` gate 至少接近 baseline，并最好接近 LF-v1。
- 视觉风险不比 LF-v1 更严重。
- 单因素变量清楚，能解释成功或失败。
- 有可保留 artifact 路径和日志记录。

回退顺序：

1. 保留 LF-v1 作为正向轻量消融。
2. ResidualCalib 保留为正向但不足的机制证据；若再试，必须直接减少 wrong-direction 或强 baseline 回退。
3. Conditional LF 当前设置已失败；若再试，必须先判断是调 mask bias、换 mask 输入，还是放弃 LF 条件化。
4. 若 CRPlus 继续失败，停止 loss 路线，把失败原因写入方法讨论。
5. 若最终没有稳定超过 baseline 的候选，论文主线转为“强 baseline + 正向 LF-v1 小收益 + 系统失败分析 + 可复现证据链”。

## 11. 推荐时间线

| 阶段 | 工作 | 产出 |
| --- | --- | --- |
| 1 | HAZE4K 数据、官方权重、baseline smoke/full eval | 数据和评测入口确认 |
| 2 | DEA-Net-CR 公平 `100k` scout | 同协议 baseline 曲线 |
| 3 | LF-v1 公平 `100k` scout + full per-image | 正向 LF 对照和风险分析 |
| 4 | Conditional LF 公平 `100k` scout | 已在 30k gate 停止，不晋级 |
| 5 | LF-v2 Haze-Aware Mask 公平 `100k` scout | 已在 30k gate 停止，不晋级 |
| 6 | ResidualCalib 公平 `100k` scout + full per-image / residual 诊断 | 正向消融但不替代 LF-v1；后续只围绕方向/幅度约束继续 |
| 7 | CRPlus-v2 公平 scout + final diagnostics | 已完成；正向 CR-only 组件候选，但不替代 LF-v1 |
| 8 | 最终候选 full eval、复杂度、可视化 | 论文主表和图 |
| 9 | 可选 TTA / 真实域测试 | 扩展章节证据 |
| 10 | 汇总方法、消融、失败讨论和局限 | 毕业论文初稿 |

## 12. 后续代码实现注意事项

- 为实现任务创建独立 feature branch。
- 保持 `code/train.py` 和 `code/eval.py` 官方入口可用。
- 新模型变体必须通过显式 option 或清晰命名启用。
- 每个新实验先写 experiment card，再训练。
- LF prior、CRPlus negative 类型、CR 权重、scout/formal 标记都应写入 run 参数记录；新 run 同时保留 `args_initial.txt`、最新 `args.txt` 和追加式 `args_history.jsonl`，避免 resume 覆盖首次启动事实。
- 新增评测或导出脚本时，必须显式区分 `.pk` 和 `.pth`。
- checkpoint、TensorBoard、推理图像和临时结果放在 `experiment/`、`trained_models/` 或外部存储，不提交 Git。
- 每次训练结果写入 `EXPERIMENT_LOG.md`；每个重要 artifact 写入 `HAZE4K_RUN_MANIFEST.md`。
