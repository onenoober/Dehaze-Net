# DEA-Net-LFCR 基于 HAZE4K 的训练与测试计划

日期：2026-05-19

## 1. 目标

以 HAZE4K 作为主数据集，以当前已经跑通的 DEA-Net-CR 作为主基线，后续改进模型暂命名为 **DEA-Net-LFCR**。

- **LF**：轻量低频/小波退化先验。
- **CR**：在现有 DEA-Net-CR 对比正则基础上的改进版对比约束。

本计划不追求大规模重写网络，而是围绕“稳步提升、可复现、可消融、可写入毕业论文”设计后续实验。预期目标如下：

- 在 HAZE4K 配对测试集上，相比本地 DEA-Net-CR 基线取得稳定 PSNR/SSIM/LPIPS 改善。
- 在视觉效果上减少雾残留、颜色偏移、过锐化和边缘 halo。
- 形成完整毕业论文证据链：理论动机、结构改动、训练策略、消融实验、复杂度分析和失败风险讨论。

## 2. 核心假设

DEA-Net 已经通过 DEConv 获得较强的局部细节建模能力，并通过 CGA 获得较强的内容引导融合能力。它当前更值得补强的部分不是重新设计主干，而是：

1. 显式建模低频雾幕和全局退化。
2. 改进训练期对比正则，使 negative 更贴近去雾任务。
3. 使用更完整的测试指标验证真实视觉质量，而不只看单一 PSNR/SSIM。

因此，后续改法遵循一个原则：

> 保持 DEConv 和 CGA 稳定，在瓶颈层、融合层和损失函数附近做轻量增强。

第一轮实验不建议直接引入大型 Transformer、Mamba、深度分支或完整双任务框架。

## 3. 数据集优先级

主数据集：

- `HAZE4K`

期望目录结构：

```text
dataset/
  HAZE4K/
    train/
      clear/
      hazy/
    test/
      clear/
      hazy/
```

主训练和测试命令从 `code/` 目录运行：

```powershell
cd code
python train.py --dataset HAZE4K ...
python eval.py --dataset HAZE4K --model_name DEA-Net-CR --pre_trained_model <checkpoint>.pth
```

注意：

- HAZE4K 预训练权重文件名以 `trained_models/HAZE4K/` 下实际下载到的文件为准。
- 上游 README 中 HAZE4K checkpoint 文件名存在不一致，不要在未检查文件夹前固定写死。

## 4. 实验命名

所有输出默认进入 `experiment/HAZE4K/<model_name>/`，建议统一使用以下名称。

| Run ID | model_name | 目的 |
| --- | --- | --- |
| H4K-B0 | `DEA-Net-CR-H4K-Baseline` | 复现本地 HAZE4K 基线 |
| H4K-A1 | `DEA-Net-LF-H4K` | 仅加入低频/小波先验 |
| H4K-A2 | `DEA-Net-CRPlus-H4K` | 仅改进对比正则 |
| H4K-A3 | `DEA-Net-LFCR-H4K` | 低频先验 + 改进 CR 的主模型 |
| H4K-A4 | `DEA-Net-LFCR-H4K-TTA` | 可选测试时真实域适配 |

每次实验至少记录：

- 分支名。
- 数据集路径和 split。
- checkpoint 文件名。
- batch size。
- patch size。
- learning rate。
- CR 权重。
- GPU 型号。
- 训练耗时。
- 最优 PSNR/SSIM 对应 step 或 epoch。
- 代表性视觉现象。

简表可以写入 `docs/EXPERIMENT_LOG.md`。

## 5. 阶段一：HAZE4K 基线复现

### 5.1 数据检查

Windows 本地检查：

```powershell
cd D:\Dehaze\Dehaze-Net
Get-ChildItem dataset\HAZE4K\train\hazy | Measure-Object
Get-ChildItem dataset\HAZE4K\train\clear | Measure-Object
Get-ChildItem dataset\HAZE4K\test\hazy | Measure-Object
Get-ChildItem dataset\HAZE4K\test\clear | Measure-Object
```

服务器检查：

```bash
cd /root/workspace/Dehaze-Net
find dataset/HAZE4K -maxdepth 3 -type d | sort
find dataset/HAZE4K/train/hazy -type f | wc -l
find dataset/HAZE4K/train/clear -type f | wc -l
find dataset/HAZE4K/test/hazy -type f | wc -l
find dataset/HAZE4K/test/clear -type f | wc -l
```

如果 hazy 和 clear 数量不一致，先不要训练。应先确认 HAZE4K 文件命名与 `code/data/data_loader.py` 中的匹配逻辑是否一致。

### 5.2 官方 checkpoint 评测

目的：确认数据、权重、评测脚本和指标计算都可用。

```bash
cd /root/workspace/Dehaze-Net/code
conda activate deanet
python eval.py \
  --dataset HAZE4K \
  --model_name DEA-Net-CR \
  --pre_trained_model <actual_haze4k_checkpoint>.pth
```

通过标准：

- 评测过程无路径错误、权重错误或 tensor shape 错误。
- PSNR/SSIM 接近官方或之前复现得到的 HAZE4K 结果。

如果结果明显偏低，优先检查：

- `clear` 与 `hazy` 是否正确匹配。
- checkpoint 是否真的是 HAZE4K 权重。
- 是否错误加载了训练态模型或未重参数化模型。

### 5.3 本地基线训练

第一轮基线必须沿用上游 DEA-Net-CR 配置，只将数据集切换为 HAZE4K。后续所有改进实验采用“两阶段训练”：

1. **scouting 短跑**：先跑 `50k` 到 `100k` step，用于判断 loss 是否稳定、PSNR 是否明显落后、视觉是否出现严重伪影。
2. **full training 完整训练**：只给基线、最优单模块和最终候选 LFCR 跑完整训练，避免把大量时间消耗在明显无效的变体上。

基线本身仍建议完整训练，因为它是所有后续提升幅度、视觉对比和复杂度报告的参照。

推荐完整训练命令：

```bash
cd /root/workspace/Dehaze-Net/code
conda activate deanet
python train.py \
  --epochs 300 \
  --iters_per_epoch 5000 \
  --finer_eval_step 1400000 \
  --w_loss_L1 1.0 \
  --w_loss_CR 0.1 \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --exp_dir ../experiment/ \
  --model_name DEA-Net-CR-H4K-Baseline \
  --dataset HAZE4K \
  --checkpoint_interval_steps 50000 \
  --eval_interval_steps 50000 \
  --early_stop_patience_evals 12 \
  --early_stop_after_step 300000
```

如果显存不足，可先降 batch size：

```bash
python train.py \
  --epochs 300 \
  --iters_per_epoch 5000 \
  --bs 8 \
  --w_loss_L1 1.0 \
  --w_loss_CR 0.1 \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --exp_dir ../experiment/ \
  --model_name DEA-Net-CR-H4K-Baseline-bs8 \
  --dataset HAZE4K
```

基线验收条件：

- 至少得到一个 best checkpoint。
- HAZE4K test PSNR/SSIM 已记录。
- 固定挑选至少 20 张代表性测试图作为视觉对比集。
- 记录失败样例，例如强雾残留、天空偏色、建筑边缘 halo、暗部噪声等。

### 5.4 短跑筛选规则

除基线外，LF、CRPlus 和 LFCR 的第一轮实验先使用短跑命令：

```bash
python train.py \
  --epochs 20 \
  --iters_per_epoch 5000 \
  --w_loss_L1 1.0 \
  --w_loss_CR 0.1 \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --exp_dir ../experiment/ \
  --model_name <candidate-scout> \
  --dataset HAZE4K \
  --checkpoint_interval_steps 10000 \
  --eval_interval_steps 10000 \
  --save_epoch_checkpoints false
```

短跑通过条件：

- `50k` step 前 loss 没有明显发散。
- `100k` step 时 PSNR 不应明显低于同等训练长度的基线短跑。
- 固定视觉样例中没有系统性偏色、halo、过锐化或大片雾残留。
- 参数量和显存增长符合轻量改造预期。

只有通过短跑的变体才进入完整训练。

## 6. 阶段二：低频/小波退化先验

模型名：

- `DEA-Net-LF-H4K`

### 6.1 优先插入位置

建议按以下顺序尝试：

1. `x8` 之后、`mix1` 之前的 bottleneck 位置。
2. 输入图像提取低频先验后，引导 bottleneck 特征。
3. `mix1` 前后。
4. `mix2` 前后。

第一版不建议改动：

- `DEConv` 内部结构。
- DEA block 数量。
- 输出重建头。

这样做的原因是：`DEConv` 已经承担了细节增强职责，低频先验更适合补充全局雾幕和整体对比度信息。

### 6.2 模块设计约束

低频先验分支建议具备以下特征：

- 从输入或中间特征提取低频成分。
- 用小型卷积 adapter 处理低频信息。
- 通过残差门控方式融合回主干。

第一版建议采用最小实现，避免一开始引入复杂依赖：

```text
input image
  -> fixed low-pass 或 1-level Haar/DWT LL
  -> lightweight adapter
  -> resize 到 x8 空间尺寸
  -> 1x1/3x3 projection 到 base_dim*4
  -> scalar gate 初始接近 0
  -> x8 + gate * prior
  -> mix1
```

如果小波依赖或边界处理带来额外风险，第一版可以先用 `avgpool/blur + upsample` 构造固定低通分支。确认有效后，再替换为 Haar/DWT 版本。

初始约束：

- 参数量相对主模型保持较小。
- 融合门控初始尽量接近 identity，避免一开始破坏 DEA-Net 已有能力。
- 输出 tensor shape 必须与原 DEA-Net 流程一致。

### 6.3 训练命令

先沿用基线训练策略，保证变量单一。LF 第一轮先短跑筛选，确认稳定后再完整训练。

```bash
python train.py \
  --epochs 20 \
  --iters_per_epoch 5000 \
  --w_loss_L1 1.0 \
  --w_loss_CR 0.1 \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --exp_dir ../experiment/ \
  --model_name DEA-Net-LF-H4K-scout \
  --dataset HAZE4K \
  --checkpoint_interval_steps 10000 \
  --eval_interval_steps 10000 \
  --save_epoch_checkpoints false
```

通过标准：

- 前 50k step 训练稳定。
- 100k step 附近验证 PSNR 不持续低于同等训练长度基线。
- 视觉结果没有更严重偏色、halo 或过度锐化。

如果不稳定：

- 降低 adapter 通道数。
- 只保留 bottleneck 位置，不在浅层多处插入。
- 增加可学习 scalar gate，并将初始贡献设小。

## 7. 阶段三：改进对比正则 CRPlus

模型名：

- `DEA-Net-CRPlus-H4K`

### 7.1 现有 CR 的局限

当前 `code/loss/cr.py` 中的 CR 逻辑是：

- anchor：模型输出。
- positive：clear GT。
- negative：hazy 输入。

这个设计已经有效，因此论文不能把“加入 CR”写成创新点。后续应写成：

> 针对 DEA-Net-CR 已有对比正则，设计更困难、更贴近雾退化的 task-aware negative。

### 7.2 候选 negative 设计

CRPlus 必须保持为“仅损失函数改动”的独立消融，不依赖 LF 分支输出，不新增推理参数。所有额外 negative 都应 `detach` 或由固定退化算子生成，避免把 CRPlus 变成隐式结构分支。

按风险从低到高测试：

| 变体 | positive | negative | 风险 |
| --- | --- | --- | --- |
| CR-B0 | clear | hazy input | 现有基线 |
| CR-P1 | clear | hazy input + fixed low-pass hazy | 低 |
| CR-P2 | clear | hazy input + frequency-degraded prediction.detach() | 中 |
| CR-P3 | clear | hazy input + weak prediction.detach() | 中 |

第一版推荐：

- 保留 VGG feature contrastive loss。
- 第一轮只实现 CR-P1：`negative = [hazy, lowpass(hazy)]`。
- `lowpass(hazy)` 使用固定低通、Gaussian blur 或 Haar LL 上采样，不调用 LF 模块。
- 多 negative 可采用逐项求 CR 后加权平均，或取 harder negative 的最大值；第一版优先使用加权平均，稳定性更高。
- 搜索 CR 权重先用 `0.05`、`0.1`，`0.2` 只作为扩展风险实验。

### 7.3 CR 权重搜索

| Run | CR 权重 | 说明 |
| --- | --- | --- |
| CR-W005 | `0.05` | 保守，适合颜色变硬时 |
| CR-W010 | `0.1` | 上游默认值 |
| CR-W020 | `0.2` | 扩展实验，约束更强，重点观察伪影 |

示例短跑命令：

```bash
python train.py \
  --epochs 20 \
  --iters_per_epoch 5000 \
  --w_loss_L1 1.0 \
  --w_loss_CR 0.05 \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --exp_dir ../experiment/ \
  --model_name DEA-Net-CRPlus-H4K-P1-w005-scout \
  --dataset HAZE4K \
  --checkpoint_interval_steps 10000 \
  --eval_interval_steps 10000 \
  --save_epoch_checkpoints false
```

同样方式再跑 `--w_loss_CR 0.1`。只有 `0.05` 和 `0.1` 都不理想但视觉上没有明显伪影时，再尝试 `0.2`。

通过标准：

- 至少一个 CRPlus 设置在同等训练长度下达到或超过基线 PSNR/SSIM。
- 视觉质量不弱于基线。
- 在与 LF 组合前，先明确最优 CR 权重。

## 8. 阶段四：组合模型 DEA-Net-LFCR

模型名：

- `DEA-Net-LFCR-H4K`

这是后续毕业论文的主模型。

### 8.1 组合规则

只有在单因素短跑结果明确后再组合：

- 如果 LF 有提升、CRPlus 中性，则采用 LF + 默认 CR。
- 如果 CRPlus 有提升、LF 中性，则以 CRPlus 作为主模型。
- 如果二者都有提升，则组合为 LFCR。
- 如果某个模块造成明显伪影，则保留为消融结果，不作为最终模型。
- 如果 LF 和 CRPlus 单独都没有收益，不建议强行训练 LFCR 完整版；优先回退到 CR 权重搜索或真实域 TTA。

### 8.2 训练命令

使用阶段三选出的最佳 CR 权重。

```bash
python train.py \
  --epochs 300 \
  --iters_per_epoch 5000 \
  --w_loss_L1 1.0 \
  --w_loss_CR <best_cr_weight> \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --exp_dir ../experiment/ \
  --model_name DEA-Net-LFCR-H4K \
  --dataset HAZE4K \
  --checkpoint_interval_steps 50000 \
  --eval_interval_steps 50000 \
  --early_stop_patience_evals 12 \
  --early_stop_after_step 300000
```

### 8.3 最终 checkpoint 选择

按以下顺序选择：

1. HAZE4K PSNR 最优。
2. SSIM 不低于基线。
3. 如果测 LPIPS，则 LPIPS 低于基线。
4. 视觉检查通过：雾残留、颜色、纹理、halo 均不能明显变差。

如果 PSNR 提升但视觉质量变差，不建议作为毕业论文最终模型，应作为“指标偏置”案例讨论。

## 9. 可选阶段五：测试时真实域适配

模型名：

- `DEA-Net-LFCR-H4K-TTA`

该阶段是扩展实验，不应阻塞 HAZE4K 主结果。

目标：

- 在不重新训练完整模型的前提下，增强真实或跨域图像的去雾稳定性。

建议方向：

- 参考 PTTD-lite 思路，对 encoder 或 bottleneck 特征做统计量调节。
- 冻结 DEA-Net-LFCR 主体权重。
- 只在测试阶段启用。

验收条件：

- HAZE4K 配对测试不崩。
- 真实图像视觉质量更自然。
- FADE/NIQE/PIQE/MUSIQ 等无参考指标至少部分改善。

## 10. 测试计划

### 10.1 HAZE4K 主测试

每个最终候选 checkpoint 都要使用与 checkpoint 格式匹配的评测入口。当前仓库有两类权重格式：

- 官方 `*.pth`：重参数化后的推理模型权重，匹配 `code/eval.py` 中的 `Backbone()`。
- 训练产生的 `best.pk` / `latest.pk`：包含 `model`、`optimizer`、step 和日志的训练态 checkpoint，匹配 `code/train.py` 中的 `DEANet()` 或后续 LFCR 训练态模型。

因此不要把 `best.pk` 直接改名为 `.pth` 交给 `eval.py`。在最终实验前必须补齐下面二选一的评测链路：

1. **训练态评测入口**：新增或扩展评测脚本，加载 `checkpoint['model']`，实例化对应训练态模型变体，例如 `DEA-Net-CR`、`DEA-Net-LF`、`DEA-Net-CRPlus`、`DEA-Net-LFCR`。
2. **导出/重参数化入口**：将训练态 checkpoint 导出为兼容 `Backbone()` 或对应 LFCR 推理模型的 `.pth`。如果 LF 模块包含不可合并的新参数，则需要建立对应推理模型，而不是只复用原始 `Backbone()`。

官方 checkpoint 评测仍使用：

```bash
cd /root/workspace/Dehaze-Net/code
conda activate deanet
python eval.py \
  --dataset HAZE4K \
  --model_name <model_name> \
  --pre_trained_model <checkpoint>.pth
```

训练态候选模型建议补充类似命令：

```bash
python eval_train_ckpt.py \
  --dataset HAZE4K \
  --model_name <model_variant> \
  --checkpoint ../experiment/HAZE4K/<run>/saved_model/best.pk \
  --save_infer_results
```

`eval_train_ckpt.py` 名称只是建议；真正实现时也可以扩展现有 `eval.py`，但必须显式区分 `.pth` 与 `.pk`。

记录：

- PSNR。
- SSIM。
- checkpoint 文件名。
- step 或 epoch。
- 评测耗时，如可获得。
- 使用的评测入口：官方 `.pth` 推理模型、训练态 `.pk` 评测，或导出后的 LFCR 推理模型。

### 10.2 补充配对指标

主训练稳定后增加：

- LPIPS。
- 可选 MAE/MSE。

这些指标用于补充 PSNR/SSIM 对过平滑、纹理不自然等问题不敏感的缺点。

### 10.3 真实或无配对指标

如果做真实图像扩展，建议使用：

- FADE。
- NIQE。
- PIQE。
- MUSIQ，如果环境支持。

建议使用 PyIQA 统一实现。注意：无参考指标不能替代 HAZE4K 上的 PSNR/SSIM，只作为真实域证据链。

### 10.4 固定视觉对比集

从 HAZE4K test 中固定选取至少 20 张图：

- 轻雾。
- 浓雾。
- 天空区域。
- 植被。
- 建筑边缘。
- 暗部区域。
- 高频纹理。

每个模型输出按同一顺序拼图：

```text
hazy input | DEA-Net-CR baseline | LF | CRPlus | LFCR | clear GT
```

视觉检查项目：

- 雾残留。
- 颜色偏移。
- 边缘 halo。
- 天空平滑度。
- 纹理恢复。
- 过锐化。
- 暗部噪声。

## 11. 消融实验表

毕业论文建议使用以下表格结构。

| ID | Model | LF prior | Improved CR | TTA | Params | FLOPs | PSNR | SSIM | LPIPS | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | DEA-Net-CR | No | No | No | TBD | TBD | TBD | TBD | TBD | 基线 |
| A1 | DEA-Net-LF | Yes | No | No | TBD | TBD | TBD | TBD | TBD | 仅结构改动 |
| A2 | DEA-Net-CRPlus | No | Yes | No | TBD | TBD | TBD | TBD | TBD | 仅损失改动 |
| A3 | DEA-Net-LFCR | Yes | Yes | No | TBD | TBD | TBD | TBD | TBD | 主模型 |
| A4 | DEA-Net-LFCR-TTA | Yes | Yes | Yes | TBD | TBD | TBD | TBD | TBD | 可选真实域扩展 |

## 12. 复杂度与部署检查

论文中建议报告：

- 参数量。
- FLOPs 或 MACs。
- 单图推理时间。
- GPU 显存占用。

至少比较：

- DEA-Net-CR baseline。
- DEA-Net-LFCR final。

目标：

- 参数量增长应保持较小。
- 推理时间增长不能明显破坏 DEA-Net 的轻量优势。

如果 LFCR 只有极小 PSNR 增益但复杂度明显上升，应简化低频先验分支。

## 13. 成功标准

最低可接受毕业结果：

- 完整复现 HAZE4K 上的 DEA-Net-CR 基线。
- LF 或 CRPlus 至少一个方向带来可测量收益，或给出充分失败分析。
- LFCR 不降低 PSNR/SSIM，并改善 LPIPS 或视觉质量中的至少一项。
- 消融表完整。

较好结果：

- DEA-Net-LFCR 在 HAZE4K 上提升约 `0.1` 到 `0.3` dB。
- SSIM 持平或提高。
- LPIPS 降低。
- 真实图像中雾残留和颜色偏移更少。

优秀结果：

- HAZE4K 指标稳定提升。
- 跨数据集或真实域指标也有改善。
- 复杂度增长较小。
- 论文贡献能清晰表述为“低频雾退化先验 + 更困难的对比正则”。

## 14. 停止与回退规则

以下情况应停止当前路线：

- 多次训练出现 loss 不稳定。
- 短跑到 `100k` step 后 PSNR 明显低于同等训练长度基线，且视觉无改善。
- 输出出现明显 halo、偏色或纹理伪影。
- 复杂度增长过大但指标收益很小。
- 训练态 checkpoint 无法形成可靠评测链路，导致结果不可复现。

回退顺序：

1. 只保留 bottleneck 位置的 LF prior。
2. 保留 DEA-Net-CR，只做 CR 权重或 negative 设计搜索。
3. 使用 DEA-Net-CR baseline + 真实域 TTA 扩展。
4. DehazeFormer 仅作为对照，不再作为主线。

## 15. 推荐时间线

| 阶段 | 工作 | 产出 |
| --- | --- | --- |
| 第 1 周 | HAZE4K 数据检查、官方权重评测、baseline smoke test | 确认基线可跑 |
| 第 2 周 | 完整训练 DEA-Net-CR HAZE4K baseline，同时保存短跑参照曲线 | 基线 checkpoint、短跑参照和指标 |
| 第 3 周 | 实现 LF prior 并做 `50k/100k` scouting | A1 短跑指标、视觉结果和是否进入完整训练的结论 |
| 第 4 周 | 实现独立 CRPlus P1，并搜索 `0.05/0.1` 权重短跑 | A2 短跑指标和最佳 CR 权重 |
| 第 5 周 | 训练通过筛选的 LF、CRPlus 或 LFCR 完整版 | 主模型 checkpoint |
| 第 6 周 | 补齐 `.pk/.pth` 评测链路、完整测试、拼图、复杂度统计 | 可复现实验表 |
| 第 7 周 | 可选 TTA 和真实域测试 | 扩展章节证据 |
| 第 8 周 | 撰写方法、消融、讨论和局限 | 毕业论文初稿 |

## 16. 后续代码实现注意事项

开始实现时应遵守：

- 为实现任务创建独立 feature branch。
- 保持 `code/train.py` 和 `code/eval.py` 官方入口可用。
- 新模型变体应通过明确命名的文件或显式选项启用。
- LF prior、CRPlus negative 类型、CR 权重、短跑/完整训练标记都应写入 `args.txt` 或实验日志，保证后续能准确复现实验。
- 新增评测脚本或导出脚本时，应同时支持保存推理图像，便于固定视觉对比集复用同一输出路径。
- checkpoint、TensorBoard、推理图像和临时结果放在 `experiment/` 或 `trained_models/`，不要提交。
- 不提交数据集、权重、日志或临时输出。
- 每次训练结果写入 `docs/EXPERIMENT_LOG.md` 或单独实验日志。
