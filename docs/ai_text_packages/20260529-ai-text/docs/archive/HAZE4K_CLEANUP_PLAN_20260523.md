# HAZE4K Cleanup Plan

日期：2026-05-23

归档状态：2026-05-24 已移入 `docs/archive/`。日常 artifact 保留/删除判断以
`docs/HAZE4K_RUN_MANIFEST.md` 为入口；本文仅用于追溯 2026-05-23 的清理过程和确认记录。

目标：为下一轮 HAZE4K 训练、测试和结果分析提供更干净、更可追溯的环境。本文只规划整理动作；真正删除、移动或压缩数据前，必须再次确认。

## 1. 流程复核结论

之前建议的方向基本合理，但顺序需要更严格：

1. 先建立 manifest，固化每个 run 的身份、结果、保留策略。
2. 再整理本地代码和脚本，让后续实验入口更少、更明确。
3. 最后再删除远端临时/错误/烟测目录。

不能先删数据再补记录。当前已经新增 `docs/HAZE4K_RUN_MANIFEST.md`，后续清理应以它为准。

## 2. Safety Rules

- 不删除 `KEEP` 项。
- `KEEP_MINIMAL` 项只在确认后删临时子目录，不删 `args.txt`、`saved_data/log.txt`、`saved_model/best.pk`、`saved_model/latest.pk`。
- `DELETE_AFTER_MANIFEST` 和 `DELETE_AFTER_CONFIRM` 项执行前必须再次列出路径，并由用户确认。
- 所有远端删除使用 `rm -rf -- <exact-path>`，禁止通配符删除。
- 删除前先运行 `du -sh` 和 `find -maxdepth 2` 复核目标。
- 删除后运行 `find`/`du` 验证目标不存在。

## 3. Stage 0 - Already Done

已完成：

- 只读检查本地 `git status`。
- 只读检查远端 `experiment/HAZE4K` run 列表和大小。
- 只读检查远端没有活跃 tmux 训练，GPU `0 MiB / 0%`。
- 新增 run manifest：`docs/HAZE4K_RUN_MANIFEST.md`。
- 新增 cleanup plan：本文件。

未做：

- 没有删除远端目录。
- 没有删除本地文件。
- 没有移动根目录 `.docx`。
- 没有合并/移动脚本。

## 3.1 Stage 2 Execution Log

执行时间：2026-05-23

已按用户确认执行 Stage 2 远端清理。删除前使用 Python `Path.resolve()` 校验所有目标都位于：

```text
/root/workspace/Dehaze-Net/experiment/HAZE4K
```

已删除：

- `experiment/HAZE4K/probe`
- `experiment/HAZE4K/DEA-Net-LF-TeacherGuard-H4K-scout-20260523-`
- `experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-002900`
- `experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-bs32-20260520-165148`
- `experiment/HAZE4K/_run_logs/probe.log`
- `experiment/HAZE4K/_run_logs/DEA-Net-LF-TeacherGuard-H4K-scout-20260523-.log`
- `experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-bs32-20260520-165148.log`

删除后已验证：

- 上述目标均不存在。
- baseline `DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk` 仍存在。
- LF-v1 `DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk` 仍存在。
- Conservative / CRPlus / TeacherGuard / PostMix 等关键失败消融 `best.pk` 仍存在。
- `per_image_eval/CR-vs-LF-v1-full-20260523/summary.json` 仍存在。
- `visual_compare/DEA-Net-CR-vs-LF-20260522/summary.json` 仍存在。
- GPU 状态仍为空闲：`0 MiB / 0%`。

## 3.2 Stage 3 Execution Log

执行时间：2026-05-23

根据用户补充规则，本阶段只删除短训练时长的流程测试：`smoke` / `dry-run` / `eval-smoke`。如果是 `10k`、`20k`、`50k`、`100k` 等较长训练，且与前面的模型尝试有关，即使结果失败也不删除。

删除前已逐项复核：

- 所有目标路径都通过 Python `Path.resolve()` 校验，位于 `/root/workspace/Dehaze-Net/experiment/HAZE4K` 下。
- 主要 `180M` 目录的 `max_observed_step` 均为 `1`，只包含 1 步 smoke 产生的 `best.pk` / `latest.pk` 或等价 checkpoint。
- 小型目录没有有效训练步数或没有 `saved_model`。
- 本批未包含任何 `10k`、`20k`、`50k`、`100k` 正式 scout / ablation run。

已删除目录：

- `experiment/HAZE4K/DEA-Net-CR-smoke-HAZE4K-20260514-032218`
- `experiment/HAZE4K/DEA-Net-CR-smoke-HAZE4K-20260515-102937`
- `experiment/HAZE4K/smoke-H4K-LF-train-20260521-0028`
- `experiment/HAZE4K/smoke-H4K-bs32-vram-20260520-164745`
- `experiment/HAZE4K/smoke-H4K-settings-20260519-230502`
- `experiment/HAZE4K/smoke-H4K-lf-postmix-`
- `experiment/HAZE4K/smoke-lf-teacher-guard-20260523`
- `experiment/HAZE4K/smoke-H4K-CRPlus-P1-20260523-011035`
- `experiment/HAZE4K/smoke-H4K-LowFreqLoss-20260523-015512`
- `experiment/HAZE4K/smoke-H4K-LF-LowFreqLoss-20260523-031517`
- `experiment/HAZE4K/smoke-H4K-settings-20260519-230213`
- `experiment/HAZE4K/smoke-H4K-settings-debug-20260519-230246`
- `experiment/HAZE4K/smoke-H4K-CRPlus-P1-20260523-010932`
- `experiment/HAZE4K/eval-H4K-official-smoke-20260519-230527`

已删除顶层 smoke 日志：

- `experiment/HAZE4K/smoke-H4K-bs32-vram-20260520-164745.log`
- `experiment/HAZE4K/smoke-H4K-bs32-vram-20260520-164745.mem.log`
- `experiment/HAZE4K/smoke-H4K-settings-20260519-230502.log`
- `experiment/HAZE4K/smoke-H4K-settings-debug-20260519-230246.log`
- `experiment/HAZE4K/eval-H4K-official-smoke-20260519-230527.log`

删除后已验证：

- 上述 19 个目标均不存在。
- baseline `DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk` 仍存在。
- LF-v1 `DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk` 仍存在。
- Conservative / CRPlus / LowFreqLoss / LF-LowFreqLoss / TeacherGuard / PostMix 等 `10k+` 失败消融 `best.pk` 仍存在。
- `per_image_eval/CR-vs-LF-v1-full-20260523/summary.json` 仍存在。
- `visual_compare/DEA-Net-CR-vs-LF-20260522/summary.json` 仍存在。
- `eval-H4K-official-full-20260520-095415` 仍存在。
- GPU 状态为空闲：`0 MiB / 0%`。

## 3.3 Stage 4 Execution Log

执行时间：2026-05-23

根据用户确认，本阶段清理 benchmark 产生的 checkpoint 目录，只保留 benchmark 汇总和日志。删除前已复核：

- `bench-H4K-speed-bs16-20260520-170842`、`bench-H4K-speed-bs24-20260520-171257`、`bench-H4K-speed-bs32-20260520-171827` 均为速度测试目录，`max_observed_step=1`，主要包含速度测试时生成的 `best.pk` / `latest.pk`。
- `bs_speed_benchmark_20260520-170637` 是早期 benchmark 尝试目录，只有小型 log / summary，无训练 checkpoint。
- 正式 benchmark 汇总目录 `bs_speed_benchmark_20260520-170841` 和 `bs_speed_benchmark_latest.log` 保留。

已删除：

- `experiment/HAZE4K/bench-H4K-speed-bs16-20260520-170842`
- `experiment/HAZE4K/bench-H4K-speed-bs24-20260520-171257`
- `experiment/HAZE4K/bench-H4K-speed-bs32-20260520-171827`
- `experiment/HAZE4K/bs_speed_benchmark_20260520-170637`

删除后已验证：

- 上述 4 个目标均不存在。
- `experiment/HAZE4K/bs_speed_benchmark_20260520-170841` 仍存在。
- `experiment/HAZE4K/bs_speed_benchmark_latest.log` 仍存在。
- baseline、LF-v1 和所有 `10k+` 失败消融的 `best.pk` 仍存在。
- GPU 状态为空闲：`0 MiB / 0%`。

## 3.4 Local Workspace Execution Log

执行时间：2026-05-23

已完成本地整理：

- 删除 Python 可再生成缓存：`code/**/__pycache__/`。
- 将根目录论文方案 `.docx` 移出 Git 仓库，保存在 `D:\Dehaze\reference\目前图像去雾基线模型深度研究与毕业论文改进方案建议.docx`。
- 将失败消融的一次性 launcher 归档到 `scripts/archive/failed-ablation-launchers/`：
  - `runyun-haze4k-crplus-scout.sh`
  - `runyun-haze4k-lowfreq-loss-scout.sh`
  - `runyun-haze4k-lf-lowfreq-loss-scout.sh`
  - `runyun-haze4k-lf-teacher-guard-scout.sh`

未删除任何本地训练代码、实验文档或可复查的失败消融脚本。

## 3.5 Documentation Cleanup Review

当前不建议直接删除任何 `docs/*.md`。这些文档的角色不同：

- `CURRENT_CONTEXT.md`：新会话入口和服务器/路径状态。
- `DEA_NET_LFCR_HAZE4K_PLAN.md`：当前 HAZE4K 训练测试主计划。
- `EXPERIMENT_LOG.md`：按时间记录所有 run、指标和 stop/continue 决策。
- `HAZE4K_RUN_MANIFEST.md`：远端 artifact 索引和保留策略。
- `HAZE4K_FAILURE_ANALYSIS_20260523.md`：多轮失败原因和下一步路线依据。
- `docs/archive/HAZE4K_CLEANUP_PLAN_20260523.md`：清理审计记录，后续只作为维护记录。
- `CORE_SERVER_RUNBOOK.md` / `WORKFLOW.md`：服务器操作和协作流程，仍有用。
- `DOWNLOADS.md` / `REPRODUCTION.md`：上游复现和下载背景，低频使用但保留成本低。

建议：

1. 不为减少文件数量而删除文档。
2. 将 `CURRENT_CONTEXT.md` 作为唯一导航入口，已补充 HAZE4K 文档角色说明。
3. 后续如果要压缩文档数量，优先把 `docs/archive/HAZE4K_CLEANUP_PLAN_20260523.md` 视为 dated audit，不再作为日常入口；不建议删除。
4. `EXPERIMENT_LOG.md` 仍保留，因为它是按时间追踪训练结果的证据链；不要被 `RUN_MANIFEST` 替代。

## 4. Stage 1 - Local Workspace Organization

### 4.1 建议保留并提交

这些文件应进入后续代码/文档基线：

- `code/evaluate_train_ckpt_per_image.py`
- `docs/HAZE4K_FAILURE_ANALYSIS_20260523.md`
- `docs/HAZE4K_RUN_MANIFEST.md`
- `docs/archive/HAZE4K_CLEANUP_PLAN_20260523.md`

### 4.2 建议确认后清理

本地安全清理候选：

- `code/**/__pycache__/`
- 根目录 `目前图像去雾基线模型深度研究与毕业论文改进方案建议.docx`

建议处理方式：

- `__pycache__`：直接删除。
- `.docx`：移动到 repo 外，例如 `D:\Dehaze\reference\`；如果要留在 repo 内，应放到明确忽略的目录，不要留在根目录。

### 4.3 脚本整理建议

当前新增且未跟踪的一次性脚本：

- `scripts/runyun-haze4k-crplus-scout.sh`
- `scripts/runyun-haze4k-lowfreq-loss-scout.sh`
- `scripts/runyun-haze4k-lf-lowfreq-loss-scout.sh`
- `scripts/runyun-haze4k-lf-teacher-guard-scout.sh`

建议不要长期散落在 `scripts/` 顶层。更好的整理方式：

1. 保留 `runyun-haze4k-lf-scout.sh` 作为 LF 通用 scout 入口。
2. 新增一个参数化脚本，例如 `scripts/runyun-haze4k-scout.sh`，通过环境变量选择 variant。
3. 将失败实验专用 launcher 放入 `scripts/archive/failed-ablation-launchers/`，或只在文档中保留命令，不提交脚本。

该阶段涉及移动文件，执行前需要确认。

## 5. Stage 2 - Remote Immediate Delete Candidates

这些是误启动、错误命名或半截 run。建议第一批删除，但仍需确认。

```bash
cd /root/workspace/Dehaze-Net
du -sh \
  experiment/HAZE4K/probe \
  experiment/HAZE4K/DEA-Net-LF-TeacherGuard-H4K-scout-20260523- \
  experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-002900 \
  experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-bs32-20260520-165148
```

确认后才执行：

```bash
cd /root/workspace/Dehaze-Net
rm -rf -- \
  experiment/HAZE4K/probe \
  experiment/HAZE4K/DEA-Net-LF-TeacherGuard-H4K-scout-20260523- \
  experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-002900 \
  experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-bs32-20260520-165148

rm -f -- \
  experiment/HAZE4K/_run_logs/probe.log \
  experiment/HAZE4K/_run_logs/DEA-Net-LF-TeacherGuard-H4K-scout-20260523-.log \
  experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-bs32-20260520-165148.log
```

预计释放空间不大，但能减少误读。

## 6. Stage 3 - Remote Smoke Cleanup Candidates

这些目录每个约 `180M`，多数只用于验证能启动。manifest 已记录后，建议第二批删除。

确认前复核：

```bash
cd /root/workspace/Dehaze-Net
du -sh \
  experiment/HAZE4K/DEA-Net-CR-smoke-HAZE4K-20260514-032218 \
  experiment/HAZE4K/DEA-Net-CR-smoke-HAZE4K-20260515-102937 \
  experiment/HAZE4K/smoke-H4K-LF-train-20260521-0028 \
  experiment/HAZE4K/smoke-H4K-bs32-vram-20260520-164745 \
  experiment/HAZE4K/smoke-H4K-settings-20260519-230502 \
  experiment/HAZE4K/smoke-H4K-lf-postmix- \
  experiment/HAZE4K/smoke-lf-teacher-guard-20260523 \
  experiment/HAZE4K/smoke-H4K-CRPlus-P1-20260523-011035 \
  experiment/HAZE4K/smoke-H4K-LowFreqLoss-20260523-015512 \
  experiment/HAZE4K/smoke-H4K-LF-LowFreqLoss-20260523-031517
```

确认后才执行：

```bash
cd /root/workspace/Dehaze-Net
rm -rf -- \
  experiment/HAZE4K/DEA-Net-CR-smoke-HAZE4K-20260514-032218 \
  experiment/HAZE4K/DEA-Net-CR-smoke-HAZE4K-20260515-102937 \
  experiment/HAZE4K/smoke-H4K-LF-train-20260521-0028 \
  experiment/HAZE4K/smoke-H4K-bs32-vram-20260520-164745 \
  experiment/HAZE4K/smoke-H4K-settings-20260519-230502 \
  experiment/HAZE4K/smoke-H4K-lf-postmix- \
  experiment/HAZE4K/smoke-lf-teacher-guard-20260523 \
  experiment/HAZE4K/smoke-H4K-CRPlus-P1-20260523-011035 \
  experiment/HAZE4K/smoke-H4K-LowFreqLoss-20260523-015512 \
  experiment/HAZE4K/smoke-H4K-LF-LowFreqLoss-20260523-031517
```

小型 smoke/debug 目录也可确认删除：

```bash
rm -rf -- \
  experiment/HAZE4K/smoke-H4K-settings-20260519-230213 \
  experiment/HAZE4K/smoke-H4K-settings-debug-20260519-230246 \
  experiment/HAZE4K/smoke-H4K-CRPlus-P1-20260523-010932 \
  experiment/HAZE4K/eval-H4K-official-smoke-20260519-230527
```

## 7. Stage 4 - Benchmark Cleanup

保留：

- `experiment/HAZE4K/bs_speed_benchmark_20260520-170841`
- `experiment/HAZE4K/bs_speed_benchmark_latest.log`

可确认删除生成 checkpoint 的 benchmark model dirs：

```bash
cd /root/workspace/Dehaze-Net
rm -rf -- \
  experiment/HAZE4K/bench-H4K-speed-bs16-20260520-170842 \
  experiment/HAZE4K/bench-H4K-speed-bs24-20260520-171257 \
  experiment/HAZE4K/bench-H4K-speed-bs32-20260520-171827 \
  experiment/HAZE4K/bs_speed_benchmark_20260520-170637
```

## 8. Stage 5 - Negative Ablation Minimalization

这一步不建议现在做。等论文/报告是否需要复查失败 checkpoint 后再决定。

候选目录：

- `DEA-Net-LF-Conservative-H4K-scout-20260522-145904`
- `DEA-Net-CRPlus-P1-w005-H4K-scout-20260523-011100`
- `DEA-Net-LowFreqLoss-w005-H4K-scout-20260523-015600`
- `DEA-Net-LF-LowFreqLoss-w001-H4K-scout-20260523-031600`
- `DEA-Net-LF-TeacherGuard-H4K-scout-20260523-112658`
- `DEA-Net-LF-PostMix-H4K-scout-20260523-133020`

如果未来要瘦身，建议只删这些子目录：

- `tensorboard/`
- `saved_plot/`
- `saved_infer/`

暂时不要删：

- `args.txt`
- `saved_data/log.txt`
- `saved_data/*.npy`
- `saved_model/best.pk`
- `saved_model/latest.pk`

## 9. Verification After Cleanup

每一批删除后执行：

```bash
cd /root/workspace/Dehaze-Net
find experiment/HAZE4K -maxdepth 2 -type d | sort
du -sh experiment/HAZE4K/*
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
```

本地执行：

```powershell
cd D:\Dehaze\Dehaze-Net
git status --short --branch
git ls-files -o --exclude-standard
```

## 10. Current Confirmation Checkpoint

Stage 2、Stage 3、Stage 4 和本地整理已完成。当前不再建议继续删除训练结果目录或文档。

如需下一步整理，建议只在用户确认后选择其一：

1. 建立一个参数化 HAZE4K scout launcher，替代多个一次性 launcher。
2. 本文件已移到 `docs/archive/`；前提是 `CURRENT_CONTEXT.md` 和 `HAZE4K_RUN_MANIFEST.md` 已覆盖日常入口信息。
3. Stage 5：只对失败消融做轻量瘦身；但根据当前规则，所有 `10k+` 且与模型尝试有关的 run 暂不处理。

收到进一步确认前，不应删除任何远端或本地数据。
