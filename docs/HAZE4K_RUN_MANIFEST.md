# HAZE4K Run Manifest

日期：2026-05-23

用途：作为 HAZE4K 训练、评估、可视化和失败消融的统一索引。后续删除或归档远端 `experiment/HAZE4K` 目录前，先查本文件，避免误删仍有分析价值的数据。

本文件来自只读盘点，没有删除或移动任何远端/本地数据。

## Keep Policy

| Policy | 含义 | 清理规则 |
| --- | --- | --- |
| `KEEP` | 当前主参考或正向候选 | 保留完整目录和 checkpoint |
| `KEEP_MINIMAL` | 失败消融，但仍有论文/分析价值 | 至少保留 `args.txt`, `saved_data/log.txt`, `saved_model/best.pk`, `saved_model/latest.pk`; 可在确认后删 TensorBoard、plots、infer 临时图 |
| `KEEP_SUMMARY` | 官方评估或汇总证据 | 保留 log / args / summary，小目录可原样保留 |
| `DELETE_AFTER_MANIFEST` | smoke 运行，只证明流程可启动 | 本 manifest 和日志记录充分后，可确认删除 |
| `DELETE_AFTER_CONFIRM` | 误启动、半截、错误命名或无有效结果 | 需要用户确认后删除 |
| `KEEP_SUMMARY_DELETE_MODEL_AFTER_CONFIRM` | benchmark 已有汇总，模型 checkpoint 价值低 | 保留 benchmark summary/log，确认后删大模型目录 |

## Fair-Comparison Boundary

HAZE4K 正式候选只承认从启动时就固定为 `100000` total steps 的 run：
`epochs=20`, `iters_per_epoch=5000`, `bs=16`, `patch_size=256`,
`w_loss_L1=1.0`, `w_loss_CR=0.1`, `start_lr=0.0001`,
`end_lr=0.000001`, 每 `10000` step 保存和验证。`20k` / `50k` 只能是该
`100k` run 内部的中途 gate。任何 `T=20000`、`T=50000` 或 resume 后改变
`epochs * iters_per_epoch` 的记录，都只能保留为 smoke/诊断/误启动证据，不进入
baseline、LF-v1 或新候选的公平对比表。

## Core Runs

| Run ID | Size | Verdict | Best / Final | Evidence | Policy |
| --- | ---: | --- | --- | --- | --- |
| `DEA-Net-CR-H4K-Baseline-scout-20260520-101334` | 186M | baseline reference | best 90k `32.2255 / 0.9844`; final 100k `32.0952 / 0.9844` | `args.txt`, `saved_data/log.txt`, `saved_model/best.pk`, `latest.pk` | `KEEP` |
| `DEA-Net-LF-H4K-scout-20260521-003100` | 184M | positive candidate | best 90k `32.4281 / 0.9845`; final 100k `32.3857 / 0.9845` | LF-v1 main positive result | `KEEP` |
| `DEA-Net-LF-Conservative-H4K-scout-20260522-145904` | 184M | negative ablation | best/final 100k `32.1083 / 0.9843` | over-constrained LF evidence | `KEEP_MINIMAL` |
| `DEA-Net-CRPlus-P1-w005-H4K-scout-20260523-011100` | 180M | negative ablation | 10k `24.9623 / 0.9504` | low-pass hazy negative failed | `KEEP_MINIMAL` |
| `DEA-Net-LowFreqLoss-w005-H4K-scout-20260523-015600` | 181M | negative ablation | 20k `27.8852 / 0.9716` | low-frequency reconstruction loss failed | `KEEP_MINIMAL` |
| `DEA-Net-LF-LowFreqLoss-w001-H4K-scout-20260523-031600` | 182M | negative ablation | 50k `31.0707 / 0.9811` | LF-v1 + lowfreq loss failed | `KEEP_MINIMAL` |
| `DEA-Net-LF-TeacherGuard-H4K-scout-20260523-112658` | 181M | negative ablation | 20k `27.7075 / 0.9704` | current teacher guard setting failed | `KEEP_MINIMAL` |
| `DEA-Net-LF-PostMix-H4K-scout-20260523-133020` | 182M | negative ablation | 50k `30.7103 / 0.9814` | post-mix structure failed | `KEEP_MINIMAL` |
| `DEA-Net-LF-ConditionalMask-H4K-gate20k-20260523-205312` | 181M | invalid for fair comparison; diagnostic only | 20k `29.0625 / 0.9734`; mask near-constant `mean~0.878735`, `std~6.85e-05` | short schedule `T=20000`; exclude from candidate tables and never resume for formal 50k comparison | `KEEP_MINIMAL` |
| `DEA-Net-LF-ConditionalMask-H4K-clean50k-20260523-224051` | small/partial | invalid launch | no formal metrics | launched with `T=50000`, then stopped after fairness correction; keep log only if needed | `DELETE_AFTER_CONFIRM` |
| `DEA-Net-LF-ConditionalMask-H4K-scout100k-20260523-224315` | 181M+ | paused fair candidate | 10k `27.1085 / 0.9638`; 20k `28.8571 / 0.9724`; latest/best at 20k | launched from commit `09880be` with `T=100000`; paused on 2026-05-24 around log step 21500; resume only with same 100k horizon | `KEEP` |

## Evaluation And Visual Evidence

| Path | Verdict | Contents | Policy |
| --- | --- | --- | --- |
| `experiment/HAZE4K/per_image_eval/CR-vs-LF-v1-full-20260523` | core full-test analysis | `summary.json`, `per_image_metrics.csv`, `group_summary.csv`, `hard_cases.json`, `analysis_report.md` | `KEEP` |
| `experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-20260522` | fixed-sample LF-v1 diagnosis | fixed samples, metrics, selected panels, objective analysis | `KEEP` |
| `experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-Conservative-20260522` | Conservative LF visual failure | fixed-sample comparison and objective analysis | `KEEP_MINIMAL` |
| `experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-gate-sweep-20260522` | gate sweep diagnosis | scale summaries, panels, objective metrics | `KEEP_MINIMAL` |
| `experiment/HAZE4K/eval-H4K-official-full-20260520-095415` | official checkpoint full eval | official `.pth` reference `34.2556 / 0.9885` | `KEEP_SUMMARY` |
| `experiment/HAZE4K/eval-HAZE4K` | old eval artifact | early official eval style artifact | `KEEP_SUMMARY` |
| `experiment/HAZE4K/eval-HAZE4K-newserver` | old eval artifact | new-server eval smoke/reference artifact | `KEEP_SUMMARY` |

## Delete-After-Confirm Candidates

这些目录来自误启动、错误命名、空/半截 run，当前没有继续分析价值。删除前仍需用户明确确认。

| Path | Size | Reason |
| --- | ---: | --- |
| `experiment/HAZE4K/probe` | 48K | deleted 2026-05-23; unintended `MODEL_NAME=probe` process, no valid eval |
| `experiment/HAZE4K/_run_logs/probe.log` | small | deleted 2026-05-23; probe log |
| `experiment/HAZE4K/DEA-Net-LF-TeacherGuard-H4K-scout-20260523-` | 44K | deleted 2026-05-23; malformed one-shot launch name, no valid eval |
| `experiment/HAZE4K/_run_logs/DEA-Net-LF-TeacherGuard-H4K-scout-20260523-.log` | small | deleted 2026-05-23; malformed run log |
| `experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-002900` | 8K | deleted 2026-05-23; empty/partial LF launch |
| `experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-bs32-20260520-165148` | 44K | deleted 2026-05-23; cancelled speed probe before useful checkpoint |

## Smoke Runs

这些 run 主要证明某个代码路径能启动。manifest 已记录后，可以确认删除远端目录以节省空间。它们不应作为论文结果。

| Run ID | Size | Notes | Policy |
| --- | ---: | --- | --- |
| `DEA-Net-CR-smoke-HAZE4K-20260514-032218` | 180M | deleted 2026-05-23; early smoke, max observed step 1 | `DELETED` |
| `DEA-Net-CR-smoke-HAZE4K-20260515-102937` | 180M | deleted 2026-05-23; early smoke, max observed step 1 | `DELETED` |
| `smoke-H4K-LF-train-20260521-0028` | 180M | deleted 2026-05-23; LF smoke, max observed step 1 | `DELETED` |
| `smoke-H4K-bs32-vram-20260520-164745` | 180M | deleted 2026-05-23; bs32 VRAM smoke, max observed step 1 | `DELETED` |
| `smoke-H4K-settings-20260519-230502` | 180M | deleted 2026-05-23; validated HAZE4K settings smoke, max observed step 1 | `DELETED` |
| `smoke-H4K-lf-postmix-` | 180M | deleted 2026-05-23; malformed smoke name, postmix smoke, max observed step 1 | `DELETED` |
| `smoke-lf-teacher-guard-20260523` | 180M | deleted 2026-05-23; teacher guard smoke, max observed step 1 | `DELETED` |
| `smoke-H4K-CRPlus-P1-20260523-011035` | 180M | deleted 2026-05-23; CRPlus smoke, max observed step 1 | `DELETED` |
| `smoke-H4K-LowFreqLoss-20260523-015512` | 180M | deleted 2026-05-23; LowFreqLoss smoke, max observed step 1 | `DELETED` |
| `smoke-H4K-LF-LowFreqLoss-20260523-031517` | 180M | deleted 2026-05-23; LF + LowFreqLoss smoke, max observed step 1 | `DELETED` |
| `smoke-H4K-settings-20260519-230213` | 28K | deleted 2026-05-23; failed/early smoke, no valid training step | `DELETED` |
| `smoke-H4K-settings-debug-20260519-230246` | 28K | deleted 2026-05-23; debug smoke, no valid training step | `DELETED` |
| `smoke-H4K-CRPlus-P1-20260523-010932` | 28K | deleted 2026-05-23; dry/failed smoke, no valid training step | `DELETED` |
| `eval-H4K-official-smoke-20260519-230527` | 12K | deleted 2026-05-23; official eval smoke; full eval exists | `DELETED` |

## Benchmark Runs

| Run ID | Size | Notes | Policy |
| --- | ---: | --- | --- |
| `bs_speed_benchmark_20260520-170841` | 116K | useful benchmark summary/logs | `KEEP_SUMMARY` |
| `bs_speed_benchmark_20260520-170637` | 28K | deleted 2026-05-23; earlier benchmark attempt | `DELETED` |
| `bench-H4K-speed-bs16-20260520-170842` | 180M | deleted 2026-05-23; generated checkpoint not needed after speed summary | `DELETED` |
| `bench-H4K-speed-bs24-20260520-171257` | 180M | deleted 2026-05-23; generated checkpoint not needed after speed summary | `DELETED` |
| `bench-H4K-speed-bs32-20260520-171827` | 180M | deleted 2026-05-23; generated checkpoint not needed after speed summary | `DELETED` |

## Local Workspace Cleanup Snapshot

Snapshot from the 2026-05-23 cleanup pass. Re-check `git status` before using
this as current truth.

- `code/evaluate_train_ckpt_per_image.py`: keep, useful analysis tool.
- `docs/HAZE4K_FAILURE_ANALYSIS_20260523.md`: keep, failure analysis.
- `docs/HAZE4K_OPTIMIZATION_WORKFLOW_REVIEW_20260523.md`: keep, method/process review for future experiment discipline.
- `docs/HAZE4K_CONDITIONAL_LF_ROUTE_AUDIT_20260523.md`: keep, pre-implementation route audit and experiment card for Conditional LF.
- `scripts/archive/failed-ablation-launchers/`: archived failed-ablation launchers retained for reproducibility, not used as daily entrypoints.
- `D:\Dehaze\reference\目前图像去雾基线模型深度研究与毕业论文改进方案建议.docx`: moved outside repo root on 2026-05-23.

Local `code/**/__pycache__` directories were deleted on 2026-05-23.

## Recommended Next State Before New Training

1. Commit or otherwise snapshot the analysis tools and docs that should survive.
2. Start the next experiment from a clean branch, ideally `codex/haze4k-lf-conditional-mask`.
3. Consider replacing archived one-off failed launchers with one parameterized scout launcher.
