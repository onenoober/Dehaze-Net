# HAZE4K Run Manifest

日期：2026-05-25

用途：作为 HAZE4K 训练、评估、可视化和失败消融的统一索引。后续删除或归档远端 `experiment/HAZE4K` 目录前，先查本文件，避免误删仍有分析价值的数据。

本文件来自只读盘点，没有删除或移动任何远端/本地数据。

## 2026-05-26 WSL Sync Snapshot

Local WSL is the durable experiment artifact home. GitHub should keep only code,
docs, commands, compact metrics, and conclusions. Cloud servers such as
`runyun-ts` are temporary compute nodes.

The 2026-05-26 small-evidence sync from `runyun-ts` is stored locally under
ignored path `experiment/sync/runyun-ts-20260526/`:

- `small-evidence-files.txt`: 236 compact evidence files selected from cloud
  `experiment/HAZE4K`.
- `small-evidence.tgz`: compact transfer archive, about 6.5 MB.
- `cloud-status.txt`, `cloud-diff-name-status.txt`, `cloud-untracked.txt`, and
  `cloud-uncommitted.patch`: inspection records from the older dirty server
  checkout.
- `cloud-untracked-small.tgz` and `untracked-small/`: compact untracked docs
  and scripts that were small enough to inspect locally.
- `large-artifacts.txt`: 5017 larger candidates intentionally not synced.

The compact evidence was extracted into local ignored `experiment/HAZE4K/`.
Do not add these files to Git. Sync selected large artifacts only when
explicitly requested.

## Keep Policy

| Policy | 含义 | 清理规则 |
| --- | --- | --- |
| `KEEP` | 当前主参考或正向候选 | 保留完整目录和 checkpoint |
| `KEEP_MINIMAL` | 失败消融，但仍有论文/分析价值 | 至少保留 `args_initial.txt`（新 run）, `args_history.jsonl`（新 run）, `args.txt`, `saved_data/log.txt`, `saved_model/best.pk`, `saved_model/latest.pk`; 可在确认后删 TensorBoard、plots、infer 临时图 |
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
| `DEA-Net-LF-ResidualCalib-H4K-scout100k-20260525-122654` | 182M+ | positive ablation; below LF-v1 | best 90k `32.3936 / 0.9845`; final 100k `32.3858 / 0.9846` | residual direction/amplitude calibration beat CR baseline by `+0.1682 dB` full-test mean delta but trailed LF-v1 by `-0.0347 dB`; alpha branch active; useful current mechanism evidence | `KEEP` |
| `DEA-Net-LF-ResidualSelector-H4K-scout100k-20260525-223844` | 180M+ | negative fair ablation; stopped at 20k gate | 20k `27.6830 / 0.9713` | launched from commit `7c93000`; fair `100000`-step selector run with `lf_residual_calibration=True`, `lf_residual_selector=True`, `lf_selector_init_bias=2.0`; 10k soft-pass was `26.7739 / 0.9615`, but 20k lagged baseline by `-1.2200 dB`, LF-v1 by `-1.1733 dB`, and ResidualCalib by `-0.8175 dB`; selector remained near init (`mean/std 0.879248/0.000051`), so stopped by PGID `50061` before 30k; do not resume this exact setting | `KEEP_MINIMAL` |
| `DEA-Net-LF-ResidualDirLoss-w005-H4K-scout100k-20260526-103853` | 180M+ | negative fair ablation; stopped at 30k hard gate | 30k `30.1058 / 0.9779` | launched from commit `521392c`; fair `100000`-step LF-v1 plus `w_loss_residual_dir=0.005`; 20k recovered to `28.9783 / 0.9733`, but 30k tied baseline and lagged LF-v1 by `-0.5195 dB` on full validation; route-specific diagnostics on a 64-image test subset were supporting mechanism evidence and also failed: direct direction loss `0.04645` worse than LF-v1-best `0.03565` and ResidualCalib-best `0.03330`, direct residual cosine `0.95355` worse than `0.96435`/`0.96670`, CR-relative wrong-direction `30/64`, LF MSE improved/regressed `17/47`; stopped by PGID `7896` at about log step `35776` before 40k, verified no process/tmux and GPU `0 MiB / 0%`; do not resume this exact setting | `KEEP_MINIMAL` |
| `DEA-Net-LF-Conservative-H4K-scout-20260522-145904` | 184M | negative ablation | best/final 100k `32.1083 / 0.9843` | over-constrained LF evidence | `KEEP_MINIMAL` |
| `DEA-Net-CRPlus-P1-w005-H4K-scout-20260523-011100` | 180M | negative ablation | 10k `24.9623 / 0.9504` | low-pass hazy negative failed | `KEEP_MINIMAL` |
| `DEA-Net-LowFreqLoss-w005-H4K-scout-20260523-015600` | 181M | negative ablation | 20k `27.8852 / 0.9716` | low-frequency reconstruction loss failed | `KEEP_MINIMAL` |
| `DEA-Net-LF-LowFreqLoss-w001-H4K-scout-20260523-031600` | 182M | negative ablation | 50k `31.0707 / 0.9811` | LF-v1 + lowfreq loss failed | `KEEP_MINIMAL` |
| `DEA-Net-LF-TeacherGuard-H4K-scout-20260523-112658` | 181M | negative ablation | 20k `27.7075 / 0.9704` | current teacher guard setting failed | `KEEP_MINIMAL` |
| `DEA-Net-LF-PostMix-H4K-scout-20260523-133020` | 182M | negative ablation | 50k `30.7103 / 0.9814` | post-mix structure failed | `KEEP_MINIMAL` |
| `DEA-Net-LF-ConditionalMask-H4K-gate20k-20260523-205312` | 181M | invalid for fair comparison; diagnostic only | 20k `29.0625 / 0.9734`; mask near-constant `mean~0.878735`, `std~6.85e-05` | short schedule `T=20000`; exclude from candidate tables and never resume for formal 50k comparison | `KEEP_MINIMAL` |
| `DEA-Net-LF-ConditionalMask-H4K-clean50k-20260523-224051` | small/partial | invalid launch | no formal metrics | launched with `T=50000`, then stopped after fairness correction; keep log only if needed | `DELETE_AFTER_CONFIRM` |
| `DEA-Net-LF-ConditionalMask-H4K-scout100k-20260523-224315` | 181M+ | negative fair ablation; stopped at 30k hard gate | 30k `30.1830 / 0.9783`; mask near-constant `mean~0.878747`, `std~0.000085` | launched from commit `09880be` with `T=100000`; resumed with same horizon on 2026-05-24; watcher stopped after 30k because it remained far below LF-v1 30k `30.6253 / 0.9783`; do not resume this exact setting | `KEEP_MINIMAL` |
| `DEA-Net-LF-HazeAwareMask-H4K-scout100k-20260524-152758` | 181M+ | negative fair ablation; stopped at 30k hard gate | 30k `30.1157 / 0.9770`; mask active `std` last `0.001312` | isolated remote copy `/root/workspace/Dehaze-Net-lf-v2-verify`; fair `T=100000` launch with haze-aware conditional LF mask; 30k tied baseline but lagged LF-v1 by `-0.5096 dB`, so stopped by PGID `17326`; useful evidence that active dark-channel/luma selection alone is not enough | `KEEP_MINIMAL` |

## Evaluation And Visual Evidence

| Path | Verdict | Contents | Policy |
| --- | --- | --- | --- |
| `experiment/HAZE4K/per_image_eval/CR-vs-LF-v1-full-20260523` | core full-test analysis | `summary.json`, `per_image_metrics.csv`, `group_summary.csv`, `hard_cases.json`, `analysis_report.md` | `KEEP` |
| `experiment/HAZE4K/per_image_eval/CR-vs-ResidualCalib-full-20260525` | ResidualCalib full-test baseline comparison | 1000 images; mean delta `+0.1682 dB`; better/worse by PSNR `547/453`; weak-baseline gain and strong-baseline regression split | `KEEP` |
| `experiment/HAZE4K/per_image_eval/LF-v1-vs-ResidualCalib-full-20260525` | ResidualCalib full-test LF-v1 comparison | 1000 images; mean delta `-0.0347 dB`; better/worse `509/491`; confirms positive ablation but not LF-v1 replacement | `KEEP` |
| `experiment/HAZE4K/residual_diagnostic/CR-vs-LF-v1-20260524` | LF-v1 root-cause diagnostic | residual direction/magnitude `summary.json`, `per_image_residual_metrics.csv`, `group_summary.csv`, `hard_cases.json`, `analysis_report.md`; corr(delta PSNR, residual cosine) `0.8775` | `KEEP` |
| `experiment/HAZE4K/residual_diagnostic/CR-vs-ResidualCalib-20260525` | ResidualCalib root-cause diagnostic vs CR | residual direction/magnitude summary; wrong-direction `163`, LF MSE improved/regressed `554/446`, corr(delta PSNR, residual cosine) `0.8490` | `KEEP` |
| `experiment/HAZE4K/residual_diagnostic/LF-v1-vs-ResidualCalib-20260525` | ResidualCalib root-cause diagnostic vs LF-v1 | residual direction/magnitude summary; wrong-direction `211`, LF MSE improved/regressed `512/488`, corr(delta PSNR, residual cosine) `0.8580` | `KEEP` |
| `experiment/HAZE4K/loss_scale/residual-dir-hardgate-review-20260526` | ResidualDirLoss 30k route-specific hard-gate review | 64-image test subset direct direction-loss/cosine comparison across CR, LF-v1, ResidualCalib, and ResidualDirLoss-30k; shows ResidualDirLoss-30k has worse direction loss/cosine | `KEEP_MINIMAL` |
| `experiment/HAZE4K/residual_diagnostic/residual-dir-hardgate-review-20260526` | ResidualDirLoss 30k CR-relative residual diagnostic | 64-image test subset residual diagnostic; ResidualDirLoss-30k wrong-direction `30/64`, LF MSE improved/regressed `17/47`, worse than LF-v1-best and ResidualCalib-best on this review set | `KEEP_MINIMAL` |
| `experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-20260522` | fixed-sample LF-v1 diagnosis | fixed samples, metrics, selected panels, objective analysis | `KEEP` |
| `experiment/HAZE4K/visual_compare/ResidualCalib-hardcases-20260525` | shared hardcase sample list | selected CR and LF-v1 relative improvement/regression samples for ResidualCalib visual triage | `KEEP` |
| `experiment/HAZE4K/visual_compare/CR-vs-ResidualCalib-hardcases-20260525` | ResidualCalib hardcase visual/objective analysis vs CR | hardcase panels and objective report; mixed outcome with color/tone risk but net positive over CR | `KEEP` |
| `experiment/HAZE4K/visual_compare/LF-v1-vs-ResidualCalib-hardcases-20260525` | ResidualCalib hardcase visual/objective analysis vs LF-v1 | hardcase panels and objective report; mixed outcome and larger LF-v1-relative hardcase regression | `KEEP` |
| `experiment/HAZE4K/three_way_eval/Baseline-LFv1-ResidualCalib-full-20260525` | three-way final-output analysis | baseline/LF-v1/ResidualCalib 1000-image metrics, group summary, hard cases, selected panels and heatmaps; winner counts `295/322/383` | `KEEP` |
| `experiment/HAZE4K/visual_compare/Baseline-LFv1-ResidualCalib-fixed20260522-20260525` | old fixed-sample three-way visual comparison | reruns the original `DEA-Net-CR-vs-LF-20260522/samples.txt` with ResidualCalib added; includes full outputs, all panels, heatmaps, and report | `KEEP` |
| `experiment/HAZE4K/three_way_eval/Baseline-LFv1-ResidualCalib-preview-20260525` | compressed three-way preview | small JPEG contact sheet for quick visual review of representative full-test and fixed-sample patterns | `KEEP_SUMMARY` |
| `experiment/HAZE4K/selector_oracle/Baseline-LFv1-ResidualCalib-full-20260525` | selector/oracle route decision | reads existing three-way CSV; full 1000-image two-way oracle LF-v1/ResidualCalib `33.0034 / 0.985299`, `+0.5751 dB` over LF-v1; three-way oracle `33.2538 / 0.985637`, `+0.8255 dB`; justifies learned selector but best simple rule is GT-aware | `KEEP` |
| `experiment/HAZE4K/selector_oracle/Baseline-LFv1-ResidualCalib-fixed20260522-20260525` | selector/oracle old adverse subset check | old 20-sample subset two-way oracle `31.8019 / 0.983492`, `+0.5439 dB` over LF-v1; three-way oracle `32.1661 / 0.983806`; diagnostic only, useful for showing the selected adverse samples also have oracle headroom, not for estimating dataset-level selector value | `KEEP` |
| `experiment/HAZE4K/selector_proxy/Baseline-LFv1-ResidualCalib-full-20260526` | initial selector proxy learnability audit | reads existing three-way CSV; 5 held-out splits show safe proxies fail the selector-v2 pass line: best safe `metadata_proxy` ridge logistic `+0.0508 dB`, oracle recovery `0.0876`, precision `0.5241`; retained as the first audit, but the strict artifact below is the current gate source | `KEEP_SUMMARY` |
| `experiment/HAZE4K/selector_proxy/Baseline-LFv1-ResidualCalib-full-strict-20260526` | strict selector proxy learnability audit | current selector-v2 gate source; safe output proxy uses explicit whitelist plus script-generated candidate-output deltas, writes `feature_lists.json`, and validates safe features; 5 held-out splits still fail: best safe `metadata_proxy` ridge logistic `+0.0508 dB`, recovery `0.0876`, precision `0.5241`; `output_proxy` ridge logistic only `+0.0268 dB`, recovery `0.0459`, precision `0.5231`; GT-aware leakage check recovers `+0.5796 dB`, so target is real but not inference-safe | `KEEP_SUMMARY` |
| `experiment/HAZE4K/selector_proxy/Baseline-LFv1-ResidualCalib-full-rich-20260526` | rich selector proxy learnability audit | current most complete CSV-derived proxy gate; evaluates 280 metadata-free rich output/agreement features from the 1000-image three-way CSV over random and degradation-held-out splits; best random metadata-free row `+0.0956 dB`, recovery `0.1643`, precision `0.6046`, but airlight/beta/combo held-out rows are only `+0.0695/+0.0755/+0.0656 dB`; all fail pass line, so do not train selector-v2 from CSV-derived proxy evidence | `KEEP_SUMMARY` |
| `experiment/HAZE4K/selector_proxy/Baseline-LFv1-ResidualCalib-full-activation-20260526` | activation-forward selector proxy learnability audit | current final deployable-proxy gate; forwards frozen CR baseline, LF-v1, and ResidualCalib best checkpoints over all 1000 HAZE4K test images; sample-size gate passed with target positives/negatives `509/491` and min split class count `55`; best metadata-free activation rows still fail: random `+0.0748 dB`, recovery `0.1247`, precision `0.5978`, with held-out families lower; GT leakage still near oracle, so stop selector-v2 search unless a future supervised/distilled selector target passes a fresh full-sample audit | `KEEP_SUMMARY` |
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
| `smoke-H4K-LF-ResidualSelector-20260525-223437` | TBD | selector code-path smoke; diagnostic `T=2`, wrote `latest.pk` at step 2 and logged selector/alpha stats; invalid for fair comparison | `KEEP_SUMMARY_DELETE_MODEL_AFTER_CONFIRM` |
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

1. Re-check `docs/CURRENT_CONTEXT.md` and `docs/README.md` before deciding what to load or edit next.
2. Do not resume the current Conditional LF, LF-v2 Haze-Aware Mask, or first
   ResidualCalib setting as an active main candidate. ResidualCalib is kept as
   positive ablation evidence, not a LF-v1 replacement.
3. If starting a new route, use a clean branch, commit/push local changes first, and launch the formal scout as a 100k-target run with 10k/20k/50k internal gates.
4. Keep archived one-off failed launchers as reproducibility evidence; prefer maintained parameterized launchers for future runs.
