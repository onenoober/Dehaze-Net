# HAZE4K Run Manifest

Created: 2026-05-25
Last reviewed: 2026-05-28

用途：作为 HAZE4K 训练、评估、可视化和失败消融 artifact 的统一索引。后续删除或归档远端 `experiment/HAZE4K` 目录前，先查本文件，避免误删仍有分析价值的数据。

本文件记录 artifact 保留/删除判断和历史盘点结果，不是当前运行状态的唯一来源。判断某个 run 是否仍在运行时，先读 `docs/CURRENT_CONTEXT.md`，再做 live process/log/checkpoint 检查。

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
| `KEEP_MINIMAL_REMOTE_MODEL` | 正向或诊断价值明确，但大 checkpoint 暂只保留在云端 | 本地保留紧凑证据和诊断；云端保留 `best.pk`/`latest.pk`，除非用户明确要求，否则不下载大模型 |
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
| `DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-192721` | small/partial | cancelled local pre-checkpoint scout | no eval/checkpoint; log reached about step `1802/100000` | branch/commit `codex/haze4k-crplus-v2` / `2e0987c`; local WSL run started while cloud server was unavailable, then stopped by request on 2026-05-26 after `runyun-ts` returned; tmux `h4k_crplusv2_100k_20260526-192721`; log `experiment/HAZE4K/_run_logs/DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-192721.log`; fair config with `w_loss_crplus_v2=0.003`; verified tmux/process absent and local GPU released; keep only as launch/speed/loss evidence, do not resume | `KEEP_SUMMARY_DELETE_MODEL_AFTER_CONFIRM` |
| `DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540` | 182M+ remote; compact local sync | positive CR-only component candidate; not LF-v1 replacement | best/final 100k `32.3633 / 0.9847`; `best.pk` and `latest.pk` step `100000` | launched from commit `24085db` in clean remote checkout `/root/workspace/Dehaze-Net-audit-sync` on branch `codex/haze4k-crplus-v2`; Tailscale status healthy on `runyun-ts`; fair config with `w_loss_crplus_v2=0.003`, `epochs=20`, `iters_per_epoch=5000`, `bs=16`, `patch_size=256`, checkpoint/eval every `10000`; curve 10k `26.7627/0.9629`, 20k `29.2182/0.9714`, 30k `30.1416/0.9769`, 40k `30.6141/0.9795`, 50k `31.3717/0.9824`, 60k `31.5598/0.9824`, 70k `31.9543/0.9838`, 80k `32.1779/0.9842`, 90k `32.3067/0.9844`, 100k `32.3633/0.9847`; final PSNR is above CR baseline but below LF-v1 and ResidualCalib, while SSIM is slightly higher; compact 100k evidence and final diagnostics synced locally, while large `best.pk`/`latest.pk` remain on runyun unless explicitly requested | `KEEP_MINIMAL_REMOTE_MODEL` |
| `DEA-Net-LFCR-v2-decay-H4K-scout100k-20260528-091455` | remote model; compact local diagnostics | negative/neutral fair ablation; not promotable | best/final 100k `32.1516 / 0.9844`; independent verify `32.1518 / 0.9844`; LF gate `0.019099` | launched from commit `9009515` in clean remote checkout `/root/workspace/Dehaze-Net-audit-sync` on branch `codex/haze4k-lfcr-v2-decay`; fair LF-v1 plus CRPlus-v2 schedule `w=0.005` through step `10000`, linearly decayed to `0.0` by step `20000`, then disabled; schedule-off mechanism partly worked and LF-v1 regression rescue remained, but final quality was below CR best, LF-v1, ResidualCalib, CRPlus-v2, and LFCR-v1 final; compact final diagnostics are synced locally under `experiment/HAZE4K/lfcr_v2_decay_diagnostics/DEA-Net-LFCR-v2-decay-H4K-scout100k-20260528-091455-final-20260528-verify`; do not sync checkpoints unless explicitly requested | `KEEP_MINIMAL_REMOTE_MODEL` |
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
| `experiment/HAZE4K/per_image_eval/CR-vs-CRPlusV2-full-100k-20260527` | CRPlus-v2 final full-test baseline comparison | 1000 images; mean delta `+0.1396 dB`, median `+0.1559`, better/worse `552/448`, weak-baseline mean delta `+0.4462`, strong-baseline mean delta `+0.0921`; confirms positive CR-only route versus CR baseline | `KEEP_SUMMARY` |
| `experiment/HAZE4K/per_image_eval/LF-v1-vs-CRPlusV2-full-100k-20260527` | CRPlus-v2 final full-test LF-v1 comparison | 1000 images; mean delta `-0.0633 dB`, median `-0.0296`, better/worse `488/512`; CRPlus-v2 helps weak LF-v1 cases (`+0.2187`) but regresses strong LF-v1 cases (`-0.1116`) | `KEEP_SUMMARY` |
| `experiment/HAZE4K/per_image_eval/ResidualCalib-vs-CRPlusV2-full-100k-20260527` | CRPlus-v2 final full-test ResidualCalib comparison | 1000 images; mean delta `-0.0286 dB`, median `-0.1235`, better/worse `461/539`; CRPlus-v2 helps weak ResidualCalib cases (`+0.4367`) but regresses strong ResidualCalib cases (`-0.1092`) | `KEEP_SUMMARY` |
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
| `experiment/HAZE4K/loss_scale/crplus-v2-baseline-train64-20260526` | CRPlus-v2 scale diagnostic | local WSL 64-image HAZE4K train center-crop diagnostic using CR baseline best checkpoint; selected combined ratio loss `0.189611`; weighted ratios to L1 `0.0197/0.0592/0.0987/0.1975` for weights `0.001/0.003/0.005/0.01`; supports first scout default `w_loss_crplus_v2=0.003` | `KEEP_SUMMARY` |
| `experiment/HAZE4K/loss_scale/crplus-v2-final-test-100k-20260527` | CRPlus-v2 final full-test loss-scale review | 1000 test images; selected ratio objective `0.357142`, trained `w=0.003` corresponds to about `0.0488` of L1; `under_dehazed_mix` remains the main active selected negative at margin `0.02` (`34.5%` active), while `hazy` is mostly easy (`2.4%` active) | `KEEP_SUMMARY` |
| `experiment/HAZE4K/loss_scale/lfcr-v1-lfv1-train256-20260527-231511` | LFCR-v1 LF-v1 checkpoint scale diagnostic | runyun 256-image HAZE4K train center-crop diagnostic using LF-v1 best checkpoint; selected ratio objective `0.184444`; weighted ratios to L1 `0.0205/0.0614/0.1024/0.2048` for weights `0.001/0.003/0.005/0.01`; supports the strong first LFCR scout weight `w_loss_crplus_v2=0.005` | `KEEP_SUMMARY` |
| `experiment/HAZE4K/DEA-Net-LFCR-v1-w005-H4K-scout100k-20260527-231728` | completed LFCR-v1 fair scout | runyun fair 100k scout launched from `codex/haze4k-lfcr-v1` checkout commit `dbd1320`, with LFCR implementation introduced at `1a32a2a`; LF-v1 architecture plus `w_loss_crplus_v2=0.005`; completed at 100k with `32.2098 / 0.9844`, `best.pk` and `latest.pk` both step `100000`, LF gate `0.0201`; early 10k was strong but final is below LF-v1 and CRPlus-v2, so keep remote checkpoint/log for diagnostics and do not promote this exact setting | `KEEP_MINIMAL_REMOTE_MODEL` |
| `experiment/HAZE4K/lfcr_v1_diagnostics/DEA-Net-LFCR-v1-w005-100k-20260528-084016` | LFCR-v1 final diagnostics | compact sync from runyun (`5.2M`, no checkpoints); full-test pairwise CSV/JSON/MD plus residual direction and CRPlus-v2 loss-scale diagnostics; key read: LFCR vs LF-v1 mean `-0.2178 dB`, improves `182/351` LF-v1 regression cases by at least `0.30 dB` but loses at least `0.30 dB` on `264/453` LF-v1 gain cases, LF gate lower than LF-v1 (`0.0201` vs `0.0339`), final CRPlus selected activity mostly `under_dehazed_mix` (`343/1000`) | `KEEP_SUMMARY` |
| `experiment/HAZE4K/lfcr_v2_decay_diagnostics/DEA-Net-LFCR-v2-decay-H4K-scout100k-20260528-091455-final-20260528-verify` | LFCR-v2 decay final diagnostics | compact sync from runyun (`5.2M`, `48` files, no checkpoints); full-test pairwise CSV/JSON/MD plus residual direction and CRPlus-v2 loss-scale diagnostics; key read: verified mean `32.1518 / 0.9844`, vs LF-v1 mean `-0.2765 dB`, vs CRPlus-v2 `-0.2131 dB`, vs LFCR-v1 `-0.0587 dB`; it improves `186/351` LF-v1 regression cases by at least `0.30 dB` and fully rescues `96`, but loses at least `0.30 dB` on `316/453` LF-v1 gain cases; residual and LF-MSE diagnostics remain weak, so the route is closed as negative/neutral | `KEEP_SUMMARY` |
| `experiment/HAZE4K/route_evidence_review/HAZE4K-route-evidence-review-20260528` | system route evidence aggregation | local read-only aggregation from existing 1000-image per-image CSVs; no model forward pass or training; writes joined five-output matrix, model summary, pairwise deltas, LF-v1 rescue/preservation table, oracle ceilings, group breakdown, and conflict-case list; key read: LF-v1 remains best standalone mean PSNR `32.4283`, while all-five oracle reaches `33.5098` (`+1.0815 dB` over LF-v1), proving complementarity but not a deployable selector | `KEEP_SUMMARY` |
| `experiment/HAZE4K/wavelet_preserve_proxy/HAZE4K-wavelet-preserve-proxy-runyun-20260528-hazy` | WaveletPreserve hazy-only proxy preflight | runyun diagnostic-only audit using `309` metadata-free hazy wavelet/degradation features and the existing route evidence matrix; primary LF-v1 preserve/intervene target has `453/351` samples; best metadata-free random split did not pass because intervene precision was `0.5541` vs required `0.60`, with held-out airlight/beta lower; do not train hazy-only WaveletPreserve from this artifact | `KEEP_SUMMARY` |
| `experiment/HAZE4K/wavelet_preserve_proxy/HAZE4K-wavelet-preserve-activation-proxy-runyun-20260528` | WaveletPreserve wavelet plus frozen activation proxy preflight | runyun diagnostic-only audit adding frozen CR/LF-v1 activation features; compact local sync includes summary/report/split CSV plus feature CSV; primary target worsened (`hazy_wavelet_plus_activation` random balanced accuracy `0.5360`, intervene precision `0.4757`), so do not train the current wavelet/activation preservation architecture scout | `KEEP_SUMMARY` |
| `experiment/HAZE4K/supervised_preserve_proxy/HAZE4K-supervised-preserve-proxy-runyun-20260528-train-p4-sklearn-liblinear` | supervised/distilled preserve-target proxy preflight | compact local sync of final sklearn-liblinear audit; runyun generated HAZE4K train teacher labels from CR and LF-v1 best checkpoints (`3000` images, `12000` patches, decisive target `4947/5108` preserve/intervene); final random-image row failed the pass line despite gain `+0.1955 dB` because balanced accuracy `0.5889`, preserve recall `0.5951`, intervene precision `0.5934`, and strong-CR regression intervene recall `0.5347` were too low; do not train this preserve-head target | `KEEP_SUMMARY` |
| `experiment/HAZE4K/DEA-Net-OfficialWarmStart-LFv1-H4K-local-scout100k-20260528-152808` | active local official warm-start LF-v1 staged fine-tune scout | launched 2026-05-28 from local WSL branch `codex/haze4k-official-warmstart-finetune` commit `b4194b1`; official checkpoint conversion report and training checkpoints/logs live here; route is isolated from cold-start fair-candidate tables; first gate is 10k | `KEEP_ACTIVE_LOCAL_MODEL` |
| `experiment/HAZE4K/_run_logs/DEA-Net-OfficialWarmStart-LFv1-H4K-local-scout100k-20260528-152808.log` | active local official warm-start LF-v1 staged fine-tune log | launch log for tmux `ow-lfv1-local-20260528-152808`; includes official-weight mapping counts, forward-equivalence check, staged-freeze config, and training progress | `KEEP_ACTIVE_LOG` |
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
| `smoke-H4K-CRPlusV2-20260526` | TBD | local CRPlus-v2 code-path smoke; diagnostic `T=2`, `bs=2`, `patch_size=64`; wrote `latest.pk` at step 2 and logged `CRPlusV2` tail `[1.4913553, 1.4035805]`; invalid for fair comparison | `KEEP_SUMMARY_DELETE_MODEL_AFTER_CONFIRM` |
| `smoke-H4K-LFCR-v1-w005-runyun-20260527-231619` | TBD | runyun LFCR-v1 code-path smoke; diagnostic `T=2`, `bs=2`, `patch_size=64`, `w_loss_crplus_v2=0.005`; wrote `latest.pk` at step 2 and logged LF prior plus CRPlus-v2 activation; invalid for fair comparison | `KEEP_SUMMARY_DELETE_MODEL_AFTER_CONFIRM` |
| `smoke-H4K-LFCR-v2-decay-runyun-20260528-091330` | TBD | runyun LFCR-v2 decay code-path smoke; diagnostic `T=2`, `bs=2`, `patch_size=64`, `w_loss_crplus_v2=0.005`, decay window `1/2`; wrote `latest.pk` at step 2 and logged `CRPlusV2_weight [0.005]` before scheduled zero; invalid for fair comparison | `KEEP_SUMMARY_DELETE_MODEL_AFTER_CONFIRM` |
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
