# Collaboration Workflow

This project is meant to be edited in small, traceable steps.

For document boundaries and where to write new analysis, start with
`docs/README.md`. This file is only for repeatable workflow and command
templates.

## Local loop
1. Create a topic branch.
2. Make a focused change.
3. Run a quick sanity check.
4. Commit with a message that describes the intent.
5. Push the branch.

Example:

```powershell
git checkout -b feat/wavelet-fusion
git add -A
git commit -m "Add wavelet fusion prototype"
git push -u origin feat/wavelet-fusion
```

## Server Loop Boundary

Current instruction: operate only on local files and GitHub unless the user
explicitly asks to sync or run on the server. The server-side commands below are
templates for later use, not permission to run them automatically.

Command validation note from 2026-05-24: the local machine is Windows
PowerShell, while the cloud server is Ubuntu/Linux. Run multi-line Linux
commands through the PowerShell here-string pattern shown below, or run the
`bash` snippets only after entering the server shell/tmux. Do not paste Linux
syntax directly into local PowerShell.

## Server loop
1. Pull the latest branch on the rented server.
2. Run training or evaluation from `code/`.
3. Save logs and checkpoints outside Git.
4. Update the experiment log with the final result.

Codex should make source changes locally, commit and push them, then pull on the
server before testing or training. Do not edit source files directly on the
server; use the server for data checks, dependency checks, evaluation, and
training logs only.

## Server GitHub sync

The private GitHub repository should be accessed from the server through SSH:

```bash
git remote set-url origin git@github.com:onenoober/Dehaze-Net.git
ssh -T git@github.com
```

The SSH test should authenticate as `onenoober`. Do not use HTTPS for the
private repo on the server unless deliberately debugging credentials.
Use `git fetch --dry-run origin` as a non-mutating connectivity check. Only run
`git pull --ff-only` after confirming the server checkout is the intended target
and `git status -sb` is clean enough for the operation.

For Conditional LF work, use the independent checkout instead of changing the
dirty main training checkout:

```powershell
@'
set -euo pipefail
cd /root/workspace/Dehaze-Net-conditional-lf
git status -sb
git fetch --dry-run origin
git fetch origin
git pull --ff-only
'@ | ssh runyun-ts "tr -d '\r' | bash -s"
```

For the older main checkout, use `/root/workspace/Dehaze-Net` only when that is
the intended target. The historical `git -c http.version=HTTP/1.1 pull
--ff-only` workaround is still useful if a host/network path hangs on plain
HTTP(S), but the current private-repo path should be SSH.

## Three-Place Source Sync

When the user asks to keep local, GitHub, and the cloud server unified, use this
order:

1. Commit locally after checks pass.
2. Push the branch to GitHub.
3. On each Git-backed server checkout, verify the target path, run
   `git status -sb`, fetch the pushed branch, and only then `git pull
   --ff-only`.
4. Verify local/GitHub/server all point at the same commit hash.

Do not edit source files directly on the server for experiment variants. If a
server copy is intentionally not a Git checkout, either recreate it from the
pushed Git source or record it as an isolated verification copy in
`CURRENT_CONTEXT.md` and `EXPERIMENT_LOG.md`.

Training run metadata has three layers:

- `args_initial.txt`: first launch parameters for new runs.
- `args.txt`: latest launch or resume parameters.
- `args_history.jsonl`: append-only launch/resume history for new runs.

For pre-existing runs created before this metadata split, treat `args.txt` as
the latest-known launch state and use saved shell scripts, logs, and
`EXPERIMENT_LOG.md` to reconstruct earlier starts.

## Check A Run

Template for later server-side checks:

```powershell
@'
set -euo pipefail
RUN='DEA-Net-LF-ConditionalMask-H4K-scout100k-20260523-224315'
ROOT='/root/workspace/Dehaze-Net-conditional-lf'
RUN_DIR="$ROOT/experiment/HAZE4K/$RUN"
LOG="$ROOT/experiment/HAZE4K/_run_logs/$RUN.log"

echo '=== process ==='
pgrep -af "$RUN|train.py" || true
echo '=== tmux ==='
tmux ls 2>/dev/null || true
echo '=== gpu ==='
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
echo '=== metrics ==='
cat "$RUN_DIR/saved_data/log.txt" 2>/dev/null || true
echo '=== last step in log ==='
/opt/anaconda/envs/py310/bin/python - <<PY
import re
from pathlib import Path
text = Path("$LOG").read_text(errors="ignore") if Path("$LOG").exists() else ""
steps = [int(x) for x in re.findall(r"step :(\\d+)/100000", text)]
print(max(steps) if steps else "no step found")
PY
'@ | ssh runyun-ts "tr -d '\r' | bash -s"
```

## Reliable remote commands

Prefer PowerShell here-strings piped to SSH for non-trivial Linux commands. This
avoids fragile nested quoting and removes Windows CRLF before Bash parses the
script:

```powershell
@'
set -euo pipefail
cd /root/workspace/Dehaze-Net-conditional-lf
source /opt/anaconda/etc/profile.d/conda.sh
conda activate py310
cd code
/opt/anaconda/envs/py310/bin/python train.py --help | head -40
'@ | ssh runyun-ts "tr -d '\r' | bash -s"
```

For simple one-shot commands, load conda explicitly before activating the
training environment:

```bash
source /opt/anaconda/etc/profile.d/conda.sh && conda activate py310
```

`runyun-ts` is the preferred SSH alias for routine work. It connects through
Tailscale Serve on `100.118.134.99:2222`, which forwards to the server-side SSH
service inside the tailnet. Keep the original public SSH target only as a
fallback for repairing Tailscale.

Validated safe probes on 2026-05-24:

- `ssh -G runyun-ts` resolves to user `root`, host `100.118.134.99`, port
  `2222`.
- `ssh -o BatchMode=yes -o ConnectTimeout=15 runyun-ts "hostname && whoami &&
  pwd"` succeeds.
- `/opt/anaconda/envs/py310/bin/python` is Python `3.10.13` with CUDA available
  on the RTX 5090 server.
- `tmux`, `git`, and `nvidia-smi` are available on the server.
- `rsync` is not installed locally or on the current server; use `scp` or a
  tar-over-SSH transfer unless rsync is deliberately installed later.

Not run as casual validation: full training, full evaluation, resume, `kill`,
package installation, symlink recreation, or `git pull` against a dirty training
checkout.

If the server restarts and `runyun-ts` times out, recover it from the public SSH
fallback by running:

```powershell
ssh runyun "bash /root/workspace/tailscale-ssh/start.sh"
```

That script restarts the userspace Tailscale daemon, the local SSHD on
`127.0.0.1:2223`, and the tailnet Serve mapping on `2222`.

The server container does not provide `systemd` or `/dev/net/tun`, so the stable
recovery path is Supervisor plus Tailscale userspace networking. The configured
Supervisor program is `runyun-tailscale-ssh`; it runs
`/root/workspace/tailscale-ssh/supervisor-keepalive.sh`, which checks tailscaled,
Serve, and the local SSHD every 60 seconds. Tailscale state is kept in
`/root/workspace/tailscale-ssh/state/tailscaled.state`.

To open the remote project in VS Code:

```powershell
code --remote ssh-remote+runyun-ts /root/workspace/Dehaze-Net
```

## Dataset and checkpoint sanity checks

Use these before a new HAZE4K training or official checkpoint evaluation. They
are checks only; fix dataset links or filenames before launching training if
counts or paths look wrong.

Local quick count, when the dataset is available on Windows:

```powershell
$paths = @(
  "dataset\HAZE4K\train\hazy",
  "dataset\HAZE4K\train\clear",
  "dataset\HAZE4K\test\hazy",
  "dataset\HAZE4K\test\clear"
)
foreach ($path in $paths) {
  if (Test-Path $path) {
    "{0}: {1}" -f $path, (Get-ChildItem -LiteralPath $path -File | Measure-Object).Count
  } else {
    "{0}: MISSING" -f $path
  }
}
```

Server count:

```powershell
@'
set -euo pipefail
ROOT=/root/workspace/Dehaze-Net
find "$ROOT/dataset/HAZE4K" -maxdepth 3 -type d | sort
find -L "$ROOT/dataset/HAZE4K/train/hazy" -type f | wc -l
find -L "$ROOT/dataset/HAZE4K/train/clear" -type f | wc -l
find -L "$ROOT/dataset/HAZE4K/test/hazy" -type f | wc -l
find -L "$ROOT/dataset/HAZE4K/test/clear" -type f | wc -l
'@ | ssh runyun-ts "tr -d '\r' | bash -s"
```

Use `find -L` for count checks because the current server exposes
`HAZE4K/*/hazy` and `HAZE4K/*/clear` as symlinks to `haze` and `gt`.

Official HAZE4K `.pth` checkpoint evaluation uses the actual downloaded
filename under `trained_models/HAZE4K/`; upstream docs have used inconsistent
names:

```powershell
@'
set -euo pipefail
cd /root/workspace/Dehaze-Net/code
/opt/anaconda/envs/py310/bin/python eval.py \
  --dataset HAZE4K \
  --model_name eval-H4K-official-full-$(date +%Y%m%d-%H%M%S) \
  --pre_trained_model <actual_haze4k_checkpoint>.pth
'@ | ssh runyun-ts "tr -d '\r' | bash -s"
```

## Fair HAZE4K training protocol

Local Windows is for coding, docs, Git, and lightweight static checks only.
Dry-run, smoke, training, benchmark, and evaluation must run on the cloud CUDA
server.

Before changing model architecture or training loss, open
`docs/HAZE4K_MODEL_CHANGE_PROTOCOL.md` and create or update the route's dated
experiment card. A long fair scout should not start until the card contains the
failure mode, mechanism hypothesis, enabled/disabled flags, route-specific
mechanism metrics, and written gate rules.

Formal HAZE4K candidate/scout runs must start with the same target horizon and
core training protocol:

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

Validation still runs every `10000` steps so the curve is comparable and
recoverable. Decision gates are not all equally strict, and they are not
PSNR/SSIM-only gates.

Before launching a fair candidate, write down the mechanism-specific diagnostic
signals that the architecture is supposed to improve. PSNR/SSIM remain the
global quality guardrails, but continuation past a weak gate requires evidence
from the route's own target:

- residual direction or calibration routes: residual-direction loss/cosine,
  wrong-direction count, low-frequency MSE delta, residual norm/error ratio,
  and strong-baseline regressions.
- selector or mask routes: selector/mask mean/std/min/max, whether the map is
  near-constant, and whether the selected branch improves the intended
  residual/error groups.
- teacher or guard routes: guard activation/weight, guarded-regression counts,
  teacher loss behavior, and whether the guard is active at the failed gate.
- frequency-reconstruction routes: low-frequency L1/MSE and color/tone
  regressions, not only the weighted training loss.

Use the list above as examples, not a fixed template. Each new route should
name its own mechanism metrics in the experiment card before the long scout.

| Step | Role | Rule |
| ---: | --- | --- |
| 10000 | sanity gate | stop only if quality collapses, training is unstable, or mechanism diagnostics show the route is inactive/degenerate |
| 20000 | early trajectory gate | stop if more than about `0.5 dB` below both references with no route-specific diagnostic upside |
| 30000 | first hard gate | for a route that is merely tied at 20k, require recovery toward the direct predecessor or clear mechanism improvement; stop if clearly below both or if route diagnostics fail |
| 50000 | promotion gate | must be at least close to baseline and preferably close to the direct predecessor, with mechanism metrics not worse than the predecessor |
| 70000 | late confirmation | continue only if still competitive and route-specific diagnostics are not worsening |
| 90000 | best-checkpoint check | compare against known best-step behavior; prepare full-test analysis if competitive |
| 100000 | final scout point | run only if the 50k/70k gates justify the remaining compute |

Do not launch formal comparisons as `epochs=4`, `epochs=10`, or a separate
50k/70k target. If a candidate fails a gate, stop the 100k-target run and record
the gate failure. If a shorter horizon is used for smoke or diagnosis, label it
diagnostic/invalid-for-comparison and keep it out of candidate metric tables.

When resuming, keep the original `epochs * iters_per_epoch` value. `train.py`
uses that value as the cosine-LR horizon, so changing it during resume changes
the learning-rate schedule and makes the result incomparable.

The Conditional LF launcher enforces this by default:
`scripts/runyun-haze4k-lf-conditional-mask-scout.sh` refuses non-100k formal
protocols unless `ALLOW_NONFAIR_PROTOCOL=1` is set. Use that override only for
dry-run/smoke/diagnostic artifacts and label the result accordingly.

## Train Checkpoint Visual Compare

For CR baseline versus an LF variant, the default baseline model is non-LF:

```bash
/opt/anaconda/envs/py310/bin/python visual_compare_train_ckpt.py \
  --baseline_checkpoint <baseline-best.pk> \
  --lf_checkpoint <lf-variant-best.pk> \
  --output_dir <compare-dir> \
  --lf_label <label>
```

## LF Residual Direction Diagnostic

Before designing another LF variant, run the residual diagnostic to check
whether the LF candidate is correcting low-frequency content in the right
direction and with the right magnitude. This is a read-only evaluation; it does
not start training.

Baseline CR versus LF-v1:

```bash
cd /root/workspace/Dehaze-Net/code
/opt/anaconda/envs/py310/bin/python diagnose_lf_residual_direction.py \
  --dataset HAZE4K \
  --split test \
  --baseline_checkpoint ../experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk \
  --current_checkpoint ../experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk \
  --current_use_lf_prior \
  --baseline_label DEA-Net-CR \
  --current_label DEA-Net-LF-v1 \
  --lowfreq_pool 8 \
  --output_dir ../experiment/HAZE4K/residual_diagnostic/CR-vs-LF-v1-20260524
```

For LF-v1 versus a later LF variant, load the baseline checkpoint with the LF
architecture and add the feature flags for the variant:

```bash
cd /root/workspace/Dehaze-Net/code
/opt/anaconda/envs/py310/bin/python diagnose_lf_residual_direction.py \
  --dataset HAZE4K \
  --split test \
  --baseline_checkpoint <lf-v1-best.pk> \
  --baseline_use_lf_prior \
  --current_checkpoint <lf-variant-best.pk> \
  --current_use_lf_prior \
  --lf_conditional_mask \
  --lf_haze_aware_mask \
  --baseline_label DEA-Net-LF-v1 \
  --current_label <label> \
  --output_dir <diagnostic-dir>
```

The script writes:

- `per_image_residual_metrics.csv`
- `group_summary.csv`
- `summary.json`
- `hard_cases.json`
- `analysis_report.md`

Important columns:

- `lf_residual_cosine`: direction match between
  `LP(current)-LP(baseline)` and `LP(GT)-LP(baseline)`.
- `lf_residual_norm_ratio`: correction magnitude relative to the target
  low-frequency correction.
- `lf_residual_error_ratio`: low-frequency residual error size relative to the
  target correction.
- `lf_mse_delta`: positive values mean the candidate made low-frequency MSE
  worse than the baseline.

Use this output to decide whether the next LF route should calibrate residual
direction/magnitude rather than add another spatial mask.

For LF-v1 versus another LF variant, explicitly enable the baseline LF
architecture so the baseline checkpoint is loaded with the correct module
shape:

```bash
/opt/anaconda/envs/py310/bin/python visual_compare_train_ckpt.py \
  --baseline_checkpoint <lf-v1-best.pk> \
  --baseline_use_lf_prior \
  --lf_checkpoint <lf-v2-best.pk> \
  --lf_conditional_mask \
  --lf_haze_aware_mask \
  --output_dir <compare-dir> \
  --lf_label <label>
```

## Candidate scout command skeleton

Use this shape only after the source branch is committed/pushed and the server
checkout is intentionally synced. Replace the feature flags and model name for
the candidate being tested, but keep the fair HAZE4K protocol unchanged:

```bash
cd /root/workspace/Dehaze-Net/code
/opt/anaconda/envs/py310/bin/python train.py \
  --epochs 20 \
  --iters_per_epoch 5000 \
  --bs 16 \
  --patch_size 256 \
  --w_loss_L1 1.0 \
  --w_loss_CR 0.1 \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --exp_dir ../experiment/ \
  --model_name <candidate-scout100k-run-id> \
  --dataset HAZE4K \
  --checkpoint_interval_steps 10000 \
  --eval_interval_steps 10000 \
  --save_epoch_checkpoints false \
  --no_pdf_plots \
  --no_tqdm
```

For LF-v1 style candidates, add only the LF feature flags under test:

```bash
  --use_lf_prior \
  --lf_prior_channels 8 \
  --lf_prior_pool 8 \
  --lf_prior_gate_init 0.0 \
  --lf_prior_injection pre_mix
```

For Conditional LF, prefer the maintained launcher
`scripts/runyun-haze4k-lf-conditional-mask-scout.sh`. If writing the command
manually, add:

```bash
  --use_lf_prior \
  --lf_prior_channels 8 \
  --lf_prior_pool 8 \
  --lf_prior_gate_init 0.0 \
  --lf_prior_injection pre_mix \
  --lf_conditional_mask \
  --lf_mask_hidden_channels 8 \
  --lf_mask_init_bias 2.0
```

For LF Residual Calibration, prefer the maintained launcher
`scripts/runyun-haze4k-lf-residual-calib-scout.sh`. If writing the command
manually, add:

```bash
  --use_lf_prior \
  --lf_prior_channels 8 \
  --lf_prior_pool 8 \
  --lf_prior_gate_init 0.0 \
  --lf_prior_injection pre_mix \
  --lf_residual_calibration \
  --lf_calib_hidden_channels 8 \
  --lf_calib_alpha_max 1.0
```

After launch, record the run id, branch/commit, protocol, checkpoint path,
metrics, and decision in `docs/EXPERIMENT_LOG.md`. Record important artifact
directories in `docs/HAZE4K_RUN_MANIFEST.md`.

## Long runs and stopping

For long HAZE4K runs, use `tmux` and write logs under
`experiment/HAZE4K/_run_logs/`. Keep launch scripts beside the logs when
possible so the exact command survives.

Do not continuously monitor long training unless the user asks. When asked to
check, report the run id, checkpoint step, metrics, log path, and whether the
GPU/process state matches the claim.

For expensive candidate resumes, prefer "run to the next hard gate, then
re-check" instead of letting the job run unattended to 100k. Use the same 100k
horizon in `train.py`; the boundary is operational, enforced by a watcher that
stops after the next evaluation point is written.

To stop a run, identify the exact model name and process group first:

```bash
MODEL="DEA-Net-LF-ConditionalMask-H4K-scout100k-20260523-224315"
pgrep -af "$MODEL|train.py"
MAIN_PID=$(pgrep -f "/opt/anaconda/envs/py310/bin/python train.py .*${MODEL}" | head -1)
PGID=$(ps -o pgid= -p "$MAIN_PID" | tr -d ' ')
kill -TERM -- -"${PGID}"
```

Then verify no matching process/tmux remains and GPU memory is released:

```bash
pgrep -af "$MODEL|train.py" || true
tmux ls 2>/dev/null || true
nvidia-smi
```

## Resume Current Conditional LF

Template for later use only. Resume only if the user explicitly asks to
continue training or sync the server. Keep the same 100k horizon and same model
name so `train.py --resume` loads `saved_model/latest.pk` from the existing run
directory.

Current pre-resume rule: the 20k point is a soft pass only. If resumed, run only
to the 30k hard gate first, then compare against baseline and LF-v1 before
spending more compute.

```powershell
@'
set -euo pipefail
ROOT='/root/workspace/Dehaze-Net-conditional-lf'
RUN='DEA-Net-LF-ConditionalMask-H4K-scout100k-20260523-224315'
SESSION="h4k_lf_condmask_100k_resume_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/experiment/HAZE4K/_run_logs"
LOG="$LOG_DIR/${RUN}-resume-$(date +%Y%m%d-%H%M%S).log"
TARGET_STEP=30000
mkdir -p "$LOG_DIR"
cd "$ROOT/code"
tmux new-session -d -s "$SESSION" "bash -lc '
  set -euo pipefail
  (
  /opt/anaconda/envs/py310/bin/python train.py \
    --resume \
    --use_lf_prior \
    --lf_prior_channels 8 \
    --lf_prior_pool 8 \
    --lf_prior_gate_init 0.0 \
    --lf_prior_injection pre_mix \
    --lf_conditional_mask \
    --lf_mask_hidden_channels 8 \
    --lf_mask_init_bias 2.0 \
    --model_name \"$RUN\" \
    --dataset HAZE4K \
    --epochs 20 \
    --iters_per_epoch 5000 \
    --bs 16 \
    --patch_size 256 \
    --num_workers 12 \
    --test_num_workers 4 \
    --pin_memory \
    --persistent_workers \
    --prefetch_factor 2 \
    --w_loss_L1 1.0 \
    --w_loss_CR 0.1 \
    --start_lr 0.0001 \
    --end_lr 0.000001 \
    --exp_dir ../experiment/ \
    --checkpoint_interval_steps 10000 \
    --eval_interval_steps 10000 \
    --save_epoch_checkpoints false \
    --no_pdf_plots \
    --no_tqdm
  ) 2>&1 | tee \"$LOG\" &
  TRAIN_PID=\$!
  while kill -0 \"\$TRAIN_PID\" 2>/dev/null; do
    if [ -f \"$ROOT/experiment/HAZE4K/$RUN/saved_data/log.txt\" ] && grep -q \"step :$TARGET_STEP \" \"$ROOT/experiment/HAZE4K/$RUN/saved_data/log.txt\"; then
      PGID=\$(ps -o pgid= -p \"\$TRAIN_PID\" | tr -d \" \")
      echo \"Reached target gate step $TARGET_STEP; stopping PGID \$PGID for review.\" | tee -a \"$LOG\"
      kill -TERM -- -\"\$PGID\" 2>/dev/null || kill -TERM \"\$TRAIN_PID\" 2>/dev/null || true
      wait \"\$TRAIN_PID\" || true
      exit 0
    fi
    sleep 60
  done
  wait \"\$TRAIN_PID\"
"
echo "SESSION=$SESSION"
echo "LOG=$LOG"
echo "TARGET_STEP=$TARGET_STEP"
'@ | ssh runyun-ts "tr -d '\r' | bash -s"
```

Do not resume the invalid short-horizon runs. Do not change `epochs` to 4, 10,
or any value that changes the 100k LR horizon.

## Revert policy
- Prefer `git revert` for undoing committed changes.
- Avoid force push unless you are deliberately resetting a short-lived feature branch.

## What should not enter Git
- Raw datasets
- Checkpoints
- Training logs
- Large inference folders
- Credentials or tokens

## Recommended branch names
- `reproduce/deanet`
- `feat/<topic>`
- `fix/<bug>`
- `docs/<topic>`
