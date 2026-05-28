# Collaboration Workflow

This project is meant to be edited in small, traceable steps.

For document boundaries and where to write new analysis, start with
`docs/README.md`. This file is only for repeatable workflow and command
templates.

## Storage Roles

- GitHub is the lightweight code and documentation repository. Do not store
  datasets, checkpoints, large generated images, large inference folders, or
  bulky experiment artifacts in GitHub.
- Local WSL is the experiment artifact home. Keep datasets, checkpoints, full
  logs, plots, previews, and other research byproducts under the local WSL
  workspace, usually in ignored paths such as `dataset/`, `trained_models/`,
  and `experiment/`.
- Cloud servers such as `autodl-dehaze` and `runyun-ts` are temporary compute
  nodes. They can produce checkpoints, logs, metrics, and analysis outputs, but
  the durable copy should eventually live in local WSL. Use `autodl-dehaze` as
  the primary server and `runyun-ts` as the secondary server.
- Sync small cloud outputs back to local WSL when they help choose the next
  research route: launch scripts, compact logs, metric summaries, CSV/JSON
  summaries, experiment-log rows, and conclusion documents.
- Sync large cloud outputs only when explicitly requested: checkpoints, full
  inference folders, large image grids, large tensor arrays, datasets, and
  archives.
- After small cloud outputs are reviewed locally, put only the distilled code,
  documentation, commands, metrics, and conclusions into GitHub.

## Local loop
1. Create a topic branch.
2. Make a focused change.
3. Run a quick sanity check.
4. Commit with a message that describes the intent.
5. Push the branch.

Example:

```bash
cd /home/ubuntu/workspace/Dehaze-Net
git checkout -b feat/wavelet-fusion
git add -A
git commit -m "Add wavelet fusion prototype"
git push -u origin feat/wavelet-fusion
```

## Server Loop Boundary

Current instruction: operate on local WSL files, GitHub, and cloud sync only
when the user asks for that scope. The server-side commands below are templates
for later use, not permission to run long jobs automatically.

Command validation note from 2026-05-26: the durable local workspace is WSL
Ubuntu with bash at `/home/ubuntu/workspace/Dehaze-Net`. Run local project
commands in WSL/bash; run multi-line cloud commands through the here-string
pattern shown below, or run the `bash` snippets only after entering the server
shell/tmux.

Command validation note from 2026-05-27: model training should run on a cloud
server by default, not local WSL, unless the user explicitly asks for local
training. The primary cloud server is the AutoDL/SeetaCloud instance reached
from WSL as `autodl-dehaze`. Use `runyun-ts` when the user asks for runyun,
when AutoDL is unavailable, or when checking runyun artifacts.

## Server loop
1. Treat the server as disposable compute, not the artifact source of truth.
2. Pull or receive the intended code branch on the rented server.
3. Run training or evaluation from `code/`.
4. Save logs, checkpoints, inference outputs, and intermediate artifacts outside Git.
5. Sync small evidence back to local WSL when useful for route decisions.
6. Sync training logs only when it is lightweight and does not delay or disturb
   active training; otherwise leave them in the original cloud run path and
   fetch summaries at the next gate or final check.
7. Sync large artifacts back to local WSL only on request.
8. Update the local experiment log with final metrics, stop/resume state, and conclusions.

Codex should make source changes locally, commit and push them, then pull on the
server before testing or training. Do not edit source files directly on the
server; use the server for data checks, dependency checks, evaluation, and
training logs only.

## Default AutoDL/SeetaCloud Server

Default for new cloud operations unless the user explicitly specifies
`runyun-ts` or another server:

```text
SSH alias: autodl-dehaze
SSH target: root@connect.bjb1.seetacloud.com
SSH port: 19285
Local WSL key: ~/.ssh/autodl_dehaze_ed25519
Project root: /root/autodl-tmp/workspace/Dehaze-Net
Python: /root/miniconda3/envs/py310/bin/python
```

WSL SSH config:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
cat >> ~/.ssh/config <<'EOF'
Host autodl-dehaze
  HostName connect.bjb1.seetacloud.com
  User root
  Port 19285
  IdentityFile ~/.ssh/autodl_dehaze_ed25519
  IdentitiesOnly yes
  ServerAliveInterval 30
  ServerAliveCountMax 6
  Compression yes
EOF
chmod 600 ~/.ssh/config
```

Basic health check:

```bash
ssh autodl-dehaze 'cd /root/autodl-tmp/workspace/Dehaze-Net && \
  hostname && git status -sb && git log -1 --oneline && \
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader && \
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate py310 && \
  python - <<'"'"'PY'"'"'
import torch, torchvision, cv2, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("cv2", cv2.__version__)
print("numpy", numpy.__version__)
PY'
```

Dataset and artifact checks on the default server:

```bash
ssh autodl-dehaze 'cd /root/autodl-tmp/workspace/Dehaze-Net && \
  for p in dataset/HAZE4K/train/hazy dataset/HAZE4K/train/clear \
           dataset/HAZE4K/test/hazy dataset/HAZE4K/test/clear; do \
    printf "%-34s " "$p"; find -L "$p" -type f | wc -l; \
  done && \
  find trained_models -maxdepth 4 -type f | sort && \
  du -sh experiment dataset trained_models && \
  find experiment/HAZE4K -mindepth 1 -maxdepth 1 | wc -l'
```

Sync ignored experiment artifacts from local WSL to the default server. Do not
use `--delete` unless deliberately mirroring after checking the remote:

```bash
cd /home/ubuntu/workspace/Dehaze-Net
rsync -avh --partial --info=progress2 \
  -e "ssh -i $HOME/.ssh/autodl_dehaze_ed25519 -p 19285" \
  experiment/ \
  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/workspace/Dehaze-Net/experiment/
```

Sync compact cloud evidence back to local WSL. Pull logs only when the transfer
is lightweight and will not slow active training; otherwise leave them on the
cloud server until the next gate or final check. Pull checkpoints only when they
are needed for resume or analysis:

```bash
cd /home/ubuntu/workspace/Dehaze-Net
RUN=<run-id>
rsync -avh --partial --exclude 'saved_model/' \
  -e "ssh -i $HOME/.ssh/autodl_dehaze_ed25519 -p 19285" \
  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/${RUN}/ \
  experiment/HAZE4K/${RUN}/
rsync -avh --partial \
  -e "ssh -i $HOME/.ssh/autodl_dehaze_ed25519 -p 19285" \
  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/_run_logs/${RUN}.log \
  experiment/HAZE4K/_run_logs/
```

Validated default-server state on 2026-05-27:

- Project root: `/root/autodl-tmp/workspace/Dehaze-Net`.
- Branch/commit: `codex/haze4k-crplus-v2`, `ae9a70c`.
- GPU: RTX 5090, driver `580.105.08`, `32607 MiB`, compute capability `12.0`.
- Environment: `py310` with `torch 2.11.0+cu128`,
  `torchvision 0.26.0+cu128`, OpenCV `4.6.0`, NumPy `1.26.4`.
- HAZE4K counts: train hazy/clear `3001/3000`, test hazy/clear `1000/1000`.
- Local `experiment/` synced to the server: about `6.3G`, `5377` files.
- Smoke run passed: `smoke-H4K-CRPlusV2-autodl-20260527-104052`.

## Secondary Runyun Server

Runyun is still an active secondary compute node. Use it when requested, when
AutoDL is unavailable, or when inspecting runyun artifacts.

```text
SSH alias: runyun-ts
Main checkout: /root/workspace/Dehaze-Net
Clean Git-backed checkout: /root/workspace/Dehaze-Net-audit-sync
Python: /opt/anaconda/envs/py310/bin/python
```

Runyun connection rule from 2026-05-27: before each runyun training/evaluation
session, first connect through the public SSH alias and refresh the Tailscale
serve tunnel, then use only the Tailscale SSH alias for repo and training work:

```powershell
ssh runyun "bash /root/workspace/tailscale-ssh/start.sh"
ssh runyun-ts "hostname && pwd"
```

Do not run training over the public `runyun` SSH path; use it only to refresh
Tailscale when needed.

Basic health check:

```powershell
ssh runyun-ts 'hostname && whoami && \
  cd /root/workspace/Dehaze-Net-audit-sync && \
  git status -sb && git log -1 --oneline && \
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader && \
  /opt/anaconda/envs/py310/bin/python - <<'"'"'PY'"'"'
import torch, torchvision, cv2, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("cv2", cv2.__version__)
print("numpy", numpy.__version__)
PY'
```

Runyun Git sync template for the clean checkout:

```powershell
@'
set -euo pipefail
BRANCH=<branch-name>
cd /root/workspace/Dehaze-Net-audit-sync
git status -sb
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
'@ | ssh runyun-ts "tr -d '\r' | bash -s"
```

Runyun HAZE4K data and checkpoint check:

```powershell
@'
set -euo pipefail
ROOT=/root/workspace/Dehaze-Net
for p in dataset/HAZE4K/train/hazy dataset/HAZE4K/train/clear \
         dataset/HAZE4K/test/hazy dataset/HAZE4K/test/clear; do
  printf "%-34s " "$p"
  find -L "$ROOT/$p" -type f | wc -l
done
find "$ROOT/trained_models" -maxdepth 4 -type f | sort
'@ | ssh runyun-ts "tr -d '\r' | bash -s"
```

Runyun official HAZE4K checkpoint eval reference:

```powershell
@'
set -euo pipefail
cd /root/workspace/Dehaze-Net/code
/opt/anaconda/envs/py310/bin/python eval.py \
  --dataset HAZE4K \
  --model_name eval-H4K-official-full-$(date +%Y%m%d-%H%M%S) \
  --pre_trained_model PSNR3426_SSIM9885.pth
'@ | ssh runyun-ts "tr -d '\r' | bash -s"
```

If `runyun-ts` times out after a server restart, recover it from the public SSH
fallback:

```powershell
ssh runyun "bash /root/workspace/tailscale-ssh/start.sh"
ssh runyun-ts "hostname && pwd"
```

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

Default AutoDL sync template:

```bash
set -euo pipefail
BRANCH=<branch-name>
cd /root/autodl-tmp/workspace/Dehaze-Net
git status -sb
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
```

Use the AutoDL template for the primary server, or the runyun template above
when the target is the secondary runyun server.

## Three-Place Source Sync

When the user asks to keep local, GitHub, and the cloud server unified, use this
order:

1. Commit locally in WSL after checks pass.
2. Push the branch to GitHub.
3. On each Git-backed server checkout, verify the target path, run
   `git status -sb`, fetch the pushed branch, and only then `git pull
   --ff-only`.
4. Verify local/GitHub/server all point at the same commit hash.

Current primary server for the WSL/GitHub/cloud handoff is `autodl-dehaze`;
`runyun-ts` is the secondary server.

Verification template:

```bash
cd /home/ubuntu/workspace/Dehaze-Net
git status -sb
git rev-parse HEAD
git rev-parse origin/<branch-name>
git ls-files experiment | wc -l
```

```bash
ssh autodl-dehaze 'cd /root/autodl-tmp/workspace/Dehaze-Net && git status -sb && git rev-parse HEAD && git rev-parse origin/<branch-name> && git ls-files experiment | wc -l'
```

The expected tracked `experiment/` file count is `0`.

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
RUN='<run-id>'
ROOT='/root/autodl-tmp/workspace/Dehaze-Net'
PY='/root/miniconda3/envs/py310/bin/python'
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
"$PY" - <<PY
import re
from pathlib import Path
text = Path("$LOG").read_text(errors="ignore") if Path("$LOG").exists() else ""
steps = [int(x) for x in re.findall(r"step :(\\d+)/100000", text)]
print(max(steps) if steps else "no step found")
PY
'@ | ssh autodl-dehaze "tr -d '\r' | bash -s"
```

## Reliable remote commands

Prefer here-strings piped to SSH for non-trivial Linux commands. This avoids
fragile nested quoting and removes Windows CRLF before Bash parses the script:

```powershell
@'
set -euo pipefail
cd /root/autodl-tmp/workspace/Dehaze-Net
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py310
cd code
/root/miniconda3/envs/py310/bin/python train.py --help | head -40
'@ | ssh autodl-dehaze "tr -d '\r' | bash -s"
```

For simple one-shot commands, load conda explicitly before activating the
training environment:

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate py310
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
ROOT=/root/autodl-tmp/workspace/Dehaze-Net
find "$ROOT/dataset/HAZE4K" -maxdepth 3 -type d | sort
find -L "$ROOT/dataset/HAZE4K/train/hazy" -type f | wc -l
find -L "$ROOT/dataset/HAZE4K/train/clear" -type f | wc -l
find -L "$ROOT/dataset/HAZE4K/test/hazy" -type f | wc -l
find -L "$ROOT/dataset/HAZE4K/test/clear" -type f | wc -l
'@ | ssh autodl-dehaze "tr -d '\r' | bash -s"
```

Use `find -L` for count checks because the current server exposes
`HAZE4K/*/hazy` and `HAZE4K/*/clear` as symlinks to `haze` and `gt`.

Official HAZE4K `.pth` checkpoint evaluation uses the actual downloaded
filename under `trained_models/HAZE4K/`; upstream docs have used inconsistent
names:

```powershell
@'
set -euo pipefail
cd /root/autodl-tmp/workspace/Dehaze-Net/code
/root/miniconda3/envs/py310/bin/python eval.py \
  --dataset HAZE4K \
  --model_name eval-H4K-official-full-$(date +%Y%m%d-%H%M%S) \
  --pre_trained_model <actual_haze4k_checkpoint>.pth
'@ | ssh autodl-dehaze "tr -d '\r' | bash -s"
```

## Fair HAZE4K training protocol

Local Windows is for coding, docs, Git, and lightweight static checks only.
Dry-run, smoke, training, benchmark, and evaluation must run on the cloud CUDA
server.

Before changing model architecture or training loss, open
`docs/HAZE4K_MODEL_CHANGE_PROTOCOL.md` and create or update the route's dated
experiment card. A long fair scout should not start until the card contains the
most-valuable-attempt rationale, failure mode, mechanism hypothesis,
enabled/disabled flags, route-specific mechanism metrics, speed metrics, and
written gate rules.

The standard question before launch is not only "could this reach the highest
final PSNR?" It is "does this have the highest route-decision value per unit of
training cost?" A candidate should either reach useful quality earlier, or use
an early gate to reduce the number of future attempts. If success and failure
would both leave the next step ambiguous, do another cheap diagnostic before a
long scout.

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

Every gate report should include these standard fields when available:

- image quality: PSNR/SSIM, matched-step references, and best/final checkpoint
  step.
- training efficiency: wall time to gate, iteration speed, steps-to-baseline or
  steps-to-current-best when applicable, and whether the curve is faster than
  the direct predecessor.
- route activity: active flags, scheduled weights at the gate, scalar gates,
  loss-component tails, and any non-finite or degenerate signals.
- mechanism evidence: the route-specific diagnostics from the experiment card.
- regression control: per-image gain/regression split, current-best rescue,
  current-best gain preservation, and weak/strong reference splits when cheap
  enough to compute.
- decision value: stop/continue decision plus what this gate taught about the
  next attempt.

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

## Candidate scout command skeleton

Use this shape only after the source branch is committed/pushed and the server
checkout is intentionally synced. Replace the feature flags and model name for
the candidate being tested, but keep the fair HAZE4K protocol unchanged:

```bash
cd /root/autodl-tmp/workspace/Dehaze-Net/code
/root/miniconda3/envs/py310/bin/python train.py \
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

Add route-specific flags only from the current route card. After launch, record
the run id, branch/commit, protocol, checkpoint path, metrics, and decision in
`docs/EXPERIMENT_LOG.md`. Record important artifact directories in
`docs/HAZE4K_RUN_MANIFEST.md`.

## Formal HAZE4K full-training command

Use this only after a cold-start candidate has passed the standard `100000`
step scout promotion rules and the corresponding CR/baseline comparison plan is
clear. For HAZE4K cold-start formal training, the default full-training target
is `300000` steps (`epochs=60`, `iters_per_epoch=5000`), not the upstream
README's ITS-only `300 * 5000` command. The HAZE4K public checkpoint is an eval
reference in the upstream README, while this repository's HAZE4K data has about
`3000` training pairs; at `bs=16`, `300000` steps is already about `1600`
random crop/rotation exposures per training image.

Do not turn a completed `100000`-step scout into this run by resuming with
`epochs=60`. Start a new formal run from step 0, because `train.py` uses
`epochs * iters_per_epoch` as the cosine-LR horizon. If the final selected
route is not LF-v1, keep the same formal schedule and core protocol, but replace
the LF-v1 flags below with the route-specific flags from the route card. A
formal superiority claim also needs a matched formal CR/baseline reference; do
not compare a `300000`-step candidate only against a `100000`-step scout.

Default AutoDL LF-v1 formal command:

```bash
cd /root/autodl-tmp/workspace/Dehaze-Net/code

RUN=DEA-Net-LF-H4K-formal300k-$(date +%Y%m%d-%H%M%S)
mkdir -p ../experiment/HAZE4K/_run_logs

/root/miniconda3/envs/py310/bin/python train.py \
  --use_lf_prior \
  --lf_prior_channels 8 \
  --lf_prior_pool 8 \
  --lf_prior_gate_init 0.0 \
  --lf_prior_injection pre_mix \
  --model_name "$RUN" \
  --dataset HAZE4K \
  --epochs 60 \
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
  --no_tqdm \
  2>&1 | tee "../experiment/HAZE4K/_run_logs/${RUN}.log"
```

For runyun, first refresh the Tailscale connection as described above, then use
the intended runyun checkout and replace the Python path with
`/opt/anaconda/envs/py310/bin/python`.

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
MODEL="<run-id>"
pgrep -af "$MODEL|train.py"
MAIN_PID=$(pgrep -f "python train.py .*${MODEL}" | head -1)
PGID=$(ps -o pgid= -p "$MAIN_PID" | tr -d ' ')
kill -TERM -- -"${PGID}"
```

Then verify no matching process/tmux remains and GPU memory is released:

```bash
pgrep -af "$MODEL|train.py" || true
tmux ls 2>/dev/null || true
nvidia-smi
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
