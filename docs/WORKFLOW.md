# Collaboration Workflow

This project is meant to be edited in small, traceable steps.

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

For Conditional LF work, use the independent checkout instead of changing the
dirty main training checkout:

```powershell
@'
set -euo pipefail
cd /root/workspace/Dehaze-Net-conditional-lf
git status -sb
git fetch origin
git pull --ff-only
'@ | ssh runyun-ts "tr -d '\r' | bash -s"
```

For the older main checkout, use `/root/workspace/Dehaze-Net` only when that is
the intended target. The historical `git -c http.version=HTTP/1.1 pull
--ff-only` workaround is still useful if a host/network path hangs on plain
HTTP(S), but the current private-repo path should be SSH.

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

## Fair HAZE4K training protocol

Local Windows is for coding, docs, Git, and lightweight static checks only.
Dry-run, smoke, training, benchmark, and evaluation must run on the cloud CUDA
server.

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

The 20k and 50k numbers are gates inside that 100k-target run. Do not launch
formal comparisons as `epochs=4`, `epochs=10`, or a separate 50k target. If a
candidate fails at 20k or 50k, stop the 100k-target run and record the gate
failure. If a shorter horizon is used for smoke or diagnosis, label it
diagnostic/invalid-for-comparison and keep it out of candidate metric tables.

When resuming, keep the original `epochs * iters_per_epoch` value. `train.py`
uses that value as the cosine-LR horizon, so changing it during resume changes
the learning-rate schedule and makes the result incomparable.

The Conditional LF launcher enforces this by default:
`scripts/runyun-haze4k-lf-conditional-mask-scout.sh` refuses non-100k formal
protocols unless `ALLOW_NONFAIR_PROTOCOL=1` is set. Use that override only for
dry-run/smoke/diagnostic artifacts and label the result accordingly.

## Long runs and stopping

For long HAZE4K runs, use `tmux` and write logs under
`experiment/HAZE4K/_run_logs/`. Keep launch scripts beside the logs when
possible so the exact command survives.

Do not continuously monitor long training unless the user asks. When asked to
check, report the run id, checkpoint step, metrics, log path, and whether the
GPU/process state matches the claim.

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
tmux ls || true
nvidia-smi
```

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
