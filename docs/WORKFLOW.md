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

Current Dehaze-Net remote target:

```powershell
ssh runyun-ts "cd /root/workspace/Dehaze-Net && git -c http.version=HTTP/1.1 pull --ff-only"
ssh runyun-ts "source /opt/anaconda/etc/profile.d/conda.sh && conda activate py310 && cd /root/workspace/Dehaze-Net/code && python eval.py ..."
ssh runyun-ts "source /opt/anaconda/etc/profile.d/conda.sh && conda activate py310 && cd /root/workspace/Dehaze-Net/code && python train.py ..."
```

Codex should make source changes locally, commit and push them, then pull on the
server before testing or training. Do not edit source files directly on the
server; use the server for data checks, dependency checks, evaluation, and
training logs only.

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

For one-shot SSH commands through `runyun-ts`, load conda explicitly before
activating the training environment:

```bash
source /opt/anaconda/etc/profile.d/conda.sh && conda activate py310
```

To open the remote project in VS Code:

```powershell
code --remote ssh-remote+runyun-ts /root/workspace/Dehaze-Net
```

For long HAZE4K runs, prefer a background `tmux` session and write logs under
`/root/workspace/Dehaze-Net/experiment/`.

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
