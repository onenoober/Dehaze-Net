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
ssh root@ssh.smoothcloud.com.cn "cd /root/workspace/Dehaze-Net && git pull --ff-only"
ssh root@ssh.smoothcloud.com.cn "cd /root/workspace/Dehaze-Net/code && python eval.py ..."
ssh root@ssh.smoothcloud.com.cn "cd /root/workspace/Dehaze-Net/code && python train.py ..."
```

Codex should make source changes locally, commit and push them, then pull on the
server before testing or training. Do not edit source files directly on the
server; use the server for data checks, dependency checks, evaluation, and
training logs only.

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
