# Download Plan

Verified on 2026-05-11 from this workspace.

## What worked

- DEA-Net repo weights on Baidu: reachable.
- RESIDE Baidu mirrors for ITS / OTS: reachable.
- Haze4K Baidu mirror from the official DMT-Net repo: reachable.
- Google Drive folder for DEA-Net weights: not reliable from this workspace; use the Baidu mirror first.
- Dropbox links from the RESIDE page: not reliable from this workspace; do not use them as the first server-side choice.

## Recommended sources

### 1. DEA-Net pretrained weights

- Baidu: `https://pan.baidu.com/s/1retfKIs_Om-D4zA45sL6Kg`
- Password: `dcyb`

Use this as the primary source.

### 2. RESIDE ITS

- Baidu: `https://pan.baidu.com/s/16rm4zUF8uVRs3Ux5T9CMMA`
- Password: `tqyh`

### 3. RESIDE OTS

- Baidu: `https://pan.baidu.com/s/1c2rW4hi`
- Password: `5vss`

### 4. HAZE4K

- Baidu: `https://pan.baidu.com/s/141MW0YAvjFcydlroQZZizA`
- Password: `cmmr`

## Best fallback order

1. Use the official Baidu mirror.
2. If you have a browser on the server, open the share page there and download directly.
3. If the server is headless, use a local download and then `rsync` to the server.
4. If you want a command-line Baidu workflow, use `BaiduPCS-Go` from its official GitHub repository and transfer the share link with the extraction code.

## Notes

- Google Drive was not a stable option in this workspace for the DEA-Net weights folder, so it should be treated as backup only.
- The upstream README has at least one filename inconsistency for HAZE4K weights, so always verify the exact checkpoint name after download.
- Do not commit the downloaded archives or extracted dataset files into Git.

## Server layout

Recommended layout:

```text
Dehaze-Net/
  code/
  dataset/
    ITS/
    OTS/
    HAZE4K/
  trained_models/
    ITS/
    OTS/
    HAZE4K/
  experiment/
  downloads/
```

