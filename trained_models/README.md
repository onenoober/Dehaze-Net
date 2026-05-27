# Pre-trained model

1. Download the pre-trained models on [[Google Drive](https://drive.google.com/drive/folders/1Rjb8dpyNnvvr0XLvIX9fg8Hdru_MhMCj?usp=sharing)] or [[Baidu Disk](https://pan.baidu.com/s/1retfKIs_Om-D4zA45sL6Kg) (password: dcyb)].
2. Make sure the file structure is consistent with the following:

```
trained_models/
├── HAZE4K
│   └── PSNR3426_SSIM9885.pth
├── ITS
│   └── PSNR4131_SSIM9945.pth
└── OTS
    └── PSNR3659_SSIM9897.pth
```

The HAZE4K checkpoint filename has appeared inconsistently in upstream notes.
Use the actual downloaded filename when running `eval.py`; the verified fork
reference is `PSNR3426_SSIM9885.pth`.
