DEA-Net Ubuntu 22.04 新手操作手册
=================================

适用场景：

- 云服务器系统：Ubuntu 22.04
- 云服务器登录用户示例：root
- 云服务器地址示例：ssh.smoothcloud.com.cn
- 项目目录：/root/workspace/Dehaze-Net
- 本地系统：Windows，使用 PowerShell 上传数据
- 本手册以当前 DEA-Net 仓库为例：git@github.com:onenoober/Dehaze-Net.git

重要原则：

- 代码放 GitHub，数据集、checkpoint、压缩包不要提交到 Git。
- 私钥文件、GitHub token 不要写进仓库，也不要发给别人。
- 大文件优先放到服务器的 downloads 目录，再在服务器端解压。
- 每做完一个核心步骤，都先执行检查命令，确认成功后再继续。


1. Linux 常用操作命令速查
=========================

下面命令都在 Ubuntu 服务器终端执行，除非特别说明是 Windows PowerShell。

1.1 路径与目录
--------------

查看当前所在目录：

```bash
pwd
```

进入目录：

```bash
cd /root/workspace
```

回到当前用户的 home 目录：

```bash
cd ~
```

注意：root 用户下，`~` 就是 `/root`。

返回上一级目录：

```bash
cd ..
```

查看当前目录文件：

```bash
ls
ls -lah
```

创建目录：

```bash
mkdir dataset
mkdir -p dataset/RESIDE/ITS
```

1.2 文件查看
------------

查看文本文件前 40 行：

```bash
head -40 Linux.md
```

查看文本文件后 40 行：

```bash
tail -40 Linux.md
```

实时查看日志：

```bash
tail -f ../experiment/train_its.log
```

分页查看文件：

```bash
less README.md
```

退出 `less`：按 `q`。

1.3 文件复制、移动、删除
------------------------

复制文件：

```bash
cp source.txt target.txt
```

复制目录：

```bash
cp -r source_dir target_dir
```

移动或重命名：

```bash
mv old_name new_name
```

删除文件：

```bash
rm file.txt
```

删除目录：

```bash
rm -r old_dir
```

新手注意：`rm -r` 会删除目录和里面所有文件，执行前先用 `ls -lah` 确认路径。

1.4 查找文件与统计空间
----------------------

查找所有 pth 权重：

```bash
find /root/workspace/Dehaze-Net -name "*.pth"
```

查看磁盘总空间：

```bash
df -h
```

查看当前目录占用：

```bash
du -sh .
```

查看数据集、权重、实验输出分别占用：

```bash
cd /root/workspace/Dehaze-Net
du -sh dataset trained_models downloads experiment
```

1.5 进程、显卡与后台任务
------------------------

查看 GPU：

```bash
nvidia-smi
```

每 1 秒刷新一次 GPU 状态：

```bash
watch -n 1 nvidia-smi
```

查看训练进程：

```bash
ps aux | grep train.py
```

停止进程：

```bash
kill 进程PID
```

如果普通 kill 停不掉：

```bash
kill -9 进程PID
```

1.6 压缩包解压
--------------

查看 zip 内部结构：

```bash
unzip -l downloads/ITS.zip | head -50
```

解压 zip：

```bash
unzip downloads/ITS.zip -d dataset/RESIDE/ITS
```

如果是分卷压缩包，且 `unzip` 失败，优先用 7z：

```bash
7z x downloads/ITS.zip -odataset/RESIDE/ITS
```

解压 tar.gz：

```bash
tar -xzf file.tar.gz -C target_dir
```

1.7 下载文件
------------

普通下载：

```bash
wget -c "下载链接" -O downloads/file.zip
```

多线程下载：

```bash
aria2c -x 16 -s 16 -c "下载链接" -d downloads -o file.zip
```

1.8 终端复制粘贴小问题
----------------------

如果你粘贴命令后出现类似：

```bash
cd ^[[200~
```

这是复制粘贴带进来的控制字符。按 `Ctrl+C` 清掉当前输入，再手动重新输入命令即可。


2. Windows 连接云服务器
=======================

本节命令在 Windows PowerShell 执行，不是在服务器里执行。

假设你的 SSH 私钥文件保存为：

```powershell
D:\Dehaze\smoothcloud.pem
```

连接服务器：

```powershell
ssh -i D:\Dehaze\smoothcloud.pem root@ssh.smoothcloud.com.cn
```

第一次连接时会出现类似提示：

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

输入：

```text
yes
```

看到类似下面内容，说明连接成功：

```text
Welcome to Ubuntu 22.04.5 LTS
(base) root@...:~#
```

连接成功后的检查：

```bash
pwd
whoami
hostname
```

期望结果：

- `whoami` 输出 root
- 当前提示符里能看到 `root@服务器名`


3. 服务器初始化
===============

本节命令在 Ubuntu 服务器执行。

进入工作目录：

```bash
mkdir -p /root/workspace
cd /root/workspace
```

安装常用工具：

```bash
apt update
apt install -y git unzip zip wget curl aria2 rsync tmux htop p7zip-full
```

检查是否成功：

```bash
git --version
unzip -v | head -2
aria2c --version | head -2
tmux -V
```

如果 `apt update` 很慢，优先换国内源。你之前已经把 Ubuntu 源换成清华源，`apt update` 速度已经正常。偶尔看到下面警告通常不影响继续：

```text
Problem unlinking the file ... Invalid cross-device link
```


4. 配置 GitHub SSH key
=====================

目标：让服务器能直接 `git clone`、`git pull`、`git push` 私有仓库。

本节命令在 Ubuntu 服务器执行。

4.1 检查是否已有 SSH key
------------------------

```bash
ls -lah ~/.ssh
```

如果已经有 `id_ed25519` 和 `id_ed25519.pub`，可以先不用新建。

4.2 生成新的 SSH key
--------------------

```bash
ssh-keygen -t ed25519 -C "dehaze-server"
```

一路回车即可。新手建议不要设置 passphrase，避免后续自动拉取代码时频繁输入密码。

检查是否生成成功：

```bash
ls -lah ~/.ssh
```

期望看到：

```text
id_ed25519
id_ed25519.pub
```

4.3 把公钥添加到 GitHub
-----------------------

显示公钥内容：

```bash
cat ~/.ssh/id_ed25519.pub
```

复制输出的整行内容，添加到 GitHub：

- GitHub 网页右上角头像
- Settings
- SSH and GPG keys
- New SSH key
- Title 可写：dehaze-server
- Key 粘贴 `id_ed25519.pub` 的内容

注意：只复制 `.pub` 公钥，不要复制 `id_ed25519` 私钥。

4.4 测试 GitHub SSH 是否通
--------------------------

```bash
ssh -T git@github.com
```

第一次会提示是否信任 GitHub 主机，输入 `yes`。

成功时通常会看到类似：

```text
Hi onenoober! You've successfully authenticated...
```

如果看到 `Permission denied (publickey)`，说明 GitHub 没有正确添加公钥，或者服务器用的不是刚才那把 key。


5. 拉取或更新 DEA-Net 仓库
==========================

本节命令在 Ubuntu 服务器执行。

如果服务器还没有仓库：

```bash
cd /root/workspace
git clone git@github.com:onenoober/Dehaze-Net.git
cd Dehaze-Net
```

如果仓库已经存在：

```bash
cd /root/workspace/Dehaze-Net
git pull --ff-only
```

检查是否成功：

```bash
pwd
git status
git remote -v
ls -lah
```

期望结果：

- `pwd` 是 `/root/workspace/Dehaze-Net`
- `git status` 显示工作区 clean，或者只显示你明确知道的改动
- `git remote -v` 指向 `github.com:onenoober/Dehaze-Net.git`
- 能看到 `code`、`dataset`、`docs`、`trained_models` 等目录

如果你还没有配置 GitHub SSH，也可以临时用 token 克隆，但不要把 token 写进文件：

```bash
export GH_TOKEN='这里临时粘贴你的token'
git -c http.https://github.com/.extraheader="AUTHORIZATION: basic $(printf "x-access-token:%s" "$GH_TOKEN" | base64 -w0)" clone https://github.com/onenoober/Dehaze-Net.git
unset GH_TOKEN
```


6. 创建 DEA-Net 所需目录
=======================

本节命令在 Ubuntu 服务器执行。

```bash
cd /root/workspace/Dehaze-Net
mkdir -p dataset/RESIDE/ITS dataset/RESIDE/OTS dataset/HAZE4K
mkdir -p trained_models/ITS trained_models/OTS trained_models/HAZE4K
mkdir -p downloads experiment
```

创建兼容软链接：

```bash
ln -sfn RESIDE/ITS dataset/ITS
ln -sfn RESIDE/OTS dataset/OTS
```

为什么要建软链接：

- `code/train.py` 默认读取 `../dataset/RESIDE/ITS/train`
- `code/eval.py` 默认读取 `../dataset/<dataset>/test`
- 软链接能避免复制两份数据，节省空间

检查是否成功：

```bash
ls -lah dataset
```

期望看到类似：

```text
ITS -> RESIDE/ITS
OTS -> RESIDE/OTS
RESIDE/
HAZE4K/
```


7. 数据集和 checkpoint 下载入口
==============================

下载链接统一看仓库文档：

```bash
cd /root/workspace/Dehaze-Net
cat docs/DOWNLOADS.md
```

当前建议：

- DEA-Net pretrained weights：优先 Baidu 链接
- RESIDE ITS：优先 Baidu 链接
- RESIDE OTS：优先 Baidu 链接
- HAZE4K：优先 Baidu 链接

因为你的云服务器下载较慢，推荐做法是：

1. Windows 本地或 U 盘先通过百度网盘客户端下载。
2. 用 PowerShell 的 `scp` 上传到云服务器。
3. 在云服务器上解压、整理目录。


8. 从 Windows 上传本地数据到云服务器
===================================

本节命令在 Windows PowerShell 执行，不是在服务器里执行。

先设置 3 个变量，后面命令会更短：

```powershell
$KEY="D:\Dehaze\smoothcloud.pem"
$HOST="root@ssh.smoothcloud.com.cn"
$REMOTE="/root/workspace/Dehaze-Net"
```

如果你的私钥不在 `D:\Dehaze\smoothcloud.pem`，把 `$KEY` 换成实际路径。

如果数据在 U 盘，比如 `E:\BaiduNetdiskDownload\dehaze`，把下面命令里的 `D:\BaiduNetdiskDownload\dehaze` 改成对应盘符。

8.1 先在服务器创建目标目录
--------------------------

```powershell
ssh -i $KEY $HOST "mkdir -p $REMOTE/downloads $REMOTE/dataset/RESIDE/OTS $REMOTE/trained_models"
```

检查是否成功：

```powershell
ssh -i $KEY $HOST "ls -lah $REMOTE"
```

8.2 上传 ITS 分卷压缩包
----------------------

你的 ITS 目前是：

```text
D:\BaiduNetdiskDownload\dehaze\ITS\ITS.z01
D:\BaiduNetdiskDownload\dehaze\ITS\ITS.z02
...
D:\BaiduNetdiskDownload\dehaze\ITS\ITS.z10
D:\BaiduNetdiskDownload\dehaze\ITS\ITS.zip
```

执行：

```powershell
Get-ChildItem "D:\BaiduNetdiskDownload\dehaze\ITS" -Filter "ITS.z*" | ForEach-Object {
  scp -i $KEY $_.FullName "${HOST}:${REMOTE}/downloads/"
}

scp -i $KEY "D:\BaiduNetdiskDownload\dehaze\ITS\ITS.zip" "${HOST}:${REMOTE}/downloads/"
```

上传后检查：

```powershell
ssh -i $KEY $HOST "ls -lh $REMOTE/downloads/ITS.*"
```

期望看到 `ITS.z01` 到 `ITS.z10`，以及 `ITS.zip`。

8.3 上传 Haze4K 压缩包
----------------------

```powershell
scp -i $KEY "D:\BaiduNetdiskDownload\dehaze\Haze4K.zip" "${HOST}:${REMOTE}/downloads/"
```

检查：

```powershell
ssh -i $KEY $HOST "ls -lh $REMOTE/downloads/Haze4K.zip"
```

8.4 上传 OTS_ALPHA 文件夹
------------------------

你的 OTS_ALPHA 当前包含：

```text
D:\BaiduNetdiskDownload\dehaze\OTS_ALPHA\clear
D:\BaiduNetdiskDownload\dehaze\OTS_ALPHA\depth
D:\BaiduNetdiskDownload\dehaze\OTS_ALPHA\haze
```

执行：

```powershell
scp -i $KEY -r "D:\BaiduNetdiskDownload\dehaze\OTS_ALPHA\clear" "${HOST}:${REMOTE}/dataset/RESIDE/OTS/"
scp -i $KEY -r "D:\BaiduNetdiskDownload\dehaze\OTS_ALPHA\depth" "${HOST}:${REMOTE}/dataset/RESIDE/OTS/"
scp -i $KEY -r "D:\BaiduNetdiskDownload\dehaze\OTS_ALPHA\haze"  "${HOST}:${REMOTE}/dataset/RESIDE/OTS/"
```

检查：

```powershell
ssh -i $KEY $HOST "find $REMOTE/dataset/RESIDE/OTS -maxdepth 2 -type d | sort | head -50"
```

注意：DEA-Net 常见读取目录名是 `hazy`，而你本地 OTS_ALPHA 里可能叫 `haze`。如果后续 OTS 测试报找不到 `hazy`，在服务器上执行：

```bash
cd /root/workspace/Dehaze-Net/dataset/RESIDE/OTS
ln -sfn haze hazy
```

8.5 上传 pretrained weights
---------------------------

```powershell
scp -i $KEY -r "D:\BaiduNetdiskDownload\dehaze\trained_models\ITS"    "${HOST}:${REMOTE}/trained_models/"
scp -i $KEY -r "D:\BaiduNetdiskDownload\dehaze\trained_models\OTS"    "${HOST}:${REMOTE}/trained_models/"
scp -i $KEY -r "D:\BaiduNetdiskDownload\dehaze\trained_models\HAZE4K" "${HOST}:${REMOTE}/trained_models/"
```

检查：

```powershell
ssh -i $KEY $HOST "find $REMOTE/trained_models -type f | sort"
```

8.6 上传很慢或中断怎么办
------------------------

- `scp` 简单直接，但中断后通常要重传当前文件。
- Windows 上可以用 WinSCP 图形界面上传，适合新手。
- 如果后续装了 rsync，可以用断点续传。

Windows PowerShell 原生通常没有 rsync，先用 `scp` 就可以。


9. 在服务器上解压和整理数据
===========================

本节命令在 Ubuntu 服务器执行。

进入项目目录：

```bash
cd /root/workspace/Dehaze-Net
```

9.1 解压 ITS
------------

先确认所有分卷都在：

```bash
ls -lh downloads/ITS.*
```

优先尝试：

```bash
unzip downloads/ITS.zip -d dataset/RESIDE/ITS
```

如果提示分卷不支持或解压失败，改用：

```bash
7z x downloads/ITS.zip -odataset/RESIDE/ITS
```

解压后检查目录：

```bash
find dataset/RESIDE/ITS -maxdepth 3 -type d | sort | head -80
```

理想结构大致是：

```text
dataset/RESIDE/ITS/train/hazy
dataset/RESIDE/ITS/train/clear
dataset/RESIDE/ITS/test/hazy
dataset/RESIDE/ITS/test/clear
```

如果多了一层 `ITS/ITS/train`，需要把里面的内容移动到正确位置。先不要急着删，先用 `find` 看清楚目录结构。

9.2 解压 Haze4K
---------------

```bash
unzip downloads/Haze4K.zip -d dataset/HAZE4K
```

检查：

```bash
find dataset/HAZE4K -maxdepth 3 -type d | sort | head -80
```

9.3 检查 OTS
------------

如果 OTS_ALPHA 已经用 scp 传成文件夹，不需要解压，直接检查：

```bash
find dataset/RESIDE/OTS -maxdepth 3 -type d | sort | head -80
du -sh dataset/RESIDE/OTS
```

如果有 OTS.zip，则：

```bash
unzip downloads/OTS.zip -d dataset/RESIDE/OTS
```


10. 配置 Python / PyTorch 环境
=============================

本节命令在 Ubuntu 服务器执行。

先检查 Python 和 Conda：

```bash
which python
python --version
conda --version
```

RTX 5090 比较新，不建议直接照旧 README 使用 PyTorch 1.10。建议优先使用较新的 PyTorch CUDA 12.x 轮子。

创建环境：

```bash
conda create -n deanet python=3.10 -y
conda activate deanet
python -m pip install --upgrade pip
```

安装 PyTorch。示例使用 CUDA 12.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

安装项目依赖：

```bash
cd /root/workspace/Dehaze-Net
pip install -r requirements.txt
```

检查 PyTorch 和 GPU：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

成功标准：

- `cuda available: True`
- 能打印出你的 GPU 名称

如果是 False，先不要训练，优先检查显卡驱动、CUDA、PyTorch 版本是否匹配。


11. 先跑官方 checkpoint 测试
===========================

本节命令在 Ubuntu 服务器执行。

先检查 checkpoint 文件名：

```bash
cd /root/workspace/Dehaze-Net
find trained_models -type f | sort
```

进入代码目录：

```bash
cd /root/workspace/Dehaze-Net/code
conda activate deanet
```

测试 ITS：

```bash
python eval.py --dataset ITS --model_name DEA-Net-CR --pre_trained_model PSNR4131_SSIM9945.pth
```

测试 OTS：

```bash
python eval.py --dataset OTS --model_name DEA-Net-CR --pre_trained_model PSNR3659_SSIM9897.pth
```

测试 HAZE4K 时，先用 `find ../trained_models/HAZE4K -type f` 查看真实文件名，再替换命令里的文件名：

```bash
python eval.py --dataset HAZE4K --model_name DEA-Net-CR --pre_trained_model 实际HAZE4K权重文件名.pth
```

保存推理图片：

```bash
python eval.py --dataset ITS --model_name DEA-Net-CR --pre_trained_model PSNR4131_SSIM9945.pth --save_infer_results
```

成功标准：

- 程序能跑完
- 不报 `FileNotFoundError`
- 能输出 PSNR / SSIM 等指标
- 如果加了 `--save_infer_results`，能在结果目录看到推理图片

如果报数据找不到，检查：

```bash
cd /root/workspace/Dehaze-Net
ls -lah dataset
find dataset -maxdepth 4 -type d | sort | head -120
```

如果报 checkpoint 找不到，检查：

```bash
find trained_models -type f | sort
```


12. 跑一个最小训练 smoke test
============================

目的：先确认训练流程能启动、能前向、能反向、能保存输出。不要一上来跑 300 epoch。

本节命令在 Ubuntu 服务器执行。

```bash
cd /root/workspace/Dehaze-Net/code
conda activate deanet
python train.py --epochs 1 --iters_per_epoch 10 --finer_eval_step 10 --w_loss_L1 1.0 --w_loss_CR 0.1 --start_lr 0.0001 --end_lr 0.000001 --exp_dir ../experiment/ --model_name smoke-test --dataset ITS
```

如果提示实验目录已存在，换一个名字：

```bash
python train.py --epochs 1 --iters_per_epoch 10 --finer_eval_step 10 --w_loss_L1 1.0 --w_loss_CR 0.1 --start_lr 0.0001 --end_lr 0.000001 --exp_dir ../experiment/ --model_name smoke-test-v2 --dataset ITS
```

成功标准：

```bash
cd /root/workspace/Dehaze-Net
find experiment -maxdepth 4 -type d | sort | head -80
find experiment -name "*.pk" -o -name "*.pth" -o -name "*.log"
```

能看到 `experiment` 下产生了新实验目录或模型文件，说明训练主流程基本跑通。


13. 跑完整训练
==============

建议用 tmux，断开 SSH 后训练不会停。

本节命令在 Ubuntu 服务器执行。

创建 tmux 会话：

```bash
tmux new -s deanet
```

在 tmux 里运行：

```bash
cd /root/workspace/Dehaze-Net/code
conda activate deanet
python train.py --epochs 300 --iters_per_epoch 5000 --finer_eval_step 1400000 --w_loss_L1 1.0 --w_loss_CR 0.1 --start_lr 0.0001 --end_lr 0.000001 --exp_dir ../experiment/ --model_name DEA-Net-CR-ITS-baseline --dataset ITS
```

退出 tmux 但保持训练运行：

```text
先按 Ctrl+B，再按 D
```

重新进入：

```bash
tmux attach -t deanet
```

查看已有 tmux 会话：

```bash
tmux ls
```

查看 GPU：

```bash
nvidia-smi
```


14. Git 常用协作命令
===================

本节命令在 Ubuntu 服务器或本地仓库执行。

开始改代码前，先确认状态：

```bash
cd /root/workspace/Dehaze-Net
git status
git pull --ff-only
```

新建实验分支：

```bash
git checkout -b exp/your-change-name
```

查看改了什么：

```bash
git status
git diff
```

提交：

```bash
git add 文件路径
git commit -m "简短说明这次修改"
```

推送：

```bash
git push -u origin exp/your-change-name
```

撤回还没提交的某个文件改动：

```bash
git restore 文件路径
```

撤回已经 `git add` 但还没 commit 的文件：

```bash
git restore --staged 文件路径
```

查看提交历史：

```bash
git log --oneline -10
```

新手注意：不要随便执行 `git reset --hard`，它会丢弃本地改动。


15. 常见问题
============

15.1 GitHub clone 失败
---------------------

检查 SSH：

```bash
ssh -T git@github.com
```

检查 remote：

```bash
git remote -v
```

如果 remote 是 HTTPS，但你想用 SSH：

```bash
git remote set-url origin git@github.com:onenoober/Dehaze-Net.git
```

15.2 scp 上传失败
-----------------

确认这条命令在 Windows PowerShell 运行：

```powershell
ssh -i $KEY $HOST "pwd"
```

确认本地文件存在：

```powershell
Test-Path "D:\BaiduNetdiskDownload\dehaze\ITS\ITS.zip"
```

确认服务器目录存在：

```powershell
ssh -i $KEY $HOST "ls -lah /root/workspace/Dehaze-Net"
```

15.3 FileNotFoundError: dataset
-------------------------------

检查目录：

```bash
cd /root/workspace/Dehaze-Net
find dataset -maxdepth 4 -type d | sort | head -120
ls -lah dataset
```

15.4 FileNotFoundError: checkpoint
----------------------------------

检查权重文件：

```bash
cd /root/workspace/Dehaze-Net
find trained_models -type f | sort
```

15.5 CUDA out of memory
-----------------------

先看显存：

```bash
nvidia-smi
```

处理思路：

- 先跑 smoke test，不要直接完整训练。
- 关闭其他占用显存的进程。
- 当前 DEA-Net 的 batch size 可能在 `code/train.py` 里写死，需要改代码里的 DataLoader batch size。

15.6 实验目录已存在
-------------------

如果提示类似：

```text
experiment/ITS/smoke-test has already existed
```

换一个 `--model_name`：

```bash
--model_name smoke-test-v2
```

15.7 服务器磁盘不够
-------------------

查看占用：

```bash
cd /root/workspace/Dehaze-Net
du -sh dataset downloads trained_models experiment
df -h
```

原则：

- `downloads` 里是压缩包，确认解压成功后可以考虑删除。
- `dataset` 和 `trained_models` 不要误删。
- 删除前先 `ls -lh` 看清楚路径。


16. 第一次推荐执行顺序
=====================

最稳妥的顺序：

1. Windows PowerShell 先确认能 SSH 登录服务器。
2. 服务器安装基础工具。
3. 服务器配置 GitHub SSH key。
4. clone 或 pull DEA-Net 仓库。
5. 创建 dataset、trained_models、downloads、experiment 目录和软链接。
6. Windows PowerShell 上传 ITS test/train、checkpoint、Haze4K 或 OTS 数据。
7. 服务器解压并用 `find` 检查目录结构。
8. 配置 conda/PyTorch 环境。
9. 先跑 `eval.py` 官方 checkpoint 测试。
10. 再跑 1 epoch、10 iter 的 smoke test。
11. 最后用 tmux 跑完整训练。

核心检查命令汇总：

```bash
cd /root/workspace/Dehaze-Net
git status
ls -lah dataset
find dataset -maxdepth 4 -type d | sort | head -120
find trained_models -type f | sort
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
nvidia-smi
df -h
```
