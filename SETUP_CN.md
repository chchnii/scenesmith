# SceneSmith 离线/受限网络环境安装指南

> 适用场景：内网 GPU 机器,仅能通过 SOCKS5 代理(如 `ssh -D` 隧道)访问外网,
> 无法直连 huggingface.co / github.com / PyPI 官方源。
> 本文档记录了在京东云 H200 节点上的完整踩坑与复现流程(2026-09)。
>
> 已有可工作环境需要迁移到其他集群时，请优先阅读
> [MIGRATION_CN.md](MIGRATION_CN.md)。其中包含当前本地代码改动、`conda-pack`、项目数据、
> `external/checkpoints`、`/root/.cache` 隐式模型、DINOv2 `/tmp` 软链接、实验阶段 checkpoint 和
> 目标机离线验收的完整流程。

## 其他机器迁移使用(环境已装好后整体搬走)

> 前提: 源码仓库、conda 环境、CUDA 工具链、checkpoints、data 五块可整体拷贝。
> 内网相通可使用 rsync；不通则制作校验和并切片后物理拷贝。完整命令见
> [MIGRATION_CN.md](MIGRATION_CN.md)。
>
> **重要更正：** 本节仅作为早期速查。普通 `tar` Conda 环境后不能假设可以执行
> `conda-unpack`；需要迁移到不同绝对路径时，必须用 `conda-pack` 生成归档。此外，本项目还依赖
> `/root/.cache/huggingface`、`/root/.cache/torch` 和 `/tmp/dinov2` 中的隐式模型/源码。
> 请按 [MIGRATION_CN.md](MIGRATION_CN.md) 操作，不要只复制下表五项。

迁移规模参考:

| 组件 | 路径 | 体积 |
|------|------|------|
| conda 环境 | `$CONDA_PREFIX`(scenesmith) | ~13G |
| CUDA 工具链 | `/path/to/cuda-12.4` | ~0.24G |
| SAM checkpoints | `scenesmith/external/checkpoints` | ~16G |
| 数据 | `scenesmith/data` | ~54G |
| Hugging Face 模型缓存 | `/root/.cache/huggingface/hub` 中的 MoGe/DFN5B | ~4.9G |
| Torch Hub 缓存 | DINOv2 权重和源码 | ~1.5G |
| 代码仓库 | `scenesmith/`(排除 data/checkpoints) | 小 |

**第一步 打包(旧机):**

```bash
# conda-pack 需要提前安装；editable 包在目标机重新安装。
conda-pack -p "$CONDA_PREFIX" --ignore-editable-packages \
  -o scenesmith_env.tar.gz

tar cJf cuda-12.4.tar.xz -C $(dirname $CUDA_HOME) $(basename $CUDA_HOME)
tar cf checkpoints.tar -C $SMITH/external checkpoints
tar cf data.tar -C $SMITH data
# 代码仓库(不带 data/checkpoints,git 或 tar 均可):
tar cf code.tar --exclude=data --exclude=external/checkpoints -C $(dirname $SMITH) $(basename $SMITH)
```

**第二步 传输(内网相通):**
`scp scenesmith_env.tar.gz cuda-12.4.tar.xz checkpoints.tar data.tar code.tar user@新机:/data/`

**第三步 解包+修复(新机,按序):**

```bash
# 3.1 conda 环境: 解压到同级目录,再用 conda-unpack 修正硬编码前缀
mkdir -p "$NEW_ENV"
tar xzf scenesmith_env.tar.gz -C "$NEW_ENV"
"$NEW_ENV/bin/conda-unpack"

# 3.2 CUDA 工具链(建议放到与旧机相同绝对路径一致的问题最小)
tar xJf cuda-12.4.tar.xz -C /path/to

# 3.3 checkpoints 和 data 解到代码仓库
tar xf checkpoints.tar -C $SMITH/external
tar xf data.tar -C $SMITH

# 3.4 重装两个可编辑(editable)包 —— 它们的 .pth 指向旧机器绝对路径,必须重装
cd $SMITH && $NEW_ENV/bin/pip install -e . --no-deps
cd $SMITH/external/SAM3 && $NEW_ENV/bin/pip install -e . --no-deps

# 3.5 更新环境变量(.bashrc): CUDA_HOME PATH LD_LIBRARY_PATH OPENAI_* HF_*
export CUDA_HOME=<新机 cuda-12.4 路径>; export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:../lib:$LD_LIBRARY_PATH
```

`/root/.cache` 不能直接整目录照搬；其中 DINOv2 仓库还是指向 `/tmp/dinov2` 的绝对软链接。模型缓存
必须按 [MIGRATION_CN.md 的第 8 节](MIGRATION_CN.md) 的 staging
方法单独打包，并在目标端设置 `HF_HOME`、`TORCH_HOME`。

**⚠️ 最关键变量: 目标 GPU 型号。** SAM3D 的 CUDA 包按 `TORCH_CUDA_ARCH_LIST="9.0"`(Hopper)编译:

- 目标机同为 Hopper(H100/H200): 整包直接搬,**零重编**,最快。
  本项目实测: 从 H200 迁到 H200,四个 CUDA 包全部直接复用,无需重编。
- 目标机是 A100/4090/L40S 等: 机器码不兼容,gsplat/nvdiffrast/kaolin/pytorch3d
  四个包**必须在目标机重编**(按 §6 tarball 方式,约半小时),其余可直接搬。

**验证(新机):**

```bash
python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK')"
python -c "import gsplat,nvdiffrast,kaolin,pytorch3d,sam3,moge,scenesmith; print('全部 OK')"
```

## 0. 前置条件

- conda 环境: Python 3.11 (`requires-python >=3.11,<3.12`)
- NVIDIA 驱动已装好(`nvidia-smi` 可用),但**不需要**预装 CUDA toolkit
- 一条可用的 SOCKS5 代理,假设在 `127.0.0.1:1080`
- proxychains4,并确认配置里有 localnet 排除(关键!否则 git 会被双重代理搞挂):

```bash
# /etc/proxychains4.conf 的 [ProxyList] 前必须有这一行:
localnet 127.0.0.0/255.0.0.0
# [ProxyList] 下: socks5 127.0.0.1 1080
```

- git 全局代理(git 自己走代理,不要套 proxychains):

```bash
git config --global http.https://github.com.proxy socks5h://127.0.0.1:1080
```

- 环境变量(写入 `~/.bashrc`):

```bash
export OPENAI_API_KEY="你的key"
export OPENAI_BASE_URL="http://你的中转站/v1"
export HF_TOKEN="hf_xxx"          # 必须是 classic Read token,fine-grained 默认无 gated 仓库权限
export HF_ENDPOINT="https://hf-mirror.com"
```

## 1. 主 Python 依赖(用 pip,不用 uv)

> 坑: uv 在慢代理下超时频繁;pip 对不稳定网络更宽容。
> 坑: `mathutils` 用 Python 3.11 编译不过 —— 不用装,`bpy` 自带。

```bash
PIP=$CONDA_PREFIX/bin/pip
# 从 pyproject.toml 手动生成 requirements(去掉 mathutils),然后:
proxychains4 -q $PIP install -r requirements.txt \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple --retries 8
$PIP install -e . --no-deps
# opencv 需要系统库:
apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

## 2. 组装 CUDA 12.4 工具链(无 root 安装 CUDA 的野路子)

> 坑1: PyPI 的 `nvidia-cuda-nvcc-cu12` **不含完整 nvcc**(只有 ptxas,给 JAX 用的)。
> 坑2: pip 的 cccl 包缺 `nv/target`(libcu++)。
> 坑3: pip 的库包只有 `libXXX.so.12`,没有链接器要的 `libXXX.so`。
> 解法: conda 包就是普通 tar,直接下载解压,不需要 conda 命令。

```bash
CUDA_HOME=/path/to/cuda-12.4   # 任选目录
mkdir -p $CUDA_HOME && cd /tmp
BASE=https://conda.anaconda.org/nvidia/linux-64
for pkg in cuda-nvcc-12.4.131-0 cuda-cccl-12.4.127-0 \
           cuda-cudart-static-12.4.127-0 cuda-cudart-dev-12.4.127-0 \
           cuda-driver-dev-12.4.127-0; do
  proxychains4 -q curl -sL -o $pkg.tar.bz2 "$BASE/$pkg.tar.bz2"
  tar xjf $pkg.tar.bz2 -C $CUDA_HOME
done

# 补齐 pip nvidia 包的头文件(cusparse.h/cublas_v2.h 等编译 torch 扩展必需):
NV=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia
$PIP install nvidia-cuda-cccl-cu12==12.4.127 nvidia-cuda-profiler-api-cu12==12.4.127
for pkg in cuda_runtime cuda_cccl cuda_profiler_api cuda_nvrtc \
           cusparse cublas cusolver cufft curand cudnn nccl nvjitlink cuda_cupti nvtx; do
  [ -d "$NV/$pkg/include" ] && cp -rn $NV/$pkg/include/* $CUDA_HOME/include/
done

# 库文件: 软链 pip 的 .so + 静态库,并给带版本的库建无版本链接:
mkdir -p $CUDA_HOME/lib64
for pkg in cuda_runtime cuda_nvrtc cublas cusparse cusolver cufft curand cudnn nvjitlink nvtx; do
  ln -sf $NV/$pkg/lib/*.so* $CUDA_HOME/lib64/
done
ln -sf $CUDA_HOME/lib/*.a $CUDA_HOME/lib64/
cd $CUDA_HOME/lib64
for f in lib*.so.*; do b=${f%%.so.*}.so; [ -e "$b" ] || ln -sf "$f" "$b"; done

# 验证(应输出 12.4 且能编译运行):
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version
# 环境变量写入 ~/.bashrc:
#   export CUDA_HOME=... ; PATH=$CUDA_HOME/bin:$PATH ; LD_LIBRARY_PATH=$CUDA_HOME/lib64:...
```

## 3. SAM 代码仓库(浅克隆,防隧道断线)

> 坑: 完整 git clone 经隧道必挂(TLS 中断)。浅取指定 commit 传输量最小。

```bash
cd external
for r in "sam-3d-objects https://github.com/facebookresearch/sam-3d-objects.git 81a82373a3a7f4cbb00bd5b32aaf6b4d0f659ddd" \
         "SAM3 https://github.com/facebookresearch/sam3.git 11dec2936de97f2857c1f76b66d982d5a001155d"; do
  set -- $r
  git init -q $1 && cd $1 && git remote add origin $2
  until git fetch --depth 1 origin $3; do sleep 15; done   # 失败自动重试
  git checkout -q FETCH_HEAD && cd ..
done
```

## 4. 模型 checkpoints(用 ModelScope,免 HF 审批!)

> HF 上 facebook/sam3 和 facebook/sam-3d-objects 是人工审批(gated: manual),
> **ModelScope 镜像无门槛**且速度快(实测 ~7MB/s)。

```bash
mkdir -p external/checkpoints
# 文件清单从 API 获取:
# https://modelscope.cn/api/v1/models/facebook/sam3/repo/files?Recursive=true
# https://modelscope.cn/api/v1/models/facebook/sam-3d-objects/repo/files?Recursive=true
# 逐个下载(URL 格式),用 curl -C - 断点续传 + 校验文件大小:
proxychains4 -q curl -sL -C - -o external/checkpoints/sam3.pt \
  "https://modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt"
# sam-3d-objects 的 checkpoints/* 同理(19 个文件,共 13.1GB),下到 external/checkpoints/ 平铺
# 注意: ss_encoder.safetensors 在 ModelScope 上是 0 字节坏文件,但 pipeline.yaml 只用 .ckpt,不需要它
touch external/checkpoints/.sam3d_objects_downloaded
```

## 5. 数据集

```bash
HF=$CONDA_PREFIX/bin/hf
# ArtVIP(HF 公开数据集,走 hf-mirror):
proxychains4 -q $HF download nepfaff/scenesmith-preprocessed-data \
    artvip/artvip_vhacd.tar.gz --repo-type dataset --local-dir . --max-workers 2
mkdir -p data/artvip_sdf && tar xzf artvip/artvip_vhacd.tar.gz -C data/artvip_sdf && rm -rf artvip

# AmbientCG CLIP embeddings(预计算好的):
proxychains4 -q $HF download nepfaff/scenesmith-preprocessed-data \
    --repo-type dataset --include "ambientcg/embeddings/**" --local-dir data/tmp
mkdir -p data/materials && mv data/tmp/ambientcg/embeddings data/materials/embeddings && rm -rf data/tmp

# AmbientCG 材质本体(2009 个,~45GB,官方脚本走代理,降低并发防隧道过载):
proxychains4 -q python scripts/download_ambientcg.py --output data/materials -c 2
```

## 6. SAM3D Python 依赖(关键坑最多的一步)

> 坑: pip 内部的 `git clone` 经隧道极易断,而**纯 HTTPS 下载很稳** ——
> 所以 git 依赖全部改用 GitHub 的 codeload tarball URL。
> 坑: 用约束文件锁 torch 版本,防止依赖解析把 torch 升级掉。

```bash
export TORCH_CUDA_ARCH_LIST="9.0"   # H200/H100=9.0, A100=8.0, RTX4090=8.9 —— 按你的卡改!
export MAX_JOBS=32
printf 'torch==2.5.1\ntorchvision==0.20.1\nnumpy<2.0\n' > /tmp/c.txt
TUNA="--index-url https://pypi.tuna.tsinghua.edu.cn/simple"
P="proxychains4 -q $PIP"

# 6.1 SAM3 包 + sam-3d-objects 基础依赖
$P install -e "./external/SAM3[notebooks]" -c /tmp/c.txt $TUNA
grep -vE "^(torch|torchvision|torchaudio|cuda-python|nvidia-|MoGe|flash_attn|bpy|wandb|jupyter|tensorboard|Flask|webdataset|sagemaker)" \
    external/sam-3d-objects/requirements.txt > /tmp/req.txt
$P install -r /tmp/req.txt -c /tmp/c.txt $TUNA

# 6.2 gsplat: tarball 不含 glm 子模块,需手动拼装
cd /tmp && proxychains4 -q curl -sL -o gsplat.tar.gz \
  "https://codeload.github.com/nerfstudio-project/gsplat/tar.gz/2323de5905d5e90e035f792fe65bad0fedd413e7"
mkdir gsplat-src && tar xzf gsplat.tar.gz -C gsplat-src --strip-components=1
proxychains4 -q curl -sL -o glm.tar.gz \
  "https://codeload.github.com/g-truc/glm/tar.gz/33b4a621a697a305bc3a7610d290677b96beb181"
mkdir -p gsplat-src/gsplat/cuda/csrc/third_party/glm
tar xzf glm.tar.gz -C gsplat-src/gsplat/cuda/csrc/third_party/glm --strip-components=1
$P install --no-build-isolation -c /tmp/c.txt $TUNA /tmp/gsplat-src

# 6.3 其余三个直接装 tarball URL(无子模块):
$P install --no-build-isolation -c /tmp/c.txt $TUNA \
  "https://codeload.github.com/NVlabs/nvdiffrast/tar.gz/refs/heads/main"
$P install --no-build-isolation -c /tmp/c.txt $TUNA \
  "https://codeload.github.com/NVIDIAGameWorks/kaolin/tar.gz/refs/tags/v0.17.0"
$P install --no-build-isolation -c /tmp/c.txt $TUNA \
  "https://codeload.github.com/facebookresearch/pytorch3d/tar.gz/refs/heads/main"

# 6.4 推理依赖 + MoGe(仓库小,git URL 也行,不稳就换 tarball):
$P install seaborn==0.13.2 gradio==5.49.0 imageio utils3d -c /tmp/c.txt $TUNA
$P install -c /tmp/c.txt $TUNA \
  "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

# 6.5 nvdiffrast CUDA 核预编译(可选,否则首次使用时 JIT):
python -c "import nvdiffrast.torch as dr; dr.RasterizeCudaContext()"

# 验证:
python -c "
for m in ['sam3','gsplat','nvdiffrast','kaolin','pytorch3d','moge']:
    __import__(m); print('OK', m)"
```

## 7. API 中转站适配(如果中转站只支持 Chat Completions)

本仓库已做如下修改(如果换机器重新 clone 原版仓库,需要重做):

| 文件 | 修改 |
|------|------|
| `configurations/*/base_*.yaml` | `model`/`summarization_model`: `gpt-5.2` → `gpt-5`(按中转站支持的模型改) |
| `main.py` | 顶部加 `from agents import set_default_openai_api; set_default_openai_api("chat_completions")` |
| `scenesmith/agent_utils/vlm_service.py` | `self._reasoning_models` 置为空集合(全部走 chat completions) |
| `scenesmith/agent_utils/turn_trimming_session.py` | 摘要调用 `responses.create` 改为 `chat.completions.create` |
| `scenesmith/agent_utils/image_generation.py` | `gpt-image-1.5` → `gpt-image-1`(按中转站支持改) |

中转站能力探测方法:

```bash
curl -s $OPENAI_BASE_URL/chat/completions -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

## 8. 踩坑速查表

| 症状 | 原因 | 解法 |
|------|------|------|
| uv sync 卡死/超时 | uv 对慢代理不友好 | 换 pip + 清华源 |
| mathutils 编译失败 `PyLong_AsInt` | 源码不兼容 Py3.11 | 不装,bpy 自带 |
| `import cv2` 报 libgthread | 缺系统库 | apt 装 libglib2.0-0 等 |
| nvcc: No such file(pip 装过) | PyPI 的 nvcc wheel 是残缺版 | conda 包手动解压(见§2) |
| `fatal error: nv/target` | pip cccl 缺 libcu++ | conda cuda-cccl 解压(见§2) |
| `cannot find -lcudart` | pip 库无 unversioned .so | 建软链(见§2) |
| `fatal error: cusparse.h` | 头文件没合并全 | 合并全部 pip nvidia 包头文件(见§2) |
| git clone 反复 TLS 中断 | 隧道掐长连接 | 浅克隆重试;或 codeload tarball |
| proxychains + git 连不上 127.0.0.1:1080 | 缺 localnet 排除 | proxychains4.conf 加 localnet(见§0) |
| HF 403(已批准+read token) | fine-grained token 无 gated 权限 | 换 classic Read token |
| HF gated 审批等不来 | sam3 是人工审批 | 用 ModelScope 镜像(见§4) |
| pip 把 torch 换版本 | 依赖解析升级 | `-c` 约束文件锁版本 |
