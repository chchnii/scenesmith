# SceneSmith 跨集群迁移手册

> 本文档面向已经在源集群正常运行 SceneSmith、希望把当前可工作的环境完整迁移到另一套集群的场景。
> 它基于 `SETUP_CN.md`、当前工作区代码和 2026-09-07 的机器实测结果整理。
>
> 本文区分四类容易混淆的内容：Python/CUDA 运行环境、项目数据、模型 checkpoint、实验阶段
> checkpoint。尤其注意：当前机器还有约 6.4 GB 的运行必需模型位于 `/root/.cache`，仅复制仓库
> 并不能在离线机器上完整运行。

## 1. 当前可工作基线

当前源机器的重要版本和路径如下。迁移前建议再次执行第 4 节的清单命令，以实际输出为准。

| 项目 | 当前值 | 约占空间 | 是否必须迁移 |
|---|---|---:|---|
| SceneSmith 工作区 | `/mnt/workspace1/users/yuxiqian/scenesmith` | 代码约 0.6 GB | 必须，且必须包含未提交改动 |
| Conda 环境 | `/mnt/workspace1/users/yuxiqian/miniconda3xx/envs/scenesmith` | 14 GB | 必须，或在目标机重建 |
| CUDA toolkit | `/mnt/workspace1/users/yuxiqian/cuda-12.4` | 241 MB | 建议迁移 |
| SAM3/SAM3D checkpoint | `external/checkpoints` | 16 GB | 必须 |
| ArtVIP 数据 | `data/artvip_sdf` | 8.9 GB | 当前配置必须 |
| AmbientCG 材质 | `data/materials` | 45 GB | 当前配置必须 |
| Hugging Face 模型缓存 | `/root/.cache/huggingface/hub` 中两个模型 | 4.9 GB | 离线运行必须 |
| Torch Hub DINOv2 权重 | `/root/.cache/torch/hub/checkpoints` | 1.5 GB | SAM3D 必须 |
| DINOv2 Torch Hub 源码 | `/tmp/dinov2` | 7.1 MB | 离线运行必须 |
| 历史实验输出 | `outputs` | 当前约 4.7 GB | 仅续跑或保留结果时需要 |

当前主要软件版本：

```text
Python             3.11.16
PyTorch            2.5.1+cu124
torchvision        0.20.1
CUDA toolkit       12.4.131
openai             2.54.0
openai-agents      0.6.9
Flask              3.0.3
Werkzeug           3.0.6
bpy                 4.5.4
open_clip_torch     3.3.0
transformers        4.39.3
huggingface_hub     0.36.2
```

当前 SceneSmith 基线 commit：

```text
67cc408fd38334b4a926efef45e284302ed5055b
```

但当前工作树含有大量尚未提交的稳定性修改和新增文件，因此只在目标机器执行 `git clone` 或
`git checkout 67cc408...` **不能**得到当前可工作的版本。必须打包当前完整工作树，或先把改动提交到
一个可访问的分支。

## 2. 当前代码中必须一同迁移的本地修改

当前工作区相对上述 commit 的重要修改包括：

- OpenAI Agents SDK 强制使用 Chat Completions，以适配当前 API 中转站。
- `gpt-5.2` 改为 `gpt-5`，图片模型按中转站能力改为 `gpt-image-1`。
- `scenesmith/_compat.py` 修复 `InputTokensDetails.cache_write_tokens` 兼容性问题。
- 对 429/无 `Retry-After` 的情况等待 65 秒，OpenAI 客户端默认最多重试 4 次。
- agent 内部 `parallel_workers` 降为 1，减少每分钟 token 限额导致的 429。
- Geometry worker 改为全新解释器进程和 Unix socket 协议，避免父进程初始化 CUDA 后重启 worker
  触发 `Cannot re-initialize CUDA in forked subprocess`。
- 修复请求序列化失败时 callback/worker-token 泄漏，以及 INIT 发送异常路径。
- Flask/Werkzeug server 改为显式管理的 `make_server(...).serve_forever()`，保证 shutdown 能真正释放端口。
- 本地 `127.0.0.1` 健康检查和渲染请求绕过 HTTP/SOCKS 代理。
- geometry server 使用 lazy preload，并把启动等待时间提高到 180 秒。
- 增加 `KeyboardInterrupt`/`CancelledError` traceback 和 server 清理日志。

特别不能遗漏的新增文件：

```text
scenesmith/_compat.py
scenesmith/agent_utils/geometry_generation_server/worker_entry.py
scenesmith/agent_utils/geometry_generation_server/worker_protocol.py
tests/unit/test_compat.py
tests/unit/test_geometry_generation_server_manager.py
tests/unit/test_retrieval_server_manager.py
```

因此本文后面使用“完整工作树归档”，而不是只生成 `git diff`。`git diff` 不包含未跟踪文件，单独迁移
patch 很容易漏掉上面的新文件。

## 3. 目标集群兼容性检查

### 3.1 操作系统和驱动

目标节点至少应满足：

- Linux x86_64。
- NVIDIA 驱动支持 CUDA 12.4 runtime；以 `nvidia-smi` 正常工作为准。
- 目标系统 glibc 不应明显旧于源环境。当前源容器用户态是 glibc 2.39；若目标集群是较老系统，优先
  在目标集群提供的相似 Ubuntu 24.04/容器镜像内运行，而不是直接搬二进制环境。
- 至少约 100 GB 可用空间；如果还要在源端同时生成所有归档，建议预留 120 GB 以上。
- `data/artvip_sdf` 有超过 12 万个文件，目标文件系统还要有足够 inode。

检查命令：

```bash
uname -m
ldd --version | head -1
nvidia-smi
df -h
df -i
```

### 3.2 GPU 架构

当前 CUDA 扩展按 Hopper 架构编译。目标 GPU 决定环境能否原样搬运：

| GPU | Compute capability | 处理方式 |
|---|---:|---|
| H100/H200 | 9.0 | 可以直接迁移当前环境，推荐 |
| A100 | 8.0 | 重编 gsplat、nvdiffrast、kaolin、pytorch3d |
| RTX 4090/L40/L40S | 8.9 | 重编上述四个 CUDA 扩展 |
| 其他架构 | 先查 capability | 按实际 `TORCH_CUDA_ARCH_LIST` 重编 |

如果目标 GPU 不同，除了重编 CUDA 扩展，还要确保目标驱动能够加载 PyTorch 2.5.1 的 CUDA 12.4
runtime。不同 GPU 架构且完全离线时，应提前把第 `SETUP_CN.md` §6 中四个项目的源码 tarball 一起带走；
仅有已经编译好的 wheel/`.so` 不足以在新架构上重编。

### 3.3 同一节点的端口

SceneSmith 默认使用 7005–7009。当前代码不支持两个默认配置的任务在同一节点上同时占用这些端口。
迁移验证时先确保端口为空：

```bash
fuser -v 7005/tcp 7006/tcp 7007/tcp 7008/tcp 7009/tcp
```

在 Slurm 等调度器中，最简单的方案是一个节点只运行一个 SceneSmith job；否则每个 job 必须覆盖全部
server 端口，而不是只改 7005。

## 4. 源集群：停止任务并生成清单

先定义源路径。下面的值是当前机器实测路径：

```bash
SMITH_SRC=/mnt/workspace1/users/yuxiqian/scenesmith
SMITH_ENV_SRC=/mnt/workspace1/users/yuxiqian/miniconda3xx/envs/scenesmith
SMITH_CUDA_SRC=/mnt/workspace1/users/yuxiqian/cuda-12.4
SMITH_CACHE_SRC=/root/.cache
SMITH_BUNDLE=/mnt/workspace1/users/yuxiqian/scenesmith_migration_bundle

mkdir -p "$SMITH_BUNDLE/manifests"
```

打包前不要有正在写 `data`、`external/checkpoints` 或 `outputs` 的任务：

```bash
pgrep -af '^/mnt/workspace1/users/yuxiqian/miniconda3xx/envs/scenesmith/bin/python main.py'
fuser -v 7005/tcp 7006/tcp 7007/tcp 7008/tcp 7009/tcp
```

保存可审计的版本清单：

```bash
cd "$SMITH_SRC"

git rev-parse HEAD > "$SMITH_BUNDLE/manifests/scenesmith-git-head.txt"
git status --short > "$SMITH_BUNDLE/manifests/scenesmith-git-status.txt"
git diff --binary > "$SMITH_BUNDLE/manifests/scenesmith-working-tree.patch"

"$SMITH_ENV_SRC/bin/python" -V \
  > "$SMITH_BUNDLE/manifests/python-version.txt" 2>&1
"$SMITH_ENV_SRC/bin/python" -m pip freeze \
  > "$SMITH_BUNDLE/manifests/pip-freeze.txt"
"$SMITH_ENV_SRC/bin/python" -m pip list --editable \
  > "$SMITH_BUNDLE/manifests/pip-editable.txt"

"$SMITH_CUDA_SRC/bin/nvcc" --version \
  > "$SMITH_BUNDLE/manifests/nvcc-version.txt" 2>&1
nvidia-smi -q \
  > "$SMITH_BUNDLE/manifests/nvidia-smi.txt" 2>&1
ldd --version \
  > "$SMITH_BUNDLE/manifests/glibc-version.txt" 2>&1
```

说明：patch 只用于审计和比对，真正迁移使用下一节的完整代码包，因为 patch 不会包含未跟踪文件。

## 5. 源集群：打包代码

下面的归档保留当前所有已修改和未跟踪源码，同时排除大数据、checkpoint、运行输出和可能包含凭据的
`.env`。根仓库的 `.git` 会保留，便于目标机器继续检查差异；打包前应确认 `.git/config` 的 remote
URL 没有嵌入访问 token。

```bash
SMITH_SRC_PARENT=$(dirname "$SMITH_SRC")
SMITH_SRC_NAME=$(basename "$SMITH_SRC")

tar \
  --exclude="$SMITH_SRC_NAME/data" \
  --exclude="$SMITH_SRC_NAME/external/checkpoints" \
  --exclude="$SMITH_SRC_NAME/outputs" \
  --exclude="$SMITH_SRC_NAME/run_logs" \
  --exclude="$SMITH_SRC_NAME/.venv" \
  --exclude="$SMITH_SRC_NAME/.env" \
  --exclude="$SMITH_SRC_NAME/.pytest_cache" \
  --exclude='__pycache__' \
  -C "$SMITH_SRC_PARENT" \
  -cf "$SMITH_BUNDLE/scenesmith-code.tar" \
  "$SMITH_SRC_NAME"
```

验证新增文件确实进入归档：

```bash
tar -tf "$SMITH_BUNDLE/scenesmith-code.tar" \
  | grep -E 'scenesmith/_compat.py|worker_entry.py|worker_protocol.py'
```

不要用“目标机重新 clone 原仓库”代替这一步，否则当前的 worker 重启、server shutdown、429 重试等
修改都会丢失。

## 6. 源集群：用 conda-pack 打包环境

### 6.1 为什么不能直接 tar 后运行 conda-unpack

`conda-unpack` 是 `conda-pack` 在生成归档时放入环境并准备好前缀替换信息的工具。普通地对现有 Conda
目录执行 `tar`，不会自动生成这些重定位信息；当前环境中也没有现成的 `conda-unpack`。所以：

- 目标路径与源路径不同：必须使用 `conda-pack`。
- 只有在目标绝对路径完全相同的情况下，普通 tar 才可作为保守备选，并且不要假设可以运行
  `conda-unpack`。

### 6.2 安装并运行 conda-pack

当前环境有两个 editable 包：

```text
scenesmith -> 当前仓库
sam3       -> external/SAM3
```

因此打包时使用 `--ignore-editable-packages`，目标端再重新执行 `pip install -e`：

```bash
/mnt/workspace1/users/yuxiqian/miniconda3xx/bin/python \
  -m pip install conda-pack

/mnt/workspace1/users/yuxiqian/miniconda3xx/bin/conda-pack \
  -p "$SMITH_ENV_SRC" \
  --ignore-editable-packages \
  -o "$SMITH_BUNDLE/scenesmith-env.tar.gz"
```

如果 `conda-pack` 报某个 Conda 包的文件被 pip 改写，不要第一时间使用
`--ignore-missing-files` 掩盖错误；先把报错保存到 manifest，并确认缺失文件不会影响运行。只有在已验证环境
可以正常导入和运行后，才把 `--ignore-missing-files` 作为最后手段。

如果目标集群强制使用相同绝对路径，也可以使用普通 tar 备份：

```bash
tar -C "$(dirname "$SMITH_ENV_SRC")" \
  -czf "$SMITH_BUNDLE/scenesmith-env-same-prefix-only.tar.gz" \
  "$(basename "$SMITH_ENV_SRC")"
```

这个备选包必须解压回：

```text
/mnt/workspace1/users/yuxiqian/miniconda3xx/envs/scenesmith
```

路径不同则不要使用该备选方案。

## 7. 源集群：打包 CUDA、项目 checkpoint 和数据

### 7.1 CUDA 12.4 工具链

```bash
tar -C "$(dirname "$SMITH_CUDA_SRC")" \
  -cJf "$SMITH_BUNDLE/cuda-12.4.tar.xz" \
  "$(basename "$SMITH_CUDA_SRC")"
```

PyTorch wheel 自带 CUDA runtime，但 gsplat、nvdiffrast、kaolin、pytorch3d 的编译/重编仍需要完整的
`CUDA_HOME` 和头文件，所以建议把这 241 MB 一起迁移。

### 7.2 SAM3 和 SAM3D 模型 checkpoint

```bash
tar -C "$SMITH_SRC" \
  -cf "$SMITH_BUNDLE/model-checkpoints.tar" \
  external/checkpoints

(cd "$SMITH_SRC" && \
  find external/checkpoints -maxdepth 1 -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum) \
  > "$SMITH_BUNDLE/manifests/model-checkpoints.sha256"
```

当前目录包含：

- `sam3.pt`，约 3.45 GB。
- SAM3D 的 `ss_generator.ckpt`、`slat_generator.ckpt`、encoder/decoder checkpoint。
- `pipeline.yaml` 以及它引用的相对 YAML/CKPT 文件。

必须打包整个 `external/checkpoints`，不要只复制 `pipeline.yaml` 或 `sam3.pt`。

### 7.3 数据

当前默认配置实际存在并使用：

```text
data/artvip_sdf   约 8.9 GB，约 124696 个文件
data/materials    约 45 GB，约 21849 个文件
```

文件很多，先归档再传输比逐文件 scp 稳定：

```bash
tar -C "$SMITH_SRC" \
  -cf "$SMITH_BUNDLE/scenesmith-data.tar" \
  data
```

材质中的 JPG/PNG 已经压缩，给整个 54 GB 数据包再做 gzip 通常节省有限但耗时明显。网络非常慢时可改用
zstd；否则保留无压缩 tar 更容易快速打包和校验。

如果以后把 `general_asset_source` 改成 `hssd` 或 `objaverse`，还必须另外迁移配置中引用但当前机器未安装
的目录：

```text
data/hssd-models
data/preprocessed
data/objathor-assets
data/partnet_mobility_sdf
```

不要因为 YAML 中存在这些路径就误以为当前 `data.tar` 已经包含它们；应以 `find data -maxdepth 1` 的实际
结果为准。

## 8. 源集群：正确打包 `/root/.cache` 中的模型

### 8.1 必须迁移的缓存内容

当前机器实际需要的缓存如下：

| 缓存 | 文件/模型 | 大小 |
|---|---|---:|
| Hugging Face | `Ruicheng/moge-vitl` 的 `model.pt` | 1.2 GB |
| Hugging Face | `apple/DFN5B-CLIP-ViT-H-14-378` 的 OpenCLIP 权重 | 3.7 GB |
| Torch Hub | `dinov2_vitb14_pretrain.pth` | 346 MB |
| Torch Hub | `dinov2_vitl14_reg4_pretrain.pth` | 1.22 GB |
| Torch Hub | DINOv2 `hubconf.py` 和源码 | 7.1 MB |

当前源机器已验证的关键缓存 SHA-256：

```text
da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f  MoGe model.pt
c07a17b547d461c60a3cce5062b26bf8545b13de602c4c59d8490361eb716033  DFN5B open_clip_pytorch_model.bin
0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73  dinov2_vitb14_pretrain.pth
36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51  dinov2_vitl14_reg4_pretrain.pth
```

加载来源对应关系：

- `external/checkpoints/pipeline.yaml` 会从 `Ruicheng/moge-vitl` 加载 MoGe。
- `scenesmith/agent_utils/clip_embeddings.py` 使用 DFN5B OpenCLIP。
- SAM3D 的 DINO backbone 使用 Torch Hub 仓库和 `dinov2_vitb14_pretrain.pth`。
- 当前缓存还包含 `dinov2_vitl14_reg4_pretrain.pth`；虽然当前默认 pipeline 未必每次使用，体积可接受时
  建议一并迁移，以免其他配置重新下载。

### 8.2 DINOv2 绝对软链接陷阱

当前路径：

```text
/root/.cache/torch/hub/facebookresearch_dinov2_main -> /tmp/dinov2
```

这是指向 `/tmp` 的绝对软链接。普通 `tar /root/.cache/torch` 会保留这个链接，但不会自动把
`/tmp/dinov2` 放入归档；目标机器会得到一个坏链接。因此必须用 `cp -aL` 解引用，把源码实体化到 staging
目录。

### 8.3 建立最小、安全的模型缓存包

不要打包整个 `/root/.cache`。其中 6.1 GB 的 pip cache、1.5 GB 的 uv cache、GL shader cache 等都可
重建，而且缓存根目录未来可能出现登录 token。只复制已确认的模型目录：

```bash
SMITH_CACHE_STAGE="$SMITH_BUNDLE/cache-stage"

mkdir -p \
  "$SMITH_CACHE_STAGE/huggingface/hub" \
  "$SMITH_CACHE_STAGE/torch/hub"

cp -a \
  "$SMITH_CACHE_SRC/huggingface/hub/models--Ruicheng--moge-vitl" \
  "$SMITH_CACHE_STAGE/huggingface/hub/"

cp -a \
  "$SMITH_CACHE_SRC/huggingface/hub/models--apple--DFN5B-CLIP-ViT-H-14-378" \
  "$SMITH_CACHE_STAGE/huggingface/hub/"

cp -a \
  "$SMITH_CACHE_SRC/torch/hub/checkpoints" \
  "$SMITH_CACHE_STAGE/torch/hub/"

# -L 是关键：把 /tmp/dinov2 的内容复制进 staging，而不是复制绝对软链接。
cp -aL \
  "$SMITH_CACHE_SRC/torch/hub/facebookresearch_dinov2_main" \
  "$SMITH_CACHE_STAGE/torch/hub/facebookresearch_dinov2_main"

if test -f "$SMITH_CACHE_SRC/torch/hub/trusted_list"; then
  cp -a "$SMITH_CACHE_SRC/torch/hub/trusted_list" \
    "$SMITH_CACHE_STAGE/torch/hub/trusted_list"
fi

test -s \
  "$SMITH_CACHE_STAGE/huggingface/hub/models--Ruicheng--moge-vitl/snapshots/ad326bfb61facd6c52b5a825bc1e34d7c97d9672/model.pt"
test -s \
  "$SMITH_CACHE_STAGE/huggingface/hub/models--apple--DFN5B-CLIP-ViT-H-14-378/snapshots/01b771ed0d1395ca5ffdd279897d665ebe00dfd2/open_clip_pytorch_model.bin"
test -s \
  "$SMITH_CACHE_STAGE/torch/hub/facebookresearch_dinov2_main/hubconf.py"

tar -C "$SMITH_CACHE_STAGE" \
  -cf "$SMITH_BUNDLE/model-cache.tar" \
  huggingface torch
```

Hugging Face snapshot 中的模型文件是指向同一模型目录下 `blobs/` 的相对软链接，使用 `cp -a` 和 tar 会
正确保留。不要只复制 `snapshots/`，否则链接指向的 `blobs/` 会丢失。

以下缓存不需要迁移：

```text
/root/.cache/pip
/root/.cache/uv
/root/.cache/nvidia/GLCache
/root/.cache/blender
/root/.cache/matplotlib
/root/.cache/huggingface/xet
```

也不要迁移 `/root/.bashrc`、Hugging Face token、OpenAI key 或任何 `.env` 文件。凭据应当在目标集群由
用户环境、Slurm secret 或集群的密钥管理系统重新注入。

## 9. 可选：迁移实验输出和阶段 checkpoint

模型 checkpoint 和实验 checkpoint 是两类不同数据：

- `external/checkpoints`：SAM3/SAM3D 模型权重，运行必须。
- `outputs/.../scene_states`：某一次场景生成的阶段状态，只在继续运行或保留结果时需要。

如果要在目标集群继续某次实验，不要只复制一个 `scene_states` 子目录。场景状态可能引用同一实验目录下的
几何、纹理、渲染结果、package 文件和组合房屋数据；应复制整个时间戳实验目录。例如：

```bash
tar -C "$SMITH_SRC" \
  -cf "$SMITH_BUNDLE/output-2026-09-06-22-51-43.tar" \
  outputs/2026-09-06/22-51-43
```

目标机解压后，从已有阶段继续：

```bash
"$SMITH_ENV_DST/bin/python" main.py \
  +name=migrated_resume \
  experiment.pipeline.start_stage=wall_mounted \
  experiment.pipeline.resume_from_path="$SMITH_DST/outputs/2026-09-06/22-51-43"
```

`resume_from_path` 逻辑会复制所需阶段状态并修正新分支中的路径。仍建议保持仓库内相对布局不变。不要对
`.blend`、模型二进制或整个输出目录进行盲目的字符串替换；需要检查旧绝对路径时使用：

```bash
rg -n --fixed-strings "/mnt/workspace1/users/yuxiqian/scenesmith" \
  "$SMITH_DST/outputs/2026-09-06/22-51-43"
```

## 10. 生成归档校验和并传输

所有包生成后：

```bash
cd "$SMITH_BUNDLE"

find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > SHA256SUMS

du -sh .
cat SHA256SUMS
```

集群之间网络互通时推荐 rsync，支持断点续传：

```bash
rsync -ah \
  --partial \
  --append-verify \
  --info=progress2 \
  "$SMITH_BUNDLE/" \
  user@target-cluster:/data/scenesmith_migration_bundle/
```

通过移动硬盘或有单文件大小限制的中转系统传输时，可以切片：

```bash
cd "$(dirname "$SMITH_BUNDLE")"
tar -cf - "$(basename "$SMITH_BUNDLE")" \
  | split -b 20G - scenesmith-migration.tar.part-

sha256sum scenesmith-migration.tar.part-* \
  > scenesmith-migration.parts.sha256
```

目标端重组时：

```bash
sha256sum -c scenesmith-migration.parts.sha256
cat scenesmith-migration.tar.part-* | tar -xf -
```

## 11. 目标集群：解包

先设置目标路径。缓存不必放在 `/root/.cache`，推荐放到用户有写权限的高速共享盘：

```bash
SMITH_BUNDLE_DST=/data/scenesmith_migration_bundle
SMITH_DST=/data/users/yourname/scenesmith
SMITH_ENV_DST=/data/users/yourname/envs/scenesmith
SMITH_CUDA_DST=/data/users/yourname/cuda-12.4
SMITH_CACHE_DST=/data/users/yourname/scenesmith-cache
```

### 11.1 校验归档

```bash
cd "$SMITH_BUNDLE_DST"
sha256sum -c SHA256SUMS
```

出现任何 `FAILED` 都应重新传输对应文件，不要继续解包。

### 11.2 代码、数据和 checkpoint

代码包包含顶层目录 `scenesmith`，解到目标父目录：

```bash
mkdir -p "$(dirname "$SMITH_DST")"
tar --no-same-owner \
  -C "$(dirname "$SMITH_DST")" \
  -xf "$SMITH_BUNDLE_DST/scenesmith-code.tar"

tar --no-same-owner \
  -C "$SMITH_DST" \
  -xf "$SMITH_BUNDLE_DST/model-checkpoints.tar"

tar --no-same-owner \
  -C "$SMITH_DST" \
  -xf "$SMITH_BUNDLE_DST/scenesmith-data.tar"
```

如果目标目录名不是 `scenesmith`，先解压到临时父目录再移动整个目录；不要把归档内容覆盖到一个不相关的
已有仓库上。

### 11.3 Conda 环境

对 conda-pack 包：

```bash
mkdir -p "$SMITH_ENV_DST"
tar --no-same-owner \
  -xzf "$SMITH_BUNDLE_DST/scenesmith-env.tar.gz" \
  -C "$SMITH_ENV_DST"

"$SMITH_ENV_DST/bin/conda-unpack"
```

`conda-unpack` 应只在环境放到最终路径后执行。执行后不要再次移动这个环境目录。

重新安装两个 editable 包，修复旧仓库的绝对路径：

```bash
"$SMITH_ENV_DST/bin/python" -m pip install \
  -e "$SMITH_DST" --no-deps

"$SMITH_ENV_DST/bin/python" -m pip install \
  -e "$SMITH_DST/external/SAM3" --no-deps
```

### 11.4 CUDA 工具链

归档中包含顶层 `cuda-12.4`：

```bash
mkdir -p "$(dirname "$SMITH_CUDA_DST")"
tar --no-same-owner \
  -xJf "$SMITH_BUNDLE_DST/cuda-12.4.tar.xz" \
  -C "$(dirname "$SMITH_CUDA_DST")"
```

### 11.5 模型缓存

```bash
mkdir -p "$SMITH_CACHE_DST"
tar --no-same-owner \
  -xf "$SMITH_BUNDLE_DST/model-cache.tar" \
  -C "$SMITH_CACHE_DST"
```

检查 DINOv2 在目标端是实体目录，不是仍指向源机器 `/tmp` 的链接：

```bash
test -f "$SMITH_CACHE_DST/torch/hub/facebookresearch_dinov2_main/hubconf.py"
test ! -L "$SMITH_CACHE_DST/torch/hub/facebookresearch_dinov2_main"
```

## 12. 目标集群：环境变量

下面这些可以写入目标用户的 shell 初始化文件或 job 脚本。路径按目标集群修改：

```bash
export SMITH_DST=/data/users/yourname/scenesmith
export SMITH_ENV_DST=/data/users/yourname/envs/scenesmith
export CUDA_HOME=/data/users/yourname/cuda-12.4
export HF_HOME=/data/users/yourname/scenesmith-cache/huggingface
export TORCH_HOME=/data/users/yourname/scenesmith-cache/torch

export PATH="$CUDA_HOME/bin:$SMITH_ENV_DST/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$SMITH_ENV_DST/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

# 防止 localhost 请求误走集群代理。
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
```

完全离线运行时可以再加：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

API 凭据不要放进迁移 tar 包。应在目标机器重新配置：

```bash
export OPENAI_API_KEY="目标集群上的密钥"
export OPENAI_BASE_URL="目标集群可访问的中转站/v1"
export OPENAI_TRACING_KEY="可选的 tracing 密钥"
```

如果目标集群仍需从 Hugging Face 下载缺失文件，再单独设置 `HF_TOKEN`；使用完整缓存且
`HF_HUB_OFFLINE=1` 时不需要携带源机器 token。

## 13. 目标集群：逐级验收

不要直接开始几个小时的完整场景生成。按以下顺序检查，能更快定位缺的是环境、数据、缓存还是服务。

### 13.1 文件完整性

```bash
cd "$SMITH_DST"

test -s external/checkpoints/sam3.pt
test -s external/checkpoints/pipeline.yaml
test -s external/checkpoints/ss_generator.ckpt
test -s external/checkpoints/slat_generator.ckpt
test -d data/artvip_sdf/embeddings
test -d data/materials/embeddings

du -sh external/checkpoints data "$HF_HOME/hub" "$TORCH_HOME/hub"
```

期望数量级：

```text
external/checkpoints   约 16 GB
data                   约 54 GB
HF hub                 约 4.9 GB
Torch hub              约 1.5 GB，加约 7 MB DINOv2 源码
```

### 13.2 Python 和 CUDA

```bash
"$SMITH_ENV_DST/bin/python" -c "
import torch
print('torch:', torch.__version__)
print('torch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
"

"$SMITH_ENV_DST/bin/python" -c "
import bpy
import flask
import gsplat
import kaolin
import moge
import nvdiffrast
import open_clip
import pytorch3d
import sam3
import scenesmith
print('核心 import 全部通过')
"
```

如果 `torch.cuda.is_available()` 为 false，先查调度器是否真的分配了 GPU，再查驱动和容器映射；不要立即
重装 PyTorch。

### 13.3 离线模型缓存

```bash
HF_HUB_OFFLINE=1 "$SMITH_ENV_DST/bin/python" -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('Ruicheng/moge-vitl', 'model.pt', local_files_only=True))
print(hf_hub_download(
    'apple/DFN5B-CLIP-ViT-H-14-378',
    'open_clip_pytorch_model.bin',
    local_files_only=True,
))
"

test -s "$TORCH_HOME/hub/checkpoints/dinov2_vitb14_pretrain.pth"
test -s "$TORCH_HOME/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
test -s "$TORCH_HOME/hub/facebookresearch_dinov2_main/hubconf.py"
```

### 13.4 当前稳定性修改的单元测试

```bash
cd "$SMITH_DST"

"$SMITH_ENV_DST/bin/python" -m pytest \
  tests/unit/test_compat.py \
  tests/unit/test_worker_pool.py \
  tests/unit/test_geometry_generation_server_manager.py \
  tests/unit/test_retrieval_server_manager.py \
  -v
```

当前代码基线预期为 `41 passed`。`PytestConfigWarning` 或 `pynvml` deprecation warning 不影响该结果；
如果集群安全沙箱禁止 Unix domain socket，worker protocol 测试可能以
`PermissionError: Operation not permitted` 假失败，需要在实际计算节点/正常容器内重跑。

检查 geometry server 的父进程不会因为导入而提前初始化 PyTorch/CUDA：

```bash
"$SMITH_ENV_DST/bin/python" -c "
from scenesmith.agent_utils.geometry_generation_server import GeometryGenerationServer
import sys
print('torch imported?', any(m.startswith('torch') for m in sys.modules))
"
```

期望：

```text
torch imported? False
```

### 13.5 端口和短任务

启动前：

```bash
fuser -v 7005/tcp 7006/tcp 7007/tcp 7008/tcp 7009/tcp
```

先跑单房间或只跑到 `floor_plan`，确认 API、Blender、材质检索和 server 生命周期正常；再运行完整家具和
多房间任务。短任务正常退出后，再执行一次 `fuser`，应没有残留端口。

## 14. 迁移后的推荐启动方式

在登录节点手工运行长任务时，使用绝对解释器路径并脱离控制终端：

```bash
cd "$SMITH_DST"
mkdir -p run_logs

RUN_LOG="run_logs/migrated_run_$(date +%Y-%m-%d_%H-%M-%S).log"

nohup setsid -f env PYTHONUNBUFFERED=1 \
  "$SMITH_ENV_DST/bin/python" \
  main.py \
  +name=migrated_run \
  experiment.num_workers=1 \
  experiment.pipeline.parallel_rooms=false \
  </dev/null >"$RUN_LOG" 2>&1 &

echo "启动日志: $RUN_LOG"
```

在 Slurm/PBS 集群上优先使用调度器的 batch job，而不是在登录节点 `nohup`。job 脚本必须显式设置第
12 节的 `CUDA_HOME`、`HF_HOME`、`TORCH_HOME`、API 变量和工作目录。多次提交相同任务前，确认调度器
不会把它们安排到同一节点并争用 7005–7009。

## 15. 常见迁移故障

| 症状 | 最可能原因 | 处理方式 |
|---|---|---|
| `conda-unpack: No such file` | 普通 tar 了环境，并非 conda-pack 包 | 回源机器用 conda-pack 重新打包，或解到完全相同绝对路径 |
| import 仍指向旧仓库 | editable `.pth` 保留旧绝对路径 | 在目标环境重新 `pip install -e` SceneSmith 和 `external/SAM3` |
| MoGe 尝试联网 | 缺 `Ruicheng/moge-vitl` cache，或 `HF_HOME` 指错 | 检查 HF cache 层级和 `model.pt` 的相对软链接 |
| OpenCLIP 尝试联网 | 缺 DFN5B cache | 搬完整 `models--apple--DFN5B-CLIP-ViT-H-14-378`，不能只搬 snapshot |
| Torch Hub 尝试下载 DINOv2 仓库 | 迁移了 `/tmp` 绝对软链接而非源码 | 用 `cp -aL` 实体化为 `$TORCH_HOME/hub/facebookresearch_dinov2_main` |
| DINOv2 权重重新下载 | `TORCH_HOME` 未设置或 checkpoint 文件名不一致 | 检查 `$TORCH_HOME/hub/checkpoints/*.pth` |
| `no kernel image is available` | CUDA 扩展为另一种 GPU 架构编译 | 按目标 capability 重编四个 CUDA 包 |
| `GLIBC_x.y not found` | 环境从较新系统搬到较旧系统 | 使用相同/更新基础镜像，或在目标机重建环境 |
| `Address already in use` | 同节点重复启动或旧 server 未退出 | 查 7005–7009；一个节点只跑一个默认端口任务 |
| 本地 health check 超时但端口存在 | localhost 请求经过代理 | 设置 `NO_PROXY/no_proxy`，并确认迁移了当前代理绕过代码 |
| 429 连续失败 | 丢失 `_compat.py` 或服务端限流更严格 | 确认当前工作树已迁移；按中转站窗口调整 65 秒 |
| worker 重启报 CUDA fork 错误 | 迁移成了上游原版 `worker_pool.py` | 确认 `worker_entry.py`、`worker_protocol.py` 和 socket/exec 改动存在 |
| 退出后端口仍占用 | 丢失 managed Werkzeug shutdown 修改 | 确认各 retrieval/geometry server manager 是当前版本 |

## 16. 最终迁移检查单

迁移完成前逐项确认：

- [ ] 当前完整工作树已迁移，不只是上游 Git commit。
- [ ] `scenesmith/_compat.py`、`worker_entry.py`、`worker_protocol.py` 存在。
- [ ] Conda 环境通过 conda-pack 重定位，两个 editable 包已重装。
- [ ] `external/checkpoints` 约 16 GB 且 `pipeline.yaml` 引用的文件齐全。
- [ ] `data/artvip_sdf` 和 `data/materials` 已完整迁移。
- [ ] MoGe 和 DFN5B 两个 Hugging Face 模型 cache 已迁移。
- [ ] DINOv2 两个 `.pth` 已迁移。
- [ ] DINOv2 Torch Hub 源码是实体目录，不是指向 `/tmp/dinov2` 的坏链接。
- [ ] `HF_HOME`、`TORCH_HOME`、`CUDA_HOME` 指向目标路径。
- [ ] API key/token 没有进入迁移包，而是在目标集群重新注入。
- [ ] 目标 GPU 架构兼容；不兼容时已重编 CUDA 扩展。
- [ ] 相关单元测试通过，父进程导入检查显示 `torch imported? False`。
- [ ] 7005–7009 启动前为空，短任务退出后也全部释放。
- [ ] 如需续跑，迁移的是完整时间戳实验目录，而非孤立的 `scene_states`。
