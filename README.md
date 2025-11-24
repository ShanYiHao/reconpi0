# Pi0-Recon-VLA: 基于辅助重建任务的 Pi0 高效微调框架

本仓库实现了一个针对 **$\pi_0$ (Pi0)** VLA 模型的轻量级微调框架。通过引入 **ReconVLA** (Reconstruction for Vision-Language-Action) 思想，在原有动作生成任务的基础上，增加了一个基于 DiT 的视觉重建辅助任务，利用 **LoRA (PEFT)** 技术实现低显存下的联合训练。

-----

## 1\. 科研工作与架构设计 (Research Methodology)

本项目旨在解决 Pi0 模型在下游任务微调中物理表征能力不足的问题。核心工作内容如下：

### 1.1 架构创新：双流联合训练 (Joint Training)

不同于原版 Pi0 仅依赖 Action Loss，本项目构建了一个 **Action-Recon 双流架构**：

  - **主干网络 (Backbone)**: 使用预训练的 PaliGemma (SigLIP + Gemma) 提取多模态特征。
  - **动作流 (Action Path)**: 特征流向原有的 Action Expert (Flow Matching) 生成动作。
  - **重建流 (Recon Path)**: **[核心工作]** 在 VLM 输出端截获 `visual_features` 和 `text_embeds`，输入到新增的 **ReconDenoiser (DiT)** 模块中，执行基于文本条件的图像重建任务。
  - **Loss 融合**: $L_{total} = L_{action} + \lambda \cdot L_{recon}$。通过重建 Loss 的梯度回传，迫使 VLM 学习更鲁棒的物理规律。

### 1.2 工程优化：LoRA 高效微调

鉴于全量微调的高显存需求，本项目采用了 **PEFT (Parameter-Efficient Fine-Tuning)** 策略：

  - **冻结 (Freeze)**: 冻结 Vision Encoder 和 LLM 的大部分参数。
  - **注入 (Inject)**: 在 Attention 层 (`q_proj`, `v_proj` 等) 注入秩为 $r=16$ 的 LoRA 适配器。
  - **全参训练**: 新增的 Recon Head (DiT) 保持全参数训练状态。

### 1.3 白盒化改造 (White-box Modification)

为了实现特征截获，对 OpenPI 源码进行了必要的**侵入式修改**：

  - 重写了 `pi0_pytorch.py` 的 `forward` 函数，使其支持输出中间层视觉特征。
  - 解决了 OpenPI 硬编码维度 (32-dim) 的问题，改为动态适配 `action_dim`。
  - 修复了多视角 (Multi-view) 数据在 DiT 输入时的维度广播问题。

-----

## 2\. 环境安装与配置 (Installation)

本项目对 `transformers` 库有特定版本依赖及补丁要求，请严格按照以下步骤操作。

### 2.1 基础环境

推荐使用 Linux 服务器环境 (CUDA 12.x)。

```bash
# 1. 创建环境
conda create -n pi0_recon python=3.11
conda activate pi0_recon

# 2. 安装 PyTorch (根据服务器 CUDA 版本调整)
pip install torch torchvision torchaudio

# 3. 安装项目依赖
pip install transformers==4.53.2 peft accelerate timm einops tqdm blobfile numpy pytest beartype jaxtyping jax jaxlib ml_collections wandb
```

### 2.2 应用 OpenPI 补丁 (关键步骤)

由于 OpenPI 修改了 SigLIP/PaliGemma 的底层实现，必须手动覆盖 `transformers` 库的源码。

**方法 A：使用一键脚本 (推荐)**

```bash
bash setup_env.sh
```

**方法 B：手动覆盖**

```bash
# 找到 transformers 安装路径
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
# 覆盖文件
cp -r src/pi0_core/transformers_replace/models/* "$SITE_PACKAGES/transformers/models/"
```

-----

## 3\. 项目结构说明 (Project Structure)

```text
pi0-recon-finetune/
├── configs/                # 存放训练配置文件
├── scripts/
│   └── train.py            # 🚀 [核心] 训练入口脚本 (集成 LoRA 与 Optimizer)
├── src/
│   ├── dataset.py          # 📋 [需修改] 数据加载器 (目前为 Mock 数据)
│   ├── wrapper.py          # 🧩 [核心] Pi0ReconWrapper，缝合 Pi0 与 Recon 模块
│   ├── pi0_core/           # 🤖 [已魔改] Pi0 模型定义 (请勿随意替换)
│   │   ├── pi0_pytorch.py  # 修改了 forward 以返回特征
│   │   └── ...
│   └── recon_core/         # 👁️ [移植] ReconVLA 的 DiT 去噪模块
│       └── denoiser_dit.py
├── debug_model.py          # ✅ 冒烟测试脚本 (用于验证架构连通性)
└── setup_env.sh            # 环境配置脚本
```

-----

## 4\. 快速上手 (Quick Start)

### 4.1 冒烟测试 (Sanity Check)

在开始训练前，运行此脚本确保模型架构正确、显存不溢出且 Loss 能够计算。
*(默认使用 Dummy 模式，不占用大量显存)*

```bash
python debug_model.py
```

**预期输出**：

> ✅ 成功！同时获得了 Action Loss 和 Recon Loss

### 4.2 启动训练

```bash
python scripts/train.py
```

-----

## 5\. 服务器集成与数据接入指南 (For Integration)

本章节用于指导如何将该框架应用到真实的机器人数据集上。

### 5.1 数据集接入 (Data Loading)

目前的 `src/dataset.py` 使用的是随机生成的假数据。在服务器上训练时，请按以下标准修改 `__getitem__`：

1.  **图像处理**:
      * 必须包含 OpenPI 默认视角的 Keys: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`。
      * **预处理**: 必须进行 Resize (224x224) 和 Normalize (Mean=0.5, Std=0.5)。
2.  **文本处理**:
      * 使用 `AutoTokenizer` (google/paligemma-3b-pt-224)。
      * 生成 `tokenized_prompt` 和对应的 Masks。
3.  **维度对齐**:
      * Action 维度需匹配机器人自由度 (例如 14)。

### 5.2 关键源文件修改点

在迁移到真实任务时，可能需要修改以下文件：

  * **`scripts/train.py`**:
      * 修改 `TrainingConfig` 类中的 `pretrained_path`，指向真实的 Pi0 `.pt` 权重文件。
      * 将 `paligemma_variant` 改回 `"gemma_2b"` (目前为了调试默认为 `"dummy"`)。
  * **`src/wrapper.py`**:
      * 如果在服务器上使用高分辨率输入 (336x336)，请将 `__init__` 中的 `n_patches` 改为 **576** (224x224 对应 256)。
  * **`src/pi0_core/pi0_pytorch.py`**:
      * 如果机器人 Action 维度发生变化 (非 14 或 7)，代码已做动态适配，只需在 Config 中修改 `action_dim` 即可。

### 5.3 显存优化建议

如果在服务器上遇到 OOM (Out of Memory)：

1.  在 `scripts/train.py` 中减小 `batch_size`。
2.  在 `src/wrapper.py` 中减小 Recon 分支的 `recon_hidden_dim` (默认 1024)。
3.  确保 `train.py` 中启用了 `gradient_checkpointing`。

-----

## 6\. 已知问题与维护

  * **Transformers 版本**: 严禁随意升级 `transformers`，否则 `src/pi0_core` 下的补丁将失效，导致 SigLIP 加载失败。
  * **JAX 依赖**: 虽然是 PyTorch 项目，但为了兼容 OpenPI 的 Config读取逻辑，环境必须安装 `jax` (CPU版即可)。