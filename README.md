# Reconpi-VLA: 基于辅助重建任务的 Pi0 高效微调框架

## 1\. 项目概述与工作内容 (Project Summary)

本项目基于 **Physical Intelligence $\pi_0$ (PyTorch版)** 模型架构，融合了 **ReconVLA** 的辅助重建思想，构建了一套支持联合训练（Joint Training）的轻量级微调框架。

**主要工作内容包括：**

1.  **架构缝合**：将 ReconVLA 中的 DiT（Diffusion Transformer）去噪头成功移植并集成到 Pi0 的 VLM 主干之后。
2.  **白盒化改造**：对 OpenPI 的源码进行了侵入式修改，打破了原有的封装，实现了中间层视觉特征（Visual Features）和文本特征（Text Embeddings）的截获与流转。
3.  **LoRA 高效微调**：使用 `peft` 库实现了 Parameter-Efficient Fine-Tuning，在冻结 2B+ 参数模型主干的同时，仅通过训练 Adapter 和 Recon Head 实现多任务优化。
4.  **多视角适配**：解决了 VLM 序列化输出与 DiT 图像化输入之间的维度冲突，支持 Aloha 等多摄像头机器人数据的并行处理。
5.  **环境解耦**：解决了 OpenPI 复杂的 JAX/Flax 依赖问题，实现了在纯 PyTorch 环境下的运行，并处理了 `transformers` 库的底层兼容性补丁。

-----

## 2\. 核心修改策略与数据流向 (Architecture & Data Flow)

### 2.1 整体架构逻辑

项目采用了 **"Backbone + Dual Heads"** 的设计模式：

  * **Backbone**: PaliGemma (SigLIP Vision Encoder + Gemma LLM)，负责提取多模态特征。
  * **Head A (Action)**: 原有的 Flow Matching Action Expert，负责生成机器人动作。
  * **Head B (Recon)**: 新增的 ReconDenoiser (DiT)，负责根据文本条件重建视觉特征。

### 2.2 关键数据流向修改 (Data Flow Modification)

这是本项目最核心的技术难点，主要体现在 `src/wrapper.py` 和 `src/pi0_core/pi0_pytorch.py` 中：

1.  **特征截获 (Feature Interception)**:

      * 修改了 `pi0_pytorch.py` 的 `forward` 函数。
      * 原逻辑仅返回 Loss，现逻辑返回字典：`{'loss': action_loss, 'visual_features': [B, N, D], 'text_embeds': [B, L, D]}`。
      * **目的**：让 Recon 分支能够获取到 Backbone 提取的高维特征，且**不阻断梯度回传**（No Detach），从而利用重建任务优化 Backbone 的表征能力。

2.  **维度重塑与对齐 (Reshape & Alignment)**:

      * **视觉特征**：Pi0 输出的是序列化特征 `[B, Total_Tokens, Dim]`（包含多视角拼接）。
          * *修改*：在 Wrapper 中将其重排为 `[B*Cams, Dim, H, W]`，以适配 DiT 对图像空间结构的输入要求。
      * **文本特征**：Pi0 输出的是序列 `[B, SeqLen, Dim]`。
          * *修改*：由于 DiT 使用 AdaLN 接收全局条件，我们对文本特征进行了 Pooling（平均池化），并扩展为 `[B*Cams, Dim, 1, 1]` 以进行广播。

3.  **Loss 融合 (Joint Optimization)**:

      * 最终 Loss 计算公式：$L_{total} = L_{action} + 0.5 \times L_{recon}$。

### 2.3 PEFT/LoRA 微调策略

为了在有限显存下训练大模型，我们并未全量更新参数，而是制定了精细的梯度更新策略：

  * **LoRA 配置 (`scripts/train.py`)**:
      * **Target Modules**: `["q_proj", "v_proj", "k_proj", "o_proj"]` (针对 Gemma 的 Attention 层)。
      * **Rank**: 16, **Alpha**: 32。
  * **参数冻结状态**:
      * 🟢 **训练**: LoRA Adapters, Recon Head (DiT, 全参)。
      * 🔒 **冻结**: SigLIP Vision Encoder, Gemma Backbone (除 LoRA 外), Action Head (可选冻结或微调)。

-----

## 3\. 仓库结构说明

```text
reconPI/
├── configs/                # 配置文件
├── scripts/
│   └── train.py            # 🚀 训练入口 (包含 LoRA 注入和 Optimizer 定义)
├── src/
│   ├── dataset.py          # 数据加载器 (需根据真实数据修改)
│   ├── wrapper.py          # 🧩 核心胶水层：处理多视角维度变换、Loss融合
│   ├── pi0_core/           # 🤖 魔改后的 Pi0 源码 (含动态维度、特征提取修改)
│   │   ├── pi0_pytorch.py  # 重点修改文件
│   │   └── ...
│   └── recon_core/         # 👁️ 移植的 ReconVLA 模块
├── debug_model.py          # ✅ 冒烟测试脚本 (用于验证架构连通性)
└── setup_env.sh            # 🛠️ 环境配置与补丁脚本
```

-----

## 4\. 服务器集成与训练指南 (For Integration)

### 4.1 环境部署

在服务器上，必须运行以下脚本以应用 OpenPI 对 `transformers` 库的底层修改：

```bash
# 1. 创建并激活环境
conda create -n pi0_recon python=3.11
conda activate pi0_recon

# 2. 运行一键配置脚本 (安装依赖 + 覆盖 transformers 源码)
bash setup_env.sh
```

### 4.2 数据集放置与接入

目前 `src/dataset.py` 使用的是 Mock 数据。接入真实数据请遵循以下规范：

1.  **数据格式**：推荐使用 LeRobot 标准或 Aloha HDF5 格式。
2.  **修改 `src/dataset.py`**:
      * **图像键名**：必须包含 `['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']`（与 OpenPI 预处理逻辑对齐）。
      * **预处理**：必须包含 `Resize(224)` 和 `Normalize(mean=0.5, std=0.5)`。
      * **文本**：使用 `google/paligemma-3b-pt-224` 的 Tokenizer 处理指令。

### 4.3 恢复“完全体”模型 (从 Dummy 切换到 Real)

本地调试时使用了 `dummy` 模式。在服务器正式训练前，需修改以下文件：

1.  **修改 `scripts/train.py`**:

      * 将 `TrainingConfig` 中的 `paligemma_variant` 改回 **`"gemma_2b"`**。
      * 设置 `pretrained_path` 为真实的 Pi0 `.pt` 权重路径。

2.  **修改 `src/wrapper.py` (视分辨率而定)**:

      * 如果输入图片是 224x224，保持 `n_patches=256` 不变。
      * 如果输入图片是 336x336，需改为 `n_patches=576`。

3.  **启动训练**:

    ```bash
    # 建议使用 tmux 后台运行
    python scripts/train.py
    ```
