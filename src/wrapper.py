import torch
import torch.nn as nn
import torch.nn.functional as F

# 引入你的两个核心模块
# 注意：这里假设你在 scripts/train.py 中运行，所以是 src.xxx
from src.recon_core.denoiser_dit import ReconDenoiser as Denoiser
# 如果 diffusion_utils 比较复杂，我们也可以在 wrapper 里写一个简单的加噪逻辑，
# 或者调用 src.recon_core.diffusion_utils.gaussian_diffusion 中的函数
# 这里为了通用性，我手写了一个简易的加噪函数，你可以替换为 reconVLA 原版的调用

class Pi0ReconWrapper(nn.Module):
    def __init__(
        self, 
        pi0_model, 
        recon_input_dim=64, # 比如 SigLIP 的输出维度
        recon_hidden_dim=32, 
        recon_depth=4
    ):
        """
        Args:
            pi0_model: 已经被加载（甚至已经加上 LoRA）的 Pi0 基础模型
            recon_input_dim: Pi0 视觉编码器输出的特征维度
        """
        super().__init__()
        self.policy = pi0_model
        
        # --- Recon 分支 (Denoiser) ---
        # 这是一个简单的 Transformer 或 MLP，用于根据噪声特征预测原特征
        # 我们这里实例化你搬运过来的 Denoiser
        self.recon_head = Denoiser(
            x_channel=recon_input_dim,
            z_channel=recon_input_dim,
            embed_dim=recon_hidden_dim,
            depth=recon_depth,
            
        )
        
        # 如果维度不匹配，加一个投影层
        self.proj = nn.Identity()
        # self.proj = nn.Linear(pi0_vision_dim, recon_input_dim) 

    def forward(self, images, text, actions=None):
        """
        Forward pass logic:
        1. Pi0 跑一遍 -> 拿到 Action Loss 和 视觉特征 (Features)
        2. Recon 跑一遍 -> 对 Features 加噪 -> 预测 -> 算 Recon Loss
        3. 加权求和
        """
        outputs = {}
        
        # =================================================
        # 1. Pi0 Forward (Action Path)
        # =================================================
        # 关键点：我们需要 pi0_model 返回的不仅仅是 loss，还有 features
        # 请确保你修改了 pi0_pytorch.py 的 forward 函数
        policy_out = self.policy(images, text, actions)
        
        # 兼容性处理：如果 policy_out 是字典
        if isinstance(policy_out, dict):
            action_loss = policy_out.get('loss', 0)
            # 必须拿到这一层！通常是 Vision Encoder 的输出
            visual_features = policy_out.get('visual_features') 
        else:
            # 如果原模型只返回 loss，这里会报错，必须去改 pi0_pytorch.py
            raise ValueError("Pi0 model output must be a dict containing 'visual_features'")

        outputs['action_loss'] = action_loss

        # =================================================
        # 2. Recon Forward (Auxiliary Task)
        # =================================================
        # 只有训练时才计算 Recon Loss
        if self.training and visual_features is not None:
            # a. Detach Features? 
            # ReconVLA 论文核心：不要 Detach！让 Recon 的梯度回传去优化 Vision Encoder
            target = visual_features # [B, Seq, Dim]
            
            # b. 准备 Timestep (0-999)
            B = target.shape[0]
            t = torch.randint(0, 1000, (B,), device=target.device).long()
            
            # c. 加噪 (Add Noise)
            noise = torch.randn_like(target)
            noisy_features = self.q_sample(target, t, noise)
            
            # d. 去噪 (Denoise Prediction)
            # 传入 noisy_features 和 timestep。
            # 有些 Denoiser 还需要 text embedding 做 condition，视你的 denoiser_vit 实现而定
            text_embeds = policy_out.get('text_embeds')
            pred_noise = self.recon_head(noisy_features, t, context=text_embeds)
            
            # e. 计算 Loss (MSE)
            # 预测噪声 vs 真实噪声
            recon_loss = F.mse_loss(pred_noise, noise)
            outputs['recon_loss'] = recon_loss
            
            # f. 合并 Loss (Total Loss)
            # 0.5 是 Recon 的权重，可调
            outputs['loss'] = action_loss + 0.5 * recon_loss
        else:
            outputs['loss'] = action_loss

        return outputs

    def q_sample(self, x_start, t, noise=None):
        """
        简易的 Diffusion 加噪过程 (Linear Schedule)
        x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 简单的线性 beta schedule，实际可以用 src/recon_core/diffusion_utils 里的
        beta_min, beta_max = 0.0001, 0.02
        betas = torch.linspace(beta_min, beta_max, 1000, device=x_start.device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # 提取当前 t 的系数
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod[t])
        
        # 广播维度 [B] -> [B, 1, 1] 以匹配特征
        while len(sqrt_alphas_cumprod.shape) < len(x_start.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)
            
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise