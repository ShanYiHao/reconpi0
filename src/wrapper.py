import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# 引入你的 Recon Denoiser
# 注意：这里假设类名是 ReconDenoiser，如果之前 grep 确认过是这个那就没问题
from src.recon_core.denoiser_dit import ReconDenoiser as Denoiser

class Pi0ReconWrapper(nn.Module):
    def __init__(
        self, 
        pi0_model, 
        recon_input_dim=2048, 
        recon_hidden_dim=1024, 
        recon_depth=4,
        n_patches=256 # 默认 patch 数量
    ):
        super().__init__()
        self.policy = pi0_model
        
        # --- Recon 分支 (Denoiser) ---
        # 初始化参数需与 ReconDenoiser.__init__ 匹配
        self.recon_head = Denoiser(
            x_channel=recon_input_dim,  # 输入图像特征维度
            z_channel=recon_input_dim,  # 条件特征维度 (文本)
            embed_dim=recon_hidden_dim, # 内部隐藏层维度
            depth=recon_depth,          # 网络深度
            n_patches=n_patches         # Patch 数量
        )

    def forward(self, images, text, actions=None):
        outputs = {}
        
        # =================================================
        # 1. Pi0 Forward (Action Path)
        # =================================================
        # 这一步会调用 pi0_pytorch.py 的 forward
        policy_out = self.policy(images, text, actions)
        
        if isinstance(policy_out, dict):
            action_loss = policy_out.get('loss', 0)
            visual_features = policy_out.get('visual_features') 
            text_embeds = policy_out.get('text_embeds') 
        else:
            # 防御性编程：如果 pi0 返回的不是字典
            raise ValueError(f"Pi0 model output expected dict, got {type(policy_out)}")

        outputs['action_loss'] = action_loss

        # =================================================
        # 2. Recon Forward (辅助任务)
        # =================================================
        # 只有在训练且成功提取到特征时才计算
        if self.training and visual_features is not None:
            # visual_features shape: [B, Total_Tokens, D]
            B, N, D = visual_features.shape
            
            # 计算 Patch 布局
            # 标准 SigLIP 224x224 -> 16x16 = 256 tokens per image
            tokens_per_img = 256 
            
            # 如果 N 不是 256 的倍数 (比如 Dummy 模式下 N 可能很小)
            # 我们做一个简单的兼容性处理，防止 math.sqrt 报错
            if N < tokens_per_img:
                # Dummy 模式兼容: 假设只有一张图，且是正方形
                tokens_per_img = N
                num_cams = 1
            else:
                num_cams = N // tokens_per_img
            
            side = int(math.sqrt(tokens_per_img))
            
            # 步骤 A: 维度重排
            # 将 [B, num_cams * H * W, D] -> [B * num_cams, D, H, W]
            # 我们把每个摄像头视角视为独立的样本进行重建
            spatial_features = rearrange(
                visual_features, 
                'b (k h w) c -> (b k) c h w', 
                k=num_cams, 
                h=side, 
                w=side
            )
            
# 步骤 B: 对齐文本条件
            if text_embeds is not None:
                # 1. [关键修复] 文本池化: [B, SeqLen, D] -> [B, D]
                # 我们取文本序列的平均值，得到一个全局的文本语义向量
                pooled_text = text_embeds.mean(dim=1) 
                
                # 2. 复制以匹配多视角: [B, D] -> [B*K, D]
                text_cond = repeat(pooled_text, 'b d -> (b k) d', k=num_cams)
                
                # 3. 伪装成 1x1 的图片传入 DiT: [B*K, D] -> [B*K, D, 1, 1]
                # 这样 DiT 内部展开后长度为 1，可以广播给任意长度的图像特征 (256)
                text_cond = rearrange(text_cond, 'b d -> b d 1 1')
            else:
                text_cond = None

            # 步骤 C: 调用 Denoiser
            recon_loss = self.recon_head(z=text_cond, target=spatial_features)
            
            if isinstance(recon_loss, dict):
                recon_loss = recon_loss['loss']
            
            # [关键修复] 1. 算出标量
            recon_loss_scalar = recon_loss.mean()
            action_loss_scalar = action_loss.mean()
            
            # [关键修复] 2. 更新 outputs 里的值为标量 (方便打印和日志)
            outputs['recon_loss'] = recon_loss_scalar
            outputs['action_loss'] = action_loss_scalar  # <--- 新增这行，覆盖原来的大 Tensor
            
            # 3. 合并 Loss
            outputs['loss'] = action_loss_scalar + 0.5 * recon_loss_scalar
            
        else:
            # [关键修复] 验证模式也要转标量
            outputs['action_loss'] = action_loss.mean() # <--- 新增这行
            outputs['loss'] = outputs['action_loss']

        return outputs