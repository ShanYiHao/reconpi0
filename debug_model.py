import torch
from dataclasses import dataclass
from src.pi0_core.pi0_pytorch import PI0Pytorch
from src.wrapper import Pi0ReconWrapper

# 1. 伪造一个 Config (模拟 OpenPi 的配置)
@dataclass
class MockConfig:
    pi05: bool = False
    paligemma_variant: str = "dummy" # 或者根据实际情况
    action_expert_variant: str = "dummy"
    dtype: str = "float32"
    action_horizon: int = 10
    action_dim: int = 14 # 假设 14 个电机

# 2. 伪造输入数据 (Mock Observation)
@dataclass
class MockObservation:
    images: dict
    image_masks: dict
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor
    token_loss_mask: torch.Tensor
    state: torch.Tensor

def test_forward():
    print(">>> 1. 初始化模型架构...")
    # ... 初始化模型的代码保持不变 ...
    # 记得把这里改成 dummy 配置
    config = MockConfig(paligemma_variant="dummy", action_expert_variant="dummy") 
    base_model = PI0Pytorch(config)
    
    # 注意：Wrapper 的 input_dim 要改成 64 (因为是 dummy 模型)
    model = Pi0ReconWrapper(
        pi0_model=base_model,
        recon_input_dim=64,   # <--- 必须是 64
        recon_hidden_dim=32,
        recon_depth=2
    )
    model.train()

    print(">>> 2. 构造假数据...")
    bs = 2
    
    expected_keys = ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
    
    fake_images = {k: torch.randn(bs, 3, 224, 224) for k in expected_keys}
    fake_masks = {k: torch.ones(bs, dtype=torch.bool) for k in expected_keys}
    
    obs = MockObservation(
        images=fake_images,
        image_masks=fake_masks,
        tokenized_prompt=torch.randint(0, 1000, (bs, 16)),
        tokenized_prompt_mask=torch.ones(bs, 16, dtype=torch.bool),
        token_ar_mask=torch.zeros(bs, 16, dtype=torch.int32),
        token_loss_mask=torch.zeros(bs, 16, dtype=torch.int32),
        state=torch.randn(bs, 14)
    )
    actions = torch.randn(bs, 10, 14)
    print(">>> 3. 运行 Forward...")
    # 这一步会触发 wrapper -> pi0 -> embed_prefix -> 返回 visual_features -> recon
    outputs = model(obs, actions)
    
    print(">>> 4. 检查输出...")
    print(f"Keys: {outputs.keys()}")
    if 'action_loss' in outputs and 'recon_loss' in outputs:
        print("✅ 成功！同时获得了 Action Loss 和 Recon Loss")
        print(f"Action Loss: {outputs['action_loss'].item()}")
        print(f"Recon Loss: {outputs['recon_loss'].item()}")
        print(f"Total Loss: {outputs['loss'].item()}")
    else:
        print("❌ 失败！缺少 Loss 项")

if __name__ == "__main__":
    try:
        test_forward()
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()