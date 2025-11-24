import sys
import os
# 把当前目录加入 path 方便 import src
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

# 引入我们自己写的模块
from src.pi0_core.pi0_pytorch import PI0Pytorch
from src.wrapper import Pi0ReconWrapper
from src.dataset import RobotDataset

# 引入配置类 (如果 openpi 有现成的 config 加载器最好，没有就 mock)
class TrainingConfig:
    # 必须指向你下载好的 PI0 checkpoint 路径
    # 如果没有，模型就是随机初始化的（什么都学不到，但能跑）
    pretrained_path = "path/to/pi0_pytorch_checkpoint" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2 # 显存小就设 1
    lr = 1e-4
    steps = 100

def main():
    print(">>> 初始化配置...")
    cfg = TrainingConfig()
    
    # 1. 加载 PI0 基础模型
    # 注意：这里你需要实例化真正的 Config 对象
    # real_config = ... load from yaml ...
    # base_model = PI0Pytorch(real_config)
    print("警告：需确保 PI0Pytorch 能够正确加载真实权重")
    # base_model = ... (此处代码需结合你真实的权重加载逻辑)
    
    # --- 模拟代码 start ---
    from dataclasses import dataclass
    @dataclass
    class MockConfig:
        pi05: bool = False
        paligemma_variant: str = "gemma_2b"
        action_expert_variant: str = "gemma_2b"
        dtype: str = "float32"
        action_horizon: int = 10
        action_dim: int = 14
    base_model = PI0Pytorch(MockConfig())
    # --- 模拟代码 end ---

    base_model.to(cfg.device)

    # 2. 注入 LoRA (关键步骤)
    print(">>> 注入 LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # 针对 Gemma/PaliGemma
        bias="none",
        task_type="CAUSAL_LM" # 或者不写
    )
    # 这一步会冻结 base_model 的大部分参数，只留 LoRA
    base_model.paligemma_with_expert = get_peft_model(base_model.paligemma_with_expert, peft_config)
    base_model.paligemma_with_expert.print_trainable_parameters()

    # 3. 包装 Recon Head
    print(">>> 挂载 Recon Head...")
    model = Pi0ReconWrapper(
        pi0_model=base_model,
        recon_input_dim=2048, 
        recon_hidden_dim=1024
    ).to(cfg.device)

    # 4. 准备优化器
    # 分组优化：LoRA 参数和 Recon 参数都要更新
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.policy.parameters() if p.requires_grad], 'lr': cfg.lr},
        {'params': model.recon_head.parameters(), 'lr': cfg.lr * 2} # Recon 稍微大点
    ])

    # 5. 准备数据
    dataset = RobotDataset(length=1000)
    # 注意：collate_fn 需要根据你的 observation 结构细写，这里为了演示略过
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # 6. 训练循环
    print(">>> 开始训练...")
    model.train()
    for step, batch in enumerate(dataloader):
        # 将数据移到 GPU
        # obs = batch['observation'].to(device)... (需手动处理字典里每个 tensor)
        
        # Forward (Mock 运行)
        # outputs = model(obs, batch['actions'])
        
        # loss = outputs['loss']
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss = ...") # loss.item()

        if step >= cfg.steps:
            break

    print(">>> 训练完成，保存模型...")
    # model.policy.save_pretrained("output/lora_adapters")
    # torch.save(model.recon_head.state_dict(), "output/recon_head.pt")

if __name__ == "__main__":
    main()