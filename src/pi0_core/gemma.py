import dataclasses
from typing import Literal

# 仅保留常量
PALIGEMMA_VOCAB_SIZE = 257_152

@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    # 删除了 lora_configs 字段，因为 PyTorch 使用 peft 库在外部控制 LoRA，
    # 不需要在这个配置文件里读取 JAX 的 LoRA 配置。

Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]

def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    
    # 合并了 normal 和 lora 版本，因为基础架构是一样的
    if variant == "gemma_300m" or variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
        
    if variant == "gemma_2b" or variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
        
    raise ValueError(f"Unknown variant: {variant}")