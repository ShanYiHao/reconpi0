import torch
from torch.utils.data import Dataset

class RobotDataset(Dataset):
    def __init__(self, length=100):
        self.length = length
        # 真实场景：这里加载 hdf5 或 jsonList

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 伪造 Aloha 的三个相机
        keys = ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
        images = {k: torch.randn(3, 224, 224) for k in keys}
        image_masks = {k: torch.tensor(True) for k in keys}
        # --- 1. 准备图像 ---
        # 假设单目相机，如果是多目，字典里加 keys
        image = torch.randn(3, 224, 224) 
        
        # --- 2. 准备指令 ---
        # 真实场景：调用 tokenizer
        # 这里假设已经 tokenize 好了
        instruction = torch.randint(0, 2000, (32,)) # 32 token length
        instruction_mask = torch.ones(32, dtype=torch.bool)

        # --- 3. 准备机械臂状态和动作 ---
        state = torch.randn(14) # 假设 14 维状态
        actions = torch.randn(10, 14) # [Horizon, Action_Dim]

        # --- 4. 封装成 OpenPi 需要的 Observation 对象 ---
        # 这一点很重要，必须匹配 PI0Pytorch 的输入签名
        # 由于 PyTorch Dataset 只能返回 tensor/dict，我们通常返回 dict，
        # 然后在 collate_fn 里转成对象，或者直接让 wrapper 接受 dict
        
        return {
            "observation": {
                "images": images,
                "image_masks": image_masks,
                "tokenized_prompt": instruction,
                "tokenized_prompt_mask": instruction_mask,
                "state": state
            },
            "actions": actions
        }

# Collate Function (批处理)
def collate_fn(batch):
    # 简单堆叠
    images_list = [b['observation']['images']['cam_high'] for b in batch]
    actions_list = [b['actions'] for b in batch]
    
    # ... 需要更复杂的逻辑把 dict of tensors 变成 tensor of dicts
    # 既然你是初学，建议在 wrapper 里处理，这里先尽量简单
    # 为了不把事情搞复杂，我们假设 batch_size=1 先跑通
    return batch[0]