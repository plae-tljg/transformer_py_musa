import json
import torch
import torch_musa
from torch.utils.data import Dataset
from typing import List, Tuple

class JSONLDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_len: int = 100, max_samples: int = 1000):
        self.data: List[Tuple[str, str]] = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 读取JSONL文件
        print("正在加载JSONL数据...")
        sample_count = 0  # 计数器
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if sample_count >= max_samples:  # 达到最大样本数就停止
                    break
                    
                item = json.loads(line)
                if item['ended']:  # 只使用完整的文本
                    text = item['text']
                    sentences = text.split('.')
                    mid_point = len(sentences) // 2
                    
                    src_text = '.'.join(sentences[:mid_point]) + '.'
                    tgt_text = '.'.join(sentences[mid_point:])
                    
                    if src_text and tgt_text:  # 确保两部分都不为空
                        self.data.append((src_text, tgt_text))
                        sample_count += 1  # 增加计数器
        
        print(f"成功加载 {len(self.data)} 条训练数据 (最大限制: {max_samples})")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_text, tgt_text = self.data[idx]
        
        # 使用 prepare_for_model 替代 encode
        src_tokens = self.tokenizer.prepare_for_model(
            src_text,
            max_length=self.max_len
        )
        
        # 对于目标文本也使用相同的方法
        tgt_tokens = self.tokenizer.prepare_for_model(
            tgt_text,
            max_length=self.max_len
        )
        
        return (
            torch.tensor(src_tokens, dtype=torch.long),
            torch.tensor(tgt_tokens, dtype=torch.long)
        )

def create_dataloaders(
    dataset: JSONLDataset,
    batch_size: int = 16,
    train_split: float = 0.9
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """创建训练集和验证集的数据加载器"""
    
    # 计算分割点
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # 随机分割数据集
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader
