import torch
import torch_musa
import torch.nn as nn
from lib00_tokenize import Llama2Tokenizer
from lib06_integrate import Transformer
from lib00b_read_data import JSONLDataset, create_dataloaders
from tqdm import tqdm
import os

device = torch.device("musa")

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(tqdm(dataloader)):
        src, tgt = src.to(device), tgt.to(device)
        
        # 准备decoder输入(移除最后一个token)和目标(移除第一个token)
        decoder_input = tgt[:, :-1]
        target = tgt[:, 1:]
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src, decoder_input)
        
        # 计算损失
        loss = criterion(
            output.contiguous().view(-1, output.size(-1)),
            target.contiguous().view(-1)
        )
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            
            decoder_input = tgt[:, :-1]
            target = tgt[:, 1:]
            
            output = model(src, decoder_input)
            
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                target.contiguous().view(-1)
            )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    # 保存最新检查点
    torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pt')
    
    # 如果是最佳模型，额外保存一份
    if is_best:
        torch.save(checkpoint, 'checkpoints/best_model.pt')

if __name__ == "__main__":
    # 初始化tokenizer
    tokenizer = Llama2Tokenizer()
    tokenizer.load("tokenizer.json")
    
    # 加载数据
    dataset = JSONLDataset('webtext.valid.jsonl', tokenizer, max_len=100, max_samples=1000)
    train_loader, val_loader = create_dataloaders(dataset, batch_size=8)
    
    # 初始化或加载模型
    try:
        checkpoint = torch.load('latest_checkpoint.pt')
        saved_vocab_size = checkpoint['model_state_dict']['encoder.embedding.tok_emb.weight'].shape[0]
        
        model = Transformer(
            src_pad_idx=tokenizer.pad_token_id,
            trg_pad_idx=tokenizer.pad_token_id,
            enc_voc_size=saved_vocab_size,
            dec_voc_size=saved_vocab_size,
            d_model=512,
            max_len=100,
            n_head=8,
            ffn_hidden=2048,
            n_layer=6,
            drop_prob=0.1,
            device=device
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"从epoch {start_epoch} 继续训练")
    except FileNotFoundError:
        print("从头开始训练新模型")
        model = Transformer(
            src_pad_idx=tokenizer.pad_token_id,
            trg_pad_idx=tokenizer.pad_token_id,
            enc_voc_size=len(tokenizer),
            dec_voc_size=len(tokenizer),
            d_model=512,
            max_len=100,
            n_head=8,
            ffn_hidden=2048,
            n_layer=6,
            drop_prob=0.1,
            device=device
        ).to(device)
        start_epoch = 0

    # 定义训练参数
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    if start_epoch > 0:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 创建保存检查点的目录
    os.makedirs('checkpoints', exist_ok=True)

    # 训练循环
    num_epochs = 5
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*50}")
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # 训练阶段
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        # 验证阶段
        val_loss = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
        
        # 保存检查点
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint(epoch, model, optimizer, train_loss, val_loss, is_best)

    print("\n训练完成!")