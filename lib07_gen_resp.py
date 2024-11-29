import torch
import torch_musa
import torch.nn as nn
from lib06_integrate import Transformer
from lib00_tokenize import Llama2Tokenizer

device = torch.device("musa")

def generate_response(model, tokenizer, question, max_length, vocab, device, start_token_id, end_token_id):
    model.eval()  # Set the model to evaluation mode

    # Tokenize the question
    question_tokens = vocab.encode(question)
    input_ids = torch.LongTensor(question_tokens).unsqueeze(0).to(device)

    # Generate response using the model's generate method
    with torch.no_grad():
        generated_ids = model.generate(
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_length=max_length
        )

    # Decode the generated sequence
    generated_tokens = generated_ids.squeeze(0).tolist()
    response = vocab.decode(generated_tokens[1:])  # Remove start token
    return response

if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = Llama2Tokenizer()
    tokenizer.load("tokenizer.json")
    
    # Initialize model
    # 首先获取保存的模型词汇表大小
    try:
        checkpoint = torch.load('checkpoints/best_model.pt')
        saved_vocab_size = checkpoint['model_state_dict']['encoder.embedding.tok_emb.weight'].shape[0]
        
        # 使用保存的词汇表大小初始化模型
        model = Transformer(
            src_pad_idx=tokenizer.pad_token_id,
            trg_pad_idx=tokenizer.pad_token_id,
            enc_voc_size=saved_vocab_size,  # 使用保存的词汇表大小
            dec_voc_size=saved_vocab_size,  # 使用保存的词汇表大小
            d_model=512,
            max_len=100,
            n_head=8,
            ffn_hidden=2048,
            n_layer=6,
            drop_prob=0.1,
            device=device
        ).to(device)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型加载成功: 使用第5个epoch的检查点 (词汇表大小: {saved_vocab_size})")
    except FileNotFoundError:
        print("使用随机初始化的模型")
        # 如果找不到检查点，使用当前tokenizer的词汇表大小
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
    
    # Test generation
    # question = "你好，请问今天天气如何？"
    question = "what is love?"
    print(f"\n输入问题: {question}")
    
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        question=question,
        max_length=100,
        vocab=tokenizer,
        device=device,
        start_token_id=tokenizer.bos_token_id,
        end_token_id=tokenizer.eos_token_id
    )
    print(response)