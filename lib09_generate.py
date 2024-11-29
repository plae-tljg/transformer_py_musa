import torch
import torch_musa
from typing import List, Optional, Iterator
from lib00_tokenize import Llama2Tokenizer
from lib06_integrate import Transformer

class StreamingGenerator:
    def __init__(
        self,
        model: Transformer,
        tokenizer: Llama2Tokenizer,
        device: torch.device,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
    
    def _nucleus_sampling(self, logits: torch.Tensor) -> int:
        # 应用温度
        logits = logits / self.temperature
        
        # 计算概率分布
        probs = torch.softmax(logits, dim=-1)
        
        # 按概率排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 应用 top-p (nucleus) 采样
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 将低于阈值的概率置为0
        sorted_probs[sorted_indices_to_remove] = 0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
        
        # 采样
        next_token = torch.multinomial(probs, num_samples=1).item()
        return next_token
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        self.model.eval()
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids]).to(self.device)
        
        # 初始化生成序列
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens or self.max_length):
                # 准备decoder输入
                decoder_input = torch.tensor([generated_tokens]).to(self.device)
                if len(decoder_input[0]) == 0:
                    decoder_input = input_ids.clone()
                
                # 获取模型输出
                output = self.model(input_ids, decoder_input)
                
                # 获取最后一个token的预测
                next_token_logits = output[0, -1, :]
                
                # 使用nucleus sampling选择下一个token
                next_token = self._nucleus_sampling(next_token_logits)
                
                # 如果生成了结束符，停止生成
                if next_token == self.tokenizer.eos_token_id:
                    break
                
                # 添加到生成序列
                generated_tokens.append(next_token)
                
                # 解码最新生成的token并yield
                new_text = self.tokenizer.decode([next_token])
                yield new_text
                
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        return "".join(self.generate_stream(prompt, max_new_tokens))

# 使用示例
if __name__ == "__main__":
    # 初始化设备
    device = torch.device("musa" if torch.cuda.is_available() else "cpu")
    
    # 加载tokenizer
    tokenizer = Llama2Tokenizer()
    tokenizer.load("tokenizer.json")
    
    # 加载模型
    checkpoint = torch.load('checkpoint_epoch_2.pt', map_location=device)
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建生成器
    generator = StreamingGenerator(model, tokenizer, device)
    
    # 测试流式生成
    # prompt = "今天天气真不错"
    prompt = "What is the meaning of life?"
    print(f"输入提示：{prompt}")
    print("生成的回复：", end="", flush=True)
    
    # 流式输出
    for token in generator.generate_stream(prompt):
        print(token, end="", flush=True)
    print()
    
    # 或者一次性生成
    response = generator.generate(prompt)
    print(f"\n完整回复：{response}") 