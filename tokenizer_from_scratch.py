from typing import List, Dict, Tuple
import regex as re
from collections import defaultdict
import torch
import torch_musa
import torch.nn.functional as F

class Llama2Tokenizer:
    def __init__(self):
        print("\n=== 初始化阶段 ===")
        # 特殊标记
        self.special_tokens = {
            "<s>": 1,
            "": 2,
            "<unk>": 0,
            "<pad>": 3
        }
        
        # 基础词汇表
        self.vocab = {}
        self.vocab.update(self.special_tokens)
        
        # 合并规则
        self.merges = {}
        
        # 检查是否有可用的GPU
        self.device = torch.device('musa' if torch.musa.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
    
    def train(self, texts: List[str], vocab_size: int = 32000, min_frequency: int = 2):
        """训练分词器"""
        print("\n=== 训练阶段开始 ===")
        
        # 1. 获取所有单词及其频率
        word_freqs = defaultdict(int)
        for text in texts:
            # 使用空格分割单词，保持中文字符完整
            words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|[0-9]+|[^a-zA-Z0-9\s\u4e00-\u9fff]+', text)
            for word in words:
                word_freqs[word] += 1
        
        # 2. 初始化词汇表和单词分割
        splits = {}
        for word in word_freqs:
            if len(word) == 1:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
            else:
                splits[word] = list(word)
        
        vocab_size_current = len(self.vocab)
        print(f"当前词汇表大小: {vocab_size_current}")
        
        # 3. 执行BPE合并
        iteration = 1
        while vocab_size_current < vocab_size:
            print(f"\n--- BPE迭代 #{iteration} ---")
            
            # 统计所有可能的合并对
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                if len(splits.get(word, [])) <= 1:
                    continue
                word_splits = splits[word]
                for i in range(len(word_splits) - 1):
                    pair = (word_splits[i], word_splits[i + 1])
                    pairs[pair] += freq
            
            if not pairs:
                break
                
            # 找出最频繁的合并对
            best_pair, freq = max(pairs.items(), key=lambda x: x[1])
            if freq < min_frequency:
                break
                
            print(f"最佳合并对: {best_pair}, 频率: {freq}")
            
            # 执行合并
            new_token = ''.join(best_pair)
            self.merges[best_pair] = new_token
            self.vocab[new_token] = len(self.vocab)
            
            # 更新所有受影响的分割
            for word in word_freqs:
                if len(splits.get(word, [])) <= 1:
                    continue
                
                i = 0
                new_splits = []
                current_splits = splits[word]
                
                while i < len(current_splits):
                    if i < len(current_splits) - 1 and (current_splits[i], current_splits[i + 1]) == best_pair:
                        new_splits.append(new_token)
                        i += 2
                    else:
                        new_splits.append(current_splits[i])
                        i += 1
                
                splits[word] = new_splits
            
            vocab_size_current += 1
            print(f"当前词汇表大小: {vocab_size_current}")
            iteration += 1
        
        print("\n训练完成后的词汇表大小:", len(self.vocab))
        print("合并规则数量:", len(self.merges))
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """基础分词 - 直接分解成字符"""
        return list(text)
    
    def _get_pairs(self, token_sequences: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """获取所有序列中的相邻对频率"""
        pairs = defaultdict(int)
        for tokens in token_sequences:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                # 跳过包含中文字符的合并
                if any('\u4e00' <= c <= '\u9fff' for c in tokens[i] + tokens[i+1]):
                    continue
                pairs[pair] += 1
        return pairs
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为标记ID"""
        print(f"\n编码文本: {text}")
        tokens = self._basic_tokenize(text)
        
        # 持续应用合并规则直到无法继续合并
        while True:
            # 标记是否在本轮中发生了任何合并
            merged = False
            # 遍历所有可能的合并
            for pair, merged_token in self.merges.items():
                # 从头开始查找可合并的对
                i = 0
                while i < len(tokens) - 1:
                    current_pair = (tokens[i], tokens[i + 1])
                    # 跳过包含中文字符的合并
                    if any('\u4e00' <= c <= '\u9fff' for c in tokens[i] + tokens[i+1]):
                        i += 1
                        continue
                    # 如果找到可以合并的对，进行合并
                    if current_pair == pair:
                        tokens[i] = merged_token
                        tokens.pop(i + 1)
                        merged = True
                    else:
                        i += 1
            # 如果这一轮没有发生合并，说明已经无法继续合并，退出循环
            if not merged:
                break
        
        result = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        print("编码结果:", result)
        return result
    
    def decode(self, ids: List[int]) -> str:
        """将标记ID解码为文本"""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(id_, "") for id_ in ids]
        result = "".join(tokens)
        print("解码结果:", result)
        return result

    def save(self, path: str):
        """保存分词器"""
        print(f"\n=== 保存分词器到: {path} ===")
        import json
        data = {
            "vocab": self.vocab,
            "merges": {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()}
        }
        print("保存的数据:", data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """加载分词器"""
        print(f"\n=== 从{path}加载分词器 ===")
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("加载的数据:", data)
        self.vocab = data["vocab"]
        self.merges = {tuple(k.split("|")): v for k, v in data["merges"].items()}
        print("词汇表:", self.vocab)
        print("合并规则:", self.merges)

if __name__ == "__main__":
    tokenizer = Llama2Tokenizer()
    # 准备包含中英文的训练文本
    texts = [
        "Hello world! 你好世界！",
        "This is a test 这是一个测试",
        "Machine learning is amazing 机器学习很神奇",
        "I love programming 我喜欢编程",
        "Natural language processing 自然语言处理",
        "深度学习 Deep learning",
        "人工智能 Artificial Intelligence",
        "大语言模型 Large Language Model"
    ]
    
    print("\n=== 训练数据 ===")
    print("训练文本:", texts)
    
    # 训练分词器
    tokenizer.train(texts, vocab_size=500)
    
    # 测试编码和解码
    test_texts = [
        "Hello 你好",
        "AI 人工智能",
        "机器学习 Machine Learning"
    ]
    
    print("\n=== 测试阶段 ===")
    for text in test_texts:
        print(f"\n测试文本: {text}")
        encoded = tokenizer.encode(text)
        print(f"编码结果: {encoded}")
        decoded = tokenizer.decode(encoded)
        print(f"解码结果: {decoded}")
    tokenizer.save("tokenizer.json")

    # # 创建分词器实例
    # tokenizer = Llama2Tokenizer()

    # # 训练分词器
    # texts = [
    #     "Hello world!",
    #     "This is a test",
    #     "Machine learning is amazing"
    # ]
    # tokenizer.train(texts, vocab_size=100)

    # # 编码文本
    # encoded = tokenizer.encode("Hello world!")
    # print(encoded)

    # # 解码标记
    # decoded = tokenizer.decode(encoded)
    # print(decoded)

    # # 保存分词器
    # tokenizer.save("tokenizer.json")

    # # 加载分词器
    # new_tokenizer = Llama2Tokenizer()
    # new_tokenizer.load("tokenizer.json")