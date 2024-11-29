from typing import List, Dict, Tuple, Optional, Union
import regex as re
from collections import defaultdict
import json
import os

class Llama2Tokenizer:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initializes the Llama2 tokenizer.
        Args:
            model_path: Path to the pretrained tokenizer (optional).
        """
        # Llama2 specific special tokens
        self.special_tokens = {
            "<s>": 1,      # Beginning of sequence
            "/s": 2,     # End of sequence
            "<unk>": 0,    # Unknown token
            "<pad>": 3,    # Padding token
            "<BOS>": 1,    # Beginning of sentence (same as <s>)
            "<EOS>": 2,    # End of sentence (same as </s>)
            "[INST]": 4,   # Instruction start
            "[/INST]": 5,  # Instruction end
            "<<SYS>>": 6,  # System prompt start
            "<</SYS>>": 7, # System prompt end
            " ": 8,        # Space token - 添加空格作为特殊token
        }
        
        # 添加这些属性
        self.bos_token_id = self.special_tokens["<s>"]
        self.eos_token_id = self.special_tokens["/s"]
        self.pad_token_id = self.special_tokens["<pad>"]
        self.unk_token_id = self.special_tokens["<unk>"]
        self.space_token_id = self.special_tokens[" "]  # 添加空格token ID

        # 初始化词汇表,从最大的特殊token ID开始
        self.vocab = {}
        self.vocab.update(self.special_tokens)
        self.next_token_id = max(self.special_tokens.values()) + 1
        self.merges = {}

        # Load pretrained tokenizer if model path is provided
        if model_path:
            self.load(model_path)

    def _basic_tokenize(self, text: str) -> List[str]:
        """Performs basic tokenization by splitting into characters."""
        tokens = list(text)  # 将文本按字符分割
        return tokens

    def _get_pairs(self, token_sequences: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Counts the frequency of all adjacent token pairs."""
        pairs = defaultdict(int)
        for sequence in token_sequences:
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                pairs[pair] += 1
        print(pairs)
        return pairs

    def train(self, texts: List[str], vocab_size: int = 32000, min_frequency: int = 2, frequency_threshold: int = 2):
        """Trains the tokenizer."""
        print("\n=== Training Llama2 tokenizer ===")

        # Preprocessing and initial tokenization
        all_tokens = []
        for text in texts:
            tokens = self._basic_tokenize(text)
            print(f"DEBUG - Text: '{text}'")
            print(f"DEBUG - Tokens with spaces: {[f'[{t}]' if t == ' ' else t for t in tokens]}")
            all_tokens.append(tokens)

        # Build initial vocabulary
        char_freqs = defaultdict(int)
        for tokens in all_tokens:
            for token in tokens:
                char_freqs[token] += 1
        
        print(f"DEBUG - Space frequency in initial vocab: {char_freqs.get(' ', 0)}")
        for token, freq in char_freqs.items():
            if freq >= min_frequency and token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # BPE training process
        while len(self.vocab) < vocab_size:
            pairs = self._get_pairs(all_tokens)
            if not pairs:
                break

            # Select all pairs with frequency greater than the threshold
            frequent_pairs = {pair: freq for pair, freq in pairs.items() if freq >= frequency_threshold}
            print("frequent_pairs", frequent_pairs)
            if not frequent_pairs:
                break

            # Merge these pairs
            for best_pair in frequent_pairs:
                new_token = ''.join(best_pair)
                self.merges[best_pair] = new_token
                self.vocab[new_token] = len(self.vocab)

                # Update sequences
                for i in range(len(all_tokens)):
                    j = 0
                    while j < len(all_tokens[i]) - 1:
                        if (all_tokens[i][j], all_tokens[i][j + 1]) == best_pair:
                            all_tokens[i][j] = new_token
                            all_tokens[i].pop(j + 1)
                        else:
                            j += 1

            print("vocab", self.vocab)
        print(self.vocab)
        print(f"Training complete, vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """Encodes text into token IDs."""
        tokens = list(text)
        ids = []

        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(0)  # 使用ID 0 作为默认值
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs into text."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        # 直接连接所有tokens,包括特殊tokens
        tokens = []
        for id_ in ids:
            token = id_to_token.get(id_, ' ')
            tokens.append(token)
        
        return ''.join(tokens)


    def save(self, path: str):
        """Saves the tokenizer to a file."""
        # 修改保存格式以匹配期望的格式
        data = {
            "model": {
                "type": "BPE",
                "vocab_size": len(self.vocab),
                "special_tokens": self.special_tokens,
                "vocab": self.vocab,
                "merges": {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()}
            }
        }
        
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """Loads the tokenizer from a file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 保持特殊tokens不变,为新tokens分配新的ID
        model_vocab = data["model"]["vocab"]
        for token, _ in model_vocab.items():
            if token not in self.special_tokens:
                self.vocab[token] = self.next_token_id
                self.next_token_id += 1
                
        # 如果merges是列表,需要转换成字典格式
        if isinstance(data["model"]["merges"], list):
            # 假设merges列表中每个元素是 "a|b" 格式的字符串
            self.merges = {tuple(merge.split('|')): i 
                          for i, merge in enumerate(data["model"]["merges"])}
        else:
            # 原来的处理方式
            self.merges = {tuple(k.split('|')): v 
                          for k, v in data["model"]["merges"].items()}

    def prepare_for_model(self, text, max_length=None):
        ids = self.encode(text)
        if max_length:
            # 先截断
            if len(ids) > max_length:
                ids = ids[:max_length]
            # 再填充
            else:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
        return ids

    def __len__(self) -> int:
        """返回词表大小"""
        return len(self.vocab)

def get_chat_format(messages: List[Dict[str, str]]) -> str:
    """Formats conversation into Llama2 format."""
    formatted = ""
    for msg in messages:
        if msg["role"] == "system":
            formatted += f"<<SYS>>{msg['content']}<</SYS>>" # Correct SYS end tag
        elif msg["role"] == "user":
            formatted += f"{msg['content']}" # Correct INST tags
        elif msg["role"] == "assistant":
            formatted += msg["content"]
    return formatted


if __name__ == "__main__":
    # Example usage
    tokenizer = Llama2Tokenizer()

    # Significantly Expanded Training Texts (with repetitions)
    texts = [
        "Hello world! This is a test.",
        "Hello world! This is another test.",  # Repetition
        "Llama2 is a large language model.",
        "Llama2 is a powerful model.",  # Variation
        "Natural language processing is amazing!",
        "Natural language understanding is crucial.", # Variation
        "This is a longer sentence with some punctuation.",
        "This is a shorter sentence.", # Variation
        "Testing different types of tokens like 123 and $4.56.",
        "Testing more tokens like 456 and £7.89.", # Variation
        "Special characters like üöäéàèìòù are also handled.",
        "More special characters like çñß are important.", # Variation
        "Multiple spaces       should be treated as one.",
        "Multiple spaces shouldn't matter.", # Variation
        "Short sentences.",
        "Short phrases.", # Variation
        "VeryLongWordWithoutSpaces",
        "VeryLongWordWithoutSpaces", # Repetition
        "AnotherLongWord",
        "I am with you Teresia", # Repetition
        "WordWithInternalCapitalization",
        "WordWithInternalCapitalization", # Repetition
        "Multiple short sentences. This is one. Here's another.",
        "Even more short sentences. One. Two. Three." # Variation
    ]


    tokenizer.train(texts, vocab_size=1000)

    # Encoding and decoding tests
    test_strings = [
        "Hello world!",
        "This is a test.",
        "A sentence with some numbers 123 and symbols $%^&.",
        "Unseen words like floccinaucinihilipilification.",  # Test <unk> token
        "VeryLongWordWithoutSpaces",
        "AnotherLongWord"
    ]
    for text in test_strings:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Original: '{text}'")
        print(f"Encoded: {encoded}")
        print(f"Decoded: '{decoded}'")
        print("-" * 20)

    # Conversation formatting example
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
    ]

    chat_text = get_chat_format(messages)
    encoded = tokenizer.prepare_for_model(chat_text, max_length=128)
    print("Encoded Chat:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded Chat:", decoded)


    # Save and load example
    tokenizer.save("llama2_tokenizer.json")
    new_tokenizer = Llama2Tokenizer()
    new_tokenizer.load("llama2_tokenizer.json")

    # Test loaded tokenizer
    for text in test_strings:
        encoded = new_tokenizer.encode(text)
        decoded = new_tokenizer.decode(encoded)
        print(f"Loaded - Original: '{text}'")
        print(f"Loaded - Encoded: {encoded}")
        print(f"Loaded - Decoded: '{decoded}'")
        print("-" * 20)

        # 使用tokenizer.json重新测试
        print("\n使用tokenizer.json进行测试:")
        print("=" * 50)
        
    tokenizer = Llama2Tokenizer()
    tokenizer.load("tokenizer.json")
    
    # 重新测试编码和解码
    for text in test_strings:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"原文: '{text}'")
        print(f"编码: {encoded}")
        print(f"解码: '{decoded}'")
        print("-" * 20)

    # 重新测试对话格式
    chat_text = get_chat_format(messages)
    encoded = tokenizer.prepare_for_model(chat_text, max_length=128)
    print("\n编码后的对话:", encoded)
    decoded = tokenizer.decode(encoded)
    print("解码后的对话:", decoded)
    print("=" * 50)

