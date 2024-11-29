import torch
import torch_musa
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from lib06_integrate import Transformer
from lib07_gen_resp import generate_response
import json


# Hyperparameters (adjust as needed)
src_vocab_size = 5000  # Example vocabulary size for source language
trg_vocab_size = 5000  # Example vocabulary size for target language
d_model = 512
n_head = 8
ffn_hidden = 2048
n_layers = 6
dropout = 0.1
max_len = 100
batch_size = 32
epochs = 10
learning_rate = 0.0001
device = "musa"

# Placeholder data (replace with your actual dataset)
src_data = torch.randint(0, src_vocab_size, (1000, max_len))  # Example source sentences
trg_data = torch.randint(0, trg_vocab_size, (1000, max_len))  # Example target sentences
src_pad_idx = 3  # <pad> token index
trg_pad_idx = 3  # <pad> token index

# Create DataLoader
dataset = TensorDataset(src_data, trg_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer, and loss function
model = Transformer(src_pad_idx, trg_pad_idx, src_vocab_size, trg_vocab_size, d_model, max_len, n_head, ffn_hidden, n_layers, dropout, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)  # Ignore padding index for loss calculation


# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        
        # Create masks
        src_mask = model.make_pad_mask(src, src, src_pad_idx, src_pad_idx)
        trg_mask = model.make_pad_mask(trg[:, :-1], trg[:, :-1], trg_pad_idx, trg_pad_idx) & model.make_casual_mask(trg[:, :-1], trg[:, :-1])
        
        output = model(src, trg[:, :-1])  # Pass the input sequence without the last token
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1) #shifted right
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch: {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")


# Load vocabulary from tokenizer.json
def load_vocab_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    class Vocabulary:
        def __init__(self, token_to_id):
            # Add special tokens
            self.stoi = {
                "<s>": 1,      # Beginning of sequence 
                "</s>": 2,     # End of sequence
                "<unk>": 0,    # Unknown token
                "<pad>": 3,    # Padding token
                "[INST]": 4,   # Instruction start
                "[/INST]": 5,  # Instruction end
                "<<SYS>>": 6,  # System prompt start  
                "<</SYS>>": 7  # System prompt end
            }
            # Add the rest of vocabulary
            self.stoi.update(token_to_id)
            self.itos = {v: k for k, v in self.stoi.items()}
            
        def encode(self, text):
            return [self.stoi.get(word, self.stoi["<unk>"]) for word in text.split()]
            
        def decode(self, indices):
            return " ".join(self.itos[idx] for idx in indices)
            
        def bos_id(self):
            return self.stoi["<s>"]
            
        def eos_id(self):
            return self.stoi["</s>"]
            
    return Vocabulary(tokenizer_data['model']['vocab'])

# Load vocabulary from tokenizer file  
vocab = load_vocab_from_json('tokenizer.json')

# Generate response using the trained model
input_sentence = "hello world"
response = generate_response(model, input_sentence, max_len, vocab, device, vocab.bos_id(), vocab.eos_id())
print(f"Chatbot: {response}")