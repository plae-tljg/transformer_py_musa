import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model, padding_idx=1)

    def forward(self, x):
        return super().forward(x)

# Example usage
vocab_size = 10
d_model = 19

layer = TokenEmbedding(vocab_size, d_model)
input_indices = torch.LongTensor([[1, 5, 2, 0, 1], [6, 2, 1, 8, 9]])

embeddings = layer(input_indices)

# Reshape the embeddings for PCA
embeddings_reshaped = embeddings.view(-1, d_model).detach().numpy()  # Detach from the computation graph and convert to NumPy

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings_reshaped)

# Plot the embeddings
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])

# Annotate points with their corresponding token indices
for i, token_index in enumerate(input_indices.view(-1)):
    plt.annotate(str(token_index.item()), (embeddings_pca[i, 0], embeddings_pca[i, 1]))


plt.title("Token Embeddings Visualized with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig("token_embeddings_pca.png")
plt.show()