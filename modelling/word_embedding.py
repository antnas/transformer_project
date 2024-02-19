import torch.nn as nn

class WordEmbedding(nn.Module):
    def __init__(self, embedding_dim, vocab_size=50000):
        super(WordEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.word_embedding(x)
