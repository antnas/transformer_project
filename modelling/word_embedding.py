import torch.nn as nn

class WordEmbedding(nn.Module):
    def _init_(self, embedding_dim, vocab_size=50000):
        super(WordEmbedding, self)._init_()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.word_embedding(x)