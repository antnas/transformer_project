import torch
import numpy as np
import torch.nn as nn
from modelling.positional_encoding import PositionalEncoding
from modelling.feed_forward import FeedForward
from modelling.functional import *
from modelling.word_embedding import WordEmbedding


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_len):
        """
        Initializes the Transformer model.

        Args:
            vocab_size (int): The size of the vocabulary
            d_model (int): The dimensionality of the embedding layer
            n_heads (int): The number of heads in the multi-head attention layers
            num_encoder_layers (int): The number of encoder layers
            num_decoder_layers (int): The number of decoder layers
            dim_feedforward (int): The dimensionality of the feedforward layer
            dropout (float): The dropout probability
            max_len (int): The maximum length of the input sequence
        """
        super(Transformer, self).__init__()

        # Store the model parameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        

        # Word Embedding Layer
        self.embedding = WordEmbedding(d_model, vocab_size)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(embedding_dim=d_model, seq_len=max_len)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            BaseTransformerLayer(
                input_dim=d_model,
                num_heads=n_heads,
                feature_dim=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                input_dim=d_model,
                num_heads=n_heads,
                feature_dim=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_decoder_layers)
        ])

        # Linear layer for final output
        self.final_linear = nn.Linear(d_model, vocab_size)
        self.final_linear.weight = self.embedding.word_embedding.weight


    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
            """
            Forward pass of the Transformer model.

            Args:
                src (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
                tgt (torch.Tensor): Target input tensor of shape (batch_size, tgt_seq_len).
                src_mask (torch.Tensor, optional): Source mask tensor of shape (batch_size, src_seq_len).
                tgt_mask (torch.Tensor, optional): Target mask tensor of shape (batch_size, tgt_seq_len).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, tgt_seq_len, vocab_size).
            """
            # Word Embedding and Positional Encoding for source and target
            src_embedded = self.embedding(src) * torch.tensor(self.d_model).sqrt()
            src_positional = self.positional_encoding(src_embedded)
            src_positional = self.dropout(src_positional)


            tgt_embedded = self.embedding(tgt) * torch.tensor(self.d_model).sqrt()
            tgt_positional = self.positional_encoding(tgt_embedded)
            tgt_positional = self.dropout(tgt_positional)

            # Encoder
            for encoder_layer in self.encoder_layers:
                src_positional = encoder_layer(src_positional, attention_mask=src_mask)

            # Decoder
            for decoder_layer in self.decoder_layers:
                tgt_positional = decoder_layer(tgt_positional, src_positional, src_mask, tgt_mask)

            # Linear layer for final output
            output = self.final_linear(tgt_positional)

            return output