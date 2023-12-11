import torch
import numpy as np
import torch.nn as nn

# 0 not padded
# 1 padded

class Attention(torch.nn.Module):
    def __init__(self, mask_future=False):
        """
        Initializes the Attention module.

        Args:
            mask_future (bool, optional): Whether to mask future positions in the attention logits. 
                                         Defaults to False.
        """
        super().__init__()
        self.mask_future = mask_future

    def forward(self, Q, K, V, padding_mask=None):
        """
        Performs forward pass of the Attention module.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, n_q, d_model).
            K (torch.Tensor): Key tensor of shape (batch_size, n_k, d_model).
            V (torch.Tensor): Value tensor of shape (batch_size, n_k, d_model).
            padding_mask (torch.Tensor, optional): Padding mask tensor of shape (batch_size, n_q).
                                                   Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_q, d_model).
        """
        print("hello")
        d_model = Q.shape[-1]  # Embedding dimension
        n_q = Q.shape[1]  # Length of input
        n_k = K.shape[1]  # Length of output

        # Calculate scaling value
        scaling_factor = 1 / torch.tensor(d_model).sqrt()

        # Calculate modelling logits
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * scaling_factor
        print(attn_logits.shape)

        # Apply future mask
        if self.mask_future:
            future_mask = torch.tril(torch.ones(n_q, n_k))
            attn_logits.masked_fill_(future_mask == 0, -torch.inf)

        # Apply padding mask
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(dim=1)
            attn_logits.masked_fill_(padding_mask == 0, -torch.inf)

        # Apply softmax to masked logits
        softmax_attn = torch.softmax(attn_logits, -1)

        # Apply modelling weighting to the values
        output = torch.bmm(softmax_attn, V)

        return output
    
class SelfAttentionHead(torch.nn.Module):
    def __init__(self, d_model, mask_future=False):
        """
        :param d_model (int): dimension of the token embeddings
        :param future_masking (bool): whether future tokens should be masked in the modelling
        """
        super().__init__()

        self.d_model = d_model
        self.mask_future = mask_future
        self.attn = Attention(mask_future=mask_future)
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_query, x_key, x_value, padding_mask=None):
        Q = self.q_proj(x_query)
        K = self.k_proj(x_key)
        V = self.v_proj(x_value)
        return self.attn(Q, K, V, padding_mask)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, mask_future=False):
        """
        :param d_model (int): dimension of the token embeddings
        :param num_heads (int): number of attention heads
        :param future_masking (bool): whether future tokens should be masked in the modelling
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.mask_future = mask_future

        self.self_attention = Attention(mask_future=mask_future)

        assert d_model % num_heads == 0, "Number of heads must evenly divide the embedding dimension"
        # This is the dimension for each head
        self.d_k = d_model // num_heads 

        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)

        self.output_transform = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_query, x_key, x_value, padding_mask=None):
        """
        :param x_query (torch.Tensor): query tensor of shape (batch_size, seq_len, d_model)
        :param x_key (torch.Tensor): key tensor of shape (batch_size, seq_len, d_model)
        :param x_value (torch.Tensor): value tensor of shape (batch_size, seq_len, d_model)
        :param padding_mask (torch.Tensor): padding mask tensor of shape (batch_size, n_q)
        :return: output tensor of shape (batch_size, seq_len, d_model)
        """
        # Calculate Q,K,V based on the linear layers
        Q = self.query_transform(x_query) # (batch_size, seq_len, d_model)
        K = self.key_transform(x_key) # (batch_size, seq_len, d_model)
        V = self.value_transform(x_value) # (batch_size, seq_len, d_model)

        # Split into d_k different chunks, one for each head
        Qs = Q.split(self.d_k, dim=-1) # (batch_size, seq_len, d_k)
        Ks = K.split(self.d_k, dim=-1) # (batch_size, seq_len, d_k)
        Vs = V.split(self.d_k, dim=-1) # (batch_size, seq_len, d_k)

        # Apply self-attention for each head
        heads = []
        for q, k, v in zip(Qs, Ks, Vs):
            heads.append(self.self_attention(q, k, v, padding_mask))
        
        heads_concat = torch.cat(heads, dim=-1)

        # Finally we apply the output transformation 
        x = self.output_transform(heads_concat)

        return x
    