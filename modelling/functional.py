import torch
import torch.nn as nn
from modelling.feed_forward import FeedForward
from modelling.attention import MultiHeadAttention


class BaseTransformerLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout):
            """
            Initializes a BaseTransformerLayer module.

            Args:
                input_dim (int): The number of features in the input.
                num_heads_heads (int): The number of heads in the multiheadattention models.
                feature_dim (int): The dimension of the feedforward network model.
                dropout (float): The dropout value.

            """
            super().__init__()
            
            self.self_attention = MultiHeadAttention(input_dim, num_heads)
            self.layer_norm_1 = nn.LayerNorm(input_dim)
            
            self.feature_transformation = FeedForward(input_dim, feature_dim)
            self.layer_norm_2 = nn.LayerNorm(input_dim)

            self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        
    def forward(self, x, attention_mask=None):
        """
        Forward pass of the BaseTransformerLayer module.

        Args:
            input (Tensor): The input sequence to the transformer layer.
            attention_mask (Tensor, optional): The mask for the attention.
        Returns:
            Tensor: The output of the transformer layer.

        """
        # First sublayer is the multi-head attention
        x_pre = x
        x = self.self_attention(x, x, x, attention_mask)
        #x *= attention_mask.unsqueeze(-1).float()
        if self.dropout:
            x = self.dropout(x)
        x = self.layer_norm_1(x + x_pre) 


        # Second sublayer is the feed forward network
        x_pre = x
        x = self.feature_transformation(x)
        #x *= attention_mask.unsqueeze(-1).float()
        if self.dropout:
            x = self.dropout(x)
        x = self.layer_norm_2(x + x_pre)
        #x *= attention_mask.unsqueeze(-1).float()

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout=0.2):
        """
        Initializes a TransformerDecoderLayer module.

        Args:
            input_dim (int): The number of features in the input.
            num_heads_heads (int): The number of heads in the multiheadattention models.
            feature_dim (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super().__init__()
        
        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=True)
        self.layer_norm_1 = nn.LayerNorm(input_dim)

        self.encoder_attention = MultiHeadAttention(input_dim, num_heads, mask_future=False)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
    
        self.feature_transformation = FeedForward(input_dim, feature_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x, encoder, encoder_attention_maks, attention_mask):
        """
        Forward pass of the TransformerDecoderLayer module.

        Args:
            input (Tensor): The input sequence to the decoder transformer layer.
            encoder (Tensor): The output of the encoder transformer layer.
            encoder_attention_maks (Tensor): The mask for the encoder attention.
            attention_mask (Tensor, optional): The mask for the decoder attention.
        Returns:
            Tensor: The output of the transformer layer.

        """
        # First sublayer is the masked multi-head attention
        x_pre = x
        x = self.self_attention(x, x, x, attention_mask)
        #x *= attention_mask.unsqueeze(-1).float()
        if self.dropout:
            x = self.dropout(x)
        x = self.layer_norm_1(x + x_pre) 

        # Second sublayer is the encoder multi-head attention
        x_pre = x
        x = self.encoder_attention(x, encoder, encoder, encoder_attention_maks)
        #x *= attention_mask.unsqueeze(-1).float()
        if self.dropout:
            x = self.dropout(x)
        x = self.layer_norm_2(x + x_pre)

        # Third sublayer is the feed forward network
        x_pre = x
        x = self.feature_transformation(x)
        #x *= attention_mask.unsqueeze(-1).float()
        if self.dropout:
            x = self.dropout(x)
        x = self.layer_norm_3(x + x_pre)
        #x *= attention_mask.unsqueeze(-1).float()

        return x