from torch import Tensor, nn

from models.transformer.attentions import MultiHeadAttention
from models.transformer.feed_forward import (
    PositionWiseFeedForward,
)


class BertLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
    ):
        super(BertLayer, self).__init__()
        self.attention = nn.Sequential(
            MultiHeadAttention(d_model, n_heads),
            nn.Dropout(dropout_rate),
        )
        self.ffn = nn.Sequential(
            PositionWiseFeedForward(d_model, d_ff),
            nn.Dropout(dropout_rate),
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, enc_input: Tensor, mask: Tensor | None = None):
        # Self attention
        attention_output: Tensor = self.attention(
            enc_input, enc_input, enc_input, mask
        )
        # Add & Norm
        attention_output = self.layer_norm1(enc_input + attention_output)

        # Feed Forward Network
        ffn_output: Tensor = self.ffn(attention_output)
        layer_output: Tensor = self.layer_norm2(attention_output + ffn_output)

        return layer_output
