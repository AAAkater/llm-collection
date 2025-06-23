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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, enc: Tensor, mask: Tensor | None = None):
        # Self attention
        attention_output: Tensor = self.attention(enc, enc, enc, mask)
        attention_output = self.norm1(enc + attention_output)  # Add & Norm

        # Feed Forward Network
        ffn_output: Tensor = self.ffn(attention_output)
        layer_output: Tensor = self.norm2(attention_output + ffn_output)

        return layer_output
