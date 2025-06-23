from torch import Tensor, nn

from models.transformer.attentions import MultiHeadAttention
from models.transformer.embeddings import TransformerEmbedding
from models.transformer.feed_forward import (
    PositionWiseFeedForward,
)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_head: int,
    ):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_head)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        enc: Tensor,
        mask: Tensor,
    ):
        residual = enc.clone()
        enc = self.attn(
            enc,
            enc,
            enc,
            mask,
        )
        # 残差连接
        enc += residual
        enc = self.norm1(enc)

        residual = enc.clone()
        enc = self.pos_ffn(enc)
        # 残差连接
        enc += residual
        enc = self.norm2(enc)

        return enc


class Encoder(nn.Module):
    """
    Transformer 模型的编码器模块。

    Args:
        enc_voc_size (int): 编码器输入词汇表的大小。
        max_seq_len (int): 输入序列的最大长度。
        d_model (int): 模型的维度。
        ffn_hidden (int): 前馈神经网络隐藏层的维度。
        n_head (int): 多头注意力机制中头的数量。
        n_layers (int): 编码器层的数量。

    Attributes:
        emb (TransformerEmbedding): 用于输入序列的嵌入层。
        layers (nn.ModuleList): 包含多个编码器层的模块列表。

    Methods:
        forward: 执行编码器的前向传播。

        Args:
            enc (Tensor): 输入序列的张量。
            src_mask (Tensor): 用于屏蔽无效位置的掩码张量。

        Returns:
            Tensor: 编码后的序列表示。
    """

    def __init__(
        self,
        enc_voc_size: int,
        max_seq_len: int,
        d_model: int,
        d_ff: int,
        n_head: int,
        n_layers: int,
    ):
        super().__init__()
        self.emb = TransformerEmbedding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            vocab_size=enc_voc_size,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    n_head=n_head,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc: Tensor, src_mask: Tensor):
        enc = self.emb(enc)

        for layer in self.layers:
            enc = layer(enc, src_mask)

        return enc
