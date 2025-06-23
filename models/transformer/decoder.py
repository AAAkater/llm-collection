from torch import Tensor, nn

from models.transformer.embeddings import TransformerEmbedding
from models.transformer.layers.my_multi_head_attention import MultiHeadAttention
from models.transformer.layers.my_position_wise_feed_forward import (
    PositionWiseFeedForward,
)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_head: int,
    ):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_head)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        dec_out: Tensor,
        enc_out: Tensor,
        tgt_mask: Tensor,
        src_mask: Tensor,
    ) -> Tensor:
        """
        执行Transformer解码器层的前向传播。

        Args:
            dec_out (Tensor): 解码器的输入张量，形状通常为(batch_size, tgt_seq_len, d_model)
            enc_out (Tensor): 编码器的输出张量，形状通常为(batch_size, src_seq_len, d_model)
            tgt_mask (Tensor): 目标序列的注意力掩码，防止关注到未来位置
            src_mask (Tensor): 源序列的注意力掩码，防止关注到填充位置

        Returns:
            Tensor: 经过解码器层处理后的输出张量,形状与dec_out相同

        Note:
            1. 执行解码器自注意力计算，使用残差连接和层归一化
            2. 执行编码器-解码器注意力计算，使用残差连接和层归一化
            3. 通过前馈网络处理，使用残差连接和层归一化
        """
        # self-attention
        residual = dec_out.clone()
        dec_out = self.dec_self_attn(
            q=dec_out, k=dec_out, v=dec_out, mask=tgt_mask
        )
        # 残差连接
        dec_out += residual
        dec_out = self.norm1(dec_out)

        # encoder-decoder attention
        residual = dec_out.clone()
        dec_out = self.enc_dec_attn(
            q=dec_out, k=enc_out, v=enc_out, mask=src_mask
        )
        # 残差连接
        dec_out += residual
        dec_out = self.norm2(dec_out)

        # position wise feed forward network
        residual = dec_out.clone()
        dec_out = self.ffn(dec_out)
        # 残差连接
        dec_out += residual
        dec_out = self.norm3(dec_out)

        return dec_out


class Decoder(nn.Module):
    def __init__(
        self,
        dec_vocab_size: int,
        max_seq_len: int,
        d_model: int,
        d_ff: int,
        n_head: int,
        n_layers: int,
    ):
        super(Decoder, self).__init__()
        self.emb = TransformerEmbedding(
            dec_vocab_size,
            d_model,
            max_seq_len,
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    d_ff,
                    n_head,
                )
                for _ in range(n_layers)
            ]
        )
        self.linear = nn.Linear(d_model, dec_vocab_size)

    def forward(
        self,
        dec_out: Tensor,
        enc_out: Tensor,
        tgt_mask: Tensor,
        src_mask: Tensor,
    ):
        tgt: Tensor = self.emb(dec_out)
        for layer in self.layers:
            tgt = layer(tgt, enc_out, tgt_mask, src_mask)
        out_put: Tensor = self.linear(tgt)
        return out_put
