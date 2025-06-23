import torch
from torch import Tensor, nn

from models.transformer.config import settings


class PositionalEncoding(nn.Module):
    """
    实现Transformer中的位置编码(Positional Encoding)模块。

    位置编码用于向输入序列注入位置信息,使模型能够利用序列中token的顺序信息。使用正弦和余弦函数的不同频率来生成位置编码。

    Args:
        d_model (int): 输入特征的维度，必须是偶数。
        max_seq_len (int, optional): 预计算位置编码的最大序列长度。默认为5000。

    Attributes:
        encoding (Tensor): 预计算好的位置编码矩阵，形状为(max_seq_len, d_model)，不参与梯度计算。

    Methods:
        forward(x): 返回输入序列对应的位置编码。

    Example:
        >>> pe = PositionalEncoding(d_model=512)
        >>> x = torch.randn(10, 32, 512)  # batch_size=10, seq_len=32
        >>> x_pe = pe(x)  # 添加位置编码
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 5000,
    ) -> None:
        super(PositionalEncoding, self).__init__()
        # 确保模型维度是偶数，因为需要成对使用sin和cos
        assert d_model % 2 == 0, f"{d_model} must be even!"

        # 创建一个形状为(max_seq_len, d_model)的零矩阵用于存储位置编码
        self.encoding = torch.zeros(max_seq_len, d_model)
        # 设置位置编码矩阵不需要计算梯度
        self.encoding.requires_grad = False

        # 生成位置索引向量 [0, 1, 2, ..., max_seq_len-1]，并扩展维度变为列向量
        pos = torch.arange(
            0,
            max_seq_len,
            dtype=torch.float,
        ).unsqueeze(dim=1)

        # 计算用于位置编码的除数项
        # 使用exp和log实现幂运算：10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(
                start=0,
                end=d_model,
                step=2,
                dtype=torch.float,
            )
            * -(torch.log(Tensor([10000.0])) / d_model)
        )

        # 使用sin函数计算偶数位置的编码
        self.encoding[:, 0::2] = torch.sin(pos / div_term)
        # 使用cos函数计算奇数位置的编码
        self.encoding[:, 1::2] = torch.cos(pos / div_term)

    def forward(self, x: Tensor) -> Tensor:
        # 获取输入张量的batch_size和序列长度
        batch_size, seq_len = x.size()
        # 返回对应序列长度的位置编码
        # 注意这里只返回编码矩阵，实际使用时需要与输入相加
        return self.encoding[:seq_len, :].to(settings.device)


class TokenEmbedding(nn.Embedding):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
    ) -> None:
        super(TokenEmbedding, self).__init__(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        drop_rate: float = 0.1,
    ) -> None:
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_seq_len)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        tok_emb: Tensor = self.tok_emb(x)
        pos_emb: Tensor = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
