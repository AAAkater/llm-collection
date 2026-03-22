import math

import torch.nn.functional as F
from torch import Tensor, nn


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        # (batch_size, n_head, seq_len, d_k)
        d_k = q.size(-1)

        # Q * K^T/sqrt(d_k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        # scores -> (batch_size, n_head, seq_len, seq_len)

        # 得出的scores是每个维度(d_1-d_v)都考虑了在当前维度(这一列)
        # 当前token对所有token的注意力后更新的新的值，
        # 换言之每个维度d是相互独立的，
        # 每个维度考虑自己的所有token的注意力，
        # 所以可以理解成1列扩展到多列

        # 加入mask矩阵
        scores += mask
        attn = F.softmax(scores, -1)

        # 返回的attn: [batch_size, n_heads, seq_len, d_k]本质上还是batch_size个句子，
        # 只不过每个句子中词向量维度512被分成了8个部分，分别由8个头各自看一部分，
        # 每个头算的是整个句子(一列)的512/8=64个维度，最后按列拼接起来
        # (seq_len , seq_len) @ (seq_len , d_k) = (seq_len , d_k)
        return attn @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_heads
        self.attention = ScaleDotProductAttention()

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: Tensor,  # (batch_size, seq_len, d_model)
        k: Tensor,  # (batch_size, seq_len, d_model)
        v: Tensor,  # (batch_size, seq_len, d_model)
        mask: Tensor,
    ) -> Tensor:
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # Split into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, n_head, seq_len, d_k)
        # where d_k = d_model // n_head
        q, k, v = self.split(q), self.split(k), self.split(v)

        out: Tensor = self.attention(q, k, v, mask)

        # Concatenate heads
        # (batch_size, n_head, seq_len, d_k) -> (batch_size, seq_len, d_model)
        out = self.concat(out)

        out: Tensor = self.linear(out)

        return out

    def split(self, x: Tensor) -> Tensor:
        # (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()

        d_k = d_model // self.n_head

        x = x.view(batch_size, seq_len, self.n_head, d_k).transpose(1, 2)
        # (batch_size, n_head, seq_len, d_k)
        return x

    def concat(self, x: Tensor) -> Tensor:
        batch_size, n_head, seq_len, d_k = x.size()

        x = (
            # (batch_size, n_head, seq_len, d_k) -> (batch_size, seq_len, n_head, d_k)
            x.transpose(1, 2)
            # 保证内存连续
            .contiguous()
            # (batch_size, seq_len, d_model)
            .view(batch_size, seq_len, n_head * d_k)
        )

        return x
