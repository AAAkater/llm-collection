import torch
from torch import Tensor, nn

from models.bert.config import settings
from models.bert.embeddings import BertEmbeddings
from models.bert.layer import BertLayer


class Bert(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 512,
        d_ff: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        dropout_rate: float = 0.1,
        with_pool: bool = False,
        with_mlm: bool = False,
        with_nsp: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = d_ff
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.with_pool = with_pool
        self.with_mlm = with_mlm
        self.with_nsp = with_nsp
        if self.with_nsp and not self.with_pool:
            self.with_pool = True
        if self.with_nsp:
            self.nsp = nn.Linear(d_model, 2)
        if self.with_pool:
            self.pool = nn.Linear(d_model, d_model)
            self.pool_activation = nn.Tanh()
        if self.with_mlm:
            self.mlm_decoder = nn.Linear(d_model, vocab_size, bias=False)
            self.mlm_decoder.bias = nn.Parameter(torch.zeros(self.vocab_size))

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BertEmbeddings(
            vocab_size=vocab_size,
            segment_vocab_size=2,
            d_model=d_model,
            max_seq_len=max_seq_len,
        )

        # multi-layers transformer blocks, deep network
        self.layers = nn.ModuleList(
            [
                BertLayer(d_model, n_heads, d_ff, dropout_rate)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor):
        x = self.embedding(x)
        for transformer_encoder in self.layers:
            x = transformer_encoder(x)
        return x
