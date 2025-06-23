import torch
from torch import Tensor, nn

from models.bert.config import settings


class BertEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super(BertEmbeddings, self).__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=d_model,
        )
        self.segment_embedding = nn.Embedding(
            num_embeddings=2,  # 1 or 0
            embedding_dim=d_model,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self, input_ids: Tensor, segment_ids: Tensor | None = None
    ) -> Tensor:
        seq_len = input_ids.size(1)
        position_ids = (
            torch.arange(seq_len, dtype=torch.long)
            .unsqueeze(0)
            .expand_as(input_ids)
            .to(device=settings.device)
        )
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        token_embeds: Tensor = self.token_embedding(input_ids)
        position_embeds: Tensor = self.position_embedding(position_ids)
        segment_embeds: Tensor = self.segment_embedding(segment_ids)
        embeddings: Tensor = token_embeds + position_embeds + segment_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
