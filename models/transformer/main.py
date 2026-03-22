import torch
from torch import Tensor, nn

from models.transformer.config import settings
from models.transformer.decoder import Decoder
from models.transformer.encoder import Encoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_pad_idx: int,
        tgt_pad_idx: int,
        tgt_sos_idx: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_head: int = 8,
        n_layer: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 5000,
    ):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.encoder = Encoder(
            encoder_vocab_size=src_vocab_size,
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layer,
            max_seq_len=max_seq_len,
            d_ff=d_ff,
        )
        self.decoder = Decoder(
            dec_vocab_size=tgt_vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            n_layers=n_layer,
        )
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
    ) -> Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_src: Tensor = self.encoder(src, src_mask)
        output: Tensor = self.decoder(tgt, enc_src, tgt_mask, src_mask)

        return self.linear(output)

    def make_src_mask(self, src: Tensor) -> Tensor:
        """
        Generates a source mask to ignore padding tokens in the input sequence.

        Args:
            src (Tensor): The input source sequence tensor with shape (batch_size, seq_len).

        Returns:
            Tensor: A boolean mask tensor with shape (batch_size, 1, 1, seq_len), where `False`
                indicates padding positions that should be ignored. The mask is created by comparing
                each token in `src` with the padding index (`self.src_pad_idx`).

        Note:
            The mask is unsqueezed twice to add dimensions for broadcasting in subsequent operations,
            typically in attention mechanisms.
        """
        # padding mask: [batch, 1, 1, src_len]
        src_pad_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_pad_mask

    def make_tgt_mask(self, tgt: Tensor):
        """
        Generates a target mask for transformer decoder self-attention.

        The mask combines padding mask (to ignore padding tokens) and subsequent mask
        (to prevent attending to future tokens) for the target sequence.

        Args:
            tgt (Tensor): Target sequence tensor of shape (batch_size, seq_len).

        Returns:
            Tensor: Combined target mask of shape (batch_size, 1, seq_len, seq_len) where:
                - False positions indicate tokens to be masked
                - True positions indicate tokens to be attended to

        The mask is constructed by:
        1. Creating padding mask from non-padding tokens (tgt != self.tgt_pad_idx)
        2. Creating subsequent mask using lower triangular matrix
        3. Combining both masks with logical AND operation
        """
        # padding mask: [batch, 1, 1, tgt_len]
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        # sub mask: [1, 1, tgt_len, tgt_len]
        tgt_len = tgt.shape[1]
        tgt_sub_mask = (
            torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool))
            .unsqueeze(0)
            .unsqueeze(0)
        ).to(settings.device)
        # merge: [batch, 1, tgt_len, tgt_len]
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask


if __name__ == "__main__":
    pass
