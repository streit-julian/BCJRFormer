import torch
import torch.nn as nn
from bcjrformer.configs.model_config import ModelConfig
from bcjrformer.models.modules import (
    TransformerEncoder,
)


class BCJRFormer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        window_block_dimension: int,
        encoded_length: int,
        device,
        mask: None | torch.Tensor = None,
        q=2,
    ):
        super().__init__()

        self.model_config = model_config
        self.device = device
        self.out_channels = 1 if q == 2 else q
        self.window_block_dimension = torch.tensor([window_block_dimension]).to(device)

        self.n = encoded_length

        self.mask = mask.to(device) if mask is not None else None

        self.n_sequence_max = model_config.inner_model_config.n_sequence_max

        self.decoder = TransformerEncoder(
            model_config.d_model,
            model_config.N_dec,
            model_config.h,
            model_config.dropout,
            mask=self.mask,
        )

        self.pos_embed = torch.nn.Parameter(torch.empty((self.n, model_config.d_model)))
        self.seq_embed = nn.Embedding(self.n_sequence_max, model_config.d_model)

        self.sequence_to_embed = (
            torch.arange(0, self.n_sequence_max)
            .reshape(-1, 1)
            .multiply(
                torch.ones(self.n * self.n_sequence_max).reshape(
                    self.n_sequence_max, self.n
                )
            )
            .reshape(-1)
            .long()
            .to(device)
        )

        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(model_config.d_model, self.out_channels)]
        )

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(window_block_dimension),
            nn.Linear(window_block_dimension, model_config.d_model),
            nn.LayerNorm(model_config.d_model),
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, y, padding_mask):
        bs = y.shape[0]
        # [batch_size, n_seq, n, window_size, window_block] -> [batch_size, n_seq *n, window_block]
        y = y.view(bs, y.shape[1] * y.shape[2], -1)

        # [batch_size, n_seq, n] -> [batch_size, 1, n_seq * n, 1]
        mask = padding_mask.view(bs, -1).unsqueeze(1).unsqueeze(-1)

        # [batch_size, n_seq * n, window_size] -> [batch_size, n_seq * n, d_model]
        x = self.to_patch_embedding(y)

        # [batch_size, n_seq * n, d_model] -> [batch_size, n_seq, n, d_model]
        x = x.view(x.shape[0], -1, self.n, self.model_config.d_model)

        # Add position embeddings (on a per sequence basis)
        x += self.pos_embed

        # [batch_size, n_seq, n, d_model] -> [batch_size, n_seq * n , d_model]
        x = x.view(x.shape[0], -1, self.model_config.d_model)

        # Add sequence embeddings (to distinguish between subsequences)
        x += self.seq_embed(self.sequence_to_embed)

        x = self.decoder(x, mask)

        x = self.oned_final_embed(x).squeeze(-1)

        if self.n_sequence_max > 1:
            mean_denominator = (
                (padding_mask.sum(axis=-1) == 0).sum(axis=1).unsqueeze(-1)
            )
            mean_nominator = (
                x.view(x.shape[0], -1, self.n, self.out_channels)
                .masked_fill(padding_mask.unsqueeze(-1), 0)
                .sum(axis=1)
            )

            x = (mean_nominator / mean_denominator.unsqueeze(-1)).squeeze(
                -1
            )  # squeeze if it only has one channel

        return x


############################################################
############################################################
