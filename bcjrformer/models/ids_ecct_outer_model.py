"""
@author: Yoni Choukroun, choukroun.yoni@gmail.com
Error Correction Code Transformer
https://arxiv.org/abs/2203.14966
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from bcjrformer.codes.linear_code import LinearCode
from bcjrformer.utils import sign_to_bin
from bcjrformer.configs.model_config import ModelConfig
from bcjrformer.models.modules import TransformerEncoder


class IdsEcctOuterTransformer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        code: LinearCode,
        device,
    ):
        super(IdsEcctOuterTransformer, self).__init__()

        self.model_config = model_config
        self.code = code

        self.dropout = model_config.dropout

        if model_config.masked_attention:
            self.pc_mask = self.get_pc_mask(code).to(device)

        # how does this denote the one-hot encoding defined according to the bit position
        self.pos_embed = torch.nn.Parameter(
            torch.empty((code.n + code.pc_matrix.shape[0], model_config.d_model))
        )
        self.pos_dropout = nn.Dropout(p=self.dropout)

        self.decoder = TransformerEncoder(
            model_config.d_model,
            model_config.N_dec,
            model_config.h,
            model_config.dropout,
            mask=self.pc_mask if model_config.masked_attention else None,
        )

        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(model_config.d_model, 1)]
        )
        self.out_fc = nn.Linear(code.n + code.pc_matrix.shape[0], code.n)

        ###
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, magnitude, syndrome):
        emb = torch.cat([magnitude, syndrome], -1).unsqueeze(-1)
        emb = self.pos_embed.unsqueeze(0) * emb

        emb = self.pos_dropout(emb)

        emb = self.decoder(emb)
        return self.out_fc(self.oned_final_embed(emb).squeeze(-1))

    def loss(self, z_pred, z2, y):
        loss = F.binary_cross_entropy_with_logits(z_pred, sign_to_bin(torch.sign(z2)))
        x_pred = sign_to_bin(torch.sign(-z_pred * torch.sign(y)))
        return loss, x_pred

    def get_pc_mask(self, code) -> torch.Tensor:
        pc = torch.tensor(code.pc_matrix)

        def build_mask(code):
            mask_size = code.n + pc.size(0)
            mask = torch.eye(mask_size, mask_size)
            for ii in range(pc.size(0)):
                idx = torch.where(pc[ii] > 0)[0]
                for jj in idx:
                    for kk in idx:
                        if jj != kk:
                            mask[jj, kk] += 1
                            mask[kk, jj] += 1
                            mask[code.n + ii, jj] += 1
                            mask[jj, code.n + ii] += 1
            src_mask = ~(mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask

        src_mask = build_mask(code)

        return src_mask


############################################################
############################################################

if __name__ == "__main__":
    pass
