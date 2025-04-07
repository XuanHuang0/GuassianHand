from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from einops import rearrange, repeat

from tgs.utils.base import BaseModule
from tgs.utils.typing import *


class TriplaneLearnablePositionalEmbedding(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        plane_size: int = 32
        num_channels: int = 1024
        n_plane: int = 1

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.embeddings = nn.Parameter(
            torch.randn(
                (self.cfg.n_plane, self.cfg.num_channels, self.cfg.plane_size, self.cfg.plane_size),
                dtype=torch.float32,
            )
            * 1
            / math.sqrt(self.cfg.num_channels)
        )

    def forward(self, batch_size: int, cond_embeddings: Float[Tensor, "B Ct"] = None) -> Float[Tensor, "B Ct Nt"]:
        embeddings = repeat(self.embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=batch_size, Np=self.cfg.n_plane)
        # embeddings = self.embeddings
        cond_embeddings = cond_embeddings.unsqueeze(1)
        # print(cond_embeddings.shape) #[8, 3, 512, 32, 32]
        # print(embeddings.shape) #[8, 3, 512, 32, 32]
        if cond_embeddings is not None:
            embeddings = embeddings + cond_embeddings
        return rearrange(
            embeddings,
            "B Np Ct Hp Wp -> B Ct (Np Hp Wp)",
        )

    def detokenize(
        self, tokens: Float[Tensor, "B Ct Nt"]
    ) -> Float[Tensor, "B Np Ct Hp Wp"]:
        batch_size, Ct, Nt = tokens.shape
        assert Nt == self.cfg.plane_size**2 * self.cfg.n_plane
        assert Ct == self.cfg.num_channels
        return rearrange(
            tokens,
            "B Ct (Np Hp Wp) -> B Np Ct Hp Wp",
            Np=self.cfg.n_plane,
            Hp=self.cfg.plane_size,
            Wp=self.cfg.plane_size,
        )
