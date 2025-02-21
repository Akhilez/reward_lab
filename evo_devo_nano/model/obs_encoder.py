import math

import torch
from torch import nn


def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with " "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class TimeStepEncoder(nn.Module):
    def __init__(self, embed_dim, max_time_steps: int):
        super(TimeStepEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.max_time_steps = max_time_steps

        self.time_embeddings = positional_encoding_1d(embed_dim, max_time_steps)

    def forward(self, embeddings, timestep: int):
        time_embeddings = self.time_embeddings[timestep]  # (1, embed_dim)
        time_embeddings = time_embeddings.unsqueeze(0).expand(embeddings.shape[0], -1, -1)  # (B, 1, embed_dim)

        # concatenate time embeddings to obs
        embeddings = torch.cat([embeddings, time_embeddings], dim=1)
        return embeddings

    @staticmethod
    def decode(embeddings):
        # embeddings: (B, T + 1, embed_dim)
        return embeddings[:, :-1]


class ObsEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim, h, w):
        super(ObsEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.h, self.w = h, w

        # self.scale_factor = 4

        # output_size=⌊(stride-input_size−kernel_size+2×padding)/stride⌋+1
        # self.h_scaled = int((self.h - (self.scale_factor * 2)) / self.scale_factor) + 1
        # self.w_scaled = int((self.w - (self.scale_factor * 2)) / self.scale_factor) + 1

        # # output_padding=desired_output_size−[(input_size−1)×stride−2×padding+kernel_size]
        # output_padding = (
        #     (self.h - ((self.h_scaled - 1) * self.scale_factor + self.scale_factor * 2)),
        #     (self.w - ((self.w_scaled - 1) * self.scale_factor + self.scale_factor * 2)),
        # )

        self.h_scaled = h // 3
        self.w_scaled = w // 3

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=7,
                stride=3,
                padding=3,
            ),
            nn.SiLU(),
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=7,
                stride=3,
                padding=3,
            )
        )
        self.decoder = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(
                embed_dim,
                embed_dim,
                kernel_size=7,
                stride=3,
                padding=3,
                output_padding=2,
            ),
            nn.SiLU(),
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.SiLU(),
            nn.ConvTranspose2d(
                embed_dim,
                embed_dim,
                kernel_size=7,
                stride=3,
                padding=3,
                output_padding=2,
            ),
            nn.SiLU(),
            nn.Conv2d(
                embed_dim,
                in_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.Sigmoid(),
        )

        # # Single conv layers
        # self.encoder = nn.Conv2d(
        #     in_channels,
        #     embed_dim,
        #     kernel_size=self.scale_factor * 2,
        #     stride=self.scale_factor,
        #     padding=0,  # self.scale_factor // 2,
        # )  # hxw -> h/4 x w/4
        #
        # self.decoder = nn.Sequential(
        #     nn.SiLU(),
        #     nn.ConvTranspose2d(
        #         embed_dim,
        #         in_channels,
        #         kernel_size=self.scale_factor * 2,
        #         stride=self.scale_factor,
        #         padding=0,  # self.scale_factor // 2,
        #         output_padding=output_padding,
        #     ),  # h/4 x w/4 -> h x w
        #     nn.Sigmoid(),
        # )

        # # 2 conv layers
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels, embed_dim, kernel_size=self.scale_factor, stride=self.scale_factor, padding=0),  # hxw -> h/4 x w/4
        #     nn.SiLU(),
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=self.scale_factor, stride=self.scale_factor, padding=0),  # h/4 x w/4 -> h/16 x w/16
        # )
        # self.decoder = nn.Sequential(
        #     nn.SiLU(),
        #     nn.ConvTranspose2d(
        #         embed_dim,
        #         embed_dim,
        #         kernel_size=self.scale_factor,
        #         stride=self.scale_factor,
        #         padding=0,  # self.scale_factor // 2,
        #         output_padding=output_padding,
        #     ),  # h/16 x w/16 -> h/4 x w/4
        #     nn.SiLU(),
        #     nn.ConvTranspose2d(
        #         embed_dim,
        #         in_channels,
        #         kernel_size=self.scale_factor,
        #         stride=self.scale_factor,
        #         padding=0,  # self.scale_factor // 2,
        #         output_padding=output_padding,
        #     ),  # h/4 x w/4 -> h x w
        #     nn.Sigmoid(),
        # )

    def forward(self, x, return_reconstruction=True):
        y = self.encoder(x)  # (B, embed_dim, h/scale, w/scale)
        embeddings = y.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)

        if return_reconstruction:
            y = self.decoder(y)  # (B, in_channels, h, w)
            return embeddings, y

        return embeddings

    def decode(self, embeddings):
        embeddings = embeddings.transpose(1, 2).reshape(-1, self.embed_dim, self.h_scaled, self.w_scaled)
        return self.decoder(embeddings)


def try_obs_encoder():
    h, w = 63, 63
    embed_dim = 64
    max_time_steps = 100

    obs_encoder = ObsEncoder(in_channels=3, embed_dim=embed_dim, h=h, w=w)
    time_encoder = TimeStepEncoder(embed_dim=embed_dim, max_time_steps=max_time_steps)

    obs = torch.randn(2, 3, h, w)
    embeddings, reconstruction = obs_encoder(obs, return_reconstruction=True)
    print(embeddings.shape, reconstruction.shape)

    # obs_reconstructed = obs_encoder.decode(embeddings)
    # print(obs_reconstructed.shape)

    # embeddings = time_encoder(embeddings, timestep=10)
    # print(embeddings.shape)


if __name__ == "__main__":
    try_obs_encoder()
