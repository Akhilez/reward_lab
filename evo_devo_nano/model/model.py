from torch import nn

from evo_devo_nano.model.obs_encoder import ObsEncoder, TimeStepEncoder
from evo_devo_nano.model.transformer import VanillaTransformer
from evo_devo_nano.model.vocab import Vocabulary


class CraftaxModel(nn.Module):
    def __init__(self, h, w, n_actions):
        super(CraftaxModel, self).__init__()
        self.n_actions = n_actions

        self.embed_dim = 256
        self.max_time_steps = 1000
        self.n_heads = 4
        self.n_layers = 6
        self.max_seq_len = 1024
        self.n_thinking_tokens = 100

        self.vocab = Vocabulary(n_actions=self.n_actions, n_thinking=self.n_thinking_tokens)
        self.embeddings = nn.Embedding(self.vocab.size, self.embed_dim)
        self.obs_encoder = ObsEncoder(in_channels=3, embed_dim=self.embed_dim, h=h, w=w)
        self.time_encoder = TimeStepEncoder(embed_dim=self.embed_dim, max_time_steps=self.max_time_steps)
        self.transformer = VanillaTransformer(
            embed_dim=self.embed_dim,
            num_heads=self.n_heads,
            num_layers=self.n_layers,
            max_len=self.max_seq_len,
        )
        self.classifier = nn.Linear(self.embed_dim, self.vocab.size, bias=False)
        # self.classifier.weight = self.embeddings.weight

    def forward(self, x):
        pass


