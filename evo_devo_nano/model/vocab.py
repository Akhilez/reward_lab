"""
- There are n_actions discrete actions.
- State, time emb and reward emb are continuous, so no vocab.
- n_special=7 special tokens for:
    - action
    - next_state
    - next_latent_obs
    - reward
    - think_start
    - think_end
    - done
- n_thinking tokens

total: n_actions + n_special + n_thinking
"""

import torch


class Vocabulary:
    def __init__(self, n_actions: int, n_thinking: int):
        self.n_actions = n_actions
        self.n_thinking = n_thinking

        self.special_tokens = [
            "action_key",
            "next_state_key",
            "next_latent_obs_key",
            "reward_key",
            "think_start",
            "think_end",
            "done",
        ]

        self.n_special = len(self.special_tokens)

        self.size = self.n_actions + self.n_special + self.n_thinking

        self.actions_mask = torch.tensor([True] * self.n_actions + [False] * (self.n_special + self.n_thinking))
        self.thinking_mask = torch.tensor([False] * (self.n_actions + self.n_special) + [True] * self.n_thinking)

        self.action_indices = self.actions_mask.nonzero(as_tuple=True)[0]
        self.thinking_indices = self.thinking_mask.nonzero(as_tuple=True)[0]

        self.idx_to_str = [f"action_{i}" for i in range(self.n_actions)] + self.special_tokens + [f"thinking_{i}" for i in range(self.n_thinking)]
        self.str_to_idx = {s: i for i, s in enumerate(self.idx_to_str)}

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.str_to_idx[item]
        elif isinstance(item, int):
            return self.idx_to_str[item]
        else:
            raise ValueError("item must be str or int")
