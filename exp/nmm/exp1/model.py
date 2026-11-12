from __future__ import annotations

import math

import torch
from einops import repeat
from functorch.einops import rearrange
from torch import nn


class Vocab:
    def __init__(self):
        self.vocab_names = {
            "padding": ["padding"],
            "pawns": ["empty_pawn", "player_pawn", "opponent_pawn", "lifted_pawn"],
            "phases": ["phase_1", "phase_2"],
            "action_types": ["action_place", "action_lift", "action_drop", "action_kill"],
            "kill_count_player": [f"pawns_dead_player_{x}" for x in range(8)],
            "kill_count_opponent": [f"pawns_dead_opponent_{x}" for x in range(8)],
            "player": ["current_player", "opponent_player"],
            "action_taken": ["action_taken"],  # Encoding
            "positions": [f"position_{x}" for x in range(24)],  # Encoding
            "pred_action": ["pred_action"],
            "next_position": ["next_position"],  # Encoding
            "next_player": ["next_player"],
            "next_phase": ["next_phase"],
            "next_action_type": ["next_action_type"],
            "next_kill_count_player": ["next_kill_count_player"],
            "next_kill_count_opponent": ["next_kill_count_opponent"],
            "pred_reward": ["pred_reward"],
            "rewards": ["reward_won", "reward_lost", "reward_killed_opponent", "reward_got_killed"],
        }

        self.i2s = [s for k, names in self.vocab_names.items() for s in self.vocab_names[k]]
        self.s2i = {s: i for i, s in enumerate(self.i2s)}
        self.size = len(self.i2s)

    def ids(self, group: str | list) -> list[int]:
        """Token ids of all names in a vocab group, or the list"""
        if isinstance(group, str):
            return [self.s2i[s] for s in self.vocab_names[group]]
        elif isinstance(group, list):
            return [self.s2i[s] for s in group]
        else:
            raise Exception(f"Unknown vocab group: {group}")

    def names(self, ids: list[int]) -> list[str]:
        return [self.i2s[id] for id in ids]


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.norm = nn.LayerNorm(embedding_dim)
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.mha = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        embeddings: [B, T, D]
        padding_mask: [B, T]. True where the element is padded.
        attention_mask: [T, T]. True where attention is not allowed.
        Returns [B, T, D]

        - Norm
        - QKV
        - MHA
        """
        normed = self.norm(embeddings)  # [B, T, D]
        qkv = self.qkv(normed)  # [B, T, D*3]
        q, k, v = qkv.chunk(3, dim=-1)  # 3 * [B, T, D]
        attention, _ = self.mha(q, k, v, key_padding_mask=padding_mask, attn_mask=attention_mask, need_weights=False)
        return embeddings + self.dropout(attention)  # [B, T, D]


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.state_attention = AttentionLayer(embedding_dim, num_heads, dropout)
        self.temporal_attention = AttentionLayer(embedding_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, embeddings: torch.Tensor, temporal_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        embeddings_context: [B, T, S, D]
        state_padding_mask: [B, T, S]. True where the element is padded.
        temporal_padding_mask: [B, T]. True where the element is padded.
        temporal_attention_mask: [T, T]. True where attention is not allowed.

        - Rearrange the inputs to work with the state attention layer.
        - Forward the state attention layer.
        - Rearrange the inputs to work with the temporal attention layer.
        - Forward the temporal attention layer.
        - Rearrange the inputs to work with the MLP layer.
        - Forward the MLP layer.
        """
        B, _, S, D = embeddings.shape

        rearranged = rearrange(embeddings, "B T S D -> (B T) S D")  # [(B T) S D]
        out_state = self.state_attention(rearranged)  # [(B, T), S, D]

        rearranged = rearrange(out_state, "(B T) S D -> (B S) T D", B=B)  # [B*S, T, D]
        temporal_padding_mask = repeat(temporal_padding_mask, "B T -> (B S) T", S=S)
        out_temporal = self.temporal_attention(rearranged, temporal_padding_mask)  # [B*S, T, D]
        out_temporal = rearrange(out_temporal, "(B S) T D -> B T S D", B=B)  # [B, T, S, D]

        out_feed_forward = self.feed_forward(out_temporal)  # [B, T, S, D]

        return out_feed_forward


class TransformerModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_temporal_sequence_length: int,
    ):
        super().__init__()
        self.vocab = Vocab()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_temporal_sequence_length = max_temporal_sequence_length

        self.embeddings = nn.Embedding(self.vocab.size, self.embedding_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    embedding_dim,
                    self.hidden_size,
                    self.num_heads,
                    self.dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)

        self.action_head = nn.Linear(embedding_dim, 24)

        self.next_phase_head = nn.Linear(embedding_dim, 2)
        self.next_action_type_head = nn.Linear(embedding_dim, 4)
        self.next_kill_count_player_head = nn.Linear(embedding_dim, 8)
        self.next_kill_count_opponent_head = nn.Linear(embedding_dim, 8)
        self.next_player_head = nn.Linear(embedding_dim, 2)
        self.reward_head = nn.Linear(embedding_dim, 1)
        self.next_board_head = nn.Linear(embedding_dim, 4)

        # Contiguous id ranges (vocab groups are laid out contiguously in i2s).
        pawn_ids = self.vocab.ids("pawns")
        self.pawn_lo, self.pawn_hi = min(pawn_ids), max(pawn_ids)
        position_ids = self.vocab.ids("positions")
        self.pos_lo, self.pos_hi = min(position_ids), max(position_ids)
        self.padding_id = self.vocab.s2i["padding"]
        self.action_taken_id = self.vocab.s2i["action_taken"]
        self.next_position_id = self.vocab.s2i["next_position"]
        self.pred_action_id = self.vocab.s2i["pred_action"]

        # Sinusoidal table for positions [0, max_T). Non-persistent: derived, so kept out of state_dict.
        self.register_buffer(
            "temporal_encodings",
            self.build_sinusoidal_encodings(self.max_temporal_sequence_length, self.embedding_dim),
            persistent=False,
        )  # (max_T, D)

    def forward(self, context_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        context_tokens: [B, T, S], int in [0, vocab_size]

        B: Batch size
        T: Temporal sequence length
        S: State tokens length: 30.
            24 (positons of pawns)
            + 1 (phase)
            + 1 (action_type)
            + 2 (kills)
            + 1 (player)
            + 1 (action_taken or pred_action in the last state)
        D: Embedding dim

        - Pawn tokens get added with positional embeddings.
        - Action location gets added with action_taken embedding.
        - Context tokens attend to themselves only.
        - Prompt tokens attend to all tokens.
        """
        b, t, s = context_tokens.shape
        device = context_tokens.device

        embeddings_context = self.embeddings(context_tokens)  # (B, T, S, D)
        embeddings_context = self.add_pawn_encodings(embeddings_context, context_tokens)  # (B, T, S, D)
        embeddings_context = self.add_action_taken_encodings(embeddings_context, context_tokens)  # (B, T, S, D)
        embeddings_context = self.add_temporal_encodings(embeddings_context)  # (B, T, S, D)

        has_action_taken = self._check_if_action_taken(context_tokens)
        if has_action_taken:
            # Setup prompt tokens
            prompt_tokens = self.get_prompt_tokens(b, device)  # (B, S), int in [0, vocab_size]
            assert prompt_tokens.shape == (b, s)

            embeddings_prompt = self.embeddings(prompt_tokens)  # (B, S, D)
            embeddings_prompt = self.add_next_position_encodings(embeddings_prompt, prompt_tokens)  # (B, S, D)

            embeddings = self.join_embeddings(embeddings_context, embeddings_prompt)  # (B, T+1, S, D)
        else:
            embeddings = embeddings_context  # (B, T, S, D)

        padding_mask = self.get_padding_mask(context_tokens, has_action_taken)  # (B, T')

        for block in self.transformer_blocks:
            embeddings = block(embeddings, padding_mask)  # (B, T', S, D)
        embeddings = self.final_norm(embeddings)  # (B, T', S, D)

        if has_action_taken:
            next_phase = embeddings[:, t, 0]  # [B, D]
            next_action_type = embeddings[:, t, 1]  # [B, D]
            next_kill_count_player = embeddings[:, t, 2]  # [B, D]
            next_kill_count_opponent = embeddings[:, t, 3]  # [B, D]
            next_player = embeddings[:, t, 4]  # [B, D]
            reward = embeddings[:, t, 5]  # [B, D]
            next_board = embeddings[:, t, 6:]  # [B, 24, D]

            out = {
                "next_phase": self.next_phase_head(next_phase),  # [B, 2]
                "next_action_type": self.next_action_type_head(next_action_type),  # [B, 4]
                "next_kill_count_player": self.next_kill_count_player_head(next_kill_count_player),  # [B, 8]
                "next_kill_count_opponent": self.next_kill_count_opponent_head(next_kill_count_opponent),  # [B, 8]
                "next_player": self.next_player_head(next_player),  # [B, 2]
                "pred_reward": self.reward_head(reward),  # [B, 1]
                "next_board": self.next_board_head(next_board),  # [B, 24, 4]
            }
            return out
        else:
            # Get the last token's output embeddings for action prediction
            action_embeddings = embeddings[:, -1, -1]  # [B, D]
            return {"pred_action": self.action_head(action_embeddings)}  # [B, 24]

    def _check_if_action_taken(self, context_tokens: torch.Tensor) -> bool:
        # Figure out what to predict: action or next state
        last_token = context_tokens[:, -1, -1]  # int
        has_action_taken = last_token.ne(self.pred_action_id)  # If last token is "pred_action", no action taken yet.
        # Make sure all true or all false.
        assert has_action_taken.all() or not has_action_taken.any()
        has_action_taken = bool(has_action_taken[0])
        return has_action_taken

    @staticmethod
    def build_sinusoidal_encodings(length: int, dim: int) -> torch.Tensor:
        """
        Returns: [length, dim], row p is the sinusoidal encoding of position p.
        """
        positions = torch.arange(length, dtype=torch.float32)  # (L,)
        half = (dim + 1) // 2
        # exp(log(10000) * [0-D/2] * 2 / D)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) * 2 / dim)  # (ceil(D/2),)
        angles = positions[:, None] * freqs[None, :]  # (L, ceil(D/2))

        encodings = torch.zeros(length, dim)  # (L, D)
        encodings[:, 0::2] = torch.sin(angles)
        encodings[:, 1::2] = torch.cos(angles[:, : dim // 2])
        return encodings

    def add_temporal_encodings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, T, S, D]
        Returns: [B, T, S, D]

        If T = 4, get sinusoidal encodings for [3, 2, 1, 0].
        Note that the 0th position is at the very end, not the first.

        The positional embeddings are of the shape [T, D]
        Rearrange it to [1, T, 1, D]
        Add to the embeddings and return
        """
        t = embeddings.shape[1]
        assert t <= self.max_temporal_sequence_length, f"T={t} > max_temporal_sequence_length"
        encodings = self.temporal_encodings[:t].flip(0)  # (T, D), positions [T-1, ..., 0]

        encodings = rearrange(encodings, "T D -> 1 T 1 D").to(embeddings.dtype)  # (1, T, 1, D)
        return embeddings + encodings  # (B, T, S, D)

    def join_embeddings(self, embeddings_context: torch.Tensor, embeddings_prompt: torch.Tensor) -> torch.Tensor:
        """
        embeddings_context: [B, T, S, D]
        embeddings_prompt: [B, S, D]

        Returns: [B, T+1, S, D]

        Basically concat
        """
        embeddings_prompt = rearrange(embeddings_prompt, "B S D -> B 1 S D")  # (B, 1, S, D)
        embeddings = torch.cat([embeddings_context, embeddings_prompt], dim=1)  # (B, T+1, S, D)
        return embeddings

    def add_pawn_encodings(self, embeddings: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, T, S, D]
        tokens: [B, T, S]

        Returns: [B, T, S, D]

        Some of the tokens in S are pawn tokens.
        The pawn token indices can be found from `vocab.vocab_names["pawns"]`
        There will be 24 of them.
        For each of the 24 locations, add the corresponding position token from `vocab.vocab_names["positions"]`
        """
        is_pawn = (tokens >= self.pawn_lo) & (tokens <= self.pawn_hi)  # (B, T, S)

        # k-th pawn token in a state -> position_k
        pawn_index = (is_pawn.long().cumsum(dim=-1) - 1).clamp(min=0)  # (B, T, S)
        position_ids = (self.pos_lo + pawn_index).clamp(max=self.pos_hi)  # (B, T, S)

        is_pawn = is_pawn.unsqueeze(-1).to(embeddings.dtype)  # (B, T, S, 1)
        position_embeddings = self.embeddings(position_ids)  # (B, T, S, D)
        position_embeddings = position_embeddings * is_pawn  # (B, T, S, D)

        return embeddings + position_embeddings  # (B, T, S, D)

    def add_action_taken_encodings(self, embeddings: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, T, S, D]
        tokens: [B, T, S]

        Returns: [B, T, S, D]

        One of the tokens in S is a position token from `vocab.vocab_names["positions"]`.
        Add the action taken embedding to it from `vocab.vocab_names["action_taken"]`.
        """
        is_position = (tokens >= self.pos_lo) & (tokens <= self.pos_hi)  # (B, T, S)
        is_position = is_position.unsqueeze(-1).to(embeddings.dtype)  # (B, T, S, 1)

        action_embedding = self.embeddings.weight[self.action_taken_id]  # (D,)
        action_embedding = action_embedding * is_position  # (B, T, S, D)

        return embeddings + action_embedding  # (B, T, S, D)

    def add_next_position_encodings(self, embeddings: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, S, D]
        tokens: [B, S]

        Returns: [B, S, D]

        If there are any position tokens in O from `vocab.vocab_names["positions"]`,
        add the next_position embedding to it from `vocab.vocab_names["next_position"]`.
        """
        is_position = (tokens >= self.pos_lo) & (tokens <= self.pos_hi)  # (B, S)
        is_position = is_position.unsqueeze(-1).to(embeddings.dtype)  # (B, S, 1)

        next_position_embedding = self.embeddings.weight[self.next_position_id]  # (D,)
        next_position_embedding = next_position_embedding * is_position  # (B, S, D)

        return embeddings + next_position_embedding  # (B, S, D)

    def get_padding_mask(self, tokens_context: torch.Tensor, has_prompt: bool) -> torch.Tensor:
        """
        tokens_context: [B, T, S]
        has_prompt: bool

        Returns: [B, T']

        Return a bool tensor with True where the fist token is padding from `vocab.vocab_names["padding"]`
        """
        b, t, s = tokens_context.shape
        t_updated = t + 1 if has_prompt else t
        padding_mask = torch.zeros((b, t_updated), device=tokens_context.device, dtype=torch.bool)
        first_tokens = tokens_context[:, :, 0]  # [B, T]
        context_mask = first_tokens == self.padding_id  # [B, T]
        padding_mask[:, :t] = context_mask
        return padding_mask

    def get_prompt_tokens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        token_strs = [
            "next_phase",
            "next_action_type",
            "next_kill_count_player",
            "next_kill_count_opponent",
            "next_player",
            "pred_reward",
        ]
        token_strs += [f"position_{i}" for i in range(24)]

        token_indices = self.vocab.ids(token_strs)
        token_indices = torch.tensor(token_indices, device=device)  # (S,)
        token_indices = repeat(token_indices, "s -> b s", b=batch_size)  # (B, S)

        return token_indices


def try_vocab():
    vocab = Vocab()
    for i, s in enumerate(vocab.i2s):
        print(f"{i}\t{s}")


def try_model():
    vocab = Vocab()
    device = "mps"

    board0 = [vocab.s2i["empty_pawn"]] * 24

    board1 = board0.copy()
    board1[0] = vocab.s2i["opponent_pawn"]

    board2 = board1.copy()
    board2[0] = vocab.s2i["player_pawn"]
    board2[1] = vocab.s2i["opponent_pawn"]

    context_tokens = [
        board0
        + vocab.ids(
            [
                "phase_1",
                "action_place",
                "pawns_dead_player_0",
                "pawns_dead_opponent_0",
                "current_player",
                "position_0",
            ]
        ),
        board1
        + vocab.ids(
            [
                "phase_1",
                "action_place",
                "pawns_dead_player_0",
                "pawns_dead_opponent_0",
                "opponent_player",
                "position_1",
            ]
        ),
        board2
        + vocab.ids(
            [
                "phase_1",
                "action_place",
                "pawns_dead_player_0",
                "pawns_dead_opponent_0",
                "current_player",
                "position_2",
            ]
        ),
    ]
    for t in range(len(context_tokens)):
        print(context_tokens[t])
        print(vocab.names(context_tokens[t]))

    context_tokens_tensor = torch.tensor(context_tokens).unsqueeze(0).to(device)  # (1, T, S)
    context_tokens_tensor = repeat(context_tokens_tensor, "1 t s -> b t s", b=4)
    print("context_tokens shape: ", context_tokens_tensor.shape)

    model = TransformerModel(
        embedding_dim=64,
        hidden_size=256,
        num_layers=6,
        num_heads=4,
        dropout=0.1,
        max_temporal_sequence_length=4,
    ).to(device)

    out = model(context_tokens_tensor)
    for k, v in out.items():
        print(k, v.shape)

    context_tokens[-1][-1] = vocab.s2i["pred_action"]
    context_tokens_tensor = torch.tensor(context_tokens).unsqueeze(0).to(device)  # (1, T, S)
    context_tokens_tensor = repeat(context_tokens_tensor, "1 t s -> b t s", b=4)

    for t in range(len(context_tokens)):
        print(context_tokens[t])
        print(vocab.names(context_tokens[t]))
    print("context_tokens shape: ", context_tokens_tensor.shape)

    out = model(context_tokens_tensor)
    for k, v in out.items():
        print(k, v.shape)


if __name__ == "__main__":
    try_vocab()
    try_model()
