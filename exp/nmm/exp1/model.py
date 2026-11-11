from __future__ import annotations

import torch
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
            "rewards": ["reward_won", "reward_lost", "reward_killed_opponent", "reward_got_killed"]
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
        attention = self.mha(q, k, v, key_padding_mask=padding_mask, attn_mask=attention_mask, need_weights=False)
        return embeddings + self.dropout(attention)  # [B, T, D]


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        max_temporal_sequence_length: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_temporal_sequence_length = max_temporal_sequence_length

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

    def forward(
        self,
        embeddings: torch.Tensor,
        state_padding_mask: torch.Tensor,
        temporal_padding_mask: torch.Tensor,
        temporal_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        embeddings_context: [B, T+1, S, D]
        state_padding_mask: [B, T+1, S]. True where the element is padded.
        temporal_padding_mask: [B, T+1]. True where the element is padded.
        temporal_attention_mask: [T+1, T+1]. True where attention is not allowed.

        - Rearrange the inputs to work with the state attention layer.
        - Forward the state attention layer.
        - Rearrange the inputs to work with the temporal attention layer.
        - Forward the temporal attention layer.
        - Rearrange the inputs to work with the MLP layer.
        - Forward the MLP layer.
        """
        B, _, S, D = embeddings.shape

        rearranged = rearrange(embeddings, "B T S D -> (B T) S D")  # [(B T+1) S D]
        state_padding_mask = rearrange(state_padding_mask, "B T S -> (B T) S")
        out_state = self.state_attention(rearranged, state_padding_mask)  # [(B, T+1), S, D]

        rearranged = rearrange(out_state, "(B T) S D -> (B S) T D", B=B)  # [B*S, T+1, D]
        out_temporal = self.temporal_attention(rearranged, temporal_padding_mask, temporal_attention_mask)
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
                    self.max_temporal_sequence_length,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)

        # Contiguous id ranges (vocab groups are laid out contiguously in i2s).
        pawn_ids = self.vocab.ids("pawns")
        self.pawn_lo, self.pawn_hi = min(pawn_ids), max(pawn_ids)
        position_ids = self.vocab.ids("positions")
        self.pos_lo, self.pos_hi = min(position_ids), max(position_ids)
        self.padding_id = self.vocab.s2i["padding"]
        self.action_taken_id = self.vocab.s2i["action_taken"]
        self.next_position_id = self.vocab.s2i["next_position"]

    def forward(self, context_tokens: torch.Tensor, prompt_tokens: torch.Tensor):
        """
        context_tokens: [B, T, S], int in [0, vocab_size]
        prompt_tokens: [B, S], int in [0, vocab_size]

        B: Batch size
        T: Temporal sequence length
        S: State tokens length: 30.
            24 (positons of pawns)
            + 1 (phase)
            + 1 (action_type)
            + 2 (kills)
            + 1 (player)
            + 1 (action_taken or padding in the last state)
        D: Embedding dim

        - Pawn tokens get added with positional embeddings.
        - Action location gets added with action_taken embedding.
        - Context tokens attend to themselves only.
        - Prompt tokens attend to all tokens.

        Steps:
        - First get all the embeddings from tokens.
        - Add all positional embeddings
        - Add all action_taken embeddings
        - Add all next_position embeddings
        - Add temporal encodings
        - Create a mask for padding
        - Forward the transformer
        - For the output prompt_tokens, forward to respective heads.
        - Return the outputs of the final heads.
        """
        b, t, s = context_tokens.shape
        embeddings_context = self.grab_embeddings(context_tokens)  # (B, T, S, D)
        embeddings_prompt = self.grab_embeddings(prompt_tokens)  # (B, S, D)

        embeddings_context = self.add_pawn_encodings(embeddings_context, context_tokens)  # (B, T, S, D)
        embeddings_context = self.add_action_taken_encodings(embeddings_context, context_tokens)  # (B, T, S, D)
        embeddings_context = self.add_temporal_encodings(embeddings_context)  # (B, T, S, D)
        embeddings_prompt = self.add_next_position_encodings(embeddings_prompt, prompt_tokens)  # (B, S, D)

        embeddings = self.join_embeddings(embeddings_context, embeddings_prompt)  # (B, T+1, S, D)

        state_padding_mask, temporal_padding_mask = self.get_state_padding_mask(context_tokens, prompt_tokens)  # (B, T+1, S), (B, T+1)
        temporal_attention_mask = self.get_temporal_attention_mask(t + 1)  # (T+1, T+1)

        for block in self.transformer_blocks:
            embeddings = block(embeddings, state_padding_mask, temporal_padding_mask, temporal_attention_mask)
        embeddings = self.final_norm(embeddings)

    def get_temporal_attention_mask(self, sequence_length: int) -> torch.Tensor:
        """
        sequence_length: T+1, int
        Returns: [T+1, T+1]

        The first T attend only to themselves.
        The last attends to all.
        """

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

    def grab_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [*, N], int in [0, vocab_size]
        Returns: [*, N, D]

        * can be any number of dimensions.
        For each int index is replaced with it's embedding from an embedding layer.
        """
        return self.embeddings(tokens)  # (*, N, D)

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

    def get_padding_masks(
        self, tokens_context: torch.Tensor, tokens_prompt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        tokens_context: [B, T, S]
        tokens_prompt: [B, S]

        Returns: [B, T+1, S], [B, T+1]

        Return a bool tensor with False where the token is padding from `vocab.vocab_names["padding"]`
        """
        tokens_prompt = rearrange(tokens_prompt, "B S -> B 1 S")
        tokens = torch.cat([tokens_context, tokens_prompt], dim=1)  # (B, T+1, S)

        state_padding_mask = tokens != self.padding_id

        temporal_tokens = tokens[:, :, 0]  # [B, T+1]
        temporal_padding_mask = temporal_tokens != self.padding_id

        return state_padding_mask, temporal_padding_mask


def try_vocab():
    vocab = Vocab()
    for i, s in enumerate(vocab.i2s):
        print(f"{i}\t{s}")


def try_model():
    vocab = Vocab()

    board0 = [vocab.s2i["empty_pawn"]] * 24

    board1 = board0.copy()
    board1[0] = vocab.s2i["opponent_pawn"]

    board2 = board1.copy()
    board2[0] = vocab.s2i["player_pawn"]
    board2[1] = vocab.s2i["opponent_pawn"]

    position_tokens = [vocab.s2i[f"position_{i}"] for i in range(24)]

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

    prompt_tokens = (
        vocab.ids(
            [
                "next_phase",
                "next_action_type",
                "next_kill_count_player",
                "next_kill_count_opponent",
                "next_player",
                "pred_reward",
            ]
        )
        + position_tokens
    )
    print(prompt_tokens)
    print(vocab.names(prompt_tokens))

    context_tokens_tensor = torch.tensor(context_tokens).unsqueeze(0)  # (1, T, S)
    prompt_tokens_tensor = torch.tensor(prompt_tokens).unsqueeze(0)  # (1, O)

    print(context_tokens_tensor.shape)
    print(prompt_tokens_tensor.shape)

    model = TransformerModel(
        embedding_dim=64,
        hidden_size=256,
        num_layers=6,
        num_heads=4,
        dropout=0.1,
        max_temporal_sequence_length=4,
    )

    model(context_tokens_tensor, prompt_tokens_tensor)


if __name__ == "__main__":
    try_vocab()
    try_model()
