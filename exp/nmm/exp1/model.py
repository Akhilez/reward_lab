from __future__ import annotations

import torch
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
        prompt_tokens: [B, O], int in [0, vocab_size]

        B: Batch size
        T: Temporal sequence length
        S: State tokens length: 30.
            24 (positons of pawns)
            + 1 (phase)
            + 1 (action_type)
            + 2 (kills)
            + 1 (player)
            + 1 (action_taken or padding in the last state)
        O: Output tokens (Can be arbitrary)
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
        - Create a mask for padding
        - Forward the transformer
        - For the output prompt_tokens, forward to respective heads.
        - Return the outputs of the final heads.
        """
        B, T, S = context_tokens.shape
        _, O = prompt_tokens.shape
        D = self.embedding_dim

        embeddings_context = self.grab_embeddings(context_tokens)  # (B, T, S, D)
        assert embeddings_context.shape == (B, T, S, D)
        embeddings_prompt = self.grab_embeddings(prompt_tokens)  # (B, O, D)
        assert embeddings_prompt.shape == (B, O, D)

        embeddings_context = self.add_pawn_encodings(embeddings_context, context_tokens)  # (B, T, S, D)
        assert embeddings_context.shape == (B, T, S, D)
        embeddings_context = self.add_action_taken_encodings(embeddings_context, context_tokens)  # (B, T, S, D)
        assert embeddings_context.shape == (B, T, S, D)
        embeddings_prompt = self.add_next_position_encodings(embeddings_prompt, prompt_tokens)  # (B, O, D)
        assert embeddings_prompt.shape == (B, O, D)

        mask_padding = self.get_padding_mask(context_tokens)  # (B, T, S)
        assert mask_padding.shape == (B, T, S)

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
        is_position = is_position.unsqueeze(-1).to(embeddings.dtype)   # (B, T, S, 1)

        action_embedding = self.embeddings.weight[self.action_taken_id]  # (D,)
        action_embedding = action_embedding * is_position  # (B, T, S, D)

        return embeddings + action_embedding  # (B, T, S, D)

    def add_next_position_encodings(self, embeddings: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, O, D]
        tokens: [B, O]

        Returns: [B, O, D]

        If there are any position tokens in O from `vocab.vocab_names["positions"]`,
        add the next_position embedding to it from `vocab.vocab_names["next_position"]`.
        """
        is_position = (tokens >= self.pos_lo) & (tokens <= self.pos_hi)  # (B, O)
        is_position = is_position.unsqueeze(-1).to(embeddings.dtype)  # (B, O, 1)

        next_position_embedding = self.embeddings.weight[self.next_position_id]  # (D,)
        next_position_embedding = next_position_embedding * is_position   # (B, O, D)

        return embeddings + next_position_embedding  # (B, O, D)

    def get_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, T, S]

        Returns: [B, T, S]

        Return a bool tensor with False where the token is padding from `vocab.vocab_names["padding"]`
        """
        return tokens != self.padding_id


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
        board0 + vocab.ids(["phase_1", "action_place", "pawns_dead_player_0", "pawns_dead_opponent_0", "current_player", "position_0"]),
        board1 + vocab.ids(["phase_1", "action_place", "pawns_dead_player_0", "pawns_dead_opponent_0", "opponent_player", "position_1"]),
        board2 + vocab.ids(["phase_1", "action_place", "pawns_dead_player_0", "pawns_dead_opponent_0", "current_player", "position_2"]),
    ]
    for t in range(len(context_tokens)):
        print(context_tokens[t])
        print(vocab.names(context_tokens[t]))

    prompt_tokens = vocab.ids(["next_phase", "next_action_type", "next_kill_count_player", "next_kill_count_opponent", "next_player"]) + position_tokens
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
