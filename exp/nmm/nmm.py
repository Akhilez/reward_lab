from __future__ import annotations

from typing import Any
import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo import AECEnv


def env(render_mode=None):
    return NineMensMorrisEnv(render_mode=render_mode)


p2n = {'player_0': 1, 'player_1': -1}
p2o = {'player_0': 'player_1', 'player_1': 'player_0'}

rows = {  # Indices for 3 in a row
    0: [(1, 2), (9, 21)],
    1: [(0, 2), (4, 7)],
    2: [(0, 1), (14, 23)],
    3: [(4, 5), (10, 18)],
    4: [(1, 7), (3, 5)],
    5: [(3, 4), (13, 20)],
    6: [(7, 8), (11, 15)],
    7: [(6, 8), (1, 4)],
    8: [(6, 7), (12, 17)],
    9: [(0, 21), (10, 11)],
    10: [(9, 11), (3, 18)],
    11: [(9, 10), (6, 15)],
    12: [(8, 17), (13, 14)],
    13: [(12, 14), (5, 20)],
    14: [(2, 23), (12, 13)],
    15: [(16, 17), (6, 11)],
    16: [(15, 17), (19, 22)],
    17: [(15, 16), (8, 12)],
    18: [(3, 10), (19, 20)],
    19: [(18, 20), (16, 22)],
    20: [(18, 19), (5, 13)],
    21: [(0, 9), (22, 23)],
    22: [(21, 23), (16, 19)],
    23: [(21, 22), (2, 14)],
}
movable = {
    0: (1, 9),
    1: (0, 2, 4),
    2: (1, 14),
    3: (4, 10),
    4: (1, 3, 5, 7),
    5: (4, 13),
    6: (7, 11),
    7: (4, 6, 8),
    8: (7, 12),
    9: (0, 21, 10),
    10: (3, 9, 11, 18),
    11: (10, 6, 15),
    12: (8, 17, 13),
    13: (12, 5, 14, 20),
    14: (2, 13, 23),
    15: (11, 16),
    16: (15, 17, 19),
    17: (16, 12),
    18: (10, 19),
    19: (16, 18, 20, 22),
    20: (19, 13),
    21: (9, 22),
    22: (21, 23, 19),
    23: (22, 14),
}
flat_to_2d = {
    0: (0, 0),
    1: (0, 3),
    2: (0, 6),
    3: (1, 1),
    4: (1, 3),
    5: (1, 5),
    6: (2, 2),
    7: (2, 3),
    8: (2, 4),
    9: (3, 0),
    10: (3, 1),
    11: (3, 2),
    12: (3, 4),
    13: (3, 5),
    14: (3, 6),
    15: (4, 2),
    16: (4, 3),
    17: (4, 4),
    18: (5, 1),
    19: (5, 3),
    20: (5, 5),
    21: (6, 0),
    22: (6, 3),
    23: (6, 6),
}


class ActionType:
    PLACE = 0  # Phase 1: Placing piece for the first time.
    LIFT = 1  # Phase 2: Lifting the piece to move it.
    DROP = 2  # Phase 2: Destination location for moving a piece.
    KILL = 3  # Phase 1 & 2: Removing opponent's piece.


class NineMensMorrisEnv(AECEnv):
    metadata = {
        'render_modes': ['human', 'ansi'],
        'name': 'nmm',
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.possible_agents: list[str] = ['player_0', 'player_1']
        self.action_spaces: dict[str, Discrete] = {agent: Discrete(24) for agent in self.possible_agents}
        self.observation_spaces: dict[str, Box] = {agent: Box(0, 4, (24,)) for agent in self.possible_agents}
        self.render_mode: str = render_mode
        self.max_steps: int = 1000

        self.agents: list[str] = self.possible_agents[:]
        self.rewards: dict[str, float] = {a: 0 for a in self.agents}
        self.terminations: dict[str, bool] = {agent: False for agent in self.agents}
        self.truncations: dict[str, bool] = {agent: False for agent in self.agents}
        self.kill_count: dict[str, int] = {agent: 0 for agent in self.agents}
        self.infos: dict[str, dict] = {agent: {} for agent in self.agents}
        self.agent_selection: str = 'player_0'
        self.board: np.ndarray = np.zeros(24, dtype=int)
        self.lifted: int | None = None
        self.must_kill: bool = False
        self.action_masks: dict[str, np.ndarray] = {a: np.ones(24) for a in self.agents}
        self.mills_placed: dict[str, int] = {a: 0 for a in self.agents}
        self.step_ = 0

    def step(self, action: int):
        p = self.agent_selection
        o = p2o[p]
        self.rewards = {p: 0, o: 0}

        # If taken too long to finish, truncate. Draw.
        self.step_ += 1
        if self.step_ > self.max_steps:
            self.truncations = {p: True, o: True}
            return

        # If done, raise exception.
        if self.terminations[p]:
            raise Exception('Already terminated')
        if self.truncations[p]:
            raise Exception('Already truncated')

        # If illegal action, raise exception.
        if self.action_masks[p][action] == 0:
            raise Exception(f'Action {action} is illegal.')

        # ============ KILLING ============

        if self.must_kill:
            self.must_kill = False
            self.board[action] = 0
            self.kill_count[o] += 1
            self.rewards = {p: 0.1, o: 0}

            # Only 2 pawns left for opponent
            if self.kill_count[o] >= 7:
                self.terminations = {p: True, o: True}
                self.rewards = {p: 1, o: -1}

            self.switch(p, o)

        # =========== PHASE 1 PLACING ================

        elif self.mills_placed[p] < 9:
            self.mills_placed[p] += 1
            self.place_mill(action, p, o)

        # =========== PHASE 2 LIFTING =============

        elif self.lifted is None:
            # Lift the piece. Store its location in self.lifted. Don't switch the player.
            self.lifted = action
            self.action_masks[p] = self._get_movable_to_positions(action)

        # ============ PHASE 2 PLACING ==============

        else:
            self.board[self.lifted] = 0
            self.lifted = None
            self.place_mill(action, p, o)

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self.__init__(self.render_mode)

    def observe(self, agent: str) -> dict[str, Any]:
        p = p2n[agent]
        o = p2n[p2o[agent]]

        obs = np.zeros(24, dtype=int)

        # All current player's positions are 1
        obs[self.board == p] = 1

        # All opponent's positions are 2
        obs[self.board == o] = 2

        # If any lifted position, it's 3
        if self.lifted is not None:
            obs[self.lifted] = 3

        return {
            'observation': obs,
            'action_mask': self.action_masks[agent].astype(int),
            'action_type': self.get_action_type(agent),
            "phase": 0 if self.is_phase_1(agent) else 1,
            "kills": self.kill_count,
        }

    def render(self):
        symbol = '•XO'
        s = [symbol[x] for x in self.board]
        s = f"""
{s[0]}-----{s[1]}-----{s[2]}
| {s[3]}---{s[4]}---{s[5]} |
| | {s[6]}-{s[7]}-{s[8]} | |
{s[9]}-{s[10]}-{s[11]}   {s[12]}-{s[13]}-{s[14]}
| | {s[15]}-{s[16]}-{s[17]} | |
| {s[18]}---{s[19]}---{s[20]} |
{s[21]}-----{s[22]}-----{s[23]}"""
        s += f'\nPlayer: {self.agent_selection}. Mills placed: {self.mills_placed[self.agent_selection]}, lifted: {self.lifted}'

        if self.render_mode == 'human':
            print(s)
            return None
        else:
            return s

    def place_mill(self, action: int, p: str, o: str) -> None:
        # Simply place the mill.
        self.board[action] = p2n[p]

        # If killable, don't switch player. Set proper action masks. Set must_kill = True.
        if self._made_3_in_a_row(action):
            # Opponent positions
            self.action_masks[p] = (self.board == p2n[o]) * 1
            self.must_kill = True
        # Else, switch player. Set proper action masks
        else:
            self.switch(p, o)

    def switch(self, p: str, o: str) -> None:
        self.agent_selection = o
        if self.is_phase_1(o):
            self.action_masks[o] = (self.board == 0) * 1
        else:  # Phase 2
            mask = self._get_movable_from_positions(o)
            self.action_masks[o] = mask
            if sum(mask) == 0:
                self.infos[o]['locked_out'] = True
                self.terminations = {p: True, o: True}
                self.rewards = {p: 1, o: -1}

    def get_action_type(self, agent: str) -> int:
        if self.must_kill:
            return ActionType.KILL
        if self.is_phase_1(agent):
            return ActionType.PLACE
        if self.lifted is not None:
            return ActionType.DROP
        return ActionType.LIFT

    def _made_3_in_a_row(self, action: int) -> bool:
        p = p2n[self.agent_selection]
        for a, b in rows[action]:
            if self.board[a] == p and self.board[b] == p:
                return True
        return False

    def _get_movable_to_positions(self, action: int) -> np.ndarray:
        m = movable[action]
        return np.array([1 if i in m and self.board[i] == 0 else 0 for i in range(24)])

    def _get_movable_from_positions(self, p: str) -> np.ndarray:
        # For each agent position, if it's movable, get 1 for that position.
        mask = np.zeros(24, dtype=int)
        for pos in np.nonzero(self.board == p2n[p])[0]:
            if sum(self._get_movable_to_positions(pos)) > 0:
                mask[pos] = 1
        return mask

    def is_phase_1(self, agent: str):
        return self.mills_placed[agent] < 9

    def state(self):
        s = self.board.astype(int).tolist()
        s.append(self.agent_selection)
        s.append(self.get_action_type(self.agent_selection))
        return tuple(s)
