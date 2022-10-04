import numpy as np


class RandomPlayer:
    def __call__(self, state, legal_actions_mask) -> int:
        return np.random.choice(len(legal_actions_mask), p=1.0 * legal_actions_mask / legal_actions_mask.sum())


class AttractorPlayer:
    def __call__(self, state, legal_actions_mask) -> int:
        """
        State is a numpy array of shape (6, 7, 7)
        legal_actions_mask is an array of shape (4)

        Returns the action that takes the agent close to the nearest food.

        planes:
        0: Boundary
        1: Terrain
        2: Food Ages
        3: Player Health
        4: Dist b/w center of the map to each point
        5: Previous path of the player health
        """
        agent = np.array(np.where(state[3] > 0)).flatten()
        visited = np.zeros_like(state[0])
        traverse(state, agent, visited)
        attainable_foods = np.array((state[2] * visited).nonzero()).T
        dist = np.array([10 ** 4] * 4)
        for food in attainable_foods:
            for i, d in enumerate(np.array([(0, -1), (-1, 0), (0, 1), (1, 0)])):
                new_position = agent + d
                try:
                    if visited[tuple(new_position)] == 1:
                        distance = np.linalg.norm(new_position - food)
                        dist[i] = min(dist[i], distance)
                except:
                    pass

        # No food found in vicinity
        # if (dist >= 10 ** 4).sum() == len(dist):
        #     if legal_actions_mask[1] == 1:  # Left is legal
        #         visits_on_opposite = state[5][:, agent[1]:].sum()
        #         dist[0] -= visits_on_opposite
        #     if legal_actions_mask[2] == 1:  # Top is legal
        #         visits_on_opposite = state[5][agent[0]:, :].sum()
        #         dist[1] -= visits_on_opposite
        #     if legal_actions_mask[3] == 1:  # Right is legal
        #         visits_on_opposite = state[5][:, :agent[1]].sum()
        #         dist[2] -= visits_on_opposite
        #     if legal_actions_mask[4] == 1:  # Bottom is legal
        #         visits_on_opposite = state[5][:agent[0], :].sum()
        #         dist[3] -= visits_on_opposite

        # Mask out the illegal actions
        dist = np.insert(dist, 0, 10 ** 5)
        dist += (1 - legal_actions_mask) * 10 ** 5

        actions = np.where(dist == np.min(dist))[0]
        action = np.random.choice(actions)
        return action


def test_line():
    agent = np.array((3, 3))
    terrain = np.array([(0, 2), (1, 1), (2, 0), (3, 3)])
    foods = np.array([(1, 0), (6, 6)])
    boundary = np.array([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])
    history = np.array([(4, 3), (4, 2)])

    state = np.zeros((6, 7, 7))

    state[3][tuple(agent)] = 1
    for t in terrain:
        state[1][tuple(t)] = 1
    for t in foods:
        state[2][tuple(t)] = 1
    for t in boundary:
        state[0][tuple(t)] = 1
    for t in history:
        state[5][tuple(t)] = 1
    state[4] = 1

    print(state.argmax(0))
    action = AttractorPlayer()(state, np.array([1, 1, 1, 0, 1]))
    assert action == 3


def traverse(state, position, visited):
    size_threshold_to_jump = 1.5

    # If position is out of bounds
    if (
        position[0] < 0
        or position[1] < 0
        or position[0] >= len(visited)
        or position[1] >= len(visited)
    ):
        return
    # if already visited, return.
    if visited[tuple(position)] != 0:
        return
    # If position is terrain, return.
    if state[1][tuple(position)] == 1 and state[3].sum() < size_threshold_to_jump:
        return
    # If position is boundary, return.
    if state[0][tuple(position)] == 1:
        return

    visited[tuple(position)] = 1
    for d in np.array([(0, -1), (-1, 0), (0, 1), (1, 0)]):
        new_position = np.array(position) + d
        traverse(state, new_position, visited)


if __name__ == "__main__":
    # for _ in range(100):
    test_line()
