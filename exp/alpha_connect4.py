from copy import deepcopy
from typing import List, Set, Any, Dict, Optional
import numpy as np
import torch
from pettingzoo.classic import connect_four_v3
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

hp = DictConfig(
    dict(
        train_iterations=100,
        n_games=10,
        mcts_sims=10,
        epochs=2,
        batch_size=8,
        dirichlet_e=0.25,
        dirichlet_alpha=0.3,
        lr=1e-3,
        weight_decay=1e-4,
        lambda_mcts_selection=1,
    )
)


class AlphaConnect4Model(nn.Module):
    pass


model = AlphaConnect4Model()
optimizer = Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
best_model: AlphaConnect4Model = deepcopy(model)  # TODO: Implement best model
tree: Dict[tuple, Any] = dict()
env = connect_four_v3.env()


class Node:
    def __init__(self, env):
        self.parents: Set[Node] = set()
        self.children: List[Node] = []
        self.env = deepcopy(env)
        self.action_mask = self.env.observation(self.env.agent_selection)["action_mask"]
        self.p: Optional[np.ndarray] = None
        self.v_sum: float = 0.0
        self.n: np.ndarray = np.zeros((len(self.action_mask),))

    def run_simulations(self, n_sims: int):
        for i_sim in range(n_sims):
            leaf = self.select()
            leaf.expand()
            value = leaf.evaluate()
            self.backup(value)

    def select(self) -> "Node":
        if len(self.children) == 0:
            return self

        q = self.v_sum / self.n
        u = self.p / (1 + self.n)

        i = (q + hp.lambda_mcts_selection * u).argmax()
        return self.children[i].select()

    def expand(self):
        if self.action_mask.sum() == 0:  # No actions to perform.
            return

        if self.env.dones[self.env.agent_selection]:
            return

        n_actions = len(self.action_mask)
        for action in range(n_actions):
            env = deepcopy(self.env)
            env.step(action)
            state = tuple(env.unwrapped.board)
            child = tree.get(state)
            if child is None:
                child = tree.setdefault(state, Node(env))
            child.parents.add(self)

            self.children.append(child)

    def evaluate(self):
        if len(self.children) == 0:
            return self.env.rewards[self.env.agent_selection]

        observation = self.env.observation(self.env.agent_selection)
        state = torch.from_numpy(observation["observation"]).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            pred_prob, pred_value = model(state)
        self.p = pred_prob[0].numpy()
        return pred_value.squeeze().item()

    def backup(self, value: float):
        self.v_sum += value
        for parent in list(self.parents):
            index = parent.children.index(self)
            parent.n[index] += 1
            parent.backup(value)

    def get_probabilities(self) -> np.ndarray:
        temperature = 1  # TODO: Reduce temperature with iterations.
        probs = self.n ** (1 / temperature)
        probs = probs * self.action_mask
        probs = probs / probs.sum()
        return probs

    def __eq__(self, other):
        return self.env.unwrapped.board == other.env.unwrapped.board


class AlphaConnect4Dataset(Dataset):
    def __init__(self, tuples):
        self.tuples = tuples

    def __getitem__(self, index):
        tup = self.tuples[index]
        return (
            tup["state"],
            (
                tup["mcts_probabilities"],
                tup["final_reward"],
            ),
        )

    def __len__(self):
        return len(self.tuples)


for i_train in range(hp.train_iterations):
    tuples_global = []

    for i_game in range(hp.n_games):
        # Play the game using MCTS and NN, store the tuple for training.
        # Each tuple will have (s, pie, z)

        env.reset()
        tree.clear()
        tree[tuple(env.unwrapped.board)] = Node(env)
        tuples_game = []
        while not any(env.dones.values()):
            node = tree[tuple(env.unwrapped.board)]
            node.run_simulations(hp.mcts_sims)
            mcts_probabilities = node.get_probabilities()

            observation = env.observe(env.agent_selection)
            state = observation["observation"]
            action_mask = observation["action_mask"]

            # Sample action from mcts_probabilities and action_mask
            legal_actions = action_mask.nonzero()[0]
            legal_probs = mcts_probabilities[legal_actions]
            dir_noise = np.random.dirichlet([hp.dirichlet_alpha] * len(legal_probs))
            legal_probs = ((1 - hp.dirichlet_e) * legal_probs) + (hp.dirichlet_e * dir_noise)
            sampled_action = np.random.choice(legal_actions, size=1, p=legal_probs)[0]

            tup = {
                "agent": env.agent_selection,
                "state": state,
                "mcts_probabilities": mcts_probabilities,
                "final_reward": None,
                "sampled_action": sampled_action,
                "i_game": i_game,
            }
            tuples_game.append(tup)

            env.step(sampled_action)

        # Update final rewards
        for tup in tuples_game:
            tup["final_reward"] = env.rewards[tup["agent"]]
            tuples_global.append(tup)

    # Batchify the tuples into training batches.
    dataloader = DataLoader(AlphaConnect4Dataset(tuples_global), batch_size=hp.batch_size, shuffle=True)

    model.train()
    for epoch in hp.epochs:
        for states, (mcts_probs, final_rewards) in dataloader:
            pred_probs, pred_values = model(states)

            # l = (z - v)^2 - pie.T * log(p)
            loss = F.mse_loss(pred_values, final_rewards) - mcts_probs @ torch.log(pred_probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
