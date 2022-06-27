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

hp = DictConfig(dict(
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
))


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
        self.children: Optional[List[Node]] = None
        self.env = deepcopy(env)
        self.p: Optional[np.ndarray] = None
        self.v_sum: Optional[np.ndarray] = None
        self.n: Optional[np.ndarray] = None

    def run_simulations(self, n_sims: int):
        # TODO: Implement run_simulations
        for i_sim in range(n_sims):
            leaf = self.select()
            leaf.expand()
            # 2. Expand and Evaluate
            # 3. Backup
            pass

    def select(self) -> 'Node':
        if len(self.children) == 0:
            return self

        q = self.v_sum / self.n
        u = self.p / (1 + self.n)

        i = (q + hp.lambda_mcts_selection * u).argmax()
        return self.children[i].select()

    def expand(self):


    def get_probabilities(self):
        # TODO: Implement get_probabilities
        # TODO: If illegal action, treat it as 0 probability.
        pass

    # TODO: Update tree when you create children.


class AlphaConnect4Dataset(Dataset):
    def __init__(self, tuples):
        self.tuples = tuples

    def __getitem__(self, index):
        tup = self.tuples[index]
        return (
            tup['state'],
            (
                tup['mcts_probabilities'],
                tup['final_reward'],
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
            state = observation['observation']
            action_mask = observation['action_mask']

            # Sample action from mcts_probabilities and action_mask
            legal_actions = action_mask.nonzero()[0]
            legal_probs = mcts_probabilities[legal_actions]
            legal_probs = (1-hp.dirichlet_e) * legal_probs + hp.dirichlet_e * np.random.dirichlet([hp.dirichlet_alpha] * len(legal_probs))
            sampled_action = np.random.choice(legal_actions, size=1, p=legal_probs)[0]

            tup = {
                'agent': env.agent_selection,
                'state': state,
                'mcts_probabilities': mcts_probabilities,
                'final_reward': None,
                'sampled_action': sampled_action,
                'i_game': i_game,
            }
            tuples_game.append(tup)

            env.step(sampled_action)

        # Update final rewards
        for tup in tuples_game:
            tup['final_reward'] = env.rewards[tup['agent']]
            tuples_global.append(tup)

    # Batchify the tuples into training batches.
    dataloader = DataLoader(AlphaConnect4Dataset(tuples_global), batch_size=hp.batch_size, shuffle=True)

    model.train()
    for epoch in hp.epochs:
        for states, (mcts_probs, final_rewards) in dataloader:
            pred_probs, pred_values = model(states)

            # l = (z - v)^2 - pie.T * log(p)
            loss = F.mse_loss(pred_values, final_rewards) - mcts_probs@torch.log(pred_probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


