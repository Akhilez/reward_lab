from collections import Counter
from copy import deepcopy
from typing import List, Set, Any, Dict, Optional
import numpy as np
import torch
from pettingzoo.classic import connect_four_v3
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import wandb
from tqdm import tqdm

hp = DictConfig(
    dict(
        train_iterations=100,
        n_games=20,
        mcts_sims=10,
        epochs=20,
        batch_size=64,
        dirichlet_e=0.25,
        dirichlet_alpha=0.3,
        lr=1e-3,
        weight_decay=1e-4,
        lambda_mcts_selection=1,
        model_depth=8,
        model_width=128,
        wandb=dict(
            mode="online",
            project="AlphaZeroConnect4",
            name="azc_" + datetime.now().strftime("%-d%H%M%S"),
        ),
        n_eval_games=10,
        min_win_rate=0.55,
        temperature_decay_rate=0.1,
    )
)


class AlphaConnect4Model(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(2, width, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.residual_stack = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(width, width, kernel_size=(3, 3), stride=1, padding=1),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                    nn.Conv2d(width, width, kernel_size=(3, 3), stride=1, padding=1),
                    nn.BatchNorm2d(width),
                )
                for _ in range(depth)
            ]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(width, 2, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6 * 7 * 2, 7),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(width, 1, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6 * 7 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, s):
        features = self.conv_block(s)
        for residual_block in self.residual_stack:
            inputs = features
            features = residual_block(features)
            features = features + inputs
            features = F.relu(features)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value


model = AlphaConnect4Model(hp.model_width, hp.model_depth)
optimizer = Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
best_model: AlphaConnect4Model = deepcopy(model)  # TODO: Implement best model
tree: Dict[tuple, Any] = dict()
temperature = 1
env = connect_four_v3.env()
wandb.init(
    name=hp.wandb.name,
    dir='../_wandb',
    config=OmegaConf.to_object(hp),
    tags=[],
    allow_val_change=True,
    save_code=True,
    project=hp.wandb.project,
    mode=hp.wandb.mode,
    # group=hp.wandb.group,
)


class Node:
    def __init__(self, env):
        self.parents: Set[Node] = set()
        self.children: List[Optional[Node]] = []
        self.env = deepcopy(env)
        self.s = np.array(self.env.unwrapped.board).reshape(6, 7)
        self.action_mask = self.env.observe(self.env.agent_selection)["action_mask"]
        self.p: Optional[np.ndarray] = None
        self.v_sum: float = 0.0
        self.n: np.ndarray = np.zeros((len(self.action_mask),))

    def run_simulations(self, n_sims: int):
        for i_sim in range(n_sims):
            leaf = self.select()
            leaf.expand()
            value = leaf.evaluate()
            leaf.backup(value)

    def select(self) -> "Node":
        if len(self.children) == 0:
            return self

        q = self.v_sum / (1 + self.n)
        u = self.p / (1 + self.n)

        selection_scores = (q + hp.lambda_mcts_selection * u)
        selection_scores = torch.softmax(torch.from_numpy(selection_scores), 0).numpy()
        selection_scores *= self.action_mask

        i = selection_scores.argmax()
        return self.children[i].select()

    def expand(self):
        if self.env.dones[self.env.agent_selection]:
            return

        n_actions = len(self.action_mask)
        for action in range(n_actions):
            if self.action_mask[action] == 0:  # illegal action
                self.children.append(None)
                continue
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

        observation = self.env.observe(self.env.agent_selection)
        state = torch.from_numpy(np.moveaxis(observation["observation"], -1, 0)).unsqueeze(0).float()
        model.eval()
        with torch.no_grad():
            pred_prob, pred_value = model(state)
        self.p = torch.softmax(pred_prob[0], 0).numpy()
        return pred_value.squeeze().item()

    def backup(self, value: float):
        self.v_sum += value
        for parent in list(self.parents):
            index = parent.children.index(self)
            parent.n[index] += 1
            parent.backup(value)

    def get_probabilities(self) -> np.ndarray:
        probs = self.n ** (1 / temperature)
        probs = probs * self.action_mask
        probs = probs / probs.sum()
        return probs

    def __eq__(self, other):
        if other is None:
            return False
        return self.env.unwrapped.board == other.env.unwrapped.board

    def __hash__(self):
        return hash(tuple(self.env.unwrapped.board))


class AlphaConnect4Dataset(Dataset):
    def __init__(self, tuples):
        self.tuples = tuples

    def __getitem__(self, index):
        tup = self.tuples[index]
        state = np.moveaxis(tup["state"], -1, 0).astype(np.float32)  # (6, 7, 2) --> (2, 6, 7)
        return (
            state,
            (
                tup["mcts_probabilities"].astype(np.float32),
                np.float32(tup["final_reward"]),
            ),
        )

    def __len__(self):
        return len(self.tuples)


for i_train in range(hp.train_iterations):
    # ----- GAME DATA GENERATION ------
    tuples_global = []
    for i_game in tqdm(range(hp.n_games), desc="Generating training data"):
        # Play the game using MCTS and NN, store the tuple for training.
        # Each tuple will have (s, pie, z)

        env.reset()
        tree.clear()
        tree[tuple(env.unwrapped.board)] = Node(env)
        tuples_game = []
        actions_counter = Counter()
        while not any(env.dones.values()):
            node = tree[tuple(env.unwrapped.board)]
            node.run_simulations(hp.mcts_sims)
            mcts_probabilities = node.get_probabilities()

            observation = env.observe(env.agent_selection)
            action_mask = observation["action_mask"]

            # Sample action from mcts_probabilities and action_mask
            legal_actions = action_mask.nonzero()[0]
            legal_probs = mcts_probabilities[legal_actions]
            dir_noise = np.random.dirichlet([hp.dirichlet_alpha] * len(legal_probs))
            legal_probs = ((1 - hp.dirichlet_e) * legal_probs) + (
                hp.dirichlet_e * dir_noise
            )
            sampled_action = np.random.choice(legal_actions, size=1, p=legal_probs)[0]

            tuples_game.append(
                {
                    "agent": env.agent_selection,
                    "state": observation["observation"],
                    "mcts_probabilities": mcts_probabilities,
                    "final_reward": None,
                    "sampled_action": sampled_action,
                    "i_game": i_game,
                }
            )
            actions_counter.update([sampled_action])

            env.step(sampled_action)

        # Update final rewards
        for tup in tuples_game:
            tup["final_reward"] = env.rewards[tup["agent"]]
            tuples_global.append(tup)

        wandb.log(
            {
                **{
                    "game_number": i_train * hp.n_games + i_game,
                    "game_length": len(tuples_game),
                    "reward_player_0": env.rewards["player_0"],
                    "reward_player_1": env.rewards["player_1"],
                    "last_frame": wandb.Image(np.array(env.unwrapped.board).reshape(6, 7) * 125)
                },
                **{f"action_count_{a}": count for a, count in actions_counter.items()},
            }
        )

    # ----- TRAINING ------

    # Batchify the tuples into training batches.
    dataloader = DataLoader(
        AlphaConnect4Dataset(tuples_global), batch_size=hp.batch_size, shuffle=True
    )

    model.train()
    for epoch in tqdm(range(hp.epochs), desc="Training"):
        losses_sum, n_batches = 0, 0
        for states, (mcts_probs, final_rewards) in dataloader:
            pred_probs, pred_values = model(states)

            # l = (z - v)^2 - pie.T * log(p)
            loss_policy = - mcts_probs.T @ torch.log(torch.softmax(pred_probs, 1))
            loss_policy = loss_policy.sum()
            loss_value = F.mse_loss(pred_values.squeeze(), final_rewards)
            loss = loss_value + loss_policy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_sum += loss.item()
            n_batches += 1
        wandb.log(
            {
                "epoch": i_train * hp.epochs + epoch,
                "loss": losses_sum / n_batches,
            }
        )

    # ---- EVAL ----

    win_rate = 0
    for i_game in range(hp.n_eval_games):
        env.reset()
        tree.clear()
        tree[tuple(env.unwrapped.board)] = Node(env)

        player = np.random.choice(['player_0', 'player_1'])

        while not any(env.dones.values()):
            if env.agent_selection == player:
                state = tuple(env.unwrapped.board)
                node = tree.get(state)
                if node is None:
                    node = tree.setdefault(state, Node(env))
                node.run_simulations(hp.mcts_sims)
                mcts_probabilities = node.get_probabilities()
                action = mcts_probabilities.argmax()
            else:
                obs = env.observe(env.agent_selection)
                action = np.random.choice(obs['action_mask'].nonzero()[0])

            env.step(action)

        win_rate += env.rewards[player]

    # TODO: I don't get the concept of temperature fully.
    # temperature = exp(- (i_train + 1) * hp.temperature_decay_rate)

    wandb.log(
        {
            "train_iteration": i_train,
            "win_rate": win_rate,
            'temperature': temperature,
        }
    )
