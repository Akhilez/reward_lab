"""
AlphaZero with pettingzoo env interface.
Background run: nohup python alpha_connect4.py > /dev/null 2>&1 &
"""
import math
from collections import Counter, deque
from copy import deepcopy
from os import makedirs
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
from torchmetrics import MeanMetric

hp = DictConfig(
    dict(
        train_iterations=1000,
        n_games=25,
        mcts_sims=100,
        epochs=10,
        batch_size=512,
        max_queue_len=10_000,
        n_eval_games=50,
        eval_every=5,
        dirichlet_e=0.25,
        dirichlet_alpha=0.3,
        lr=1e-1,
        weight_decay=1e-5,
        lambda_mcts_selection=1,
        win_rate_threshold=0.50,
        model_depth=4,
        model_width=128,
        wandb=dict(
            mode="online",
            project="AlphaZeroConnect4",
            name="azc_" + datetime.now().strftime("%-d%H%M%S"),
            dir="../_wandb",
        ),
        temperature_decay_rate=0.1,
        device="cuda:3",
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
            nn.Softmax(),
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


class SimpleAlphaConnect4Model(nn.Module):
    def __init__(
        self, in_size: int, units: List[int], out_size: int, flatten: bool = False
    ):
        super().__init__()

        flatten_layer = [nn.Flatten()] if flatten else []
        self.first = nn.Sequential(
            *flatten_layer, nn.Linear(in_size, units[0]), nn.ReLU(), nn.Dropout(0.3)
        )

        self.hidden = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(units[i], units[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
                for i in range(len(units) - 1)
            ]
        )

        self.out = nn.Linear(units[-1], out_size)

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        x = x.flatten(1)
        x = self.out(x)
        policy = x[:, :-1]
        value = torch.tanh(x[:, -1:])
        return policy, value


model = AlphaConnect4Model(hp.model_width, hp.model_depth).to(hp.device)
# model = SimpleAlphaConnect4Model(2 * 6 * 7, [64, 64, 64, 64], 7 + 1, flatten=True).to(hp.device)
optimizer = Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
prev_model = deepcopy(model)
tree: Dict[tuple, Any] = dict()
temperature = 1
env = connect_four_v3.env()
makedirs(hp.wandb.dir, exist_ok=True)
wandb.init(
    name=hp.wandb.name,
    dir=hp.wandb.dir,
    config=OmegaConf.to_object(hp),
    tags=[],
    allow_val_change=True,
    save_code=True,
    project=hp.wandb.project,
    mode=hp.wandb.mode,
    # group=hp.wandb.group,
)
game_length_metric = MeanMetric()
loss_metric = MeanMetric()


class Node:
    def __init__(self, env):
        self.parents: Set[Node] = set()
        self.children: List[Optional[Node]] = []
        self.env = deepcopy(env)
        self.s = np.array(self.env.unwrapped.board).reshape(6, 7)
        self.action_mask = self.env.observe(self.env.agent_selection)["action_mask"]
        self.p: Optional[np.ndarray] = None
        self.v: float = 0.0
        self.n: np.ndarray = np.zeros((len(self.action_mask),))
        self.q: np.ndarray = np.zeros((len(self.action_mask),))

    def run_simulations(self, n_sims: int, model, root, eval_mode=False):
        # Resetting parents because backing up the parents is unnecessary and time-consuming
        node.parents = set()
        for i_sim in range(n_sims):
            leaf = self.select(eval_mode)
            leaf.expand(root)
            leaf.evaluate(model)
            leaf.backup(-self.v)

    def select(self, eval_mode=False) -> "Node":
        if len(self.children) == 0:
            return self

        c_puct = hp.lambda_mcts_selection
        u = self.q + c_puct * self.p * math.sqrt(sum(self.n)) / (1 + self.n)

        u = torch.softmax(torch.from_numpy(u), dim=0)
        if False:  # eval_mode:
            # Set u = 0 for invalid actions.
            u *= self.action_mask
            # When not eval mode, get the max u
            # TODO: Handle case where multiple actions have equal max value. Must be random b/w them.
            i = u.argmax().item()
        else:
            # Add dirichlet noise during self-play
            # TODO: What is dirichlet noise???
            u += torch.rand((len(self.n),)) * 0.10
            # Set u = 0 for invalid actions.
            u *= self.action_mask
            i = torch.multinomial(u, 1)[0].item()

        return self.children[i].select(eval_mode=eval_mode)

    def expand(self, root):
        """Just create child nodes with default values."""

        if (
            self.env.terminations[self.env.agent_selection]
            or self.env.truncations[self.env.agent_selection]
        ):
            return

        n_actions = len(self.action_mask)
        for action in range(n_actions):
            if self.action_mask[action] == 0:  # illegal action
                self.children.append(None)
                continue
            env = deepcopy(self.env)
            env.step(action)
            state = tuple(env.unwrapped.board)
            child = root.get(state)
            if child is None:
                child = root.setdefault(state, Node(env))
            child.parents.add(self)
            self.children.append(child)

    def evaluate(self, model):
        """
        Sets child action probs from NN and returns value.
        If terminal state, value = final reward.
        Else, value comes from neural network.
        """

        if len(self.children) == 0:
            self.v = self.env.rewards[self.env.agent_selection]
            # print("Reached terminal")
            return

        observation = self.env.observe(self.env.agent_selection)
        state = (
            torch.from_numpy(np.moveaxis(observation["observation"], -1, 0))
            .unsqueeze(0)
            .float()
        )
        model.eval()
        with torch.no_grad():
            pred_prob, pred_value = model(state.to(hp.device))
        self.p = pred_prob[0].cpu().numpy()
        self.v = pred_value.squeeze().item()

    def backup(self, value: float):
        for node in list(self.parents):
            i = node.children.index(self)
            # Q[s][a] = (N[s][a]*Q[s][a] + v)/(N[s][a]+1)
            node.q[i] = (node.n[i] * node.q[i] + value) / (node.n[i] + 1)
            node.n[i] += 1
            node.backup(-value)

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


tuples_global = deque([], maxlen=hp.max_queue_len)
for i_train in range(hp.train_iterations):
    iter_log = {"train_iteration": i_train}  # 'temperature': temperature,

    # ----- GAME DATA GENERATION ------
    game_length_metric.reset()
    for i_game in tqdm(range(hp.n_games), desc="Generating training data"):
        # Play the game using MCTS and NN, store the tuple for training.
        # Each tuple will have (s, pie, z)

        env.reset()
        tree.clear()
        tree[tuple(env.unwrapped.board)] = Node(env)
        tuples_game = []
        actions_counter = Counter()
        while not (any(env.terminations.values()) or any(env.truncations.values())):
            node = tree[tuple(env.unwrapped.board)]
            node.run_simulations(hp.mcts_sims, model, tree, False)
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
                    # "board": tuple(env.unwrapped.board),
                }
            )
            actions_counter.update([sampled_action])

            env.step(sampled_action)

        # Update final rewards
        for tup in tuples_game:
            tup["final_reward"] = env.rewards[tup["agent"]]
            tuples_global.append(tup)

        game_length_metric(len(tuples_game))
        # wandb.log(
        #     {
        #         **{
        #             # "game_number": i_train * hp.n_games + i_game,
        #             # "game_length": len(tuples_game),
        #             # "reward_player_0": env.rewards["player_0"],
        #             # "reward_player_1": env.rewards["player_1"],
        #             # "last_frame": wandb.Image(np.array(env.unwrapped.board).reshape(6, 7) * 125)
        #         },
        #         # **{f"action_count_{a}": count for a, count in actions_counter.items()},
        #     }
        # )
    iter_log['dataset_size'] = len(tuples_global)
    iter_log['game_length_avg'] = game_length_metric.compute()

    # ----- TRAINING ------

    # Batchify the tuples into training batches.
    dataloader = DataLoader(AlphaConnect4Dataset(tuples_global), batch_size=hp.batch_size, shuffle=True)

    loss_metric.reset()
    model.train()
    for epoch in tqdm(range(hp.epochs), desc="Training"):
        for states, (mcts_probs, final_rewards) in dataloader:
            pred_probs, pred_values = model(states.to(hp.device))

            # l = (z - v)^2 - pie.T * log(p)
            loss_policy = -mcts_probs.T.to(hp.device) @ torch.log(pred_probs)
            loss_value = F.mse_loss(pred_values.squeeze(), final_rewards.squeeze().to(hp.device))
            loss = loss_value + loss_policy.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_metric(loss.item())
        # wandb.log(
        #     {
        #         "epoch": i_train * hp.epochs + epoch,
        #         "loss": losses_sum / n_batches,
        #     }
        # )
    iter_log['loss_avg'] = loss_metric.compute()

    # ---- EVAL ----

    if i_train % hp.eval_every == 0 and i_train > 0:

        frames = []
        trees = {"player_0": dict(), "player_1": dict()}
        win_rate = 0
        for i_game in tqdm(range(hp.n_eval_games), desc="Evaluating"):
            env.reset()
            for t in trees.values():
                t.clear()
                t[tuple(env.unwrapped.board)] = Node(env)

            is_player_0 = np.random.random() > 0.5
            player = "player_0" if is_player_0 else "player_1"
            player_model = {
                "player_0": model if is_player_0 else prev_model,
                "player_1": prev_model if is_player_0 else model,
            }

            while not (any(env.terminations.values()) or any(env.truncations.values())):
                state = tuple(env.unwrapped.board)
                t = trees[env.agent_selection]
                node = t.get(state)
                if node is None:
                    node = t.setdefault(state, Node(env))
                node.run_simulations(
                    hp.mcts_sims, player_model[env.agent_selection], t, eval_mode=True
                )
                mcts_probabilities = node.get_probabilities()
                action = mcts_probabilities.argmax()
                env.step(action)

                if i_game == hp.n_eval_games - 1:
                    frames.append(np.array(env.unwrapped.board).reshape(6, 7) * 125)

            win_rate += max(0, env.rewards[player])
        win_rate /= hp.n_eval_games
        print(f"Win rate: {win_rate}")

        if win_rate > hp.win_rate_threshold:
            prev_model = deepcopy(model)
            torch.save(model.state_dict(), "model.pth")
            wandb.save("model.pth")
        # else:
        #     model = deepcopy(prev_model)

        iter_log['win_rate'] = win_rate
        iter_log["eval_game"] = wandb.Video(
            np.array(frames).reshape((-1, 1, 6, 7)), fps=4, format="gif"
        )

    # -------------- Logging and stuff -------------

    # TODO: I don't get the concept of temperature fully.
    # temperature = exp(- (i_train + 1) * hp.temperature_decay_rate)

    wandb.log(iter_log)


# pid: 1163598, 1168936
