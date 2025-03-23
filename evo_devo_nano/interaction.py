from jax import random
from craftax import craftax_env



class Memory:
    def __init__(self, size: int):
        self.size = size
        self.actions = []
        self.observations = []
        self.rewards = []
        self.dones = []
        self.done_mask = []
        self.infos = []

    def append(self, action, observation, reward, done, info):
        self.actions.append(action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.dones.append(done)
        self.done_mask.append(1-done)
        self.infos.append(info)

        if len(self.actions) > self.size:
            self.actions.pop(0)
            self.observations.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            self.done_mask.pop(0)
            self.infos.pop(0)


class Interaction:
    def __init__(self):
        self.env = craftax_env.make_craftax_env_from_params(classic=True, symbolic=False, auto_reset=True)
        self.key = random.key(0)
        self.key, subkey = random.split(self.key)
        self.obs, self._state = self.env.reset(subkey)
        self.h, self.w, _ = self.obs.shape
        self.n_actions = self.env.action_space().n

        self.time_step = 0
        self.memory = Memory(size=5)

    def step(self, action):
        self.key, subkey = random.split(self.key)
        obs_next, self._state, reward, done, info = self.env.step(subkey, self._state, int(action))
        self.time_step = 0 if done else self.time_step + 1
        self.memory.append(action, self.obs, reward, done, info)
        self.obs = obs_next

    def reset(self):
        self.key, subkey = random.split(self.key)
        self.obs, self._state = self.env.reset(subkey)
        self.memory = Memory(size=5)


class BatchedInteractions:
    def __init__(self, batch_size: int):
        self.interactions = [Interaction() for _ in range(batch_size)]

        self.h, self.w = self.interactions[0].h, self.interactions[0].w
        self.n_actions = self.interactions[0].n_actions

    def step(self, actions):
        assert len(self.interactions) == len(self.interactions)
        for interaction, action in zip(self.interactions, actions):
            interaction.step(action)

    def reset(self):
        for interaction in self.interactions:
            interaction.reset()

    def __getitem__(self, idx):
        return self.interactions[idx]

    def __len__(self):
        return len(self.interactions)

    @property
    def obs(self):
        return [interaction.obs for interaction in self.interactions]

    @property
    def time_step(self):
        return [interaction.time_step for interaction in self.interactions]
