import numpy as np
from dm_control.composer import Entity, Observables, observable, Task, Environment
from dm_control.composer.observation.observable import MJCFFeature, Generic
from dm_control.locomotion.arenas import Floor
from dm_control.mjcf import RootElement
from gymnasium.utils.env_checker import check_env
from gymnasium import Env, spaces


class Roller(Entity):
    def _build(self):
        self._model = RootElement(model="roller")
        self._model.worldbody.add(
            "geom",
            name="torso",
            type="box",
            size=(0.1, 0.15, 0.03),
            rgba=(0.75, 1, 0.75, 1),
            pos=(0, 0.15, 0),
            mass=0.5,
        )

        # --------------------------------------------------------------
        left_wheel = self._model.worldbody.add(
            "body",
            pos=(-0.15, 0, 0),
            name="left_wheel",
        )
        left_wheel_joint = left_wheel.add(
            "joint",
            type="hinge",
            name="left_wheel_joint",
            axis=(1, 0, 0),
            damping=1.5,
        )
        left_wheel.add(
            "geom",
            name="left_tyre",
            type="cylinder",
            size=(0.1, 0.05),
            euler=(0, 90, 0),
            mass=0.1,
            rgba=(1, 0.75, 0.75, 1),
        )

        # --------------------------------------------------------------
        right_wheel = self._model.worldbody.add(
            "body",
            pos=(0.15, 0, 0),
            name="right_wheel",
        )
        right_wheel_joint = right_wheel.add(
            "joint",
            type="hinge",
            name="right_wheel_joint",
            axis=(1, 0, 0),
            damping=1.5,
        )
        right_wheel.add(
            "geom",
            name="right_tyre",
            type="cylinder",
            size=(0.1, 0.05),
            euler=(0, 90, 0),
            mass=0.1,
            rgba=(0.75, 0.75, 1, 1),
        )

        # --------------------------------------------------------------
        self._model.actuator.add(
            "velocity",
            name="left_motor",
            joint=left_wheel_joint,
            ctrllimited=True,
            ctrlrange=[-5, 5],
        )
        self._model.actuator.add(
            "velocity",
            name="right_motor",
            joint=right_wheel_joint,
            ctrllimited=True,
            ctrlrange=[-5, 5],
        )

    @property
    def mjcf_model(self):
        return self._model

    def _build_observables(self):
        return Roller.Observables(self)

    class Observables(Observables):
        @observable
        def joint_velocities(self):
            return MJCFFeature("qvel", self._entity.mjcf_model.find_all("joint"))


class RollToTargetTask(Task):
    NUM_SUBSTEPS = 50  # The number of physics substeps per control timestep.

    def __init__(self):
        self._arena = Floor()
        self._arena.mjcf_model.compiler.angle = "degree"
        self._arena.mjcf_model.worldbody.add("light", pos=(0, 0, 4))
        self.control_timestep = self.NUM_SUBSTEPS * self.physics_timestep

        self.roller = Roller()
        self._arena.add_free_entity(self.roller)

        self._target_site = self._arena.mjcf_model.worldbody.add(
            "site",
            name="target_site",
            type="sphere",
            rgba=(1, 0.2, 0.2, 1),
            size=(0.1, 0.1, 0.1),
            pos=(0.5, 0, 0.1),
        )
        
        def vector_to_target(physics):
            target_pos = physics.bind(self._target_site).pos
            # local_vector = self.roller.global_vector_to_local_frame(physics, target_pos)
            roller_pos, _ = self.roller.get_pose(physics)
            return roller_pos - target_pos

        self._vec2target_observable = Generic(vector_to_target)
        self._vec2target_observable.enabled = True
        self.roller.observables.joint_velocities.enabled = True

    @property
    def root_entity(self):
        return self._arena

    def get_reward(self, physics):
        pos = self._vec2target_observable(physics)
        distance = (pos ** 2).sum() ** 0.5
        initial = 0.50
        reward = (initial - distance) / initial
        return reward
        # phys = physics.bind(self._target_site)
        # return 0

    def initialize_episode(self, physics, random_state):
        self.roller.set_pose(physics, position=(0, 0, 0.2))

    @property
    def task_observables(self):
        return {"vector_to_target": self._vec2target_observable}


class RollToTargetEnv(Env):
    reward_range = (-float("inf"), float("inf"))
    max_episode_steps = 64
    # spec: EnvSpec | None = None

    def __init__(self):
        self.env = Environment(RollToTargetTask())
        self.timestep = None

    @property
    def action_space(self):
        spec = self.env.action_spec()
        return spaces.Box(
            low=0,
            high=1,
            shape=spec.shape,
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        spec = self.env.observation_spec()
        shape = sum([v.shape[-1] for v in spec.values()])
        return spaces.Box(low=-9999999, high=9999999, shape=(shape,), dtype=np.float32)

    def step(self, actions):
        actions = self._normalize_actions(actions)
        self.timestep = self.env.step(actions)
        terminated = self.timestep.reward > 0.9
        return self.observe(), self.timestep.reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.timestep = self.env.reset()
        return self.observe(), {}

    def observe(self):
        keys = [
            "vector_to_target",
            "roller/joint_velocities",
        ]
        obs = np.concatenate(
            [self.timestep.observation[k].flatten() for k in keys],
            axis=0,
            dtype=np.float32,
        )
        return obs

    def render(self):
        return self.env.physics.render()

    def _normalize_actions(self, actions):
        spec = self.env.action_spec()
        actions = spec.minimum + (spec.maximum - spec.minimum) * actions
        return actions


def test_gym_env():
    from matplotlib import pyplot as plt
    from gymnasium.wrappers import TimeLimit

    check_env(RollToTargetEnv())

    env = RollToTargetEnv()
    env = TimeLimit(env, 100)
    rewards = []

    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if truncated or terminated:
            break

    plt.plot(rewards)
    plt.show()


def test_dm_env():
    from matplotlib import pyplot as plt
    from matplotlib import cm
    from matplotlib.animation import ArtistAnimation
    from copy import deepcopy

    with open("roller.xml", "w") as fp:
        fp.write(RollToTargetTask().root_entity.mjcf_model.to_xml_string())

    env = Environment(RollToTargetTask())

    print(f"{env.action_spec()=}")
    print(f"{env.observation_spec()=}")

    time_step = env.reset()
    print(f"{time_step=}")

    duration = 5  # Seconds
    framerate = 30
    frames = []
    ticks = []
    rewards = []
    observations = []

    spec = env.action_spec()
    time_step = env.reset()
    fig = plt.figure()

    print(f"{spec.minimum=}\n{spec.maximum=}\n{spec.shape=}")
    print(f"{time_step=}")

    while env.physics.data.time < duration:
        action = np.random.uniform(spec.maximum, spec.maximum, spec.shape)
        time_step = env.step(action)

        frames.append([plt.imshow(env.physics.render(), cmap=cm.Greys_r, animated=True)])
        rewards.append(time_step.reward)
        observations.append(deepcopy(time_step.observation))
        ticks.append(env.physics.data.time)

    ArtistAnimation(fig, frames, interval=1000 / framerate, blit=True, repeat_delay=1000).save("roller.mp4")

    plt.close()
    plt.plot(ticks, rewards)
    plt.show()


if __name__ == "__main__":
    test_dm_env()
    # test_gym_env()
