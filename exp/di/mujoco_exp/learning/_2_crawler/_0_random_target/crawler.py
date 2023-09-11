from random import random

import numpy as np
from dm_control.composer import Entity, Observables, observable, Task, Environment, variation
from dm_control.composer.observation.observable import MJCFFeature, Generic
from dm_control.locomotion.arenas import Floor
from dm_control.composer.variation import distributions, noises
from dm_control.mjcf import RootElement, get_attachment_frame
from gymnasium.utils.env_checker import check_env
from gymnasium import Env, spaces


class Crawler(Entity):
    def _build(self, n_legs=3):
        self._rgba = (0.5, 0.8, 0.8, 1)
        self._size = 0.1

        self._model = RootElement(model="crawler")
        self._torso = self._model.worldbody.add(
            "geom",
            name="torso",
            type="ellipsoid",
            size=(self._size, self._size, self._size / 2),
            rgba=self._rgba,
        )

        for i in range(n_legs):
            theta = 2 * i * np.pi / n_legs
            pos = np.array([np.cos(theta), np.sin(theta), 0]) * self._size
            site = self._model.worldbody.add("site", pos=pos, euler=[0, 0, theta])
            site.attach(self._build_leg())

        self._orientation = self._model.sensor.add("framequat", name="orientation", objtype="geom", objname=self._torso)
        self._linear_velocity = self._model.sensor.add(
            "framelinvel", name="linear_velocity", objtype="geom", objname=self._torso
        )
        self._angular_velocity = self._model.sensor.add(
            "frameangvel", name="angular_velocity", objtype="geom", objname=self._torso
        )

    @property
    def mjcf_model(self):
        return self._model

    @property
    def orientation(self):
        """Ground truth orientation sensor."""
        return self._orientation

    @property
    def linear_velocity(self):
        """Ground truth orientation sensor."""
        return self._linear_velocity

    @property
    def angular_velocity(self):
        """Ground truth orientation sensor."""
        return self._angular_velocity

    def _build_leg(self):
        model = RootElement(model="leg")

        # Defaults:
        model.default.joint.damping = 2
        model.default.joint.type = "hinge"
        model.default.geom.type = "capsule"
        model.default.geom.rgba = self._rgba
        model.default.position.ctrllimited = True
        model.default.position.ctrlrange = (-0.5, 0.5)

        # Thigh:
        thigh = model.worldbody.add("body", name="thigh")
        hip = thigh.add("joint", axis=[0, 0, 1])
        thigh.add("geom", fromto=[0, 0, 0, self._size, 0, 0], size=[self._size / 4])

        # Hip:
        shin = thigh.add("body", name="shin", pos=[self._size, 0, 0])
        knee = shin.add("joint", axis=[0, 1, 0])
        shin.add("geom", fromto=[0, 0, 0, 0, 0, -self._size], size=[self._size / 5])

        # Position actuators:
        model.actuator.add("position", name="hip_joint", joint=hip, kp=10)
        model.actuator.add("position", name="knee_joint", joint=knee, kp=10)

        return model

    def _build_observables(self):
        return Crawler.Observables(self)

    class Observables(Observables):
        @observable
        def joint_positions(self):
            return MJCFFeature("qpos", self._entity.mjcf_model.find_all("joint"))

        @observable
        def joint_velocities(self):
            return MJCFFeature("qvel", self._entity.mjcf_model.find_all("joint"))

        @observable
        def orientation(self):
            return MJCFFeature("sensordata", self._entity.orientation)

        @observable
        def linear_velocity(self):
            return MJCFFeature("sensordata", self._entity.linear_velocity)

        @observable
        def angular_velocity(self):
            return MJCFFeature("sensordata", self._entity.angular_velocity)


class UniformCircle(variation.Variation):
    """A uniformly sampled horizontal point on a circle of radius `distance`."""

    # Akhil: Basically sampling from a 2D donut.

    def __init__(self, distance):
        self._distance = distance  # Akhil: Depth of the donut
        self._heading = distributions.Uniform(0, 2 * np.pi)  # Akhil: Angle (line in cricket bowling)

    def __call__(self, initial_value=None, current_value=None, random_state=None):
        distance, heading = variation.evaluate((self._distance, self._heading), random_state=random_state)
        return distance * np.cos(heading), distance * np.sin(heading), 0


class WalkToTargetTask(Task):
    NUM_SUBSTEPS = 50  # The number of physics substeps per control timestep.

    def __init__(self):
        self._arena = Floor()
        self._arena.mjcf_model.worldbody.add("light", pos=(0, 0, 4))
        self.control_timestep = self.NUM_SUBSTEPS * self.physics_timestep

        self.crawler = Crawler()
        self._arena.add_free_entity(self.crawler)

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
            # return self.crawler.global_vector_to_local_frame(physics, target_pos)[:2]
            # local_vector = self.roller.global_vector_to_local_frame(physics, target_pos)
            roller_pos, roller_quat = self.crawler.get_pose(physics)
            return target_pos - roller_pos

        self._vec2target_observable = Generic(vector_to_target)

        self.crawler.observables.joint_positions.enabled = True
        self.crawler.observables.joint_velocities.enabled = True
        self.crawler.observables.orientation.enabled = True
        # self.crawler.observables.linear_velocity.enabled = True
        # self.crawler.observables.angular_velocity.enabled = True
        self._vec2target_observable.enabled = True

    @property
    def root_entity(self):
        return self._arena

    def get_reward(self, physics):
        pos = self._vec2target_observable(physics)
        distance = (pos ** 2).sum() ** 0.5
        velocity_linear, velocity_angular = self.crawler.get_velocity(physics)
        joint_pos = self.crawler.observables.joint_positions(physics)
        joint_pos_penalty = (joint_pos ** 2).sum() ** 0.5
        height = self.crawler.get_pose(physics)[0][2]
        return (
            - 5.0 * distance
            + 2.0 * np.linalg.norm(velocity_linear)
            - 0.5 * np.linalg.norm(velocity_angular)
            - 1e-3 * joint_pos_penalty
            - 5.0 * height
        )

    def initialize_episode(self, physics, random_state):
        # Set random angle on z axis  # https://quaternions.online/
        self.crawler.set_pose(
            physics, position=(random(), random(), 0.2), quaternion=(random(), 0, 0, random() * 2 - 1)
        )
        target_pos = variation.evaluate(UniformCircle(distributions.Uniform(0.5, 0.75)))
        physics.bind(self._target_site).pos = target_pos

    @property
    def task_observables(self):
        return {"vector_to_target": self._vec2target_observable}


class WalkToTargetEnv(Env):
    reward_range = (-float("inf"), float("inf"))
    max_episode_steps = 64
    # spec: EnvSpec | None = None

    def __init__(self):
        self.env = Environment(WalkToTargetTask())
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
        terminated = False  # self.timestep.reward > 0.9
        return self.observe(), self.timestep.reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.timestep = self.env.reset()
        return self.observe(), {}

    def observe(self):
        keys = [
            "vector_to_target",
            "crawler/joint_positions",
            "crawler/joint_velocities",
            "crawler/orientation",
        ]
        obs = np.concatenate([self.timestep.observation[k].flatten() for k in keys], axis=0, dtype=np.float32)
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

    check_env(WalkToTargetEnv())

    env = WalkToTargetEnv()
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

    with open("crawler.xml", "w") as fp:
        fp.write(WalkToTargetTask().root_entity.mjcf_model.to_xml_string())

    env = Environment(WalkToTargetTask())

    print(f"{env.action_spec()=}")
    print(f"{env.observation_spec()=}")

    time_step = env.reset()
    print(f"{time_step=}")

    duration = 2  # Seconds
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
        action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
        time_step = env.step(action)

        frames.append([plt.imshow(env.physics.render(), cmap=cm.Greys_r, animated=True)])
        rewards.append(time_step.reward)
        observations.append(deepcopy(time_step.observation))
        ticks.append(env.physics.data.time)

    ArtistAnimation(fig, frames, interval=1000 / framerate, blit=True, repeat_delay=1000).save("crawler.mp4")

    plt.close()
    plt.plot(ticks, rewards)
    plt.show()


if __name__ == "__main__":
    test_dm_env()
    test_gym_env()
