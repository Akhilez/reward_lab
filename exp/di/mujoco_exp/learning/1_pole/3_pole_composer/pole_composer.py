import copy

from PIL import Image
from dm_control import composer
import numpy as np
from dm_control import mjcf
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions, noises
from dm_control.locomotion.arenas import floors
from matplotlib import pyplot as plt, animation, cm


random_state = np.random.RandomState(2)
NUM_SUBSTEPS = 25  # The number of physics substeps per control timestep.


def build_pole():
    arena = mjcf.RootElement()

    # # <option gravity="0 0 -9.8" viscosity="0.1" wind="0 0 0" />
    # arena.option.gravity = [0, 0, -9.8]
    # arena.option.viscosity = 0.1
    # arena.option.wind = [0, 0, 0]
    # arena.worldbody.add("light", pos=[0, 0, 7], dir=[0, 0, -1])

    # <geom type="plane" size="1 1 0.1" rgba=".9 0.9 0.9 1"/>
    # arena.worldbody.add("geom", type="plane", size=[1, 1, 0.1], rgba=[0.9, 0.9, 0.9, -1])

    # <body pos="0 0 0.3" euler="0 0 0">
    body = arena.worldbody.add("body", pos=[0, 0, 0.3], euler=[0, 0, 0])
    # <geom type="cylinder" size="0.01 0.1" pos="0 0 0" mass="0.05"/>
    body.add("geom", type="cylinder", size=[0.01, 0.1], pos=[0, 0, 0], mass=0.05)
    # <geom type="box" size="0.02 0.02 0.02" pos="0 0 -0.1" mass="1"/>
    body.add("geom", type="box", size=[0.02, 0.02, 0.02], pos=[0, 0, -0.1], mass=1)

    # <joint name="main_hinge" type="hinge" pos="0 0 0.1" axis="0 1 0" damping="0.1" range="-360 360" limited="true" />
    body.add(
        "joint",
        name="main_hinge",
        type="hinge",
        pos=[0, 0, 0.1],
        axis=[0, 1, 0],
        damping=0.1,
        range=[-6.28, 6.28],
        # range=[-360, 360],
        limited=True,
    )

    # <site name="end_of_pole" pos="0 0 -0.1" size="0.025"/>
    body.add("site", name="end_of_pole", pos=[0, 0, -0.1], size=[0.025])

    arena.sensor.add("jointvel", joint="main_hinge")  # <jointvel joint="main_hinge" />
    # <framepos objtype="site" objname="end_of_pole"/>
    arena.sensor.add("framepos", name="tip_sensor", objtype="site", objname="end_of_pole")

    # <motor joint="main_hinge" ctrllimited="true" ctrlrange="-10 10"  />
    arena.actuator.add("motor", name="pole_actuator", joint="main_hinge", ctrllimited=True, ctrlrange=[-2.5, 2.5])

    return arena


class Pole(composer.Entity):
    def _build(self):
        self._model = build_pole()

    def _build_observables(self):
        return PoleObservables(self)

    @property
    def mjcf_model(self):
        return self._model

    @property
    def actuator(self):
        return self._model.find("actuator", "pole_actuator")

    @property
    def tip(self):
        return self._model.find("sensor", "tip_sensor")


class PoleObservables(composer.Observables):
    @composer.observable
    def joint_velocity(self):
        joint = self._entity.mjcf_model.find("joint", "main_hinge")
        return observable.MJCFFeature("qvel", joint)

    @composer.observable
    def joint_pos(self):
        joint = self._entity.mjcf_model.find("joint", "main_hinge")
        return observable.MJCFFeature("qpos", joint)

    @composer.observable
    def tip_pos(self):
        return observable.MJCFFeature(
            "sensordata",
            self._entity.tip,
            buffer_size=NUM_SUBSTEPS,
            aggregator="mean",
        )


class BalancePoleOnTop(composer.Task):
    def __init__(self, pole):
        self._pole = pole
        self._arena = floors.Floor()
        self._arena.attach(self._pole)
        self._arena.mjcf_model.worldbody.add("light", pos=[0, 0, 7], dir=[0, 0, -1])

        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # Configure and enable observables
        pos_corrptor = noises.Additive(distributions.Normal(scale=0.01))
        self._pole.observables.joint_pos.corruptor = pos_corrptor
        self._pole.observables.joint_pos.enabled = True
        vel_corruptor = noises.Multiplicative(distributions.LogNormal(sigma=0.01))
        self._pole.observables.joint_velocity.corruptor = vel_corruptor
        self._pole.observables.joint_velocity.enabled = True
        self._pole.observables.tip_pos.enabled = True

        self.control_timestep = NUM_SUBSTEPS * self.physics_timestep

    @property
    def root_entity(self):
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)

    def get_reward(self, physics):
        """
        The position can be from 0.2 to 0.6.
        We want the reward to start from a threshold height.
        """
        pos = self._pole.observables.tip_pos(physics)[2]
        threshold = 0.5
        pos = max(threshold, pos)
        pos = (pos - threshold) / (0.6 - threshold)  # normalize (t, 0.6) to (0, 1)
        return pos


if __name__ == '__main__':

    task = BalancePoleOnTop(Pole())
    env = composer.Environment(task, random_state=random_state)

    env.reset()
    Image.fromarray(env.physics.render()).save("pole_env_start.png")

    # Simulate episode with random actions
    duration = 10  # Seconds
    framerate = 30
    frames = []
    ticks = []
    rewards = []
    observations = []
    joint_positions = []
    joint_velocities = []

    spec = env.action_spec()
    time_step = env.reset()
    fig = plt.figure()

    print(f"Action:\n{spec.minimum=}\n{spec.maximum=}\n{spec.shape=}\n")
    print(f"Observation: {env.observation_spec()=}")
    print(f"{time_step=}")

    while env.physics.data.time < duration:
        action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
        time_step = env.step(action)

        frames.append([plt.imshow(env.physics.render(), cmap=cm.Greys_r, animated=True)])
        rewards.append(time_step.reward)
        observations.append(copy.deepcopy(time_step.observation))
        ticks.append(env.physics.data.time)
        joint_positions.append(time_step.observation['unnamed_model/joint_pos'].squeeze())
        joint_velocities.append(time_step.observation['unnamed_model/joint_velocity'].squeeze())

    animation.ArtistAnimation(fig, frames, interval=1000 / framerate, blit=True, repeat_delay=1000).save("pole.mp4")
    plt.close()

    _, ax = plt.subplots(3, 1, sharex=True, figsize=(4, 8))
    ax[0].plot(ticks, rewards)
    ax[0].set_title("reward")

    ax[1].plot(ticks, joint_positions)
    ax[1].set_title("joint_pos")

    ax[2].plot(ticks, joint_velocities)
    ax[2].set_title("joint_vel")
    ax[-1].set_xlabel("time")

    # for i, key in enumerate(time_step.observation):
    #     data = np.asarray([observations[j][key] for j in range(len(observations))])
    #     ax[i + 1].plot(ticks, data, label=key)
    #     ax[i + 1].set_ylabel(key)
    plt.show()
