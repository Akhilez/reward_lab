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


class Leg(object):
    """A 2-DoF leg with position actuators."""

    def __init__(self, length, rgba):
        self.model = mjcf.RootElement()

        # Defaults:
        self.model.default.joint.damping = 2
        self.model.default.joint.type = "hinge"
        self.model.default.geom.type = "capsule"
        self.model.default.geom.rgba = rgba  # Continued below...

        # Thigh:
        self.thigh = self.model.worldbody.add("body")
        self.hip = self.thigh.add("joint", axis=[0, 0, 1])
        self.thigh.add("geom", fromto=[0, 0, 0, length, 0, 0], size=[length / 4])

        # Hip:
        self.shin = self.thigh.add("body", pos=[length, 0, 0])
        self.knee = self.shin.add("joint", axis=[0, 1, 0])
        self.shin.add("geom", fromto=[0, 0, 0, 0, 0, -length], size=[length / 5])

        # Position actuators:
        self.model.actuator.add("position", joint=self.hip, kp=10)
        self.model.actuator.add("position", joint=self.knee, kp=10)


BODY_RADIUS = 0.1
BODY_SIZE = (BODY_RADIUS, BODY_RADIUS, BODY_RADIUS / 2)
random_state = np.random.RandomState(42)


def make_creature(num_legs):
    """Constructs a creature with `num_legs` legs."""
    rgba = random_state.uniform([0, 0, 0, 1], [1, 1, 1, 1])
    model = mjcf.RootElement()
    model.compiler.angle = "radian"  # Use radians.

    # Make the torso geom.
    model.worldbody.add("geom", name="torso", type="ellipsoid", size=BODY_SIZE, rgba=rgba)

    # Attach legs to equidistant sites on the circumference.
    for i in range(num_legs):
        theta = 2 * i * np.pi / num_legs
        hip_pos = BODY_RADIUS * np.array([np.cos(theta), np.sin(theta), 0])
        hip_site = model.worldbody.add("site", pos=hip_pos, euler=[0, 0, theta])
        leg = Leg(length=BODY_RADIUS, rgba=rgba)
        hip_site.attach(leg.model)

    return model


# ========= New stuff ==========


class Creature(composer.Entity):
    """A multi-legged creature derived from `composer.Entity`."""

    def _build(self, num_legs):
        self._model = make_creature(num_legs)

    def _build_observables(self):
        return CreatureObservables(self)

    @property
    def mjcf_model(self):
        return self._model

    @property
    def actuators(self):
        return tuple(self._model.find_all("actuator"))


# Add simple observable features for joint angles and velocities.
class CreatureObservables(composer.Observables):
    @composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all("joint")
        return observable.MJCFFeature("qpos", all_joints)

    @composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all("joint")
        return observable.MJCFFeature("qvel", all_joints)


# @title The `Button` class

NUM_SUBSTEPS = 25  # The number of physics substeps per control timestep.


class Button(composer.Entity):
    """A button Entity which changes colour when pressed with certain force."""

    def _build(self, target_force_range=(5, 10)):
        self._min_force, self._max_force = target_force_range
        self._mjcf_model = mjcf.RootElement()
        self._geom = self._mjcf_model.worldbody.add("geom", type="cylinder", size=[0.25, 0.02], rgba=[1, 0, 0, 1])
        self._site = self._mjcf_model.worldbody.add(
            "site", type="cylinder", size=self._geom.size * 1.01, rgba=[1, 0, 0, 0]
        )
        self._sensor = self._mjcf_model.sensor.add("touch", site=self._site)
        self._num_activated_steps = 0

    def _build_observables(self):
        return ButtonObservables(self)

    @property
    def mjcf_model(self):
        return self._mjcf_model

    # Update the activation (and colour) if the desired force is applied.
    def _update_activation(self, physics):
        current_force = physics.bind(self.touch_sensor).sensordata[0]
        self._is_activated = current_force >= self._min_force and current_force <= self._max_force
        physics.bind(self._geom).rgba = [0, 1, 0, 1] if self._is_activated else [1, 0, 0, 1]
        self._num_activated_steps += int(self._is_activated)

    def initialize_episode(self, physics, random_state):
        self._reward = 0.0
        self._num_activated_steps = 0
        self._update_activation(physics)

    def after_substep(self, physics, random_state):
        self._update_activation(physics)

    @property
    def touch_sensor(self):
        return self._sensor

    @property
    def num_activated_steps(self):
        return self._num_activated_steps


class ButtonObservables(composer.Observables):
    """A touch sensor which averages contact force over physics substeps."""

    @composer.observable
    def touch_force(self):
        return observable.MJCFFeature(
            "sensordata",
            self._entity.touch_sensor,
            buffer_size=NUM_SUBSTEPS,
            aggregator="mean",
        )


# @title Random initializer using `composer.variation`


class UniformCircle(variation.Variation):
    """A uniformly sampled horizontal point on a circle of radius `distance`."""

    # Akhil: Basically sampling from a 2D donut.

    def __init__(self, distance):
        self._distance = distance  # Akhil: Depth of the donut
        self._heading = distributions.Uniform(0, 2 * np.pi)  # Akhil: Angle (line in cricket bowling)

    def __call__(self, initial_value=None, current_value=None, random_state=None):
        distance, heading = variation.evaluate((self._distance, self._heading), random_state=random_state)
        return distance * np.cos(heading), distance * np.sin(heading), 0


# @title The `PressWithSpecificForce` task


class PressWithSpecificForce(composer.Task):
    def __init__(self, creature):
        self._creature = creature
        self._arena = floors.Floor()
        self._arena.add_free_entity(self._creature)
        self._arena.mjcf_model.worldbody.add("light", pos=(0, 0, 4))
        self._button = Button()
        self._arena.attach(self._button)

        # Configure initial poses
        self._creature_initial_pose = (0, 0, 0.15)
        button_distance = distributions.Uniform(0.15, 0.175)
        self._button_initial_pose = UniformCircle(button_distance)

        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # Configure and enable observables
        pos_corrptor = noises.Additive(distributions.Normal(scale=0.01))
        self._creature.observables.joint_positions.corruptor = pos_corrptor
        self._creature.observables.joint_positions.enabled = True
        vel_corruptor = noises.Multiplicative(distributions.LogNormal(sigma=0.01))
        self._creature.observables.joint_velocities.corruptor = vel_corruptor
        self._creature.observables.joint_velocities.enabled = True
        self._button.observables.touch_force.enabled = True

        def to_button(physics):
            button_pos, _ = self._button.get_pose(physics)
            return self._creature.global_vector_to_local_frame(physics, button_pos)

        self._task_observables = {"button_position": observable.Generic(to_button)}

        for obs in self._task_observables.values():
            obs.enabled = True

        self.control_timestep = NUM_SUBSTEPS * self.physics_timestep

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)
        creature_pose, button_pose = variation.evaluate(
            (self._creature_initial_pose, self._button_initial_pose),
            random_state=random_state,
        )
        self._creature.set_pose(physics, position=creature_pose)
        self._button.set_pose(physics, position=button_pose)

    def get_reward(self, physics):
        return self._button.num_activated_steps / NUM_SUBSTEPS


# @title Instantiating an environment{vertical-output: true}

creature = Creature(num_legs=4)
task = PressWithSpecificForce(creature)
random_state = np.random.RandomState(42)
env = composer.Environment(task, random_state=random_state)

env.reset()
Image.fromarray(env.physics.render()).save("env_start.png")

# Simulate episode with random actions
duration = 4  # Seconds
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

    # TODO: How to set limits on the action ranges?
    # TODO: How long will one action last? Currently it advances 0.05. How to modify that?
    action = random_state.uniform(-np.ones_like(spec.minimum), np.ones_like(spec.maximum), spec.shape)
    # action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
    time_step = env.step(action)

    camera0 = env.physics.render()
    # camera1 = env.physics.render(camera_id=1, height=200, width=200)
    frames.append([plt.imshow(camera0, cmap=cm.Greys_r, animated=True)])
    rewards.append(time_step.reward)
    observations.append(copy.deepcopy(time_step.observation))
    ticks.append(env.physics.data.time)

# html_video = display_video(frames, framerate=1.0 / env.control_timestep())
animation.ArtistAnimation(fig, frames, interval=1000 / framerate, blit=True, repeat_delay=1000).save("creature.mp4")

# Show video and plot reward and observations
num_sensors = len(time_step.observation)

plt.close()
plt.plot(ticks, rewards)
plt.show()
# _, ax = plt.subplots(1 + num_sensors, 1, sharex=True, figsize=(4, 8))
# ax[0].plot(ticks, rewards)
# ax[0].set_ylabel("reward")
# ax[-1].set_xlabel("time")
#
# for i, key in enumerate(time_step.observation):
#     data = np.asarray([observations[j][key] for j in range(len(observations))])
#     ax[i + 1].plot(ticks, data, label=key)
#     ax[i + 1].set_ylabel(key)

# html_video
