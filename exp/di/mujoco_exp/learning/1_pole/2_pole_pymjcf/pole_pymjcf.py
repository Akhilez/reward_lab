from random import random

from dm_control import mujoco, mjcf
from dm_control.mujoco.wrapper.mjbindings import enums
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import matplotlib.animation as animation


duration = 2  # (seconds)
framerate = 60  # (Hz)
actuator_strength = 0.2  # [0 to 1]

# Visualize the joint axis
scene_option = mujoco.wrapper.core.MjvOption()
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True
scene_option.frame = enums.mjtFrame.mjFRAME_GEOM
scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True


def build_mjcf():
    arena = mjcf.RootElement()

    # <option gravity="0 0 -9.8" viscosity="0.1" wind="0 0 0" />
    arena.option.gravity = [0, 0, -9.8]
    arena.option.viscosity = 0.1
    arena.option.wind = [0, 0, 0]

    arena.worldbody.add("light", pos=[0, 0, 7], dir=[0, 0, -1])

    # <geom type="plane" size="1 1 0.1" rgba=".9 0.9 0.9 1"/>
    arena.worldbody.add("geom", type="plane", size=[1, 1, 0.1], rgba=[0.9, 0.9, 0.9, -1])

    # <body pos="0 0 0.3" euler="0 0 0">
    body = arena.worldbody.add("body", pos=[0, 0, 0.3], euler=[0, 0, 0])
    pole_body = body.add("body")
    # <geom type="cylinder" size="0.01 0.1" pos="0 0 0" mass="0.05"/>
    pole_body.add("geom", type="cylinder", size=[0.01, 0.1], pos=[0, 0, 0], mass=0.05)
    # <geom type="box" size="0.02 0.02 0.02" pos="0 0 -0.1" mass="1"/>
    pole_body.add("geom", type="box", size=[0.02, 0.02, 0.02], pos=[0, 0, -0.1], mass=1)

    # <joint name="main_hinge" type="hinge" pos="0 0 0.1" axis="0 1 0" damping="0.1" range="-360 360" limited="true" />
    body.add(
        "joint",
        name="main_hinge",
        type="hinge",
        pos=[0, 0, 0.1],
        axis=[0, 1, 0],
        damping=0.1,
        range=[-360, 360],
        limited=True,
    )

    # <site name="end_of_pole" pos="0 0 -0.1" size="0.025"/>
    body.add("site", name="end_of_pole", pos=[0, 0, -0.1], size=[0.025])

    arena.sensor.add("jointvel", joint="main_hinge")  # <jointvel joint="main_hinge" />
    # <framepos objtype="site" objname="end_of_pole"/>
    arena.sensor.add("framepos", objtype="site", objname="end_of_pole")

    # <motor joint="main_hinge" ctrllimited="true" ctrlrange="-10 10"  />
    arena.actuator.add("motor", joint="main_hinge", ctrllimited=True, ctrlrange=[-10, 10])

    return arena


# physics = mujoco.Physics.from_xml_path("/Users/akhildevarashetti/code/reward_lab/exp/di/mujoco_exp/learning/1_pole/pole.xml")
arena = build_mjcf()
physics = mjcf.Physics.from_mjcf_model(arena)

print(arena.to_xml_string())

plot_data = {
    "end_x": [],
    "end_y": [],
    "end_z": [],
}

frames = []  # for storing the generated images
fig = plt.figure()
physics.reset()  # Reset state and time
while True:
    time_prev = physics.data.time
    while physics.data.time - time_prev < 1.0 / framerate:
        physics.step()

    pixels = physics.render(scene_option=scene_option)
    frames.append([plt.imshow(physics.render(), cmap=cm.Greys_r, animated=True)])

    plot_data["end_x"].append(physics.named.data.site_xpos["end_of_pole", "x"])
    plot_data["end_y"].append(physics.named.data.site_xpos["end_of_pole", "y"])
    plot_data["end_z"].append(physics.named.data.site_xpos["end_of_pole", "z"])

    physics.named.data.ctrl[0] += (random() * 2 - 1) * actuator_strength

    if physics.data.time >= duration:
        break

ani = animation.ArtistAnimation(fig, frames, interval=1000 / framerate, blit=True, repeat_delay=1000)
ani.save("movie.mp4")
plt.close()

for label, data in plot_data.items():
    plt.plot(plot_data[label], label=label)
plt.legend()
plt.show()
