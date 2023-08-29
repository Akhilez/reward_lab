import numpy as np
from dm_control import mjcf

_rgba = (0.5, 0.8, 0.8, 1)
_size = 0.1


def build_crawler(n_legs=4):
    _model = mjcf.RootElement(model="crawler")
    _model.worldbody.add(
        "geom",
        name="torso",
        type="ellipsoid",
        size=(_size, _size, _size / 2),
        rgba=_rgba,
    )

    for i in range(n_legs):
        theta = 2 * i * np.pi / n_legs
        pos = np.array([np.cos(theta), np.sin(theta), 0]) * _size
        site = _model.worldbody.add("site", pos=pos, euler=[0, 0, theta])
        site.attach(_build_leg())

    return _model


def _build_leg():
    model = mjcf.RootElement(model="leg")

    # Defaults:
    model.default.joint.damping = 2
    model.default.joint.type = "hinge"
    model.default.geom.type = "capsule"
    model.default.geom.rgba = _rgba
    model.default.position.ctrllimited = True
    model.default.position.ctrlrange = (-0.5, 0.5)

    # Thigh:
    thigh = model.worldbody.add("body", name="thigh")
    hip = thigh.add("joint", axis=[0, 0, 1])
    thigh.add("geom", fromto=[0, 0, 0, _size, 0, 0], size=[_size / 4])

    # Hip:
    shin = thigh.add("body", name="shin", pos=[_size, 0, 0])
    knee = shin.add("joint", axis=[0, 1, 0])
    shin.add("geom", fromto=[0, 0, 0, 0, 0, -_size], size=[_size / 5])

    # Position actuators:
    model.actuator.add("position", name="hip_joint", joint=hip, kp=10)
    model.actuator.add("position", name="knee_joint", joint=knee, kp=10)

    return model


def main():
    from random import random
    from dm_control.locomotion.arenas import Floor
    from matplotlib import pyplot as plt
    from matplotlib import cm
    from matplotlib.animation import ArtistAnimation

    _arena = Floor().mjcf_model
    _arena.worldbody.add("light", pos=(0, 0, 4))

    site = _arena.worldbody.add("site", pos=[0, 0, 0.2])
    crawler = site.attach(build_crawler(n_legs=3))
    crawler.add("freejoint")

    physics = mjcf.Physics.from_mjcf_model(_arena)

    # physics.bind(crawler).pos[:] = [0, 0, 0.5]

    duration = 2  # (seconds)
    framerate = 60  # (Hz)
    actuator_strength = 0.2  # [0 to 1]
    frames = []  # for storing the generated images
    fig = plt.figure()
    physics.reset()  # Reset state and time
    while True:
        time_prev = physics.data.time
        while physics.data.time - time_prev < 1.0 / framerate:
            physics.step()

        frames.append([plt.imshow(physics.render(), cmap=cm.Greys_r, animated=True)])
    
        physics.named.data.ctrl[0] += (random() * 2 - 1) * actuator_strength
    
        if physics.data.time >= duration:
            break

    ani = ArtistAnimation(fig, frames, interval=1000 / framerate, blit=True, repeat_delay=1000)
    ani.save("temp.mp4")
    print()







if __name__ == '__main__':
    main()




