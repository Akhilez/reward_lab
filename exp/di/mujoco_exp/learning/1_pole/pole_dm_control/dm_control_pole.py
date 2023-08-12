from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import matplotlib.animation as animation


duration = 2  # (seconds)
framerate = 60  # (Hz)

# Visualize the joint axis
scene_option = mujoco.wrapper.core.MjvOption()
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True
scene_option.frame = enums.mjtFrame.mjFRAME_GEOM
scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True


physics = mujoco.Physics.from_xml_path("/exp/di/mujoco_exp/learning/1_pole/pole.xml")


frames = []  # for storing the generated images
fig = plt.figure()
physics.reset()  # Reset state and time
while True:
    time_prev = physics.data.time
    while physics.data.time - time_prev < 1.0 / framerate:
        physics.step()

    pixels = physics.render(scene_option=scene_option)
    frames.append([plt.imshow(physics.render(), cmap=cm.Greys_r, animated=True)])

    if physics.data.time >= duration:
        break

ani = animation.ArtistAnimation(fig, frames, interval=1000/framerate, blit=True, repeat_delay=1000)
ani.save('movie.mp4')
