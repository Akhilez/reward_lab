from os.path import dirname, join

import mujoco as mj
import random


def controller(model, data):
    # put the controller here. This function is called inside the simulation.
    pass


def main():
    xml_path = "pole.xml"  # xml file (assumes this is in the same folder as this file)
    simend = 5  # simulation time

    # get the full path
    xml_path = join(dirname(__file__) + "/" + xml_path)

    # MuJoCo data structures
    model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
    data = mj.MjData(model)  # MuJoCo data

    # set the controller
    mj.set_mjcb_control(controller)

    while True:
        time_prev = data.time

        while data.time - time_prev < 1.0 / 60.0:
            mj.mj_step(model, data)

        print(f"{data.sensordata[0]=:.3f}\t{data.actuator_force[0]=:.2f}")
        data.ctrl[0] += random.random() * 2 - 1

        if data.time >= simend:
            break


if __name__ == "__main__":
    main()
