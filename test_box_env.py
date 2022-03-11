import math
import os
import numpy as np


def test_mujoco():
    from mujoco_py import load_model_from_xml, MjSim, MjViewer
    f=open("dreamer/envs/box.xml","r")
    MODEL_XML = f.read()
    model=load_model_from_xml(MODEL_XML)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    for i in range(1000):
        sim.step()
        viewer.render()

def test_env():
    from dreamer.envs.box import Box
    env=Box()
    env.reset()
    for i in range(10):
        env.step(-1)
        print(env.get_obs())
    
if __name__=="__main__":
    #test_env()
    test_mujoco()
