from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os
import numpy as np
from box import Box


def test_mujoco():
    f=open("dreamer/envs/box.xml","r")
    MODEL_XML = f.read()
    model=load_model_from_xml(MODEL_XML)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    for i in range(1000):
        sim.step()
        viewer.render

def test_env():
    env=Box()
    env.reset()
    for i in range(10):
        env.step(1)
    
if __name__=="__main__":
    test_env()
