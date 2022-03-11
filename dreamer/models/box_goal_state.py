import argparse
import numpy as np

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree
from dreamer.envs.box import Physics


def goal_obs(bx=0):
    f=open("dreamer/envs/box.xml","r")
    MODEL_XML = f.read().format(bx=bx)
    physics = Physics.from_xml_string(MODEL_XML, common.ASSETS)
    print("goal:", physics.box_position())
    return physics.render(64, 64, camera_id=0).transpose(2, 0, 1).copy()


if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bx', help='box position', default=0)
    obs = goal_obs(args.bx)
    f='box_goal_state_'+str(bx)
    np.save(f,obs)