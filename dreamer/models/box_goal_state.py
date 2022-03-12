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


def goal_obs(bx=0.24):
    f=open("dreamer/envs/box.xml","r")
    MODEL_XML = f.read()
    physics = Physics.from_xml_string(MODEL_XML, common.ASSETS)
    with physics.reset_context():
         physics.named.data.qpos['slider'][0]=bx
    print("goal:", physics.box_position())
    return physics.render(64, 64, camera_id=0).transpose(2, 0, 1).copy()


if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bx', help='box position', default=0)
    args = parser.parse_args()
    obs0 = goal_obs(2.0)
    obs1 = goal_obs(-3)
    #f='box_goal_state_'+str(args.bx)
    #np.save(f,obs)
    print(obs0, obs1)
    print((obs0-obs1)[obs0 != obs1])
