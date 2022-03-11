from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree
from dreamer.envs.box import Physics


def goal_obs():
    f=open("dreamer/envs/box.xml","r")
    MODEL_XML = f.read()
    physics = Physics.from_xml_string(MODEL_XML, common.ASSETS)
    return physics.render(64, 64, camera_id=0).transpose(2, 0, 1).copy()