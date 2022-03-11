from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os
import numpy as np

from rlpyt.envs.base import Env, EnvStep
from rlpyt.utils.collections import namedarraytuple
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox

from dreamer.envs.env import EnvInfo

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree

class Physics(mujoco.Physics):
  def box_position(self):
    return self.named.data.qpos['slider'][0]

class Box(Env):

    def __init__(self, size=(64, 64)):

        self.physics = Physics.from_xml_string(common.read_model('dreamer/envs/box.xml'), common.ASSETS)
        self._size = size
        camera = 0
        self._camera = camera
        self._obs=None

    @property
    def observation_space(self):
        return IntBox(low=0, high=255, shape=(3,) + self._size,
                      dtype="uint8")

    @property
    def action_space(self):
        return FloatBox(low=-1.0, high=1.0)

    def step(self, action):
        self.physics.set_control(action)
        self.physics.step()
        obs = self.render()
        reward = 0
        done = False
        self._obs=self.physics.box_position()

        info = EnvInfo(np.array(1, np.float32), None, done)
        return EnvStep(obs, reward, done, info)

    def reset(self):
        self._physics.reset_context()
        obs = self.render()
        return obs

    def render(self, *args, **kwargs):
        return self.physics.render(*self._size, camera_id=self._camera).transpose(2, 0, 1).copy()

    def get_obs(self):
        return self._obs
