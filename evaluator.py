import datetime
import torch

from rlpyt.utils.buffer import numpify_buffer, torchify_buffer
from rlpyt.utils.logging import logger

class Evaluator:
    def __init__(self, agent, env, T=100):
        self.env = env
        self.agent = agent
        self.T = T

    def ctrl(self, itr):
        logger.log("Start evaluating: "f"{itr}")
        self.agent.reset()
        self.agent.eval_mode(itr)

        observation = torchify_buffer(self.env.reset())

        for t in range(self.T):
            act, _ = self.agent.step(observation, None, None)
            action = numpify_buffer(act_pyt)
            obs, r, d, env_info = self.env.step(action)

            if d:
                logger.log("Done in " f"{t} steps.")
                break
        
        logger.log("position: "f"{self.env._env.physics.bounded_position()} and velocity: "f"{self.env._env.physics.velocity()}.")

        



