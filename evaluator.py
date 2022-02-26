import datetime
import torch

from rlpyt.utils.buffer import numpify_buffer, torchify_buffer
from rlpyt.utils.logging import logger

class Evaluator:
    def __init__(self, agent, env, T=50):
        self.env = env
        self.agent = agent
        self.T = T

    def ctrl(self, itr):
        logger.log("Start evaluating: "f"{itr}")
        self.agent.reset()
        self.agent.eval_mode(itr)

        observation = torchify_buffer(self.env.reset())
        action = torch.zeros(1, 1, device=self.agent.device)
        reward = None
        logger.log("position: "f"{self.env.get_obs()}")
        for t in range(self.T):
            observation = observation.unsqueeze(0)
            action, _ = self.agent.step(observation, action, reward)
            act = numpify_buffer(action)[0] 
            obs, r, d, env_info = self.env.step(action)

            if d:
                logger.log("Done in " f"{t} steps.")
                break

            observation = torch.tensor(obs)
        
        logger.log("position: "f"{self.env.get_obs()}")

        



