import datetime
import torch
from tqdm import tqdm

from rlpyt.utils.buffer import numpify_buffer, torchify_buffer
from rlpyt.utils.logging import logger

class Evaluator:
    def __init__(self, agent, env, T=100):
        self.env = env
        self.agent = agent
        self.T = T

    def ctrl(self, itr):
        logger.log("\nStart evaluating: "f"{itr}")
        self.agent.reset()
        self.agent.eval_mode(itr)

        observation = torchify_buffer(self.env.reset())
        action = torch.zeros(1, 1, device=self.agent.device)
        reward = None

        tot=0
        for t in tqdm(range(self.T), desc='mpc'):
            observation = observation.unsqueeze(0)
            action, _ = self.agent.step(observation, action, reward)
            act = numpify_buffer(action)[0] 
            obs, r, d, env_info = self.env.step(action)
            tot+=r
            if d:
                logger.log("Done in " f"{t} steps.")
                break

            observation = torch.tensor(obs)
        
        logger.log("position: "f"{self.env.get_obs()}, reward: "f"{tot}")

        



