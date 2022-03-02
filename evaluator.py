import datetime
import os
import argparse
import torch
from tqdm import tqdm

from dreamer.agents.dmc_dreamer_agent import DMCDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.envs.dmc import DeepMindControl
from dreamer.envs.time_limit import TimeLimit
from dreamer.envs.action_repeat import ActionRepeat
from dreamer.envs.normalize_actions import NormalizeActions
from dreamer.envs.wrapper import make_wapper

from rlpyt.utils.buffer import numpify_buffer, torchify_buffer
from rlpyt.utils.logging import logger

class Evaluator:
    def __init__(self, agent, env, T=100):
        self.env = env
        self.agent = agent
        self.T = T
        self.dtype=torch.float

    def ctrl(self, itr, verbose=False):
        logger.log("\nStart evaluating: "f"{itr}")
        self.agent.reset()
        self.agent.eval_mode(itr)

        observation = torchify_buffer(self.env.reset())
        action = torch.zeros(1, 1, device=self.agent.device,dtype=self.dtype)
        reward = None

        tot=0
        for t in tqdm(range(self.T), desc='mpc'):
            if verbose:
                logger.log("position: "f"{self.env.get_obs()}")
            observation = observation.unsqueeze(0)
            action, _ = self.agent.step(observation, action, reward)
            act = numpify_buffer(action)[0] 
            if verbose:
                logger.log(f"{action}")
            obs, r, d, env_info = self.env.step(action)
            tot+=r
            if d:
                logger.log("Done in " f"{t} steps.")
                break

            observation = torch.tensor(obs)

        logger.log("position: "f"{self.env.get_obs()}, reward: "f"{tot}")

        

def eval(load_model_path, game="cartpole_balance",itr=10):
    params = torch.load(load_model_path) if load_model_path else {}
    agent_state_dict = params.get('agent_state_dict')
    optimizer_state_dict = params.get('optimizer_state_dict')
    action_repeat = 2
    factory_method = make_wapper(
        DeepMindControl,
        [ActionRepeat, NormalizeActions, TimeLimit],
        [dict(amount=action_repeat), dict(), dict(duration=1000 / action_repeat)])

    agent = DMCDreamerAgent(train_noise=0.3, eval_noise=0, expl_type="additive_gaussian",
                              expl_min=None, expl_decay=None, initial_model_state_dict=agent_state_dict)
    env=factory_method(name=game)
    agent.initialize(env.spaces)
    evaluator=Evaluator(agent, env)
    
    for i in tqdm(range(itr)):
        evaluator.ctrl(i,verbose=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='DMC game', default='cartpole_balance')
    parser.add_argument('--run-ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--load-model-path', help='load model from path', type=str)  # path to params.pkl
    
    parser.add_argument('--itr', help='total iter', type=int,default=10)  # path to params.pkl
    default_log_dir = os.path.join(
        os.path.dirname(__file__),
        'data',
        'test',
        datetime.datetime.now().strftime("%Y%m%d"))
    parser.add_argument('--log-dir', type=str, default=default_log_dir)
    args = parser.parse_args()
    log_dir = os.path.abspath(args.log_dir)
    i = args.run_ID
    while os.path.exists(os.path.join(log_dir, 'run_' + str(i))):
        i += 1
    print(f'Using run id = {i}')
    args.run_ID = i
    eval(
        args.load_model_path,
        game=args.game,
        itr=args.itr
        )

