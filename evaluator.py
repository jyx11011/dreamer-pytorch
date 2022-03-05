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
from dreamer.models.rnns import get_feat

from rlpyt.utils.buffer import numpify_buffer, torchify_buffer
from rlpyt.utils.logging import logger

class Evaluator:
    def __init__(self, agent, env, T=100):
        self.env = env
        self.agent = agent
        self.T = T

    def ctrl(self, itr, verbose=False):
        logger.log("\nStart evaluating: "f"{itr}")
        self.agent.reset()
        self.agent.eval_mode(itr)

        observation = torchify_buffer(self.env.reset()).type(torch.float) / 255.0 - 0.5
        action = torch.zeros(1, 1, device=self.agent.device)
        reward = None

        tot=0
        for t in tqdm(range(self.T), desc='mpc'):
            if verbose:
                logger.log("position: "f"{self.env.get_obs()}")
            observation = observation.unsqueeze(0).type(torch.float) / 255.0 - 0.5
            action, _ = self.agent.step(observation, action, reward)
            act = numpify_buffer(action)[0] 
            print(action)
            obs, r, d, env_info = self.env.step(action)
            tot+=r
            if d:
                logger.log("Done in " f"{t} steps.")
                break

            observation = torch.tensor(obs).type(torch.float) / 255.0 - 0.5

        logger.log("position: "f"{self.env.get_obs()}, reward: "f"{tot}")


    def eval_model(self, T=10):
        model = self.agent.model
        logger.log("\nStart evaluating model")
        observations = [torch.tensor(self.env.reset())]
        action = torch.rand(T, 1, 1) * 2 - 1
        for i in range(T):
            obs, r, d, env_info = self.env.step(action[i][0][0])
            observations.append(torch.tensor(obs))
        observations = torch.stack(observations[:-1], dim=0).unsqueeze(1)
        observations = observations.type(torch.float) / 255.0 - 0.5
        
	embed = model.observation_encoder(observations)
        prev_state = model.representation.initial_state(1)
        prior, post = model.rollout.rollout_representation(T, embed, action, prev_state)
        feat = get_feat(post)
        image_pred = model.observation_decoder(feat)
        print(observations-image_pred.mean)
        '''
        for i in range(T):
            print(i)
            print(observations[i], image_pred[i])        
        '''

def eval(load_model_path, game="cartpole_balance",itr=10, eval_model=False):
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
    
    if eval_model:
        evaluator.eval_model()
    else:
        for i in tqdm(range(itr)):
            evaluator.ctrl(i,verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='DMC game', default='cartpole_balance')
    parser.add_argument('--run-ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--load-model-path', help='load model from path', type=str)  # path to params.pkl
    parser.add_argument('--model', help='evaluate model', action='store_true')


    parser.add_argument('--itr', help='total iter', type=int,default=10)  # path to params.pkl
    default_log_dir = os.path.join(
        os.path.dirname(__file__),
        'data',
        'test',
        datetime.datetime.now().strftime("%Y%m%d"))
    parser.add_argument('--log-dir', type=str, default=default_log_dir)
    args = parser.parse_args()
    log_dir = os.path.abspath(args.log_dir)
    '''
    i = args.run_ID
    
    while os.path.exists(os.path.join(log_dir, 'run_' + str(i))):
        i += 1
    print(f'Using run id = {i}')
    args.run_ID = i
    '''
    eval(
        args.load_model_path,
        game=args.game,
        itr=args.itr,
        eval_model=args.model
        )

