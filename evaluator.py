import datetime
import os
import argparse
import torch
from tqdm import tqdm
import numpy as np

from dreamer.agents.dmc_dreamer_agent import DMCDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.envs.dmc import DeepMindControl
from dreamer.envs.time_limit import TimeLimit
from dreamer.envs.action_repeat import ActionRepeat
from dreamer.envs.normalize_actions import NormalizeActions
from dreamer.envs.wrapper import make_wapper
from dreamer.models.rnns import get_feat
from dreamer.utils.configs import configs, load_configs

from rlpyt.utils.buffer import numpify_buffer, torchify_buffer
from rlpyt.utils.logging import logger

class Evaluator:
    def __init__(self, agent, env, T=100, cuda_idx=None):
        self.env = env
        self.agent = agent
        self.T = T
        self.cuda_idx = cuda_idx
        self.action_dim=env.spaces.action.shape[0]

    def ctrl(self, itr, verbose=False, log_path=None):
        logger.log("\nStart evaluating: "f"{itr}")
        self.agent.reset()
        self.agent.eval_mode(itr)
        self.agent.model.update_mpc_planner()
        device = torch.device("cuda:" + str(self.cuda_idx)) if self.cuda_idx is not None else torch.device("cpu")

        observation = torchify_buffer(self.env.reset()).type(torch.float)
        action = torch.zeros(1, self.action_dim, device=self.agent.device).to(device)
        reward = None

        observations = []
        actions = []
        tot=0
        for t in tqdm(range(self.T), desc='mpc'):
            if verbose:
                logger.log("position: "f"{self.env.get_obs()}")
            observation = observation.unsqueeze(0).type(torch.float).to(device)
            observations.append(self.env.get_obs())
            action, _ = self.agent.step(observation, action, reward)
            act = numpify_buffer(action)[0] 
            actions.append(act)
            print(action)
            obs, r, d, env_info = self.env.step(action)
            tot+=r
            if d:
                logger.log("Done in " f"{t} steps.")
                break
            if verbose:
                print(r)
            observation = torch.tensor(obs).type(torch.float)

        if log_path is not None:
            np.savez(log_path, observations=observations, actions=actions)
        logger.log("position: "f"{self.env.get_obs()}, reward: "f"{tot}")


    def eval_model(self, T=10):
        model = self.agent.model
        self.agent.reset()
        self.agent.eval_mode(0)
        self.agent.model.update_mpc_planner()
        device = torch.device("cuda:" + str(self.cuda_idx)) if self.cuda_idx is not None else torch.device("cpu")

        logger.log("\nStart evaluating model")

        observation = torchify_buffer(self.env.reset()).type(torch.float)
        observations = [observation]
        action = torch.zeros(1, self.action_dim, device=self.agent.device).to(device)
        reward = None
        actions = []
        tot=0
        for t in range(T):
            observation = observation.unsqueeze(0).to(device)
            action, _ = self.agent.step(observation, action.to(device), reward)
            actions.append(action)
            act = numpify_buffer(action)[0] 
            print(action[0])
            obs, r, d, env_info = self.env.step(action)
            observation = torch.tensor(obs).type(torch.float)
            observations.append(observation)

        observations = torch.stack(observations[:-1], dim=0).unsqueeze(1).to(device)
        observations = observations.type(torch.float) / 255.0 - 0.5
        actions = torch.stack(actions, dim=0).to(device)
        with torch.no_grad():
            embed = model.observation_encoder(observations)
            prev_state = model.representation.initial_state(1, device=device)
            prior, post = model.rollout.rollout_representation(T, embed, actions, prev_state)
            feat = get_feat(post)
            image_pred = model.observation_decoder(feat)
        diff=torch.abs(observations-image_pred.mean)
        print(torch.sum(torch.where(diff>0.01,1,0)))
        '''
        for i in range(T):
            print(i)
            print(observations[i], image_pred[i])        
        '''

def eval(load_model_path, cuda_idx=None, game="cartpole_balance",itr=10, eval_model=None, 
        save=True, log_dir=None):
    domain, task = game.split('_',1)
    if '_' in task:
        d,task=task.split('_')
        domain+='_'+d
    params = torch.load(load_model_path) if load_model_path else {}
    agent_state_dict = params.get('agent_state_dict')
    optimizer_state_dict = params.get('optimizer_state_dict')
    action_repeat = configs.action_repeat
    factory_method = make_wapper(
        DeepMindControl,
        [ActionRepeat, NormalizeActions, TimeLimit],
        [dict(amount=action_repeat), dict(), dict(duration=1000 / action_repeat)])

    agent = DMCDreamerAgent(train_noise=0.3, eval_noise=0, expl_type="additive_gaussian",
                              expl_min=None, expl_decay=None, initial_model_state_dict=agent_state_dict,
                               model_kwargs={"domain": domain, "task": task, "cuda_idx": cuda_idx})
    env=factory_method(name=game)
    agent.initialize(env.spaces)
    agent.to_device(cuda_idx)
    evaluator=Evaluator(agent, env, cuda_idx=cuda_idx)
    
    if eval_model is not None:
        evaluator.eval_model(T=eval_model)
    else:
        for i in tqdm(range(itr)):
            path = None
            if log_dir is not None:
                path = os.path.join(log_dir, 'iter_'+str(i))
            evaluator.ctrl(i,verbose=True, log_path=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--timesteps', type=int,default=None)
    parser.add_argument('--iter', type=int, default=None)
    parser.add_argument('--max_linesearch_iter', type=float, default=None)
    parser.add_argument('--linesearch_decay', type=float, default=None)
    parser.add_argument('--eps', type=float, default=None)
    parser.add_argument('--detach_unconverged', type=bool, default=None)
    parser.add_argument('--backprop', type=bool, default=None)
    parser.add_argument('--delta_u', type=float, default=None)
    parser.add_argument('--eval_buffer_size', type=float, default=None)

    parser.add_argument('--game', help='DMC game', default='cartpole_balance')
    parser.add_argument('--cuda-idx', help='cuda', type=int, default=None)
    parser.add_argument('--run-ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--load-model-path', help='load model from path', type=str)  # path to params.pkl
    parser.add_argument('--model', help='evaluate model', type=int, default=None)
    parser.add_argument('--itr', help='total iter', type=int,default=10)  # path to params.pkl

    parser.add_argument('--save', help='save', type=bool,default=True)  # path to params.pkl
    
    args = parser.parse_args()

    load_dir = os.path.dirname(args.load_model_path)
    load_configs(load_dir=load_dir)
    configs.update(args)

    log_dir = os.path.join(os.path.dirname(args.load_model_path), 'eval_log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    i = args.run_ID
    while os.path.exists(os.path.join(log_dir, 'run_' + str(i))):
        i += 1
    print(f'Using run id = {i}')
    args.run_ID = i
    log_dir = os.path.join(log_dir, 'run_'+str(i))
    os.mkdir(log_dir)
    
    configs.save(log_dir)
    eval(
        args.load_model_path,
        cuda_idx=args.cuda_idx,
        game=args.game,
        itr=args.itr,
        eval_model=args.model,
        save=args.save,
        log_dir=log_dir
        )
 
