import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

class WeightModel():
    def __init__(self, size: int, goal_state, device, lr=0.001):
        super().__init__()
        self.goal_state=torch.clone(goal_state).squeeze(0).requires_grad_()
        self.w=torch.ones(size,dtype=torch.double,requires_grad=True).to(device)
        self.lr=lr

    def grad(self, states, rewards):
        states=torch.clone(states).requires_grad_()
        diff=torch.mul(self.w, states)-self.goal_state
        e=torch.sum(torch.mul(diff, diff),dim=1)
        loss=nn.MSELoss(reduction='mean')(e,rewards)
        dloss_dw = grad(outputs=loss, inputs=self.w)
        self.w-=self.lr*dloss_dw[0]
        return loss

import datetime
import os
import argparse
from tqdm import tqdm

from dreamer.agents.dmc_dreamer_agent import DMCDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.envs.dmc import DeepMindControl
from dreamer.envs.time_limit import TimeLimit
from dreamer.envs.action_repeat import ActionRepeat
from dreamer.envs.normalize_actions import NormalizeActions
from dreamer.envs.wrapper import make_wapper
from dreamer.models.rnns import get_feat
from dreamer.utils.configs import configs, load_configs
from dreamer.models.mpc_planner import load_goal_state

class LearnWeight:
    def __init__(self, agent, env, cuda_idx=None, game='cartpole_balance',lr=0.001):
        self.env = env
        self.agent = agent
        self.cuda_idx = cuda_idx
        self.game = game
        self.obs=None
        self.reward=None
        self.device=torch.device("cuda:" + str(self.cuda_idx)) if self.cuda_idx is not None else torch.device("cpu")
        self.w=WeightModel(configs.stochastic_size+configs.deterministic_size, self.goal(),device=self.device,lr=lr)
    
    def goal(self):
        g=load_goal_state(torch.float).to(self.device)
        with torch.no_grad():
            state = self.agent.model.get_state_representation(g)
            feat = get_feat(state)
        return feat[0]

    def collect(self, data_path,B=1000,T=100):
        model = self.agent.model
        self.obs=None
        self.reward=None
        print("Start collecting data")
        for b in tqdm(range(B)):
            observations=[]
            reward=[]
            actions=[torch.zeros(1,1)]
            self.env.reset()
            for t in range(T):
                action=torch.rand(1,1)*2-1
                actions.append(action)
                obs, r, d, env_info = self.env.step(action)
                observation = torch.tensor(obs)
                observations.append(observation)
                reward.append(r)
            reward=torch.tensor(reward).to(self.device)
            observations = torch.stack(observations, dim=0).unsqueeze(1).to(self.device)
            observations = observations.type(torch.float) / 255.0 - 0.5
            actions = torch.stack(actions, dim=0).to(self.device)
            with torch.no_grad():
                embed = model.observation_encoder(observations)
                prev_state=model.representation.initial_state(1, device=self.device, dtype=torch.float)
                _, post = model.rollout.rollout_representation(T, embed, actions, prev_state)
                feat = get_feat(post).squeeze()
            if self.obs is None:
                self.obs=feat
                self.reward=reward
            else:
                self.obs=torch.cat((self.obs, feat))
                self.reward=torch.cat((self.reward, reward))
        self.reward=self.reward.to(self.device)
        np.savez(data_path[:-4], obs=self.obs.cpu(), reward=self.reward.cpu())


    def load(self, path):
         data=np.load(path,allow_pickle=True)
         self.obs=torch.tensor(data['obs']).to(self.device)
         self.reward=torch.tensor(data['reward']).to(self.device)

    def train(self, log_file,e=100):
        print("Start training")
        for i in tqdm(range(e)):
            n=len(self.obs)
            perm=torch.randperm(n)
            s=0
            for j in range(0,n,100):
                loss=self.w.grad(self.obs[perm[j:j+100]], self.reward[perm[j:j+100]])
                s+=loss
            print(s)
        np.save(log_file,self.w.w.cpu())
        
        


def train(cuda_idx=None, game="cartpole_balance",path=None,
        B=1000, T=100, lr=0.001,data_path=None,
        log_file=log_file):
    domain, task = game.split('_')
    domain, task = game.split('_',1)
    if '_' in task:
        d,task=task.split('_')
        domain+='_'+d
    params = torch.load(path)
    agent_state_dict = params.get('agent_state_dict')
    optimizer_state_dict = params.get('optimizer_state_dict')
    action_repeat = configs.action_repeat
    factory_method = make_wapper(
        DeepMindControl,
        [ActionRepeat, NormalizeActions, TimeLimit],
        [dict(amount=action_repeat), dict(), dict(duration=1000 / action_repeat)])

    agent = DMCDreamerAgent(train_noise=0.3, eval_noise=0, expl_type="additive_gaussian",
                              expl_min=None, expl_decay=None, initial_model_state_dict=agent_state_dict,
                               model_kwargs={"domain": domain, "task": task, "cuda_idx": cuda_idx,"wpath":None})
    env=factory_method(name=game)
    agent.initialize(env.spaces)
    agent.to_device(cuda_idx)
    lw=LearnWeight(agent, env,cuda_idx=cuda_idx,lr=lr)
    if not os.path.exists(data_path):
        lw.collect(data_path,B=B,T=T)
    else:
        lw.load(data_path)
    lw.train(log_file)   
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='DMC game', default='cartpole_balance')
    parser.add_argument('--cuda-idx', help='cuda', type=int, default=None)
    parser.add_argument('--model', help='model path', type=str, default=None)
    parser.add_argument('--B', help='', type=int, default=1000)
    parser.add_argument('--T', help='', type=int, default=100)
    parser.add_argument('--lr', help='', type=float, default=0.001)
    parser.add_argument('--data',type=str,default='data')
    args = parser.parse_args()
    data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data+'.npz')
    
    i = 0
    log_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'weight')
    while os.path.exists(os.path.join(log_dir, 'w_' + str(i))):
        i += 1
    log_file=os.path.join(log_dir,'w_'+str(i))
    print(log_file)
    train(game=args.game,cuda_idx=args.cuda_idx,path=args.model,
            B=args.B, T=args.T,lr=args.lr,data_path=data_path,
            log_file=log_file)
