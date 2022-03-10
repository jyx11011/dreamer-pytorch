import os
import gym
import numpy as np
import torch
import torch.autograd
import torch.nn.functional as tf
from mpc import mpc

class Dynamics(torch.nn.Module):
    def __init__(self, dynamics):
        super(Dynamics, self).__init__()
        self._dynamics=dynamics

    def forward(self, state, action):
        stoch, deter = torch.split(state, [self._dynamics._stoch_size, self._dynamics._deter_size], dim=-1)
        rnn_input = self._dynamics._rnn_input_model(torch.cat([action, stoch], dim=-1))
        deter_state = self._dynamics._cell(rnn_input, deter)
        mean, std = torch.chunk(self._dynamics._stochastic_prior_model(deter_state), 2, dim=-1)
        std = tf.softplus(std) + 0.1
        dist = self._dynamics._dist(mean, std)
        stoch_state = dist.rsample()
        return torch.cat((stoch_state, deter_state), dim=-1)

class PendulumCost(torch.nn.Module):
    def __init__(self, reward):
        super().__init__()
        self._reward = reward

    def forward(self, state):
        s = state[:,:-1]
        u = state[:,-1:][0]
        sc = self._reward(s)[0]
        return  -1000*sc + 0.001 * torch.mul(u,u)

class MPC_planner:
    def __init__(self, nx, nu, dynamics, reward,
            timesteps=40,
            goal_weights=None, ctrl_penalty=0.001, iter=10,
            action_low=-1.0, action_high=1.0):
        self._timesteps=timesteps
        self._u_init = None
        self._iter = iter
        self._nx = nx
        self._nu = nu
        self._action_low = action_low
        self._action_high = action_high
        self._dtype=torch.float
        self._dynamics = Dynamics(dynamics)#.to("cuda")
        self._cost = PendulumCost(reward)

    '''
    def set_goal_state(self, state):
        self._cost = PendulumCost(state)
        self._u_init = None
    '''

    def reset(self):
        self._u_init = None

    def get_next_action(self, state, num=1, mode='sample'):
        if num > self._timesteps:
            num = self._timesteps
        n_batch = state.shape[0]
        if self._u_init is None:
            self._u_init=torch.rand(self._timesteps, n_batch, self._nu)*2-1
        state = torch.clone(state)
        with torch.enable_grad():
            ctrl = mpc.MPC(self._nx, self._nu, self._timesteps, 
                        u_lower=self._action_low * torch.ones(self._timesteps, n_batch, self._nu,device=state.device), 
                        u_upper=self._action_high * torch.ones(self._timesteps, n_batch, self._nu,device=state.device), 
                        lqr_iter=self._iter, 
                        n_batch=n_batch,
                        u_init=self._u_init,
                        max_linesearch_iter=10,
                        linesearch_decay=0.2,
                        exit_unconverged=False, 
                        detach_unconverged = False, 
                        backprop=False,
                        verbose=1,
                        eps=1e-2,
                        #delta_u=0.5,
                        grad_method=mpc.GradMethods.AUTO_DIFF)
            nominal_states, nominal_actions, nominal_objs = ctrl(state, self._cost, self._dynamics)
        action = nominal_actions[:num]
        if mode == 'eval':
            self._u_init = torch.cat((nominal_actions[num:], torch.zeros(num, n_batch, self._nu, dtype=self._dtype,device=action.device)), dim=0)
        return action

def load_goal_state(dtype, domain = "cartpole", task = "balance"):
    goal_state_obs = np.load(os.getcwd()+'/dreamer/models/'+domain+'/'+domain+'_'+task+'.npy')
    return torch.tensor(goal_state_obs / 255.0 - 0.5, dtype=dtype).unsqueeze(0)
