import os
import gym
import numpy as np
import torch
import torch.autograd
import torch.nn.functional as tf
from mpc import mpc

class Dynamics(torch.nn.Module):
    def __init__(self, dynamics):
        super().__init__()
        self._dynamics=dynamics

    def forward(self, state, action):
        #stoch, deter = torch.split(state, [self._dynamics._stoch_size, self._dynamics._deter_size], dim=-1)
        #rnn_input = self._dynamics._rnn_input_model(torch.cat([action, stoch], dim=-1))
        #deter_state = self._dynamics._cell(rnn_input, deter)
        #mean, std = torch.chunk(self._dynamics._stochastic_prior_model(deter_state), 2, dim=-1)
        #std = tf.softplus(std) + 0.1
        #dist = self._dynamics._dist(mean, std)
        #stoch_state = dist.rsample()
        #return torch.cat((stoch_state, deter_state), dim=-1)
        #return torch.cat((stoch, deter), dim=-1)
        s=state.requires_grad()
        s=s*0.1
        print(state.requires_grad,s.requires_grad)
        return s

class MPC_planner:
    def __init__(self, timesteps, n_batch, nx, nu, dynamics,
            goal_weights=None, ctrl_penalty=0.001, iter=5,
            action_low=None, action_high=None, eps=0.01):
        self._timesteps = timesteps
        self._n_batch = n_batch
        self._u_init = None
        self._iter = iter
        self._nx = nx
        self._nu = nu
        self._action_low = torch.ones([timesteps, n_batch, nu]) * action_low
        self._action_high = torch.ones([timesteps, n_batch, nu]) * action_high
        self._eps = eps
        self._dtype=torch.float

        if goal_weights is None:
            goal_weights = torch.ones(nx, dtype=self._dtype)
        self._goal_weights = goal_weights
        q = torch.cat((
            goal_weights,
            ctrl_penalty * torch.ones(nu, dtype=self._dtype)
        ))
        self._Q = torch.diag(q).repeat(timesteps, n_batch, 1, 1).type(self._dtype)  # T x B x nx+nu x nx+nu
        self._dynamics = Dynamics(dynamics)

    def set_goal_state(self, state):
        goal_state = torch.clone(state)[0]
        px = -torch.sqrt(self._goal_weights) * goal_state
        p = torch.cat((px, torch.zeros(self._nu, dtype=self._dtype)))
        p = p.repeat(self._timesteps, self._n_batch, 1)
        self._cost = mpc.QuadCost(self._Q, p)
        self._u_init = None

    def get_next_action(self, state):
        ctrl = mpc.MPC(self._nx, self._nu, self._timesteps, 
                        u_lower=self._action_low, u_upper=self._action_high, 
                        lqr_iter=self._iter, eps=self._eps, n_batch=1,
                        u_init=self._u_init,
                        exit_unconverged=False, backprop=False, verbose=1, 
                        grad_method=mpc.GradMethods.AUTO_DIFF)
        nominal_states, nominal_actions, nominal_objs = ctrl(state, self._cost, self._dynamics)
        action = nominal_actions[0] 
        self._u_init = torch.cat((nominal_actions[1:], torch.zeros(1, self._n_batch, self._nu, dtype=self._dtype)), dim=0)

        return action

def load_goal_state(dtype):
    domain = "cartpole"
    task = "balance"
    goal_state_obs = np.load(os.getcwd()+'/dreamer/models/'+domain+'/'+domain+'_'+task+'.npy')
    return torch.tensor(goal_state_obs / 255.0 - 0.5, dtype=dtype).unsqueeze(0)
