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


class MPC_planner:
    def __init__(self, nx, nu, dynamics,
            timesteps=10,
            goal_weights=None, ctrl_penalty=0.001, iter=5,
            action_low=None, action_high=None):
        self._timesteps=timesteps
        self._u_init = None
        self._iter = iter
        self._nx = nx
        self._nu = nu
        self._action_low = action_low.item()
        self._action_high = action_high.item()
        self._dtype=torch.float

        if goal_weights is None:
            goal_weights = torch.ones(nx, dtype=self._dtype)
        self._goal_weights = goal_weights
        q = torch.cat((
            goal_weights,
            ctrl_penalty * torch.ones(nu, dtype=self._dtype)
        ))
        self._Q = torch.diag(q).repeat(timesteps, 1, 1).type(self._dtype)
        self._dynamics = Dynamics(dynamics)#.to("cuda")

    def set_goal_state(self, state):
        goal_state = torch.clone(state)[0]
        self._goal_weights=self._goal_weights.to(state.device)
        px = -torch.sqrt(self._goal_weights) * goal_state
        p = torch.cat((px, torch.zeros(self._nu, dtype=self._dtype,device=state.device)))
        p = p.repeat(self._timesteps, 1)
        self._Q=self._Q.to(state.device)
        self._cost = mpc.QuadCost(self._Q, p)
        self._u_init = None
        self._dynamics=self._dynamics.to(state.device)

    def get_next_action(self, state):
        n_batch = state.shape[0]
        #self._u_init=torch.rand(self._timesteps, n_batch, self._nu)*2-1
        state = torch.clone(state)
        with torch.enable_grad():
            ctrl = mpc.MPC(self._nx, self._nu, self._timesteps, 
                        u_lower=self._action_low * torch.ones(self._timesteps, n_batch, self._nu,device=state.device), 
                        u_upper=self._action_high * torch.ones(self._timesteps, n_batch, self._nu,device=state.device), 
                        lqr_iter=self._iter, 
                        n_batch=n_batch,
                        u_init=self._u_init,
                        max_linesearch_iter=10,
                        linesearch_decay=0.5,
                        exit_unconverged=False, 
                        backprop=True, 
                        detach_unconverged = False, 
                        verbose=0, 
                        grad_method=mpc.GradMethods.AUTO_DIFF)
            nominal_states, nominal_actions, nominal_objs = ctrl(state, self._cost, self._dynamics)
        action = nominal_actions[0]
        self._u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, self._nu, dtype=self._dtype,device=action.device)), dim=0)
        return action

def load_goal_state(dtype):
    domain = "cartpole"
    task = "balance"
    goal_state_obs = np.load(os.getcwd()+'/dreamer/models/'+domain+'/'+domain+'_'+task+'.npy')
    return torch.tensor(goal_state_obs / 255.0 - 0.5, dtype=dtype).unsqueeze(0)
