import datetime
import os
import argparse
import torch
from tqdm import tqdm
from torch.autograd import Function, Variable
import numpy as np
from mpc import mpc

from dreamer.agents.dmc_dreamer_agent import DMCDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.envs.dmc import DeepMindControl
from dreamer.envs.time_limit import TimeLimit
from dreamer.envs.action_repeat import ActionRepeat
from dreamer.envs.normalize_actions import NormalizeActions
from dreamer.envs.wrapper import make_wapper

class MPC_planner:
    def __init__(self, nx, nu, dynamics,
            timesteps=50,
            goal_weights=None, ctrl_penalty=0.001, iter=50,
            action_low=-1.0, action_high=1.0):
        self._timesteps=timesteps
        self._u_init = None
        self._iter = iter
        self._nx = nx
        self._nu = nu
        self._action_low = action_low
        self._action_high = action_high
        self._dtype=torch.float

        if goal_weights is None:
            goal_weights = torch.rand(nx, dtype=self._dtype)
        self._goal_weights = goal_weights
        q = torch.cat((
            goal_weights,
            ctrl_penalty * torch.ones(nu, dtype=self._dtype)
        ))
        self._Q = torch.diag(q).repeat(timesteps, 1, 1).type(self._dtype)
        self._dynamics = dynamics

    def set_goal_state(self, state):
        goal_state = torch.clone(state)
        self._goal_weights=self._goal_weights.to(state.device)
        px = -torch.sqrt(self._goal_weights) * goal_state
        p = torch.cat((px, torch.zeros(self._nu, dtype=self._dtype,device=state.device)))
        p = p.repeat(self._timesteps, 1)
        self._Q=self._Q.to(state.device)
        self._cost = mpc.QuadCost(self._Q, p)
        self._u_init = None
    
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
                        max_linesearch_iter=20,
                        linesearch_decay=0.2,
                        exit_unconverged=False, 
                        backprop=False,
                        verbose=0,
                        delta_u=0.5,
                        eps=1e-5,
                        grad_method=mpc.GradMethods.AUTO_DIFF)
            nominal_states, nominal_actions, nominal_objs = ctrl(state, self._cost, self._dynamics)
        action = nominal_actions[:num]
        #if mode == 'eval':
        #    self._u_init = torch.cat((nominal_actions[num:], torch.zeros(num, n_batch, self._nu, dtype=self._dtype,device=action.device)), dim=0)
        return action


class CartpoleDx(torch.nn.Module):
    def __init__(self, params=None):
        super().__init__()

        self.n_state = 5
        self.n_ctrl = 1

        # model parameters
        # gravity, masscart, masspole, length
        self.params = Variable(torch.Tensor((9.8, 1.0, 0.1, 1)))
        self.theta_threshold_radians = np.pi#12 * 2 * np.pi / 360
        self.dt = 0.02

        self.lower = -1
        self.upper = 1

        # 0  1      2        3   4
        # x dx cos(th) sin(th) dth
        self.goal_state = torch.Tensor(  [ 0.,  0.,  1., 0.,   0.])
        self.goal_weights = torch.Tensor([0.1, 0.1,  1., 1., 0.1])
        self.ctrl_penalty = 0.001

        self.mpc_eps = 1e-4
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 10

    def forward(self, state, u):

        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()
        gravity, masscart, masspole, length = torch.unbind(self.params)
        polemass_length = masspole * length
        x, dx, cos_th, sin_th, dth = torch.unbind(state, dim=1)
        total_mass = masspole + masscart
        th = torch.atan2(sin_th, cos_th)
        u=u[:,0]
        th_acc = (-u * cos_th - masspole*length*dth**2*sin_th*cos_th+total_mass*gravity*sin_th)\
            / length / (masscart + masspole * sin_th**2)
        xacc = (u + masspole *sin_th*(length*dth**2-gravity*cos_th))/(masscart+masspole*sin_th**2)
        x = x + self.dt * dx
        dx = dx + self.dt * xacc
        th = th + self.dt * dth
        dth = dth + self.dt * th_acc
        state = torch.stack((
            x, dx, torch.cos(th), torch.sin(th), dth
        ), 1)

        return state


def ctrl(game='cartpole_balance'):
    action_repeat=2
    factory_method = make_wapper(
        DeepMindControl,
        [ActionRepeat, NormalizeActions, TimeLimit],
        [dict(amount=action_repeat), dict(), dict(duration=1000 / action_repeat)])
    env=factory_method(name=game)

    cartpole = CartpoleDx()
    planner=MPC_planner(5, 1, cartpole, goal_weights=cartpole.goal_weights)
    planner.set_goal_state(cartpole.goal_state)

    env.reset()
    obs=env.get_obs()
    x=obs['position'][0]
    cos=obs['position'][1]
    sin=obs['position'][2]
    dx=obs['velocity'][0]
    dth=obs['velocity'][1]
    state=torch.tensor([[x, dx, cos, sin, dth]], dtype=torch.float)
    tot=0
    r=0
    #actions = planner.get_next_action(state, mode='eval',num=50)
    for t in tqdm(range(500), desc='mpc'):
        print("position: "f"{env.get_obs()}, reward: "f"{r}")
        action = planner.get_next_action(state, mode='eval')
        print(action)
        print(cartpole(state, action[0]))
        obs, r, d, env_info = env.step(action[0].item())
        tot+=r
        if d:
            print("Done in " f"{t} steps.")
            break
        obs=env.get_obs()
        x=obs['position'][0]
        cos=obs['position'][1]
        sin=obs['position'][2]
        dx=obs['velocity'][0]
        dth=obs['velocity'][1]
        state=torch.tensor([[x, dx, cos, sin, dth]], dtype=torch.float)
    
    print("position: "f"{env.get_obs()}, reward: "f"{tot}")

if __name__ == "__main__":
    ctrl()
