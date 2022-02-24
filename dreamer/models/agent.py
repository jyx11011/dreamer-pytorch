import numpy as np
import torch
import torch.nn as nn
from rlpyt.utils.buffer import buffer_func
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, to_onehot, from_onehot

from dreamer.models.observation import ObservationDecoder, ObservationEncoder
from dreamer.models.rnns import RSSMState, RSSMRepresentation, RSSMTransition, RSSMRollout, get_feat

from dreamer.models.mpc_planner import MPC_planner, load_goal_state

from dreamer.utils.module import get_parameters, FreezeParameters

class AgentModel(nn.Module):
    def __init__(
            self,
            action_shape,
            stochastic_size=3,
            deterministic_size=20,
            hidden_size=20,
            image_shape=(3, 64, 64),
            action_dist='one_hot',
            dtype=torch.float,
            use_pcont=False,
            pcont_layers=3,
            pcont_hidden=20,
            **kwargs,
    ):
        super().__init__()
        self.observation_encoder = ObservationEncoder(shape=image_shape)
        encoder_embed_size = self.observation_encoder.embed_size
        decoder_embed_size = stochastic_size + deterministic_size
        self.observation_decoder = ObservationDecoder(embed_size=decoder_embed_size, shape=image_shape)
        self.action_shape = action_shape
        output_size = np.prod(action_shape)
        self.transition = RSSMTransition(output_size, stochastic_size, deterministic_size, hidden_size)
        self.representation = RSSMRepresentation(self.transition, encoder_embed_size, output_size, stochastic_size,
                                                 deterministic_size, hidden_size)
        self.rollout = RSSMRollout(self.representation, self.transition)
        feature_size = stochastic_size + deterministic_size
        self.action_size = output_size
        self.action_dist = action_dist
        self.dtype = dtype
        
        self.mpc_planner = MPC_planner(feature_size, output_size, self.transition, 
                action_low = kwargs['action_low'], action_high = kwargs['action_high'])
        self.goal_state = load_goal_state(dtype)
        self.mpc_planner.set_goal_state(self.zero_action(self.goal_state))
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        if use_pcont:
            self.pcont = DenseModel(feature_size, (1,), pcont_layers, pcont_hidden, dist='binary')
        self.goal_state = self.goal_state.cuda()

    def forward(self, observation: torch.Tensor, prev_action: torch.Tensor = None, prev_state: RSSMState = None,
            rand=False):
        state = self.get_state_representation(observation, prev_action, prev_state)
        if rand:
            action = torch.randn(*prev_action.shape)
        else:
            action = self.policy(state)
        return action, state

    def policy(self, state: RSSMState):
        feat = get_feat(state)
        action = self.mpc_planner.get_next_action(feat)
        '''
        if self.action_dist == 'tanh_normal':
            if self.training:  # use agent.train(bool) or agent.eval()
                action = action_dist.rsample()
            else:
                action = action_dist.mode()
        elif self.action_dist == 'one_hot':
            action = action_dist.sample()
            # This doesn't change the value, but gives us straight-through gradients
            action = action + action_dist.probs - action_dist.probs.detach()
        elif self.action_dist == 'relaxed_one_hot':
            action = action_dist.rsample()
        else:
            action = action_dist.sample()
        '''
        return action

    def get_state_representation(self, observation: torch.Tensor, prev_action: torch.Tensor = None,
                                 prev_state: RSSMState = None):
        """

        :param observation: size(batch, channels, width, height)
        :param prev_action: size(batch, action_size)
        :param prev_state: RSSMState: size(batch, state_size)
        :return: RSSMState
        """
        obs_embed = self.observation_encoder(observation)
        if prev_action is None:
            prev_action = torch.zeros(observation.size(0), self.action_size,
                                      device=observation.device, dtype=observation.dtype)
        if prev_state is None:
            prev_state = self.representation.initial_state(prev_action.size(0), device=prev_action.device,
                                                           dtype=prev_action.dtype)
        _, state = self.representation(obs_embed, prev_action, prev_state)
        return state

    def get_state_transition(self, prev_action: torch.Tensor, prev_state: RSSMState):
        """

        :param prev_action: size(batch, action_size)
        :param prev_state: RSSMState: size(batch, state_size)
        :return: RSSMState
        """
        state = self.transition(prev_action, prev_state)
        return state

    def forward(self, observation: torch.Tensor, prev_action: torch.Tensor = None, prev_state: RSSMState = None):
        return_spec = ModelReturnSpec(None, None)
        raise NotImplementedError()

    def zero_action(self, obs):
        with torch.no_grad():
            latent = self.representation.initial_state(len(obs), device=obs.device)
            action = torch.zeros(1,self.action_size, device=obs.device)
            embed = self.observation_encoder(obs)
            latent, _ = self.representation(embed, action, latent)
            feat = get_feat(latent)
        return feat

    def update_mpc_planner(self):
        self.mpc_planner.set_goal_state(self.zero_action(self.goal_state))


class AtariDreamerModel(AgentModel):
    def forward(self, observation: torch.Tensor, prev_action: torch.Tensor = None, prev_state: RSSMState = None, rand = False):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        observation = observation.reshape(T * B, *img_shape).type(self.dtype) / 255.0 - 0.5
        prev_action = prev_action.reshape(T * B, -1).to(self.dtype)
        if prev_state is None:
            prev_state = self.representation.initial_state(prev_action.size(0), device=prev_action.device,
                                                           dtype=self.dtype)
        state = self.get_state_representation(observation, prev_action, prev_state)

        if rand:
            action = torch.randn(*prev_action.shape)
        else:
            action = self.policy(state)
        return_spec = ModelReturnSpec(action, state)
        return_spec = buffer_func(return_spec, restore_leading_dims, lead_dim, T, B)
        return return_spec


ModelReturnSpec = namedarraytuple('ModelReturnSpec', ['action', 'state'])
