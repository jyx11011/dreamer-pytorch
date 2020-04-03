import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as tf
from rlpyt.utils.collections import namedarraytuple

RSSMState = namedarraytuple('RSSMState', ['mean', 'std', 'stoch', 'deter'])


def stack_states(rssm_states: list, dim):
    return RSSMState(
        torch.stack([state.mean for state in rssm_states], dim=dim),
        torch.stack([state.std for state in rssm_states], dim=dim),
        torch.stack([state.stoch for state in rssm_states], dim=dim),
        torch.stack([state.deter for state in rssm_states], dim=dim),
    )


class TransitionBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prev_action, prev_state):
        """:return: next state"""
        raise NotImplementedError


class RepresentationBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, obs_embed, prev_action, prev_state):
        """:return: next state"""
        raise NotImplementedError


class RollOutModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, steps, obs_embed, prev_action, prev_state):
        raise NotImplementedError


class RSSMTransition(TransitionBase):
    def __init__(self, action_size, stochastic_size=30, deterministic_size=200, hidden_size=200, activation=nn.ELU,
                 distribution=td.Normal):
        super().__init__()
        self._action_size = action_size
        self._stoch_size = stochastic_size
        self._deter_size = deterministic_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._cell = nn.GRUCell(hidden_size, deterministic_size)
        self._rnn_input_model = self._build_rnn_input_model()
        self._stochastic_prior_model = self._build_stochastic_model()
        self._dist = distribution

    def _build_rnn_input_model(self):
        rnn_input_model = [nn.Linear(self._action_size + self._stoch_size, self._hidden_size)]
        rnn_input_model += [self._activation()]
        return nn.Sequential(*rnn_input_model)

    def _build_stochastic_model(self):
        stochastic_model = [nn.Linear(self._hidden_size, self._hidden_size)]
        stochastic_model += [self._activation()]
        stochastic_model += [nn.Linear(self._hidden_size, 2 * self._stoch_size)]
        return nn.Sequential(*stochastic_model)

    def initial_state(self, batch_size, **kwargs):
        return RSSMState(
            torch.zeros(batch_size, self._stoch_size, **kwargs),
            torch.zeros(batch_size, self._stoch_size, **kwargs),
            torch.zeros(batch_size, self._stoch_size, **kwargs),
            torch.zeros(batch_size, self._deter_size, **kwargs),
        )

    def forward(self, prev_action: torch.Tensor, prev_state: RSSMState):
        rnn_input = self._rnn_input_model(torch.cat([prev_action, prev_state.stoch], dim=-1))
        deter_state = self._cell(rnn_input, prev_state.deter)
        mean, std = torch.chunk(self._stochastic_prior_model(deter_state), 2, dim=-1)
        std = tf.softplus(std) + 0.1
        dist = self._dist(mean, std)
        stoch_state = dist.rsample()
        return RSSMState(mean, std, stoch_state, deter_state)


class RSSMRepresentation(RepresentationBase):
    def __init__(self, transition_model: RSSMTransition, obs_embed_size, action_size, stochastic_size=30,
                 deterministic_size=200, hidden_size=200, activation=nn.ELU, distribution=td.Normal):
        super().__init__()
        self._transition_model = transition_model
        self._obs_embed_size = obs_embed_size
        self._action_size = action_size
        self._stoch_size = stochastic_size
        self._deter_size = deterministic_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._dist = distribution
        self._stochastic_posterior_model = self._build_stochastic_model()

    def _build_stochastic_model(self):
        stochastic_model = [nn.Linear(self._deter_size + self._obs_embed_size, self._hidden_size)]
        stochastic_model += [self._activation()]
        stochastic_model += [nn.Linear(self._hidden_size, 2 * self._stoch_size)]
        return nn.Sequential(*stochastic_model)

    def initial_state(self, batch_size, **kwargs):
        return RSSMState(
            torch.zeros(batch_size, self._stoch_size, **kwargs),
            torch.zeros(batch_size, self._stoch_size, **kwargs),
            torch.zeros(batch_size, self._stoch_size, **kwargs),
            torch.zeros(batch_size, self._deter_size, **kwargs),
        )

    def forward(self, obs_embed: torch.Tensor, prev_action: torch.Tensor, prev_state: RSSMState):
        prior_state = self._transition_model(prev_action, prev_state)
        x = torch.cat([prior_state.deter, obs_embed], -1)
        mean, std = torch.chunk(self._stochastic_posterior_model(x), 2, dim=-1)
        std = tf.softplus(std) + 0.1
        dist = self._dist(mean, std)
        stoch_state = dist.rsample()
        posterior_state = RSSMState(mean, std, stoch_state, prior_state.deter)
        return prior_state, posterior_state


class RSSMRollout(RollOutModule):
    def __init__(self, representation_model: RSSMRepresentation, transition_model: RSSMTransition):
        super().__init__()
        self.representation_model = representation_model
        self.transition_model = transition_model

    def forward(self, steps: int, obs_embed: torch.Tensor, action: torch.Tensor, prev_state: RSSMState):
        return self.rollout_representation(steps, obs_embed, action, prev_state)

    def rollout_representation(self, steps: int, obs_embed: torch.Tensor, action: torch.Tensor,
                               prev_state: RSSMState):
        """
        Roll out the model with actions and observations from data.
        :param steps: number of steps to roll out
        :param obs_embed: size(batch_size, time_steps, embedding_size)
        :param action: size(batch_size, time_steps, action_size)
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: prior, posterior states. size(batch_size, time_steps, state_size)
        """
        priors = []
        posteriors = []
        for t in range(steps):
            prior_state, posterior_state = self.representation_model(obs_embed[:, t], action[:, t], prev_state)
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state
        prior = stack_states(priors, dim=1)
        post = stack_states(posteriors, dim=1)
        return prior, post

    def rollout_transition(self, steps: int, action: torch.Tensor, prev_state: RSSMState):
        """
        Roll out the model with actions from data.
        :param steps: number of steps to roll out
        :param action: size(batch_size, time_steps, action_size)
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: prior states. size(batch_size, time_steps, state_size)
        """
        priors = []
        state = prev_state
        for t in range(steps):
            state = self.transition_model(action[:, t], state)
            priors.append(state)
        return stack_states(priors, dim=1)

    def rollout_policy(self, steps: int, policy, prev_action: torch.Tensor, prev_state: RSSMState):
        """
        Roll out the model with a policy function.
        :param steps: number of steps to roll out
        :param policy: RSSMState -> action
        :param prev_action: size(batch_size, action_size)
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: prior, posterior states. size(batch_size, time_steps, state_size)
        """
        state, action = prev_state, prev_action
        priors = []
        actions = []
        for t in range(steps):
            state = self.transition_model(action, state)
            action = policy(state)
            priors.append(state)
            actions.append(action)
        priors = stack_states(priors, dim=1)
        actions = torch.stack(actions, dim=1)
        return priors, actions