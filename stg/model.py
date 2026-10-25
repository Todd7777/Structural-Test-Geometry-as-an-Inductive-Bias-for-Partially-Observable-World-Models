from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RSSMState:
    deterministic: torch.Tensor
    stochastic: torch.Tensor

    def detach(self):
        return RSSMState(self.deterministic.detach(), self.stochastic.detach())


class RSSM(nn.Module):
    def __init__(self, obs_dim, action_dim, deterministic_dim=200, stochastic_dim=56, hidden_dim=128):
        super().__init__()
        if min(obs_dim, action_dim, deterministic_dim, stochastic_dim, hidden_dim) < 1:
            raise ValueError("Model dimensions must be positive")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim
        self.latent_dim = deterministic_dim + stochastic_dim
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRUCell(stochastic_dim + action_dim, deterministic_dim)
        self.prior = nn.Sequential(nn.Linear(deterministic_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * stochastic_dim))
        self.posterior = nn.Sequential(nn.Linear(deterministic_dim + hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * stochastic_dim))
        self.observation_head = nn.Sequential(nn.Linear(self.latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, obs_dim))
        self.reward_head = nn.Sequential(nn.Linear(self.latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.value_head = nn.Sequential(nn.Linear(self.latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def initial(self, batch, device=None):
        parameter = next(self.parameters())
        return RSSMState(torch.zeros(batch, self.deterministic_dim, device=device or parameter.device, dtype=parameter.dtype), torch.zeros(batch, self.stochastic_dim, device=device or parameter.device, dtype=parameter.dtype))

    def _normal(self, parameters):
        mean, raw_scale = parameters.chunk(2, dim=-1)
        return mean, F.softplus(raw_scale) + 1e-4

    def observe_step(self, observation, prev_action, state, sample=False):
        deterministic = self.gru(torch.cat((state.stochastic, prev_action), dim=-1), state.deterministic)
        prior_mean, prior_scale = self._normal(self.prior(deterministic))
        mean, scale = self._normal(self.posterior(torch.cat((deterministic, self.encoder(observation)), dim=-1)))
        stochastic = mean + scale * torch.randn_like(mean) if sample else mean
        kl = (torch.log(prior_scale / scale) + (scale.square() + (mean - prior_mean).square()) / (2 * prior_scale.square()) - 0.5).sum(dim=-1)
        new_state = RSSMState(deterministic, stochastic)
        z = torch.cat((deterministic, mean), dim=-1)
        return new_state, z, kl

    def imagine_step(self, action, state, sample=False):
        deterministic = self.gru(torch.cat((state.stochastic, action), dim=-1), state.deterministic)
        mean, scale = self._normal(self.prior(deterministic))
        stochastic = mean + scale * torch.randn_like(mean) if sample else mean
        return RSSMState(deterministic, stochastic), torch.cat((deterministic, mean), dim=-1)

    def state_from_latent(self, z):
        return RSSMState(z[..., :self.deterministic_dim], z[..., self.deterministic_dim:])

    def observe_sequence(self, observations, prev_actions, mask=None, sample=False):
        if observations.ndim != 3 or prev_actions.shape[:2] != observations.shape[:2]:
            raise ValueError("Expected observations [batch,time,obs] and aligned previous actions")
        if observations.shape[-1] != self.obs_dim or prev_actions.shape[-1] != self.action_dim:
            raise ValueError("Observation and action feature dimensions must match the model")
        if mask is not None and mask.shape != observations.shape[:2]:
            raise ValueError("History mask must have shape [batch,time]")
        if observations.shape[1] == 0:
            raise ValueError("History must contain at least one observation")
        state = self.initial(observations.shape[0], observations.device)
        states, latents, divergences = [], [], []
        for t in range(observations.shape[1]):
            candidate, z, kl = self.observe_step(observations[:, t], prev_actions[:, t], state, sample=sample)
            if mask is not None:
                valid = mask[:, t].bool().unsqueeze(-1)
                state = RSSMState(torch.where(valid, candidate.deterministic, state.deterministic), torch.where(valid, candidate.stochastic, state.stochastic))
                z = torch.where(valid, z, latents[-1] if latents else torch.zeros_like(z))
                kl = kl * valid.squeeze(-1)
            else:
                state = candidate
            states.append(state)
            latents.append(z)
            divergences.append(kl)
        return states, torch.stack(latents, dim=1), torch.stack(divergences, dim=1)

    def encode_history(self, observations, prev_actions, mask=None):
        return self.observe_sequence(observations, prev_actions, mask=mask)[1][:, -1]

    def decode(self, z):
        return self.observation_head(z), self.reward_head(z).squeeze(-1), self.value_head(z).squeeze(-1)

    def base_loss(self, observations, prev_actions, rewards, returns, mask=None, kl_weight=1.0):
        _, z, kl = self.observe_sequence(observations, prev_actions, mask=mask, sample=True)
        reconstruction, reward, value = self.decode(z)
        valid = torch.ones_like(kl) if mask is None else mask.to(kl.dtype)
        denominator = valid.sum().clamp_min(1)
        terms = {
            "observation_loss": ((reconstruction - observations).square().mean(-1) * valid).sum() / denominator,
            "reward_loss": ((reward - rewards).square() * valid).sum() / denominator,
            "value_loss": ((value - returns).square() * valid).sum() / denominator,
            "kl_loss": (kl * valid).sum() / denominator,
        }
        loss = terms["observation_loss"] + terms["reward_loss"] + terms["value_loss"] + kl_weight * terms["kl_loss"]
        return loss, {name: item.detach() for name, item in terms.items()}
