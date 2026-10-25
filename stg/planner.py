import torch
from torch.nn import functional as F

from .model import RSSMState


class CEMPlanner:
    def __init__(self, model, horizon=10, population=200, iterations=10, elites=20, discount=0.99):
        if horizon < 1 or iterations < 1 or not 1 <= elites <= population:
            raise ValueError("Require positive horizon and iterations and 1 <= elites <= population")
        if not 0 <= discount <= 1:
            raise ValueError("Discount must lie in [0,1]")
        self.model = model
        self.horizon, self.population, self.iterations, self.elites = horizon, population, iterations, elites
        self.discount = discount

    @torch.no_grad()
    def plan(self, state, generator=None):
        if isinstance(state, torch.Tensor):
            state = self.model.state_from_latent(state)
        state = state.detach()
        batch = len(state.deterministic)
        action_count = self.model.action_dim
        probabilities = torch.full((batch, self.horizon, action_count), 1 / action_count, device=state.deterministic.device)
        for _ in range(self.iterations):
            draws = torch.multinomial(probabilities.reshape(-1, action_count), self.population, replacement=True, generator=generator)
            sequences = draws.reshape(batch, self.horizon, self.population).permute(0, 2, 1)
            imagined = RSSMState(state.deterministic.repeat_interleave(self.population, dim=0), state.stochastic.repeat_interleave(self.population, dim=0))
            scores = torch.zeros(batch * self.population, device=state.deterministic.device)
            for t in range(self.horizon):
                action = F.one_hot(sequences[:, :, t].reshape(-1), action_count).to(state.deterministic.dtype)
                imagined, z = self.model.imagine_step(action, imagined)
                _, reward, value = self.model.decode(z)
                scores.add_(self.discount ** t * reward)
            scores.add_(self.discount ** self.horizon * value)
            scores = scores.reshape(batch, self.population)
            if not torch.isfinite(scores).all():
                raise ValueError("Planner encountered nonfinite model predictions")
            elite_indices = scores.topk(self.elites, dim=-1).indices
            elites = sequences.gather(1, elite_indices.unsqueeze(-1).expand(-1, -1, self.horizon))
            probabilities = F.one_hot(elites, action_count).float().mean(dim=1).clamp_min(1e-6)
            probabilities /= probabilities.sum(-1, keepdim=True)
        best = scores.argmax(-1)
        first_action = sequences[torch.arange(batch, device=best.device), best, 0]
        action = F.one_hot(first_action, action_count).to(state.deterministic.dtype)
        return action, scores[torch.arange(batch, device=best.device), best]
