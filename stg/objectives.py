import math

import torch
from torch import nn
from torch.nn import functional as F


def profile_distances(profiles, continuous=False):
    if profiles.ndim != 3 or min(profiles.shape) < 1:
        raise ValueError("Profiles must have shape [history,test,outcome_or_sample]")
    if not torch.isfinite(profiles).all():
        raise ValueError("Profiles must be finite")
    if continuous:
        if (profiles < 0).any() or (profiles > 1).any():
            raise ValueError("Continuous empirical returns must be normalized to [0,1]")
        ordered = profiles.sort(dim=-1).values
        return torch.cdist(ordered.flatten(1), ordered.flatten(1), p=1) / (profiles.shape[1] * profiles.shape[2])
    if (profiles < 0).any() or not torch.allclose(profiles.sum(-1), torch.ones_like(profiles[..., 0]), atol=1e-4):
        raise ValueError("Categorical response probabilities must be nonnegative and sum to one")
    return torch.cdist(profiles.flatten(1), profiles.flatten(1), p=1) / (2 * profiles.shape[1])


class MetricDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(3 * latent_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, left, right):
        return self.network(torch.cat((left + right, left * right, (left - right).abs()), dim=-1)).squeeze(-1)


class QuotientLoss(nn.Module):
    def __init__(self, decay=0.99, lower=0.1, upper=10.0, epsilon=1e-12):
        super().__init__()
        if not 0 <= decay < 1 or not 0 < lower <= upper or epsilon <= 0:
            raise ValueError("Invalid quotient scale configuration")
        self.decay, self.lower, self.upper, self.epsilon = decay, lower, upper, epsilon
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, native_distances, targets):
        if native_distances.numel() == 0:
            return native_distances.sum() * 0
        with torch.no_grad():
            denominator = native_distances.square().sum()
            if self.training and denominator > self.epsilon:
                optimum = (native_distances.detach() * targets.detach()).sum() / denominator
                updated = self.decay * self.scale + (1 - self.decay) * optimum
                self.scale.copy_(updated.clamp(self.lower, self.upper))
        return (self.scale.detach() * native_distances - targets.detach()).square().mean()


def neighborhood_contrastive(z, distances, threshold=0.2, temperature=0.1):
    count = z.shape[0]
    diagonal = torch.eye(count, dtype=torch.bool, device=z.device)
    positives = (distances < threshold) & ~diagonal
    valid = positives.any(dim=-1)
    if not valid.any():
        return z.sum() * 0
    logits = -torch.cdist(z, z).square() / temperature
    logits = logits.masked_fill(diagonal, -torch.inf)
    log_probabilities = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    selected = torch.where(positives, log_probabilities, torch.zeros_like(log_probabilities))
    return -(selected.sum(dim=-1)[valid] / positives.sum(dim=-1)[valid]).mean()


def bivariate_nll(parameters, values):
    mean = parameters[..., :2]
    scale = F.softplus(parameters[..., 2:4]) + 1e-4
    correlation = 0.999 * parameters[..., 4].tanh()
    standardized = (values - mean.unsqueeze(-2)) / scale.unsqueeze(-2)
    x, y = standardized.unbind(-1)
    residual = 1 - correlation.square()
    quadratic = (x.square() + y.square() - 2 * correlation.unsqueeze(-1) * x * y) / residual.unsqueeze(-1)
    return math.log(2 * math.pi) + scale.log().sum(-1).unsqueeze(-1) + 0.5 * residual.log().unsqueeze(-1) + 0.5 * quadratic


def mixture_nll(parameters, values):
    logits, mean, raw_scale = parameters.chunk(3, dim=-1)
    scale = F.softplus(raw_scale) + 1e-4
    sample = values.squeeze(-1).unsqueeze(-1)
    log_density = -0.5 * ((sample - mean.unsqueeze(-2)) / scale.unsqueeze(-2)).square() - scale.log().unsqueeze(-2) - 0.5 * math.log(2 * math.pi)
    return -torch.logsumexp(logits.log_softmax(-1).unsqueeze(-2) + log_density, dim=-1)


def sampled_mixture_wasserstein(parameters, empirical, samples=1000, temperature=0.5):
    logits, mean, raw_scale = parameters.chunk(3, dim=-1)
    scale = F.softplus(raw_scale) + 1e-4
    expanded = logits.unsqueeze(-2).expand(*logits.shape[:-1], samples, logits.shape[-1])
    assignments = F.gumbel_softmax(expanded, tau=temperature, hard=False, dim=-1)
    noise = torch.randn_like(expanded)
    predicted = (assignments * (mean.unsqueeze(-2) + scale.unsqueeze(-2) * noise)).sum(-1)
    predicted = predicted.sort(dim=-1).values
    observed = empirical.sort(dim=-1).values
    predicted_count, observed_count = predicted.shape[-1], observed.shape[-1]
    boundaries = torch.unique(torch.cat((torch.arange(predicted_count + 1, device=predicted.device) / predicted_count, torch.arange(observed_count + 1, device=predicted.device) / observed_count)), sorted=True)
    midpoints = 0.5 * (boundaries[:-1] + boundaries[1:])
    left = (midpoints * predicted_count).long().clamp_max(predicted_count - 1)
    right = (midpoints * observed_count).long().clamp_max(observed_count - 1)
    return ((predicted[..., left] - observed[..., right]).abs() * boundaries.diff()).sum(-1).mean()


class AuxiliaryObjective(nn.Module):
    def __init__(self, method, latent_dim=256, test_count=100, outcome_count=4, belief_kind="categorical", belief_dim=4, continuous_responses=False, mixture_samples=1000):
        super().__init__()
        belief_kind = "bivariate" if belief_kind == "coordinates" else belief_kind
        self.method = method.upper()
        if self.method not in {"NONE", "RNC", "TR", "MD", "BD-S", "QP", "PH"}:
            raise ValueError(f"Unknown objective {method}")
        if belief_kind not in {"categorical", "factorized", "bivariate", "mixture"}:
            raise ValueError(f"Unknown belief head {belief_kind}")
        if belief_kind == "factorized" and (not isinstance(belief_dim, (list, tuple)) or not belief_dim or min(belief_dim) < 1):
            raise ValueError("Factorized belief_dim must list each factor's positive category count")
        self.test_count, self.outcome_count = test_count, outcome_count
        self.belief_kind, self.belief_dim = belief_kind, belief_dim
        self.continuous_responses, self.mixture_samples = continuous_responses, mixture_samples
        if self.method == "MD":
            self.head = MetricDecoder(latent_dim)
        elif self.method == "QP":
            self.head = QuotientLoss()
        elif self.method == "TR":
            output = test_count * (15 if continuous_responses else outcome_count)
            self.head = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, output))
        elif self.method in {"BD-S", "PH"}:
            dimension = 2 if self.method == "PH" else {"bivariate": 5, "mixture": 15}.get(belief_kind, sum(belief_dim) if isinstance(belief_dim, (list, tuple)) else belief_dim)
            self.head = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, dimension))

    def forward(self, z, profiles, beliefs=None, realized=None, target_distances=None):
        if self.method == "NONE":
            return z.sum() * 0, {}
        if self.method in {"QP", "MD", "RNC"}:
            distances = profile_distances(profiles.detach(), self.continuous_responses) if target_distances is None else target_distances.detach()
            if distances.shape != (len(z), len(z)):
                raise ValueError("Target distances must have shape [batch,batch]")
            if not torch.isfinite(distances).all() or (distances < 0).any():
                raise ValueError("Target distances must be finite and nonnegative")
            if self.method == "RNC":
                return neighborhood_contrastive(z, distances), {}
            left, right = torch.triu_indices(len(z), len(z), offset=1, device=z.device)
            target = distances[left, right]
            if not len(left):
                return z.sum() * 0, {}
            if self.method == "MD":
                prediction = self.head(z[left], z[right])
                return F.mse_loss(prediction, target), {}
            native = torch.linalg.vector_norm(z[left] - z[right], dim=-1)
            loss = self.head(native, target)
            return loss, {"qp_scale": self.head.scale.detach()}
        prediction = self.head(z)
        if self.method == "TR":
            width = 15 if self.continuous_responses else self.outcome_count
            prediction = prediction.reshape(len(z), self.test_count, width)
            if profiles.ndim != 3 or profiles.shape[:2] != prediction.shape[:2] or (not self.continuous_responses and profiles.shape[-1] != width):
                raise ValueError("TR targets must match the configured test and outcome dimensions")
            loss = sampled_mixture_wasserstein(prediction, profiles.detach(), samples=self.mixture_samples) if self.continuous_responses else -(profiles.detach() * prediction.log_softmax(-1)).sum(-1).mean()
            return loss, {}
        if self.method == "PH":
            if realized is None or realized.shape != prediction.shape:
                raise ValueError("PH needs realized 2D coordinates")
            return F.mse_loss(prediction, realized.detach()), {}
        if beliefs is None:
            raise ValueError("BD-S needs posterior targets")
        if self.belief_kind == "categorical":
            return -(beliefs.detach() * prediction.log_softmax(-1)).sum(-1).mean(), {}
        if self.belief_kind == "factorized":
            targets = beliefs if isinstance(beliefs, (list, tuple)) else beliefs.split(self.belief_dim, dim=-1)
            logits = prediction.split(self.belief_dim, dim=-1)
            return sum(-(target.detach() * logit.log_softmax(-1)).sum(-1).mean() for target, logit in zip(targets, logits)), {}
        values, weights = beliefs["values"].detach(), beliefs["weights"].detach()
        if (weights < 0).any() or not torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]), atol=1e-4):
            raise ValueError("Posterior weights must sum to one")
        losses = bivariate_nll(prediction, values) if self.belief_kind == "bivariate" else mixture_nll(prediction, values)
        return (losses * weights).sum(-1).mean(), {}
