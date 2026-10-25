import pytest
import torch
from torch import nn
from torch.nn import functional as F

from stg.model import RSSM, RSSMState
from stg.objectives import AuxiliaryObjective, MetricDecoder, QuotientLoss, bivariate_nll, mixture_nll, neighborhood_contrastive, profile_distances
from stg.planner import CEMPlanner


def test_metric_decoder_exact_capacity_and_symmetry():
    torch.manual_seed(4)
    decoder = MetricDecoder()
    assert sum(parameter.numel() for parameter in decoder.parameters()) == 262913
    left, right = torch.randn(7, 256), torch.randn(7, 256)
    assert torch.equal(decoder(left, right), decoder(right, left))


def test_qp_scale_is_detached_moving_average_and_skips_collapse():
    objective = QuotientLoss()
    native = torch.tensor([1.0, 2.0], requires_grad=True)
    targets = torch.tensor([2.0, 4.0], requires_grad=True)
    loss = objective(native, targets)
    assert objective.scale.item() == pytest.approx(1.01)
    loss.backward()
    assert native.grad is not None
    assert targets.grad is None
    assert not objective.scale.requires_grad
    scale = objective.scale.clone()
    collapse = torch.zeros(2, requires_grad=True)
    collapsed_loss = objective(collapse, targets)
    assert torch.equal(objective.scale, scale)
    assert torch.isfinite(collapsed_loss)
    objective.eval()
    objective(native, 20 * targets)
    assert torch.equal(objective.scale, scale)


def test_qp_clips_extreme_scale():
    objective = QuotientLoss(decay=0)
    objective(torch.tensor([1.0]), torch.tensor([1000.0]))
    assert objective.scale.item() == 10
    objective(torch.tensor([1.0]), torch.tensor([0.0]))
    assert objective.scale.item() == pytest.approx(0.1)


def test_profile_distance_matches_tv_and_empirical_wasserstein():
    profiles = torch.tensor([[[1.0, 0.0], [0.5, 0.5]], [[0.0, 1.0], [0.5, 0.5]], [[1.0, 0.0], [0.5, 0.5]]])
    distance = profile_distances(profiles)
    assert distance[0, 1].item() == pytest.approx(0.5)
    assert distance[0, 2].item() == 0
    assert torch.equal(distance, distance.T)
    samples = torch.tensor([[[0.0, 0.5, 1.0]], [[0.5, 0.75, 1.0]]])
    assert profile_distances(samples, continuous=True)[0, 1].item() == pytest.approx(0.25)
    with pytest.raises(ValueError):
        profile_distances(profiles * 2)


def test_rnc_ignores_no_positive_anchors_and_remains_differentiable():
    z = torch.randn(3, 5, requires_grad=True)
    distances = torch.tensor([[0.0, 0.1, 0.8], [0.1, 0.0, 0.9], [0.8, 0.9, 0.0]])
    loss = neighborhood_contrastive(z, distances)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    no_positives = torch.ones(3, 3) - torch.eye(3)
    assert neighborhood_contrastive(z, no_positives).item() == 0


def test_rssm_observation_mask_and_latent_contract():
    torch.manual_seed(5)
    model = RSSM(6, 3, deterministic_dim=8, stochastic_dim=4, hidden_dim=10)
    observations = torch.randn(2, 5, 6)
    actions = F.one_hot(torch.randint(3, (2, 5)), 3).float()
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    _, latents, kl = model.observe_sequence(observations, actions, mask)
    assert latents.shape == (2, 5, 12)
    assert kl.shape == (2, 5)
    assert torch.equal(latents[0, 2], latents[0, 4])
    assert torch.allclose(model.encode_history(observations, actions, mask)[0], model.encode_history(observations[:1, :3], actions[:1, :3])[0], atol=1e-6)
    perturbed = observations.clone()
    perturbed[0, 3:] = 1000
    assert torch.equal(model.encode_history(observations, actions, mask)[0], model.encode_history(perturbed, actions, mask)[0])
    decoded, reward, value = model.decode(latents)
    assert decoded.shape == observations.shape
    assert reward.shape == value.shape == (2, 5)
    assert (kl >= -1e-5).all()


def test_rssm_base_loss_reaches_encoder_and_dynamics():
    torch.manual_seed(6)
    model = RSSM(6, 3, deterministic_dim=8, stochastic_dim=4, hidden_dim=10)
    observations = torch.randn(2, 4, 6)
    actions = F.one_hot(torch.randint(3, (2, 4)), 3).float()
    loss, diagnostics = model.base_loss(observations, actions, torch.rand(2, 4), torch.rand(2, 4))
    loss.backward()
    for component in (model.encoder, model.gru, model.prior, model.posterior, model.observation_head, model.reward_head, model.value_head):
        assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0 for parameter in component.parameters())
    assert all(torch.isfinite(item) for item in diagnostics.values())


@pytest.mark.parametrize("method", ["NONE", "QP", "MD", "TR", "RNC", "BD-S", "PH"])
def test_objectives_have_finite_gradients(method):
    torch.manual_seed(7)
    z = torch.randn(5, 12, requires_grad=True)
    profiles = torch.rand(5, 3, 4).softmax(-1)
    beliefs = torch.rand(5, 4).softmax(-1)
    objective = AuxiliaryObjective(method, latent_dim=12, test_count=3, outcome_count=4)
    loss, _ = objective(z, profiles, beliefs, realized=torch.randn(5, 2))
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    if method not in {"NONE", "RNC"}:
        assert z.grad.abs().sum() > 0


@pytest.mark.parametrize("kind,dimension", [("bivariate", 2), ("mixture", 1)])
def test_continuous_belief_heads_use_weighted_posterior_nll(kind, dimension):
    z = torch.randn(3, 12, requires_grad=True)
    objective = AuxiliaryObjective("BD-S", latent_dim=12, belief_kind=kind)
    posterior = {"values": torch.rand(3, 4, dimension), "weights": torch.rand(3, 4).softmax(-1)}
    loss, _ = objective(z, torch.ones(3, 1, 1), posterior)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_gaussian_nll_matches_torch_distribution():
    raw = torch.tensor([[0.2, 0.3, 0.7, -0.4, 0.25]])
    values = torch.tensor([[[0.4, 0.6], [-0.2, 0.3]]])
    scale = F.softplus(raw[0, 2:4]) + 1e-4
    rho = 0.999 * raw[0, 4].tanh()
    covariance = torch.diag(scale.square())
    covariance[0, 1] = covariance[1, 0] = rho * scale.prod()
    expected = -torch.distributions.MultivariateNormal(raw[0, :2], covariance_matrix=covariance).log_prob(values[0])
    assert torch.allclose(bivariate_nll(raw, values)[0], expected, atol=1e-6)


def test_mixture_nll_matches_torch_distribution():
    parameters = torch.randn(2, 15)
    values = torch.rand(2, 7, 1)
    logits, mean, raw_scale = parameters.chunk(3, -1)
    mixture = torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=logits), torch.distributions.Normal(mean, F.softplus(raw_scale) + 1e-4))
    expected = -mixture.log_prob(values.squeeze(-1).T).T
    assert torch.allclose(mixture_nll(parameters, values), expected, atol=1e-6)


def test_continuous_response_head_runs_and_backpropagates():
    torch.manual_seed(9)
    z = torch.randn(3, 12, requires_grad=True)
    objective = AuxiliaryObjective("TR", latent_dim=12, test_count=2, continuous_responses=True, mixture_samples=20)
    loss, _ = objective(z, torch.rand(3, 2, 7))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_factorized_belief_head_sums_factor_losses():
    z = torch.randn(2, 8, requires_grad=True)
    objective = AuxiliaryObjective("BD-S", latent_dim=8, belief_kind="factorized", belief_dim=[3, 2])
    targets = [torch.rand(2, 3).softmax(-1), torch.rand(2, 2).softmax(-1)]
    loss, _ = objective(z, torch.ones(2, 1, 1), targets)
    logits = objective.head(z).split([3, 2], -1)
    expected = sum(-(target * logit.log_softmax(-1)).sum(-1).mean() for target, logit in zip(targets, logits))
    assert torch.equal(loss, expected)
    loss.backward()
    assert torch.isfinite(z.grad).all()


class ActionRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_dim = 3

    def imagine_step(self, action, state):
        return RSSMState(action, state.stochastic), action

    def decode(self, z):
        reward = z[..., 2]
        return z, reward, torch.full_like(reward, 2.0)


def test_cem_uses_model_reward_terminal_value_discount_and_seed():
    model = ActionRewardModel()
    planner = CEMPlanner(model, horizon=3, population=300, iterations=4, elites=20, discount=0.5)
    state = RSSMState(torch.zeros(2, 3), torch.zeros(2, 1))
    action, scores = planner.plan(state, generator=torch.Generator().manual_seed(8))
    repeated_action, repeated_scores = planner.plan(state, generator=torch.Generator().manual_seed(8))
    assert torch.equal(action, torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))
    assert torch.allclose(scores, torch.full((2,), 2.0))
    assert torch.equal(action, repeated_action)
    assert torch.equal(scores, repeated_scores)
    assert not scores.requires_grad
