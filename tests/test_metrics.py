import numpy as np
import pytest

from stg.metrics import (
    empirical_wasserstein_1d,
    iqd,
    normalized_wasserstein_1d,
    paired_nested_bootstrap,
    rank_correlation,
    structural_distance,
    total_variation,
    validate_probabilities,
)


def test_probability_validation_and_total_variation():
    p = np.array([[1.0, 0.0], [0.25, 0.75]])
    q = np.array([[0.0, 1.0], [0.75, 0.25]])
    np.testing.assert_allclose(total_variation(p, q), [1.0, 0.5])
    np.testing.assert_allclose(total_variation(p, p), 0)
    for invalid in ([0.2, 0.2], [-0.1, 1.1], [np.nan, 1]):
        with pytest.raises(ValueError):
            validate_probabilities(invalid)
    with pytest.raises(ValueError):
        total_variation(p, [0.5, 0.5])


def test_wasserstein_unequal_samples_exact_cdf_integration():
    assert empirical_wasserstein_1d([0, 2], [1]) == pytest.approx(1.0)
    assert empirical_wasserstein_1d([0, 0, 3], [0, 3]) == pytest.approx(0.5)
    assert empirical_wasserstein_1d([2], [2, 2, 2]) == 0
    assert normalized_wasserstein_1d([0, 2], [1], value_range=(0, 2)) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        normalized_wasserstein_1d([2], [0])
    with pytest.raises(ValueError):
        empirical_wasserstein_1d([], [1])


@pytest.mark.parametrize("kind", ["categorical", "continuous"])
def test_structural_distance_is_a_pseudometric(kind):
    rng = np.random.default_rng(91)
    profiles = rng.uniform(size=(6, 4, 5))
    if kind == "categorical":
        profiles /= profiles.sum(axis=-1, keepdims=True)
    profiles[1] = profiles[0]
    distances = structural_distance(profiles, weights=[0.1, 0.2, 0.3, 0.4], kind=kind)
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_allclose(np.diag(distances), 0)
    assert distances[0, 1] == 0
    assert (distances >= 0).all() and (distances <= 1).all()
    for intermediate in range(6):
        assert (distances <= distances[:, intermediate, None] + distances[intermediate, None, :] + 1e-12).all()
    with pytest.raises(ValueError):
        structural_distance(profiles, weights=[0, 0.2, 0.3, 0.5], kind=kind)
    with pytest.raises(ValueError):
        structural_distance(profiles, weights=[1, 1, 1, 1], kind=kind)


def test_weighted_structural_distance_has_manual_solution():
    categorical = np.array([[[1, 0], [0.5, 0.5]], [[0, 1], [0.5, 0.5]]])
    assert structural_distance(categorical, [0.25, 0.75])[0, 1] == pytest.approx(0.25)
    continuous = np.array([[[0, 0], [0, 1]], [[1, 1], [0.5, 0.5]]])
    assert structural_distance(continuous, [0.25, 0.75], kind="continuous")[0, 1] == pytest.approx(0.625)


def _euclidean(points):
    return np.linalg.norm(points[:, None] - points[None, :], axis=-1)


def test_iqd_scale_translation_rotation_and_coordinate_dependence():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    targets = _euclidean(points)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    result = iqd(_euclidean(7 * points @ rotation + 10), targets)
    assert result.score == pytest.approx(0.0, abs=1e-15)
    assert result.scale == pytest.approx(1 / 7)
    assert iqd(_euclidean(points * [4.0, 1.0]), targets).score > 0
    pairs = np.triu_indices(3, k=1)
    assert iqd(targets[pairs] * 1e100, targets[pairs]).score == pytest.approx(0, abs=1e-15)
    assert iqd(targets[pairs] * 1e-100, targets[pairs]).scale == pytest.approx(1e100)


def test_iqd_collapsed_and_zero_target_conventions():
    assert iqd([0, 0], [1, 2]).score == 1
    assert iqd([0, 0], [1, 2]).scale == 0
    assert iqd([0, 0], [0, 0]).score == 0
    assert iqd([1, 2], [0, 0]).score == 0
    assert iqd([1, 0], [0, 1]).score == 1
    for latent, target in (([-1], [1]), ([1], [1, 2]), ([[0, 1], [0, 0]], [[0, 1], [1, 0]])):
        with pytest.raises(ValueError):
            iqd(latent, target)


def test_spearman_uses_average_ranks_for_ties():
    assert rank_correlation([1, 2, 2, 3], [9, 4, 4, 1]) == pytest.approx(-1)
    assert rank_correlation([1, 2, 3], [1, 2, 3]) == pytest.approx(1)
    with pytest.raises(ValueError):
        rank_correlation([1, 1], [1, 2])


def _records(task="grid", seed=10, outcomes=(0, 1, 0, 1)):
    return [dict(task=task, seed=seed, episode=episode, success=success) for episode, success in enumerate(outcomes)]


def test_nested_bootstrap_preserves_pairing_and_uses_observed_data():
    records = _records() + _records(seed=11, outcomes=(1, 1, 0, 1))
    result = paired_nested_bootstrap(records, list(reversed(records)), replicates=100, rng_seed=19)
    assert result["estimate"] == result["lower"] == result["upper"] == 0
    first = _records(outcomes=(1, 1, 1, 1)) + _records(seed=11, outcomes=(1, 1, 1, 1))
    second = _records(outcomes=(0, 0, 0, 0)) + _records(seed=11, outcomes=(0, 0, 0, 0))
    result = paired_nested_bootstrap(first, second, replicates=100, rng_seed=19)
    assert result["estimate"] == result["lower"] == result["upper"] == 1
    assert result["units"] == "success_fraction"


def test_nested_bootstrap_fixed_tasks_and_equal_seed_weighting():
    first = _records(task="a", seed=10, outcomes=(1,)) + _records(task="a", seed=11, outcomes=(0,) * 8)
    first += _records(task="b", seed=10, outcomes=(1, 1)) + _records(task="b", seed=11, outcomes=(1,))
    second = [{**record, "success": 0} for record in first]
    result = paired_nested_bootstrap(first, second, replicates=500, rng_seed=123)
    assert result["estimate"] == pytest.approx(0.75)
    assert result["tasks"] == ["a", "b"]
    assert result["seeds_per_task"] == 2
    assert result == paired_nested_bootstrap(first, second, replicates=500, rng_seed=123)


def test_nested_bootstrap_shares_seed_resamples_across_tasks():
    first = _records(task="a", seed=10, outcomes=(1,)) + _records(task="a", seed=11, outcomes=(0,))
    first += _records(task="b", seed=10, outcomes=(0,)) + _records(task="b", seed=11, outcomes=(1,))
    second = [{**record, "success": 0} for record in first]
    result = paired_nested_bootstrap(first, second, replicates=500)
    assert result["estimate"] == result["lower"] == result["upper"] == 0.5


def test_nested_bootstrap_rejects_unpaired_duplicate_or_incomplete_records():
    records = _records()
    with pytest.raises(ValueError, match="matched"):
        paired_nested_bootstrap(records, records[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        paired_nested_bootstrap(records + records[:1], records)
    with pytest.raises(ValueError, match="same seed"):
        incomplete = records + _records(task="other", seed=11)
        paired_nested_bootstrap(incomplete, incomplete)
    with pytest.raises(ValueError, match="binary"):
        invalid = [{**records[0], "success": 0.4}]
        paired_nested_bootstrap(invalid, invalid)
    with pytest.raises(ValueError):
        paired_nested_bootstrap([], [])
