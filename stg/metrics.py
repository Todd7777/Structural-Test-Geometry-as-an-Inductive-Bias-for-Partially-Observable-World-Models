from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class IQDResult:
    score: float
    scale: float


def _finite_array(values, name):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def validate_probabilities(values, axis=-1, tolerance=1e-6):
    array = _finite_array(values, "probabilities")
    if array.ndim == 0:
        raise ValueError("probabilities need a category axis")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if (array < 0).any() or (array > 1).any():
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(array.sum(axis=axis), 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("probabilities must sum to one along the category axis")
    return array


def total_variation(p, q, axis=-1):
    p = validate_probabilities(p, axis=axis)
    q = validate_probabilities(q, axis=axis)
    if p.shape != q.shape:
        raise ValueError("categorical distributions must have the same shape")
    p = p / p.sum(axis=axis, keepdims=True)
    q = q / q.sum(axis=axis, keepdims=True)
    return np.clip(np.abs(p - q).sum(axis=axis) * 0.5, 0.0, 1.0)


def empirical_wasserstein_1d(x, y):
    x = _finite_array(x, "first sample")
    y = _finite_array(y, "second sample")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Wasserstein inputs must be one-dimensional samples")
    x = np.sort(x)
    y = np.sort(y)
    support = np.unique(np.concatenate((x, y)))
    if support.size == 1:
        return 0.0
    p_cdf = np.searchsorted(x, support[:-1], side="right") / x.size
    q_cdf = np.searchsorted(y, support[:-1], side="right") / y.size
    return float(np.dot(np.abs(p_cdf - q_cdf), np.diff(support)))


def _return_bounds(value_range):
    bounds = _finite_array(value_range, "value range")
    if bounds.shape != (2,) or bounds[1] <= bounds[0]:
        raise ValueError("value_range must contain increasing lower and upper bounds")
    return float(bounds[0]), float(bounds[1])


def normalized_wasserstein_1d(x, y, value_range=(0.0, 1.0)):
    lower, upper = _return_bounds(value_range)
    x = _finite_array(x, "first sample")
    y = _finite_array(y, "second sample")
    if (x < lower).any() or (x > upper).any() or (y < lower).any() or (y > upper).any():
        raise ValueError("samples fall outside the declared return range")
    return empirical_wasserstein_1d(x, y) / (upper - lower)


def structural_distance(profiles, weights=None, kind="categorical", value_range=(0.0, 1.0)):
    profiles = _finite_array(profiles, "response profiles")
    if profiles.ndim != 3:
        raise ValueError("response profiles must have shape [histories, tests, outcomes_or_samples]")
    n_histories, n_tests, _ = profiles.shape
    if weights is None:
        weights = np.full(n_tests, 1.0 / n_tests)
    else:
        weights = _finite_array(weights, "test weights")
    if weights.shape != (n_tests,) or (weights <= 0).any():
        raise ValueError("each structural test needs one strictly positive weight")
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("structural-test weights must sum to one")
    weights = weights / weights.sum()
    distances = np.zeros((n_histories, n_histories), dtype=np.float64)
    if kind == "categorical":
        profiles = validate_probabilities(profiles)
        profiles = profiles / profiles.sum(axis=-1, keepdims=True)
        for i in range(n_histories):
            per_test = 0.5 * np.abs(profiles[i] - profiles[i + 1 :]).sum(axis=-1)
            distances[i, i + 1 :] = per_test @ weights
    elif kind == "continuous":
        lower, upper = _return_bounds(value_range)
        if (profiles < lower).any() or (profiles > upper).any():
            raise ValueError("samples fall outside the declared return range")
        profiles = np.sort(profiles, axis=-1)
        for i in range(n_histories):
            per_test = np.abs(profiles[i] - profiles[i + 1 :]).mean(axis=-1) / (upper - lower)
            distances[i, i + 1 :] = per_test @ weights
    else:
        raise ValueError("kind must be 'categorical' or 'continuous'")
    return distances + distances.T


def _pair_distances(values, name):
    distances = _finite_array(values, name)
    if (distances < 0).any():
        raise ValueError(f"{name} must be nonnegative")
    if distances.ndim == 2:
        if distances.shape[0] != distances.shape[1] or distances.shape[0] < 2:
            raise ValueError(f"{name} must be a square matrix with at least two histories")
        if not np.allclose(distances, distances.T, rtol=0.0, atol=1e-10):
            raise ValueError(f"{name} matrix must be symmetric")
        if not np.allclose(np.diag(distances), 0.0, rtol=0.0, atol=1e-10):
            raise ValueError(f"{name} matrix must have a zero diagonal")
        return distances[np.triu_indices(distances.shape[0], k=1)]
    if distances.ndim != 1:
        raise ValueError(f"{name} must be a pair vector or symmetric distance matrix")
    return distances


def iqd(latent_distances, target_distances):
    latent = _pair_distances(latent_distances, "latent distances")
    target = _pair_distances(target_distances, "target distances")
    if latent.shape != target.shape:
        raise ValueError("latent and target distances must describe the same pairs")
    target_max = target.max()
    latent_max = latent.max()
    if target_max == 0:
        return IQDResult(0.0, 0.0)
    if latent_max == 0:
        return IQDResult(1.0, 0.0)
    z = latent / latent_max
    t = target / target_max
    normalized_scale = float(np.dot(z, t) / np.dot(z, z))
    scale = float((target_max / latent_max) * normalized_scale)
    if not np.isfinite(scale):
        raise ValueError("optimal scale exceeds the floating-point range")
    score = float(np.linalg.norm(normalized_scale * z - t) / np.linalg.norm(t))
    return IQDResult(float(np.clip(score, 0.0, 1.0)), scale)


def _average_ranks(values):
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    ranks = np.empty(values.size, dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) * 0.5
    return ranks


def rank_correlation(x, y):
    x = _finite_array(x, "first variable")
    y = _finite_array(y, "second variable")
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("rank correlation needs matching vectors with at least two observations")
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = np.linalg.norm(x_rank) * np.linalg.norm(y_rank)
    if denominator == 0:
        raise ValueError("rank correlation is undefined for a constant variable")
    return float(np.clip(np.dot(x_rank, y_rank) / denominator, -1.0, 1.0))


def _record_map(records, name):
    mapped = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"{name} records must be mappings")
        if not {"task", "seed", "episode", "success"}.issubset(record):
            raise ValueError(f"{name} records require task, seed, episode, and success")
        task, seed, episode = record["task"], record["seed"], record["episode"]
        if not isinstance(task, str) or not task:
            raise ValueError("task identifiers must be nonempty strings")
        if not isinstance(seed, (str, int)) or isinstance(seed, bool):
            raise ValueError("seed identifiers must be strings or integers")
        if not isinstance(episode, (str, int)) or isinstance(episode, bool):
            raise ValueError("episode identifiers must be strings or integers")
        success = record["success"]
        if not isinstance(success, (bool, int, float, np.integer, np.floating)) or success not in (0, 1):
            raise ValueError("success values must be binary zero or one")
        key = (task, seed, episode)
        if key in mapped:
            raise ValueError(f"{name} contains duplicate task/seed/episode identifiers")
        mapped[key] = float(success)
    if not mapped:
        raise ValueError(f"{name} has no evaluation records")
    return mapped


def paired_nested_bootstrap(
    records_a: Iterable[Mapping],
    records_b: Iterable[Mapping],
    replicates=10000,
    confidence=0.95,
    rng_seed=0,
):
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    a = _record_map(records_a, "first method")
    b = _record_map(records_b, "second method")
    if a.keys() != b.keys():
        raise ValueError("methods must have exactly matched task/seed/episode identifiers")
    tasks = sorted({task for task, _, _ in a})
    task_seeds = {task: {seed for t, seed, _ in a if t == task} for task in tasks}
    common_seeds = task_seeds[tasks[0]]
    if any(seeds != common_seeds for seeds in task_seeds.values()):
        raise ValueError("fixed-task suite bootstrap requires the same seed identifiers across tasks")
    seeds = sorted(common_seeds, key=lambda value: (type(value).__name__, repr(value)))
    rng = np.random.default_rng(rng_seed)
    selected_seeds = rng.integers(len(seeds), size=(replicates, len(seeds)))
    distribution = np.zeros(replicates, dtype=np.float64)
    observed_task_means = []
    episode_counts = []
    for task in tasks:
        task_replicates = np.zeros(replicates, dtype=np.float64)
        seed_means = []
        for seed_index, seed in enumerate(seeds):
            episodes = sorted(
                (episode for t, s, episode in a if t == task and s == seed),
                key=lambda value: (type(value).__name__, repr(value)),
            )
            differences = np.array([a[(task, seed, episode)] - b[(task, seed, episode)] for episode in episodes])
            seed_means.append(float(differences.mean()))
            episode_counts.append({"task": task, "seed": seed, "episodes": len(episodes)})
            rows, _ = np.nonzero(selected_seeds == seed_index)
            draws = rng.integers(len(episodes), size=(rows.size, len(episodes)))
            np.add.at(task_replicates, rows, differences[draws].mean(axis=1))
        distribution += task_replicates / len(seeds)
        observed_task_means.append(float(np.mean(seed_means)))
    distribution /= len(tasks)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(distribution, [tail, 1.0 - tail])
    return {
        "estimate": float(np.mean(observed_task_means)),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": float(confidence),
        "replicates": replicates,
        "rng_seed": rng_seed,
        "tasks": tasks,
        "seeds_per_task": len(seeds),
        "episode_counts": episode_counts,
        "contrast": "first_method_minus_second_method",
        "units": "success_fraction",
        "resampling": "shared_matched_seeds_then_paired_episodes_within_each_fixed_task_and_seed",
    }
