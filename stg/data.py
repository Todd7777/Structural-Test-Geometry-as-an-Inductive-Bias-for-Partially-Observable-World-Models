import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

from .environments import DiagnosticEnv
from .targets import estimate_profiles, expected_profiles, make_tests


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discounted_returns(rewards, discount=0.99):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        returns[index] = running
        running = float(rewards[index]) + discount * running
    return returns


def collect(path, histories=10000, seed=0, split="train", rollouts=100, test_seed=0, exact=False, discount=0.99):
    if histories < 1 or rollouts < 1 or not 0 <= discount <= 1:
        raise ValueError("Require positive history/rollout counts and discount in [0,1]")
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Dataset output must have the .npz extension")
    if path.exists():
        raise FileExistsError(f"Dataset already exists: {path}")
    env = DiagnosticEnv(split=split, seed=seed)
    tests = make_tests(seed=test_seed)
    actions_rng = np.random.default_rng(seed)
    target_rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    count = env.horizon + 1
    observations = np.zeros((histories, count, env.observation_dim), dtype=np.float32)
    actions = np.zeros((histories, count, env.action_count), dtype=np.float32)
    rewards = np.zeros((histories, count), dtype=np.float32)
    returns = np.zeros_like(rewards)
    mask = np.zeros_like(rewards, dtype=bool)
    history_mask = np.zeros_like(mask)
    profiles, beliefs, realized = [], [], []
    for index in range(histories):
        observed = env.reset()
        observations[index, 0] = observed
        cutoff = int(actions_rng.integers(count))
        snapshot = env.snapshot() if cutoff == 0 else None
        goal = np.asarray(env.realized_target, dtype=np.float32).copy()
        end = 0
        for time in range(1, count):
            action = int(actions_rng.integers(env.action_count))
            observed, reward, done, _ = env.step(action)
            observations[index, time] = observed
            actions[index, time, action] = 1.0
            rewards[index, time] = reward
            end = time
            if time == cutoff:
                snapshot = env.snapshot()
            if done:
                break
        cutoff = min(cutoff, end)
        if snapshot is None:
            snapshot = env.snapshot()
        mask[index, :end + 1] = True
        history_mask[index, :cutoff + 1] = True
        returns[index, :end + 1] = discounted_returns(rewards[index, :end + 1], discount)
        profiles.append(expected_profiles(snapshot, tests) if exact else estimate_profiles(snapshot, tests, rollouts=rollouts, rng=target_rng))
        beliefs.append(np.asarray(snapshot.posterior, dtype=np.float32))
        realized.append(goal)
    metadata = {
        "format_version": 1,
        "provenance": "new-post-acceptance-reference",
        "domain": "gridworld-reference",
        "split": split,
        "seed": seed,
        "test_seed": test_seed,
        "histories": histories,
        "rollouts": None if exact else rollouts,
        "target_estimator": "exact-finite-expectation" if exact else "posterior-monte-carlo",
        "categorical_pseudocount": None if exact else 0.5,
        "tests": len(tests),
        "test_weight": 0.01,
        "discount": discount,
        "episode_keys": [f"gridworld:{seed}:{index}" for index in range(histories)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, observations=observations, prev_actions=actions, rewards=rewards,
                        returns=returns, mask=mask, history_mask=history_mask,
                        profiles=np.asarray(profiles, dtype=np.float32),
                        belief_weights=np.asarray(beliefs, dtype=np.float32),
                        belief_values=np.broadcast_to(env.belief_coordinates, (histories, 4, 2)).copy(),
                        realized=np.asarray(realized, dtype=np.float32),
                        metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    return metadata


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
        metadata = json.loads(str(archive["metadata"]))
    required = {"observations", "prev_actions", "rewards", "returns", "mask", "history_mask", "profiles", "belief_weights", "belief_values", "realized"}
    if not required.issubset(arrays) or metadata.get("format_version") != 1:
        raise ValueError("Unsupported dataset format")
    n, t, _ = arrays["observations"].shape
    if n < 1 or t < 1 or len(metadata.get("episode_keys", [])) != n:
        raise ValueError("Dataset must contain identified nonempty histories")
    if len(set(metadata["episode_keys"])) != n:
        raise ValueError("Episode identifiers must be unique")
    for name, value in arrays.items():
        if len(value) != n or not np.isfinite(value).all():
            raise ValueError(f"Invalid array: {name}")
    if arrays["prev_actions"].shape[:2] != (n, t):
        raise ValueError("Actions must align with observations")
    for key in ("rewards", "returns", "mask", "history_mask"):
        if arrays[key].shape != (n, t):
            raise ValueError(f"Invalid shape: {key}")
    for key in ("mask", "history_mask"):
        values = arrays[key].astype(bool)
        if not values[:, 0].all() or np.any(np.diff(values.astype(int), axis=1) > 0):
            raise ValueError(f"{key} must be a nonempty contiguous prefix")
    if np.any(arrays["history_mask"] & ~arrays["mask"]):
        raise ValueError("History prefix cannot exceed the recorded episode")
    if arrays["profiles"].ndim != 3 or (arrays["profiles"] < 0).any() or not np.allclose(arrays["profiles"].sum(-1), 1, atol=1e-5):
        raise ValueError("Profiles must be categorical probability distributions")
    if (arrays["belief_weights"] < 0).any() or not np.allclose(arrays["belief_weights"].sum(-1), 1, atol=1e-5):
        raise ValueError("Belief weights must form probability distributions")
    return arrays, metadata


class BalancedSampler:
    def __init__(self, profiles, groups=20, seed=0):
        if groups < 1 or len(profiles) < 1:
            raise ValueError("Need positive group and history counts")
        flattened = np.asarray(profiles, dtype=np.float64).reshape(len(profiles), -1)
        count = min(groups, len(np.unique(flattened, axis=0)))
        if count == 1:
            labels = np.zeros(len(profiles), dtype=int)
        else:
            _, labels = kmeans2(flattened, count, iter=20, minit="++", rng=np.random.default_rng(seed))
        self.members = [np.flatnonzero(labels == group) for group in np.unique(labels)]

    def sample(self, batch_size, rng):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        assignments = rng.permutation(len(self.members))[np.arange(batch_size) % len(self.members)]
        rng.shuffle(assignments)
        return np.asarray([rng.choice(self.members[group]) for group in assignments])
