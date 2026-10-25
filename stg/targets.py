from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .environments import ACTION_DELTAS, GOAL_COORDINATES, HistorySnapshot, move, validate_posterior


@dataclass(frozen=True)
class StructuralTest:
    probe: tuple | None
    policy: str
    target_corner: int
    horizon: int
    weight: float = 0.01

    def __post_init__(self):
        if self.probe is not None and self.probe not in (*GOAL_COORDINATES, (3, 3)):
            raise ValueError("A teleport probe must name a corner or the centre.")
        if self.policy not in {"random", "x_first", "y_first"}:
            raise ValueError("Unknown continuation policy.")
        if not 0 <= self.target_corner < 4 or self.horizon < 1:
            raise ValueError("A test needs a valid target corner and a positive horizon.")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("Test weights must be positive and finite.")


def make_tests(seed=0):
    rng = np.random.default_rng(seed)
    probes = (*GOAL_COORDINATES, (3, 3), None)
    tests = []
    for probe in probes:
        for variant in range(5):
            policy = "random" if variant < 3 else ("x_first" if variant == 3 else "y_first")
            tests.append(
                StructuralTest(probe, policy, int(rng.integers(4)), int(rng.integers(3, 8)))
            )
    for _ in range(70):
        tests.append(
            StructuralTest(
                probes[int(rng.integers(len(probes)))],
                "random",
                int(rng.integers(4)),
                int(rng.integers(3, 8)),
            )
        )
    return tuple(tests)


def _action_probabilities(position, test):
    if test.policy == "random":
        return np.full(len(ACTION_DELTAS), 1 / len(ACTION_DELTAS))
    target = GOAL_COORDINATES[test.target_corner]
    axes = (0, 1) if test.policy == "x_first" else (1, 0)
    action = 0
    for axis in axes:
        difference = target[axis] - position[axis]
        if difference:
            action = (2 if difference > 0 else 1) if axis == 0 else (4 if difference > 0 else 3)
            break
    result = np.zeros(len(ACTION_DELTAS))
    result[action] = 1
    return result


def _validate_inputs(snapshot, tests):
    if not isinstance(snapshot, HistorySnapshot):
        raise TypeError("Structural targets require a history snapshot, not realized simulator state.")
    probabilities = validate_posterior(snapshot.posterior)
    if not tests:
        raise ValueError("At least one structural test is required.")
    if not np.isclose(sum(test.weight for test in tests), 1.0):
        raise ValueError("Structural test weights must sum to one.")
    return probabilities


@lru_cache(maxsize=16384)
def _reach_probability(start, goal, test):
    if start == goal:
        return 1.0
    states = {start: 1.0}
    reached = 0.0
    for _ in range(test.horizon):
        next_states = {}
        for position, probability in states.items():
            for action, action_probability in enumerate(_action_probabilities(position, test)):
                if action_probability == 0:
                    continue
                successor = move(position, action)
                mass = probability * action_probability
                if successor == goal:
                    reached += mass
                else:
                    next_states[successor] = next_states.get(successor, 0.0) + mass
        states = next_states
    return min(1.0, max(0.0, reached))


def expected_profiles(snapshot, tests):
    probabilities = _validate_inputs(snapshot, tests)
    profiles = np.zeros((len(tests), 4), dtype=np.float64)
    uniform = np.full(4, 0.25)
    for test_index, test in enumerate(tests):
        start = snapshot.position if test.probe is None else test.probe
        for goal_index, goal in enumerate(GOAL_COORDINATES):
            if probabilities[goal_index] == 0:
                continue
            reach = _reach_probability(start, goal, test)
            conditional = uniform * (1 - reach)
            conditional[goal_index] += reach
            profiles[test_index] += probabilities[goal_index] * conditional
    return profiles.astype(np.float32)


def estimate_profiles(snapshot, tests, rollouts=100, rng=None, smoothing=0.5):
    probabilities = _validate_inputs(snapshot, tests)
    if not isinstance(rollouts, (int, np.integer)) or rollouts < 1:
        raise ValueError("Rollouts must be a positive integer.")
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("Categorical pseudocount smoothing must be nonnegative and finite.")
    rng = np.random.default_rng() if rng is None else rng
    counts = np.zeros((len(tests), 4), dtype=np.float64)
    goal_coordinates = np.asarray(GOAL_COORDINATES)
    action_deltas = np.asarray(ACTION_DELTAS)
    for test_index, test in enumerate(tests):
        goals = rng.choice(4, size=rollouts, p=probabilities)
        start = snapshot.position if test.probe is None else test.probe
        positions = np.broadcast_to(start, (rollouts, 2)).copy()
        selected_goals = goal_coordinates[goals]
        reached = np.all(positions == selected_goals, axis=-1)
        for _ in range(test.horizon):
            if reached.all():
                break
            if test.policy == "random":
                actions = rng.integers(len(ACTION_DELTAS), size=rollouts)
            else:
                target = goal_coordinates[test.target_corner]
                difference = target - positions
                actions = np.zeros(rollouts, dtype=np.int64)
                axes = (1, 0) if test.policy == "x_first" else (0, 1)
                for axis in axes:
                    mask = difference[:, axis] != 0
                    positive, negative = (2, 1) if axis == 0 else (4, 3)
                    actions[mask] = np.where(difference[mask, axis] > 0, positive, negative)
            successors = np.clip(positions + action_deltas[actions], 1, 5)
            positions = np.where(reached[:, None], positions, successors)
            reached |= np.all(positions == selected_goals, axis=-1)
        outcomes = np.where(reached, goals, rng.integers(4, size=rollouts))
        counts[test_index] = np.bincount(outcomes, minlength=4)
    return ((counts + smoothing) / (rollouts + 4 * smoothing)).astype(np.float32)


def profile_weights(tests):
    weights = np.asarray([test.weight for test in tests], dtype=np.float32)
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Structural test weights must be positive and sum to one.")
    return weights
