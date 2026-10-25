from dataclasses import dataclass

import numpy as np


GOAL_COORDINATES = ((1, 1), (1, 5), (5, 1), (5, 5))
ACTION_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def move(position, action):
    if not isinstance(action, (int, np.integer)) or not 0 <= action < len(ACTION_DELTAS):
        raise ValueError("Action must be an integer from 0 to 4.")
    delta = ACTION_DELTAS[int(action)]
    return tuple(int(min(5, max(1, position[k] + delta[k]))) for k in range(2))


def validate_posterior(posterior):
    probabilities = np.asarray(posterior, dtype=np.float64)
    if probabilities.shape != (4,) or not np.isfinite(probabilities).all():
        raise ValueError("The posterior must contain four finite probabilities.")
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
        raise ValueError("Posterior probabilities must be nonnegative and sum to one.")
    return probabilities.copy()


@dataclass(frozen=True)
class HistorySnapshot:
    position: tuple
    elapsed: int
    posterior: tuple

    def __post_init__(self):
        if len(self.position) != 2 or any(int(v) != v or not 1 <= v <= 5 for v in self.position):
            raise ValueError("The public position must be an interior grid coordinate.")
        if not isinstance(self.elapsed, (int, np.integer)) or self.elapsed < 0:
            raise ValueError("Elapsed steps must be a nonnegative integer.")
        probabilities = validate_posterior(self.posterior)
        object.__setattr__(self, "position", tuple(int(v) for v in self.position))
        object.__setattr__(self, "posterior", tuple(float(v) for v in probabilities))


class DiagnosticEnv:
    observation_dim = 3
    action_count = 5
    belief_kind = "coordinates"
    belief_dim = 2
    outcome_count = 4
    horizon = 50
    domain = "gridworld"

    def __init__(self, domain="gridworld", split="train", seed=0):
        if domain not in {"gridworld", "hidden_goal_gridworld"}:
            raise ValueError("This reference release implements only Hidden-Goal GridWorld.")
        if split not in {"train", "id", "shift"}:
            raise ValueError("Split must be train, id, or shift.")
        self.split = split
        self._rng = np.random.default_rng(seed)
        self._position = (3, 3)
        self._elapsed = 0
        self._done = True
        self._reward = 0.0
        self._goal_index = 0
        self._posterior = np.full(4, 0.25, dtype=np.float64)

    @property
    def posterior(self):
        return self._posterior.copy()

    @property
    def belief_coordinates(self):
        return np.asarray(GOAL_COORDINATES, dtype=np.float32) / 6.0

    @property
    def realized_target(self):
        return self.belief_coordinates[self._goal_index].copy()

    @property
    def position(self):
        return self._position

    @property
    def elapsed(self):
        return self._elapsed

    def _observation(self):
        return np.asarray((*np.asarray(self._position) / 6.0, self._reward), dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._position = (3, 3)
        self._elapsed = 0
        self._done = False
        self._reward = 0.0
        if self.split == "shift":
            self._goal_index = 3
            self._posterior = np.full(4, 0.25, dtype=np.float64)
        elif self.split == "id":
            self._goal_index = int(self._rng.integers(3))
            self._posterior = np.full(4, 0.25, dtype=np.float64)
        else:
            self._goal_index = int(self._rng.integers(3))
            self._posterior = np.asarray([1 / 3, 1 / 3, 1 / 3, 0], dtype=np.float64)
        return self._observation()

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before stepping a finished episode.")
        self._position = move(self._position, action)
        self._elapsed += 1
        success = self._position == GOAL_COORDINATES[self._goal_index]
        reward = float(success)
        self._reward = reward
        likelihood = np.asarray(
            [(self._position == goal) == success for goal in GOAL_COORDINATES],
            dtype=np.float64,
        )
        weights = self._posterior * likelihood
        normalizer = weights.sum()
        if normalizer <= 0:
            raise RuntimeError("The observation has zero likelihood under the history posterior.")
        self._posterior = weights / normalizer
        self._done = bool(success or self._elapsed >= self.horizon)
        return self._observation(), reward, self._done, {"success": bool(success)}

    def snapshot(self):
        return HistorySnapshot(self._position, self._elapsed, tuple(self._posterior))

    def sample_belief(self, rng, size=None):
        indices = rng.choice(4, size=size, p=self._posterior)
        return self.belief_coordinates[indices]


HiddenGoalGridWorld = DiagnosticEnv
