import dataclasses

import numpy as np
import pytest

from stg.environments import DiagnosticEnv, GOAL_COORDINATES, HistorySnapshot, move
from stg.targets import StructuralTest, _reach_probability, estimate_profiles, expected_profiles, make_tests, profile_weights


def test_train_prior_excludes_held_out_goal_and_shift_prior_does_not_reveal_it():
    train = DiagnosticEnv(seed=0)
    identically_distributed = DiagnosticEnv(split="id", seed=0)
    shift = DiagnosticEnv(split="shift", seed=0)
    np.testing.assert_equal(train.reset(), shift.reset())
    identically_distributed.reset()
    np.testing.assert_allclose(train.posterior, [1 / 3, 1 / 3, 1 / 3, 0])
    np.testing.assert_allclose(identically_distributed.posterior, np.full(4, 0.25))
    np.testing.assert_allclose(shift.posterior, np.full(4, 0.25))
    assert train.observation_dim == 3
    assert shift.realized_target.tolist() == shift.belief_coordinates[3].tolist()


def test_negative_evidence_eliminates_visited_goal():
    env = DiagnosticEnv(split="shift")
    env.reset()
    for action in (1, 1, 3, 3):
        _, reward, done, info = env.step(action)
    assert env.position == (1, 1)
    assert reward == 0
    assert not done
    assert not info["success"]
    np.testing.assert_allclose(env.posterior, [0, 1 / 3, 1 / 3, 1 / 3])


def test_success_reveals_goal_and_requires_reset():
    env = DiagnosticEnv(split="shift")
    env.reset()
    for action in (2, 2, 4, 4):
        observation, reward, done, info = env.step(action)
    assert reward == 1
    assert observation[2] == 1
    assert done and info["success"]
    np.testing.assert_array_equal(env.posterior, [0, 0, 0, 1])
    with pytest.raises(RuntimeError):
        env.step(0)


def test_fifty_step_horizon_and_valid_actions():
    env = DiagnosticEnv()
    env.reset()
    for _ in range(49):
        assert not env.step(0)[2]
    assert env.step(0)[2]
    assert move((1, 5), 1) == (1, 5)
    with pytest.raises(ValueError):
        move((3, 3), 5)
    with pytest.raises(ValueError):
        move((3, 3), 1.5)


def test_reset_seed_reproduces_realized_goal_without_adding_it_to_snapshot():
    env = DiagnosticEnv()
    env.reset(seed=18)
    goal = env.realized_target
    env.reset(seed=18)
    np.testing.assert_equal(goal, env.realized_target)
    assert set(dataclasses.asdict(env.snapshot())) == {"position", "elapsed", "posterior"}


def test_invalid_posterior_is_rejected():
    with pytest.raises(ValueError):
        HistorySnapshot((3, 3), 0, (1, 1, 1, 1))
    with pytest.raises(ValueError):
        HistorySnapshot((3, 3), 0, (np.nan, 0, 0, 1))


def test_test_library_is_frozen_normalized_and_has_paper_family_counts():
    tests = make_tests(seed=20)
    assert tests == make_tests(seed=20)
    assert len(tests) == 100
    assert sum(test.policy == "random" for test in tests) == 88
    assert sum(test.policy == "x_first" for test in tests) == 6
    assert sum(test.policy == "y_first" for test in tests) == 6
    np.testing.assert_allclose(profile_weights(tests), np.full(100, 0.01))


def test_exact_profiles_integrate_belief_and_continuation_randomness():
    tests = (StructuralTest((1, 1), "x_first", 0, 3, 1.0),)
    snapshot = HistorySnapshot((3, 3), 2, (0.5, 0.5, 0, 0))
    np.testing.assert_allclose(expected_profiles(snapshot, tests), [[0.625, 0.125, 0.125, 0.125]])
    other = HistorySnapshot((3, 3), 2, (0, 1, 0, 0))
    np.testing.assert_allclose(expected_profiles(other, tests), [[0.25, 0.25, 0.25, 0.25]])


def test_expected_profiles_are_normalized_and_change_with_belief():
    tests = make_tests()
    first = HistorySnapshot((3, 3), 4, (1, 0, 0, 0))
    second = HistorySnapshot((3, 3), 4, (0, 1, 0, 0))
    first_profiles = expected_profiles(first, tests)
    second_profiles = expected_profiles(second, tests)
    assert first_profiles.shape == (100, 4)
    np.testing.assert_allclose(first_profiles.sum(axis=-1), 1, atol=1e-6)
    assert np.max(np.abs(first_profiles - second_profiles)) > 0.5


def test_sampled_profiles_converge_to_exact_posterior_expectation():
    tests = (
        StructuralTest((1, 1), "x_first", 0, 3, 0.5),
        StructuralTest(None, "random", 0, 3, 0.5),
    )
    snapshot = HistorySnapshot((3, 3), 4, (0.2, 0.3, 0.5, 0))
    exact = expected_profiles(snapshot, tests)
    sampled = estimate_profiles(snapshot, tests, rollouts=10000, rng=np.random.default_rng(31))
    np.testing.assert_allclose(sampled, exact, atol=0.025)


def test_profiles_depend_only_on_the_passed_history_not_realized_goal():
    env = DiagnosticEnv(seed=1)
    env.reset()
    snapshot = env.snapshot()
    tests = make_tests()
    first = expected_profiles(snapshot, tests)
    env.reset(seed=92)
    second = expected_profiles(snapshot, tests)
    np.testing.assert_array_equal(first, second)
    with pytest.raises(TypeError):
        expected_profiles(env, tests)


def test_belief_sampling_cannot_invent_an_excluded_goal():
    env = DiagnosticEnv(seed=2)
    env.reset()
    samples = env.sample_belief(np.random.default_rng(3), size=1000)
    assert samples.shape == (1000, 2)
    assert not np.all(samples == np.asarray(GOAL_COORDINATES[3]) / 6, axis=-1).any()


def test_exact_profile_cache_reuses_frozen_test_dynamics_without_changing_probabilities():
    test = StructuralTest(None, "random", 0, 4, 1.0)
    snapshot = HistorySnapshot((3, 3), 0, (0.5, 0.5, 0, 0))
    _reach_probability.cache_clear()
    first = expected_profiles(snapshot, (test,))
    hits_before = _reach_probability.cache_info().hits
    second = expected_profiles(snapshot, (test,))
    np.testing.assert_array_equal(first, second)
    assert _reach_probability.cache_info().hits > hits_before


def test_sampled_profile_pseudocounts_are_explicit_and_validated():
    test = StructuralTest((1, 1), "x_first", 0, 3, 1.0)
    snapshot = HistorySnapshot((3, 3), 0, (1, 0, 0, 0))
    unsmoothed = estimate_profiles(snapshot, (test,), rollouts=10, smoothing=0)
    np.testing.assert_array_equal(unsmoothed, [[1, 0, 0, 0]])
    smoothed = estimate_profiles(snapshot, (test,), rollouts=10)
    np.testing.assert_allclose(smoothed, [[10.5 / 12, 0.5 / 12, 0.5 / 12, 0.5 / 12]])
    with pytest.raises(ValueError):
        estimate_profiles(snapshot, (test,), smoothing=-1)
