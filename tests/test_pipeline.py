import json

import numpy as np
import pytest
import torch

from stg.cli import main
from stg.data import BalancedSampler, collect, discounted_returns, load_dataset
from stg.model import RSSM
from stg.training import TrainConfig, diagnose, evaluate, load_model, train


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cache")
    training = directory / "train.npz"
    heldout = directory / "heldout.npz"
    collect(training, histories=12, seed=52, exact=True)
    collect(heldout, histories=6, seed=53, split="id", exact=True)
    return training, heldout, directory


def test_returns_are_future_rewards_not_current_reward():
    np.testing.assert_allclose(discounted_returns(np.asarray([0, 0, 1]), discount=0.5), [0.5, 1, 0])


def test_collection_is_seeded_and_histories_are_consistent(caches, tmp_path):
    original, _, _ = caches
    copy = tmp_path / "copy.npz"
    collect(copy, histories=12, seed=52, exact=True)
    arrays, metadata = load_dataset(original)
    repeated, _ = load_dataset(copy)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], repeated[key])
    assert metadata["provenance"] == "new-post-acceptance-reference"
    assert (arrays["belief_weights"][:, 3] == 0).all()
    for index in range(12):
        prefix = arrays["history_mask"][index]
        position = tuple(np.rint(arrays["observations"][index, np.flatnonzero(prefix)[-1], :2] * 6).astype(int))
        if position in ((1, 1), (1, 5), (5, 1)):
            goal_index = ((1, 1), (1, 5), (5, 1)).index(position)
            last_reward = arrays["observations"][index, np.flatnonzero(prefix)[-1], 2]
            assert arrays["belief_weights"][index, goal_index] == (1 if last_reward else 0)
    with pytest.raises(FileExistsError):
        collect(original, histories=1)


def test_balanced_sampler_handles_identical_profiles():
    sampler = BalancedSampler(np.full((4, 100, 4), 0.25), groups=20)
    indices = sampler.sample(256, np.random.default_rng(0))
    assert len(indices) == 256
    assert indices.min() >= 0 and indices.max() < 4


@pytest.mark.parametrize("method", ["NONE", "RNC", "TR", "MD", "BD-S", "QP", "PH"])
def test_every_method_trains_and_checkpoint_loads(caches, tmp_path, method):
    training, _, _ = caches
    config = TrainConfig(method=method, steps=2, batch_size=2, auxiliary_batch_size=4,
                         deterministic_dim=12, stochastic_dim=4, hidden_dim=16, threads=1)
    result = train(training, tmp_path / method, config)
    model, checkpoint = load_model(result["checkpoint"])
    assert checkpoint["configuration"]["method"] == method
    log = [json.loads(line) for line in (tmp_path / method / "training.jsonl").read_text().splitlines()]
    assert len(log) == 2
    assert all(np.isfinite(row["loss"]) for row in log)
    assert model.latent_dim == 16
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    torch.manual_seed(config.seed)
    initial = RSSM(model.obs_dim, model.action_dim, config.deterministic_dim, config.stochastic_dim, config.hidden_dim)
    assert not torch.equal(initial.encoder[0].weight, model.encoder[0].weight)


def test_deployment_and_heldout_diagnostics(caches, tmp_path):
    training, heldout, _ = caches
    config = TrainConfig(steps=1, batch_size=2, auxiliary_batch_size=4,
                         deterministic_dim=12, stochastic_dim=4, hidden_dim=16, threads=1)
    trained = train(training, tmp_path / "trained", config)
    result = evaluate(trained["checkpoint"], tmp_path / "evaluation.json", episodes=1, seed=80,
                      split="shift", horizon=2, population=4, iterations=1, elites=2)
    assert result["records"][0]["success"] in (0, 1)
    assert result["records"][0]["length"] <= 50
    diagnostic = diagnose(trained["checkpoint"], heldout, tmp_path / "diagnostic.json", pairs=10)
    assert 0 <= diagnostic["iqd"] <= 1
    assert diagnostic["pairs"] == 10
    with pytest.raises(ValueError, match="held-out"):
        diagnose(trained["checkpoint"], training, tmp_path / "bad.json")


def test_training_rejects_heldout_cache_and_unknown_settings(caches, tmp_path):
    _, heldout, _ = caches
    with pytest.raises(ValueError, match="training-split"):
        train(heldout, tmp_path / "bad", TrainConfig(steps=1))
    with pytest.raises(ValueError, match="Unknown"):
        TrainConfig.from_mapping({"epohcs": 1})


def test_same_seed_history_reuse_is_rejected_even_across_split_labels(caches, tmp_path):
    training, _, _ = caches
    heldout = tmp_path / "same-seed-id.npz"
    collect(heldout, histories=2, seed=52, split="id", exact=True)
    config = TrainConfig(steps=1, batch_size=1, auxiliary_batch_size=2,
                         deterministic_dim=8, stochastic_dim=2, hidden_dim=8, threads=1)
    trained = train(training, tmp_path / "trained", config)
    with pytest.raises(ValueError, match="disjoint"):
        diagnose(trained["checkpoint"], heldout, tmp_path / "reuse.json")


def test_cli_collects_and_does_not_overwrite(tmp_path, capsys):
    main(["collect", "--output", str(tmp_path / "cli.npz"), "--histories", "2", "--exact"])
    assert json.loads(capsys.readouterr().out)["histories"] == 2
    with pytest.raises(SystemExit) as failure:
        main(["collect", "--output", str(tmp_path / "cli.npz"), "--histories", "2", "--exact"])
    assert failure.value.code == 2
