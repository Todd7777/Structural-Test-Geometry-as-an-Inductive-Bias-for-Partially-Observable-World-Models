import importlib.metadata
import json
import platform
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .data import BalancedSampler, file_hash, load_dataset
from .environments import DiagnosticEnv
from .metrics import iqd
from .model import RSSM
from .objectives import AuxiliaryObjective
from .planner import CEMPlanner


@dataclass
class TrainConfig:
    method: str = "QP"
    seed: int = 10
    steps: int = 200000
    batch_size: int = 50
    auxiliary_batch_size: int = 256
    clusters: int = 20
    deterministic_dim: int = 200
    stochastic_dim: int = 56
    hidden_dim: int = 128
    learning_rate: float = 0.0003
    auxiliary_weight: float | None = None
    kl_weight: float = 1.0
    gradient_clip: float = 100.0
    threads: int = 4
    device: str = "cpu"

    def __post_init__(self):
        self.method = self.method.upper()
        weights = {"NONE": 0.0, "RNC": 1.0, "TR": 0.5, "MD": 1.0, "BD-S": 0.5, "QP": 1.0, "PH": 1.0}
        if self.method not in weights:
            raise ValueError(f"Unsupported method: {self.method}")
        if self.auxiliary_weight is None:
            self.auxiliary_weight = weights[self.method]
        for key in ("steps", "batch_size", "auxiliary_batch_size", "clusters", "deterministic_dim", "stochastic_dim", "hidden_dim", "threads"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("learning_rate", "gradient_clip"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not np.isfinite(self.auxiliary_weight) or not np.isfinite(self.kl_weight) or self.auxiliary_weight < 0 or self.kl_weight < 0:
            raise ValueError("Loss coefficients must be finite and nonnegative")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("Device must be cpu, cuda, or mps")

    @classmethod
    def from_mapping(cls, mapping):
        unknown = set(mapping) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**mapping)


def seed_runtime(seed, threads):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)


def _tensor_arrays(arrays, device):
    return {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}


def train(dataset_path, output, config):
    arrays, metadata = load_dataset(dataset_path)
    if metadata.get("split") != "train":
        raise ValueError("Training accepts only a training-split cache")
    if metadata.get("domain") != "gridworld-reference":
        raise ValueError("The executable training pipeline currently supports the reference GridWorld only")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output must be empty: {output}")
    rng = seed_runtime(config.seed, config.threads)
    device = torch.device(config.device)
    model = RSSM(arrays["observations"].shape[-1], arrays["prev_actions"].shape[-1],
                 config.deterministic_dim, config.stochastic_dim, config.hidden_dim).to(device)
    auxiliary = AuxiliaryObjective(config.method, latent_dim=model.latent_dim,
                                   test_count=arrays["profiles"].shape[1], outcome_count=arrays["profiles"].shape[2],
                                   belief_kind="bivariate", belief_dim=2).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(auxiliary.parameters()), lr=config.learning_rate, betas=(0.9, 0.999))
    sampler = BalancedSampler(arrays["profiles"], groups=config.clusters, seed=config.seed)
    data = _tensor_arrays(arrays, device)
    output.mkdir(parents=True, exist_ok=True)
    configuration = asdict(config)
    runtime = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "PyYAML")}
    runtime["python"] = platform.python_version()
    (output / "config.json").write_text(json.dumps({"training": configuration, "dataset": metadata, "runtime": runtime}, indent=2) + "\n")
    model.train()
    auxiliary.train()
    with (output / "training.jsonl").open("w") as log:
        for step in range(config.steps):
            base_indices = torch.as_tensor(rng.integers(len(arrays["observations"]), size=config.batch_size), device=device)
            base, details = model.base_loss(data["observations"][base_indices], data["prev_actions"][base_indices],
                                            data["rewards"][base_indices], data["returns"][base_indices],
                                            mask=data["mask"][base_indices], kl_weight=config.kl_weight)
            extra = base * 0
            if config.method != "NONE":
                indices = torch.as_tensor(sampler.sample(config.auxiliary_batch_size, rng), device=device)
                z = model.encode_history(data["observations"][indices], data["prev_actions"][indices], data["history_mask"][indices])
                extra, diagnostics = auxiliary(z, data["profiles"][indices],
                                                beliefs={"values": data["belief_values"][indices], "weights": data["belief_weights"][indices]} if config.method == "BD-S" else None,
                                                realized=data["realized"][indices] if config.method == "PH" else None)
                details.update(diagnostics)
            loss = base + config.auxiliary_weight * extra
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at update {step + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(auxiliary.parameters()), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            if step == 0 or step == config.steps - 1 or (step + 1) % 100 == 0:
                record = {"step": step + 1, "loss": float(loss.detach()), "auxiliary_loss": float(extra.detach()), "gradient_norm": float(norm)}
                record.update({key: float(value) for key, value in details.items()})
                log.write(json.dumps(record, sort_keys=True) + "\n")
                log.flush()
    checkpoint = {
        "format_version": 1,
        "provenance": "new-post-acceptance-reference",
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "auxiliary": {key: value.detach().cpu() for key, value in auxiliary.state_dict().items()},
        "configuration": configuration,
        "dataset": metadata,
        "dataset_sha256": file_hash(dataset_path),
        "observation_dim": model.obs_dim,
        "action_dim": model.action_dim,
        "runtime": runtime,
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    return {"checkpoint": str(output / "checkpoint.pt"), "updates": config.steps, "method": config.method, "provenance": checkpoint["provenance"]}


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("provenance") != "new-post-acceptance-reference":
        raise ValueError("Unsupported reference checkpoint")
    config = TrainConfig.from_mapping(checkpoint["configuration"])
    torch.set_num_threads(config.threads)
    model = RSSM(checkpoint["observation_dim"], checkpoint["action_dim"], config.deterministic_dim, config.stochastic_dim, config.hidden_dim)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def _write_new_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


@torch.no_grad()
def evaluate(checkpoint_path, output, episodes=100, seed=10000, split="id", horizon=10, population=200, iterations=10, elites=20, device="cpu"):
    if episodes < 1 or split not in {"id", "shift"}:
        raise ValueError("Evaluation needs positive episodes and id or shift split")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    model, checkpoint = load_model(checkpoint_path, device)
    torch.manual_seed(seed)
    planner = CEMPlanner(model, horizon=horizon, population=population, iterations=iterations, elites=elites, discount=checkpoint["dataset"]["discount"])
    generator = torch.Generator(device=device).manual_seed(seed)
    records = []
    for episode in range(episodes):
        env = DiagnosticEnv(split=split, seed=seed + episode)
        observation = env.reset()
        previous_action = torch.zeros(1, env.action_count, device=device)
        state = model.initial(1, device=device)
        total = 0.0
        for time in range(env.horizon):
            observed = torch.as_tensor(observation, device=device).unsqueeze(0)
            state, _, _ = model.observe_step(observed, previous_action, state)
            previous_action, _ = planner.plan(state, generator=generator)
            observation, reward, done, info = env.step(int(previous_action.argmax(-1)))
            total += reward
            if done:
                break
        records.append({"task": f"gridworld-reference:{split}", "seed": checkpoint["configuration"]["seed"],
                        "episode": f"{seed}:{episode}", "success": int(info["success"]), "return": total,
                        "length": time + 1, "method": checkpoint["configuration"]["method"]})
    result = {"provenance": "new-reference-evaluation", "checkpoint_sha256": file_hash(checkpoint_path),
              "split": split, "evaluation_seed": seed, "episodes": episodes,
              "success_rule": "Reach the hidden goal within 50 ordinary deployment actions",
              "success_rate": float(np.mean([row["success"] for row in records])),
              "planner": {"horizon": horizon, "population": population, "iterations": iterations, "elites": elites},
              "records": records}
    _write_new_json(output, result)
    return result


@torch.no_grad()
def diagnose(checkpoint_path, dataset_path, output, pairs=2000, seed=20000, device="cpu"):
    if pairs < 1:
        raise ValueError("Pair count must be positive")
    model, checkpoint = load_model(checkpoint_path, device)
    arrays, metadata = load_dataset(dataset_path)
    if metadata.get("split") == "train" or set(metadata["episode_keys"]) & set(checkpoint["dataset"]["episode_keys"]):
        raise ValueError("IQD requires held-out episodes disjoint from training")
    if metadata["domain"] != checkpoint["dataset"]["domain"] or metadata["test_seed"] != checkpoint["dataset"]["test_seed"]:
        raise ValueError("Held-out cache must use the same domain and fixed test family")
    n = len(arrays["observations"])
    count = min(pairs, n * (n - 1) // 2)
    if count < 1:
        raise ValueError("Need at least two held-out histories")
    rng = np.random.default_rng(seed)
    selected = set()
    while len(selected) < count:
        left, right = sorted(rng.integers(n, size=2).tolist())
        if left != right:
            selected.add((left, right))
    pair_indices = np.asarray(sorted(selected))
    z = []
    for start in range(0, n, 128):
        selection = slice(start, start + 128)
        z.append(model.encode_history(torch.as_tensor(arrays["observations"][selection], device=device),
                                      torch.as_tensor(arrays["prev_actions"][selection], device=device),
                                      torch.as_tensor(arrays["history_mask"][selection], device=device)).cpu().numpy())
    z = np.concatenate(z)
    left, right = pair_indices.T
    native = np.linalg.norm(z[left] - z[right], axis=-1)
    target = np.abs(arrays["profiles"][left] - arrays["profiles"][right]).sum(-1).mean(-1) / 2
    fitted = iqd(native, target)
    result = {"provenance": "new-reference-diagnostic", "checkpoint_sha256": file_hash(checkpoint_path),
              "dataset_sha256": file_hash(dataset_path), "split": metadata["split"], "pair_seed": seed,
              "pairs": count, "iqd": fitted.score, "scale": fitted.scale,
              "target_second_moment": float(np.mean(target ** 2)),
              "zero_target_convention": "IQD is reported as zero when all target distances are zero; the paper's normalization is undefined there"}
    _write_new_json(output, result)
    return result
