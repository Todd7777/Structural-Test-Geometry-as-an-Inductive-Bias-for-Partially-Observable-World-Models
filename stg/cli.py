import argparse
import json
from pathlib import Path

import yaml

from .data import collect, file_hash
from .metrics import paired_nested_bootstrap
from .training import TrainConfig, diagnose, evaluate, train


def _records(paths):
    records = []
    for path in paths:
        contents = Path(path).read_text()
        try:
            document = json.loads(contents)
        except json.JSONDecodeError:
            document = [json.loads(line) for line in contents.splitlines() if line.strip()]
        records.extend(document["records"] if isinstance(document, dict) else document)
    return records


def parser():
    root = argparse.ArgumentParser(description="New post-acceptance reference implementation; not the archived six-task experimental code")
    commands = root.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("collect", help="Build a new reference GridWorld history/response cache")
    cache.add_argument("--output", required=True)
    cache.add_argument("--histories", type=int, default=10000)
    cache.add_argument("--seed", type=int, default=0)
    cache.add_argument("--split", choices=["train", "id", "shift"], default="train")
    cache.add_argument("--rollouts", type=int, default=100)
    cache.add_argument("--test-seed", type=int, default=0)
    cache.add_argument("--exact", action="store_true", help="Use exact finite-state expectation instead of sampled response estimates")
    cache.add_argument("--discount", type=float, default=0.99)
    fitting = commands.add_parser("train", help="Fit an RSSM and selected auxiliary objective to a training cache")
    fitting.add_argument("--data", required=True)
    fitting.add_argument("--output", required=True)
    fitting.add_argument("--config")
    fitting.add_argument("--method", choices=["NONE", "RNC", "TR", "MD", "BD-S", "QP", "PH"])
    fitting.add_argument("--seed", type=int)
    fitting.add_argument("--steps", type=int)
    fitting.add_argument("--device", choices=["cpu", "cuda", "mps"])
    evaluation = commands.add_parser("evaluate", help="Plan using deployment actions only and save new episode-level measurements")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--episodes", type=int, default=100)
    evaluation.add_argument("--seed", type=int, default=10000)
    evaluation.add_argument("--split", choices=["id", "shift"], default="id")
    evaluation.add_argument("--horizon", type=int, default=10)
    evaluation.add_argument("--population", type=int, default=200)
    evaluation.add_argument("--iterations", type=int, default=10)
    evaluation.add_argument("--elites", type=int, default=20)
    evaluation.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    diagnostic = commands.add_parser("diagnose", help="Measure IQD on separately collected held-out histories")
    diagnostic.add_argument("--checkpoint", required=True)
    diagnostic.add_argument("--data", required=True)
    diagnostic.add_argument("--output", required=True)
    diagnostic.add_argument("--pairs", type=int, default=2000)
    diagnostic.add_argument("--seed", type=int, default=20000)
    diagnostic.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    bootstrap = commands.add_parser("bootstrap", help="Compute a paired nested bootstrap from matched measured episode records")
    bootstrap.add_argument("--first", nargs="+", required=True)
    bootstrap.add_argument("--second", nargs="+", required=True)
    bootstrap.add_argument("--output", required=True)
    bootstrap.add_argument("--replicates", type=int, default=10000)
    bootstrap.add_argument("--seed", type=int, default=0)
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "collect":
            result = collect(args.output, args.histories, args.seed, args.split, args.rollouts, args.test_seed, args.exact, args.discount)
        elif args.command == "train":
            mapping = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
            if not isinstance(mapping, dict):
                raise ValueError("Configuration must be a mapping")
            for key in ("method", "seed", "steps", "device"):
                if getattr(args, key) is not None:
                    mapping[key] = getattr(args, key)
            result = train(args.data, args.output, TrainConfig.from_mapping(mapping))
        elif args.command == "evaluate":
            result = evaluate(args.checkpoint, args.output, args.episodes, args.seed, args.split, args.horizon, args.population, args.iterations, args.elites, args.device)
            result = {key: value for key, value in result.items() if key != "records"}
        elif args.command == "diagnose":
            result = diagnose(args.checkpoint, args.data, args.output, args.pairs, args.seed, args.device)
        else:
            result = paired_nested_bootstrap(_records(args.first), _records(args.second), replicates=args.replicates, rng_seed=args.seed)
            result["provenance"] = "computed-from-supplied-episode-records"
            result["source_sha256"] = {"first": [file_hash(path) for path in args.first], "second": [file_hash(path) for path in args.second]}
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as stream:
                stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as error:
        root.exit(2, f"Error: {error}\n")
