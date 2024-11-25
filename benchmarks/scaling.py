"""Script to run when benchmarking our solver."""

import argparse
import dataclasses
import itertools
import json
import operator as op
import os
import pathlib
import shutil
import time
from datetime import datetime

import cpuinfo
import sklearn
import z3
from sklearn.datasets import make_blobs

import z3ml

parser = argparse.ArgumentParser(description="Scaling Benchmarks.")
parser.add_argument("--classes", type=int, required=True, help="")
parser.add_argument("--features", type=int, required=True, help="")
parser.add_argument("--samples", type=int, required=True, help="")
parser.add_argument("--model", choices=["linear"], default="linear", help="")
parser.add_argument("--results", default="results.json", help="")
parser.add_argument("--seed", type=int, default=1337, help="")
parser.add_argument("--trial", type=int, default=1, help="")


@dataclasses.dataclass(frozen=True)
class Config:
    num_classes: int
    num_features: int
    num_samples: int
    seed: int
    model: str
    sk_learn_version: str
    z3ml_version: str
    z3_version: str
    processor: str
    memory: int

    def __str__(self):
        return json.dumps(
            dict(sorted(dataclasses.asdict(self).items(), key=op.itemgetter(0)))
        )

    @classmethod
    def from_string(cls, s):
        return cls(**json.loads(s))


def temp_file(path):
    head = os.path.dirname(path)
    tail = os.path.basename(path)
    return os.path.join(head, f".{tail}.tmp")


def backup_file(path):
    head = os.path.dirname(path)
    tail = os.path.basename(path)
    ts = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    return os.path.join(head, f".{tail}.bk.{ts}")


def read_results(path):
    """Read results, but use shadow files to make sure we never actually delete things."""
    temp = temp_file(path)
    pathlib.Path(path).touch(exist_ok=True)
    # Create a shadow file
    shutil.copy(path, temp)
    # Create a backup
    shutil.copy(temp, backup_file(path))
    with open(path) as f:
        # json.load doesn't handle an empty file.
        f_content = f.read()
        if f_content:
            results = json.loads(f_content)
        else:
            results = []
    return results


def write_results(results, path):
    temp = temp_file(path)
    try:
        # Save the results.
        with open(path, "w") as wf:
            json.dump(results, wf)
    except:
        # If there was an error, restore the previous results file.
        if os.path.exists(temp):
            shutil.copy(temp, path)
        raise
    finally:
        os.remove(temp)


def main(args):
    # Setup Configuration
    config = Config(
        num_classes=args.classes,
        num_features=args.features,
        num_samples=args.samples,
        seed=args.seed,
        model=args.model,
        sk_learn_version=sklearn.__version__,
        z3ml_version=z3ml.__version__,
        z3_version="missing",
        processor=cpuinfo.get_cpu_info()["brand_raw"],
        memory=os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
    )
    results = read_results(args.results)
    # Generate Dataset
    X, y = make_blobs(
        n_samples=args.samples,
        n_features=config.num_features,
        centers=config.num_classes,
        random_state=config.seed,
    )
    # Get model and loss based on config
    if args.model == "linear":
        if args.classes == 2:
            ml = z3ml.models.z3MultiClassLinear(config.num_features, config.num_classes)
            loss_fn = z3ml.losses.multiclass_loss
        else:
            ml = z3ml.models.z3Linear(config.num_features)
            loss_fn = z3ml.losses.threshold_loss
    else:
        raise ValueError("Don't understand what model this is.")

    # Create the constraints
    constraints = z3ml.train.train(X, y, ml, loss_fn)

    # Solve the SAT problem.
    s = z3.Solver()
    s.add(*itertools.chain(*constraints))
    tic = time.time()
    status = s.check()
    if status == z3.unsat:
        raise ValueError("This dataset is UNSAT, try changing the seed.")
    m = s.model()
    toc = time.time()
    # Calculate the time taken
    results.append({**dataclasses.asdict(config), **{"time": toc - tic}})

    write_results(results, args.results)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
