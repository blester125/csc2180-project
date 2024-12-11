#!/usr/bin/env python3

import argparse
import itertools
import time

import get_mnist
import numpy as np
import z3

import z3ml

parser = argparse.ArgumentParser(description="Train a OneVsOne classifier on MNIST.")
parser.add_argument(
    "--sample",
    default=None,
    type=int,
    help="How many examples to sample for each classifer.",
)


def main(args):
    X_train, y_train, X_test, y_test = get_mnist.mnist()
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    ml = z3ml.models.z3OneVsOne(
        classes=sorted(np.unique(y_train).tolist()),
        # classes=(0, 1, 2),
        n_features=X_train.shape[-1],
    )

    X_train_, y_train_ = ml.filter_data(X_train, y_train)

    if args.sample is not None:
        sampled = {}
        for classes, data in X_train_.items():
            sampled[classes] = np.random.choice(
                np.arange(data.shape[0]), size=args.sample, replace=False
            )
        X_train_ = {k: v[sampled[k]] for k, v in X_train_.items()}
        y_train_ = {k: v[sampled[k]] for k, v in y_train_.items()}

    tic = time.time()
    constraints = z3ml.train.train(X_train_, y_train_, ml, z3ml.losses.one_vs_one_loss)
    toc = time.time()
    print(f"Constraint Generation Time: {toc - tic:.4f}")

    solvers = {}
    times = {}
    for classes, cs in constraints.items():
        solvers[classes] = z3.Solver()
        solvers[classes].add(*itertools.chain(*cs))
        tic = time.time()
        status = solvers[classes].check()
        toc = time.time()
        times[classes] = toc - tic
        if status == z3.sat:
            ml.models[classes] = ml.models[classes].realize(solvers[classes].model())
        else:
            print(f"{classes[0]} vs {classes[1]} was UNSAT, skipping...")
            ml.models.pop(classes, None)
    # Testing
    # ml.models.pop((2, 3), None)
    print("Solve Times")
    for classes, t in times.items():
        print(f"  {classes[0]} vs {classes[1]}: {t:.4f}")
    print(f"Total Solve Time: {sum(times.values()):.4f}")

    X_test_, y_test_ = ml.filter_data(X_test, y_test)

    perf = {}
    for classes, labels in y_test_.items():
        if classes in ml.models:
            preds = ml.models[classes].predict(X_test_[classes])
            perf[classes] = np.mean(preds == labels)
    preds = ml.predict(X_test)
    perf["All way"] = np.mean(preds == y_test)
    print("Performance:")
    for classes, acc in perf.items():
        if classes == "All way":
            continue
        print(f"  {classes[0]} vs {classes[1]}: {acc * 100:.4f}")
    print(f"Multiclass Performance: {perf['All way'] * 100:.4f}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
