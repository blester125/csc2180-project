#!/usr/bin/env bash

TRIALS=${TRIALS:-5}
CLASSES=${CLASSES:-4}
FEATURES=${FEATURES:-3}

for samples in 10 100 200 500 1000 1500 2000 5000 10000; do
    for trial in $(seq "${TRIALS}"); do
        ret=1
        while [[ "${ret}" -ne 0 ]]; do
            echo "python scaling.py --features ${FEATURES} --samples ${samples} --classes ${CLASSES}"
            python scaling.py --features "${FEATURES}" --samples "${samples}" --classes "${CLASSES}"
            ret=$?
        done
        ret=1
        while [[ "${ret}" -ne 0 ]]; do
            echo "python scaling.py --features ${FEATURES} --samples ${samples} --classes ${CLASSES} --one-vs-one"
            python scaling.py --features "${FEATURES}" --samples "${samples}" --classes "${CLASSES}" --one-vs-one
            ret=$?
        done
    done
done
