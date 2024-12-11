#!/usr/bin/env bash

TRIALS=${TRIALS:-5}
CLASSES=${CLASSES:-4}
SAMPLES=${SAMPLES:-100}

for trial in $(seq "${TRIALS}"); do
    for features in 2 3 4 8 16 32 64 128 512 784 1024; do
        ret=1
        while [[ "${ret}" -ne 0 ]]; do
            echo "python scaling.py --features ${features} --samples ${SAMPLES} --classes ${CLASSES}"
            python scaling.py --features "${features}" --samples "${SAMPLES}" --classes "${CLASSES}"
            ret=$?
        done
        ret=1
        while [[ "${ret}" -ne 0 ]]; do
            echo "python scaling.py --features ${features} --samples ${SAMPLES} --classes ${CLASSES} --one-vs-one"
            python scaling.py --features "${features}" --samples "${SAMPLES}" --classes "${CLASSES}" --one-vs-one
            ret=$?
        done
    done
done
