#!/usr/bin/env bash

TRIALS=${TRIALS:-5}
FEATURES=${FEATURES:-2}
SAMPLES=${SAMPLES:-100}

for trial in $(seq "${TRIALS}"); do
    for classes in 2 3 4 6 8 12 16 24 32 64; do
        ret=1
        while [[ "${ret}" -ne 0 ]]; do
            echo "python scaling.py --features ${FEATURES} --samples ${SAMPLES} --classes ${classes}"
            python scaling.py --features "${FEATURES}" --samples "${SAMPLES}" --classes "${classes}"
            ret=$?
        done
        ret=1
        while [[ "${ret}" -ne 0 ]]; do
            echo "python scaling.py --features ${FEATURES} --samples ${SAMPLES} --classes ${classes} --one-vs-one"
            python scaling.py --features "${FEATURES}" --samples "${SAMPLES}" --classes "${classes}" --one-vs-one
            ret=$?
        done
    done
done
