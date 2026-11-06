#!/bin/bash

PROPS=(
    "fsp3"
    "n_star"
    "n_ring"
    "n_atoms"
)

for prop in "${PROPS[@]}"
do
    echo "====================================="
    echo "Running optimization for ${prop}"
    echo "====================================="

    python experiments/optimization/optimization.py \
        --prop ${prop} \
        --method wave_sup \
        --steps 1000 \
        --step_size 0.1

    echo "Finished ${prop}"
    echo
done

echo "All optimizations completed!"
