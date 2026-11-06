#!/bin/bash

for prop in logp fsp3 n_star n_ring n_atoms
do
    python experiments/supervised/train_wavepde_prop.py \
        --prop $prop \
        --model.learning_rate 1e-3 \
        --model.pde_function wave \
        --data.n 50000 \
        -e 50
done
