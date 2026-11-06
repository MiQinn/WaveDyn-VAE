#!/bin/bash

for prop in logp fsp3 n_star n_ring n_atoms; do

  echo "=============================="
  echo "Training predictor for $prop (no early stopping)"
  echo "=============================="

  python experiments/supervised/train_prop_predictor.py \
    --data.prop $prop \
    --model.optimizer adam \
    -e 50 \
    --data.n 18341 \
    --data.batch_size 256 \
    --data.seed 42 \
    --no_early_stop \
    --bucket

done
