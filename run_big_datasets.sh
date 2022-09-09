#! /bin/bash

for RANDOM_SEED in 0 1 2 3 4 5 6 7 8 9
do
    for BUDGET in 0.1 0.2 0.3 0.4 0.5;
    do
        python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.99 --dataset_name="chess" --budget=$BUDGET --random_seed=$RANDOM_SEED --seed_size=1000 --verbose=0
        python main.py --method="fixed_uncertainty" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.99 --dataset_name="chess" --budget=$BUDGET --random_seed=$RANDOM_SEED --seed_size=1000 --verbose=0
        python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.85 --dataset_name="chess" --budget=$BUDGET --random_seed=$RANDOM_SEED --seed_size=1000 --verbose=0
        python main.py --method="classification_margin" --base_model="mlp" --prediction_threshold=0.75 --dataset_name="chess" --budget=$BUDGET --random_seed=$RANDOM_SEED --seed_size=1000 --verbose=0
    done
done