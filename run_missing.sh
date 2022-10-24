#! /bin/bash

for RANDOM_SEED in 0 1 2 3 4 5 6 7 8 9;
do
    python main.py --method="all_labeled" --base_model="mlp" --dataset_name="firewall" --random_seed=$RANDOM_SEED --seed_size=1000 --verbose=0
    python main.py --method="all_labeled_ensemble" --base_model="mlp" --dataset_name="firewall" --random_seed=$RANDOM_SEED --seed_size=1000 --verbose=0
    for BUDGET in 0.1 0.2 0.3 0.4 0.5;
    do
        python main.py --method="random" --base_model="mlp"  --dataset_name="firewall" --budget=$BUDGET --random_seed=$RANDOM_SEED --seed_size=1000 --verbose=0
    done
done


python tune_hyperparamters.py --method="consensus_entropy" --base_model="mlp" --prediction_threshold=0.3 --dataset_name="nursery" --budget=0.3 --seed_size=100 --random_seed=42 --verbose=0
python tune_hyperparamters.py --method="max_disagreement" --base_model="mlp" --prediction_threshold=0.00001 --dataset_name="nursery" --budget=0.3 --seed_size=100 --random_seed=42 --verbose=0

for DATASET in "wine" "abalone";
do
    python tune_hyperparamters.py --method="vote_entropy" --base_model="mlp" --prediction_threshold=2.1 --dataset_name="$DATASET" --budget=0.3 --seed_size=100 --random_seed=42 --verbose=0
    python tune_hyperparamters.py --method="consensus_entropy" --base_model="mlp" --prediction_threshold=0.3 --dataset_name="$DATASET" --budget=0.3 --seed_size=100 --random_seed=42 --verbose=0
    python tune_hyperparamters.py --method="max_disagreement" --base_model="mlp" --prediction_threshold=0.00001 --dataset_name="$DATASET" --budget=0.3 --seed_size=100 --random_seed=42 --verbose=0
    python tune_hyperparamters.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --dataset_name="$DATASET" --budget=0.3 --seed_size=100 --random_seed=42 --verbose=0
done

python tune_hyperparamters.py --method="vote_entropy" --base_model="mlp" --prediction_threshold=2.1 --dataset_name="abalone" --budget=0.3 --seed_size=500 --random_seed=42 --verbose=0
python tune_hyperparamters.py --method="consensus_entropy" --base_model="mlp" --prediction_threshold=0.3 --dataset_name="abalone" --budget=0.3 --seed_size=500 --random_seed=42 --verbose=0
python tune_hyperparamters.py --method="max_disagreement" --base_model="mlp" --prediction_threshold=0.00001 --dataset_name="abalone" --budget=0.3 --seed_size=500 --random_seed=42 --verbose=0
python tune_hyperparamters.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --dataset_name="abalone" --budget=0.3 --seed_size=500 --random_seed=42 --verbose=0
