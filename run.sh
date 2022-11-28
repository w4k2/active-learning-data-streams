#! /bin/bash

do_experiments () {
    echo "executing experiments for dataset $1"
    
    # variable budget
    for RANDOM_SEED in 0 1 2 3 4 5 6 7 8 9;
    do
        python main.py --method="all_labeled" --base_model="mlp" --dataset_name="$1" --random_seed=$RANDOM_SEED --seed_size=1000 --verbose=0 &
        python main.py --method="all_labeled_ensemble" --base_model="mlp" --dataset_name="$1" --random_seed=$RANDOM_SEED --seed_size=1000 --verbose=0 &
        for BUDGET in 0.1 0.2 0.3 0.4 0.5;
        do
            python main.py --method="random" --base_model="mlp"  --dataset_name="$1" --budget=$BUDGET --random_seed=$RANDOM_SEED --seed_size=1000 --verbose=0 &
        done
        wait
    done
    for BUDGET in 0.1 0.2 0.3 0.4 0.5;
    do
        echo "experiments with budget = $BUDGET"
        python tune_hyperparamters.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.8 --dataset_name="$1" --budget=$BUDGET --random_seed=42 --seed_size=1000 --verbose=0 &
        python tune_hyperparamters.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.995 --dataset_name="$1" --budget=$BUDGET --random_seed=42 --seed_size=1000 --verbose=0 &
        python tune_hyperparamters.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=$BUDGET --random_seed=42 --seed_size=1000 --verbose=0 &
        python tune_hyperparamters.py --method="classification_margin" --base_model="mlp" --prediction_threshold=0.4 --dataset_name="$1" --budget=$BUDGET --random_seed=42 --seed_size=1000 --verbose=0 &
        python tune_hyperparamters.py --method="vote_entropy" --base_model="mlp" --prediction_threshold=10.0 --dataset_name="$1" --budget=$BUDGET --random_seed=42 --seed_size=1000 --verbose=0 &
        python tune_hyperparamters.py --method="consensus_entropy" --base_model="mlp" --prediction_threshold=1.0 --dataset_name="$1" --budget=$BUDGET --random_seed=42 --seed_size=1000 --verbose=0 &
        python tune_hyperparamters.py --method="max_disagreement" --base_model="mlp" --prediction_threshold=20.0 --dataset_name="$1" --budget=$BUDGET --random_seed=42 --seed_size=1000 --verbose=0 &
        python tune_hyperparamters.py --method="min_margin" --base_model="mlp" --prediction_threshold=1.0 --dataset_name="$1" --budget=$BUDGET --random_seed=42 --seed_size=1000 --verbose=0 &
        echo "waiting for experiments to finish"
        wait
    done

    # base model ng (naive bayes)
    echo "experiments with Naive Bayes classifier"
    python tune_hyperparamters.py --method="ours" --base_model="ng" --beta1=0.9 --prediction_threshold=0.8 --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0 &
    python main.py --method="all_labeled" --base_model="ng" --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0 &
    python main.py --method="all_labeled_ensemble" --base_model="ng" --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0 &
    python main.py --method="random" --base_model="ng"  --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0 &
    python tune_hyperparamters.py --method="fixed_uncertainty" --base_model="ng" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0 &
    python tune_hyperparamters.py --method="variable_uncertainty" --base_model="ng" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0 &
    python tune_hyperparamters.py --method="classification_margin" --base_model="ng" --prediction_threshold=0.9 --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0 &
    python tune_hyperparamters.py --method="vote_entropy" --base_model="ng" --prediction_threshold=2.1 --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0 &
    python tune_hyperparamters.py --method="consensus_entropy" --base_model="ng" --prediction_threshold=0.3 --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0 &
    python tune_hyperparamters.py --method="max_disagreement" --base_model="ng" --prediction_threshold=0.00001 --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0 &
    python tune_hyperparamters.py --method="min_margin" --base_model="ng" --prediction_threshold=1.0 --dataset_name="$1" --budget=0.5 --seed_size=$SEED_SIZE --random_seed=45 --verbose=0 &
    echo "waiting for experiments to finish"
    wait


    # variable seed size
    for RANDOM_SEED in 0 1 2 3 4 5 6 7 8 9;
    do
        python main.py --method="all_labeled" --base_model="mlp" --dataset_name="$1" --budget=0.3 --seed_size=100 --random_seed=$RANDOM_SEED --verbose=0 &
        python main.py --method="all_labeled_ensemble" --base_model="mlp" --dataset_name="$1" --budget=0.3 --seed_size=100 --random_seed=$RANDOM_SEED --verbose=0 &
        for SEED_SIZE in 100 200 500 1000;
        do
            python main.py --method="random" --base_model="mlp"  --dataset_name="$1" --budget=0.3 --seed_size=$SEED_SIZE --random_seed=$RANDOM_SEED --verbose=0 &
        done
        wait
    done
    for SEED_SIZE in 100 200 500 1000;
    do
        echo "experiments with seed size = $SEED_SIZE"
        python tune_hyperparamters.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --dataset_name="$1" --budget=0.3 --seed_size=$SEED_SIZE --random_seed=42 --verbose=0 &
        python tune_hyperparamters.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.3 --seed_size=$SEED_SIZE --random_seed=42 --verbose=0 &
        python tune_hyperparamters.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.3 --seed_size=$SEED_SIZE --random_seed=42 --verbose=0 &
        python tune_hyperparamters.py --method="classification_margin" --base_model="mlp" --prediction_threshold=0.9 --dataset_name="$1" --budget=0.3 --seed_size=$SEED_SIZE --random_seed=42 --verbose=0 &
        python tune_hyperparamters.py --method="vote_entropy" --base_model="mlp" --prediction_threshold=2.1 --dataset_name="$1" --budget=0.3 --seed_size=$SEED_SIZE --random_seed=42 --verbose=0 &
        python tune_hyperparamters.py --method="consensus_entropy" --base_model="mlp" --prediction_threshold=0.3 --dataset_name="$1" --budget=0.3 --seed_size=$SEED_SIZE --random_seed=42 --verbose=0 &
        python tune_hyperparamters.py --method="max_disagreement" --base_model="mlp" --prediction_threshold=0.00001 --dataset_name="$1" --budget=0.3 --seed_size=$SEED_SIZE --random_seed=42 --verbose=0 &
        python tune_hyperparamters.py --method="min_margin" --base_model="mlp" --prediction_threshold=1.0 --dataset_name="$1" --budget=0.3 --seed_size=$SEED_SIZE --random_seed=42 --verbose=0 &
        echo "waiting for experiments to finish"
        wait
    done
}

do_experiments "abalone"
do_experiments "wine"
do_experiments "mushroom"
do_experiments "nursery"
do_experiments "chess"
do_experiments "firewall"
do_experiments "bank_marketing"
do_experiments "adult"
do_experiments "accelerometer"
