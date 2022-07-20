#! /bin/bash

do_experiments () {
    # variable budget
    python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --dataset_name="$1" --budget=0.5 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="all_labeled" --base_model="mlp" --dataset_name="$1" --budget=0.5 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="all_labeled_ensemble" --base_model="mlp" --dataset_name="$1" --budget=0.5 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="random" --base_model="mlp"  --dataset_name="$1" --budget=0.5 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.5 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.5 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="classification_margin" --base_model="mlp" --prediction_threshold=0.9 --dataset_name="$1" --budget=0.5 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="vote_entropy" --base_model="mlp" --prediction_threshold=2.1 --dataset_name="$1" --budget=0.5 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="consensus_entropy" --base_model="mlp" --prediction_threshold=0.3 --dataset_name="$1" --budget=0.5 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="max_disagreement" --base_model="mlp" --prediction_threshold=0.00001 --dataset_name="$1" --budget=0.5 --random_seed=1410 --seed_size=1000 --verbose=0
    # python plots.py --dataset_name="$1" --budget=0.5 --random_seed=1410 --seed_size=1000

    python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --dataset_name="$1" --budget=0.4 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="all_labeled" --base_model="mlp" --dataset_name="$1" --budget=0.4 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="all_labeled_ensemble" --base_model="mlp" --dataset_name="$1" --budget=0.4 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="random" --base_model="mlp"  --dataset_name="$1" --budget=0.4 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.4 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.4 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="classification_margin" --base_model="mlp" --prediction_threshold=0.9 --dataset_name="$1" --budget=0.4 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="vote_entropy" --base_model="mlp" --prediction_threshold=2.1 --dataset_name="$1" --budget=0.4 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="consensus_entropy" --base_model="mlp" --prediction_threshold=0.3 --dataset_name="$1" --budget=0.4 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="max_disagreement" --base_model="mlp" --prediction_threshold=0.00001 --dataset_name="$1" --budget=0.4 --random_seed=1410 --seed_size=1000 --verbose=0
    # python plots.py --dataset_name="$1" --budget=0.4 --random_seed=1410 --seed_size=1000

    python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --dataset_name="$1" --budget=0.3 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="all_labeled" --base_model="mlp" --dataset_name="$1" --budget=0.3 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="all_labeled_ensemble" --base_model="mlp" --dataset_name="$1" --budget=0.3 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="random" --base_model="mlp"  --dataset_name="$1" --budget=0.3 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.3 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.3 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="classification_margin" --base_model="mlp" --prediction_threshold=0.9 --dataset_name="$1" --budget=0.3 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="vote_entropy" --base_model="mlp" --prediction_threshold=2.1 --dataset_name="$1" --budget=0.3 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="consensus_entropy" --base_model="mlp" --prediction_threshold=0.3 --dataset_name="$1" --budget=0.3 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="max_disagreement" --base_model="mlp" --prediction_threshold=0.00001 --dataset_name="$1" --budget=0.3 --random_seed=1410 --seed_size=1000 --verbose=0
    # python plots.py --dataset_name="$1" --budget=0.3 --random_seed=1410 --seed_size=1000

    python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --dataset_name="$1" --budget=0.2 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="all_labeled" --base_model="mlp" --dataset_name="$1" --budget=0.2 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="all_labeled_ensemble" --base_model="mlp" --dataset_name="$1" --budget=0.2 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="random" --base_model="mlp"  --dataset_name="$1" --budget=0.2 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.2 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.2 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="classification_margin" --base_model="mlp" --prediction_threshold=0.9 --dataset_name="$1" --budget=0.2 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="vote_entropy" --base_model="mlp" --prediction_threshold=2.1 --dataset_name="$1" --budget=0.2 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="consensus_entropy" --base_model="mlp" --prediction_threshold=0.3 --dataset_name="$1" --budget=0.2 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="max_disagreement" --base_model="mlp" --prediction_threshold=0.00001 --dataset_name="$1" --budget=0.2 --random_seed=1410 --seed_size=1000 --verbose=0
    # python plots.py --dataset_name="$1" --budget=0.2 --random_seed=1410 --seed_size=1000

    python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --dataset_name="$1" --budget=0.1 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="all_labeled" --base_model="mlp" --dataset_name="$1" --budget=0.1 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="all_labeled_ensemble" --base_model="mlp" --dataset_name="$1" --budget=0.1 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="random" --base_model="mlp"  --dataset_name="$1" --budget=0.1 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.1 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.1 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="classification_margin" --base_model="mlp" --prediction_threshold=0.9 --dataset_name="$1" --budget=0.1 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="vote_entropy" --base_model="mlp" --prediction_threshold=2.1 --dataset_name="$1" --budget=0.1 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="consensus_entropy" --base_model="mlp" --prediction_threshold=0.3 --dataset_name="$1" --budget=0.1 --random_seed=1410 --seed_size=1000 --verbose=0
    python main.py --method="max_disagreement" --base_model="mlp" --prediction_threshold=0.00001 --dataset_name="$1" --budget=0.1 --random_seed=1410 --seed_size=1000 --verbose=0
    # python plots.py --dataset_name="$1" --budget=0.1 --random_seed=1410 --seed_size=1000


    # base model ng (naive bayes)
    python main.py --method="ours" --base_model="ng" --beta1=0.9 --prediction_threshold=0.8 --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0
    python main.py --method="all_labeled" --base_model="ng" --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0
    python main.py --method="all_labeled_ensemble" --base_model="ng" --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0
    python main.py --method="random" --base_model="ng"  --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0
    python main.py --method="fixed_uncertainty" --base_model="ng" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0
    python main.py --method="variable_uncertainty" --base_model="ng" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0
    python main.py --method="classification_margin" --base_model="ng" --prediction_threshold=0.9 --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0
    python main.py --method="vote_entropy" --base_model="ng" --prediction_threshold=2.1 --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0
    python main.py --method="consensus_entropy" --base_model="ng" --prediction_threshold=0.3 --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0
    python main.py --method="max_disagreement" --base_model="ng" --prediction_threshold=0.00001 --dataset_name="$1" --budget=0.5 --random_seed=45 --verbose=0
    # python plots.py --dataset_name="$1" --budget=0.5 --base_model="ng" --random_seed=45


    # variable seed size
    python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --dataset_name="$1" --budget=0.3 --seed_size=100 --random_seed=1410 --verbose=0
    python main.py --method="all_labeled" --base_model="mlp" --dataset_name="$1" --budget=0.3 --seed_size=100 --random_seed=1410 --verbose=0
    python main.py --method="all_labeled_ensemble" --base_model="mlp" --dataset_name="$1" --budget=0.3 --seed_size=100 --random_seed=1410 --verbose=0
    python main.py --method="random" --base_model="mlp"  --dataset_name="$1" --budget=0.3 --seed_size=100 --random_seed=1410 --verbose=0
    python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.3 --seed_size=100 --random_seed=1410 --verbose=0
    python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.3 --seed_size=100 --random_seed=1410 --verbose=0
    python main.py --method="classification_margin" --base_model="mlp" --prediction_threshold=0.9 --dataset_name="$1" --budget=0.3 --seed_size=100 --random_seed=1410 --verbose=0
    python main.py --method="vote_entropy" --base_model="mlp" --prediction_threshold=2.1 --dataset_name="$1" --budget=0.3 --seed_size=100 --random_seed=1410 --verbose=0
    python main.py --method="consensus_entropy" --base_model="mlp" --prediction_threshold=0.3 --dataset_name="$1" --budget=0.3 --seed_size=100 --random_seed=1410 --verbose=0
    python main.py --method="max_disagreement" --base_model="mlp" --prediction_threshold=0.00001 --dataset_name="$1" --budget=0.3 --seed_size=100 --random_seed=1410 --verbose=0
    # python plots.py --dataset_name="$1" --budget=0.3 --seed_size=100 --random_seed=1410

    python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --dataset_name="$1" --budget=0.3 --seed_size=500 --random_seed=1410 --verbose=0
    python main.py --method="all_labeled" --base_model="mlp" --dataset_name="$1" --budget=0.3 --seed_size=500 --random_seed=1410 --verbose=0
    python main.py --method="all_labeled_ensemble" --base_model="mlp" --dataset_name="$1" --budget=0.3 --seed_size=500 --random_seed=1410 --verbose=0
    python main.py --method="random" --base_model="mlp"  --dataset_name="$1" --budget=0.3 --seed_size=500 --random_seed=1410 --verbose=0
    python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.3 --seed_size=500 --random_seed=1410 --verbose=0
    python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.3 --seed_size=500 --random_seed=1410 --verbose=0
    python main.py --method="classification_margin" --base_model="mlp" --prediction_threshold=0.9 --dataset_name="$1" --budget=0.3 --seed_size=500 --random_seed=1410 --verbose=0
    python main.py --method="vote_entropy" --base_model="mlp" --prediction_threshold=2.1 --dataset_name="$1" --budget=0.3 --seed_size=500 --random_seed=1410 --verbose=0
    python main.py --method="consensus_entropy" --base_model="mlp" --prediction_threshold=0.3 --dataset_name="$1" --budget=0.3 --seed_size=500 --random_seed=1410 --verbose=0
    python main.py --method="max_disagreement" --base_model="mlp" --prediction_threshold=0.00001 --dataset_name="$1" --budget=0.3 --seed_size=500 --random_seed=1410 --verbose=0
    # python plots.py --dataset_name="$1" --budget=0.3 --seed_size=500 --random_seed=1410

    python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --dataset_name="$1" --budget=0.3 --seed_size=1000 --random_seed=1410 --verbose=0
    python main.py --method="all_labeled" --base_model="mlp" --dataset_name="$1" --budget=0.3 --seed_size=1000 --random_seed=1410 --verbose=0
    python main.py --method="all_labeled_ensemble" --base_model="mlp" --dataset_name="$1" --budget=0.3 --seed_size=1000 --random_seed=1410 --verbose=0
    python main.py --method="random" --base_model="mlp"  --dataset_name="$1" --budget=0.3 --seed_size=1000 --random_seed=1410 --verbose=0
    python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.3 --seed_size=1000 --random_seed=1410 --verbose=0
    python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --dataset_name="$1" --budget=0.3 --seed_size=1000 --random_seed=1410 --verbose=0
    python main.py --method="classification_margin" --base_model="mlp" --prediction_threshold=0.9 --dataset_name="$1" --budget=0.3 --seed_size=1000 --random_seed=1410 --verbose=0
    python main.py --method="vote_entropy" --base_model="mlp" --prediction_threshold=2.1 --dataset_name="$1" --budget=0.3 --seed_size=1000 --random_seed=1410 --verbose=0
    python main.py --method="consensus_entropy" --base_model="mlp" --prediction_threshold=0.3 --dataset_name="$1" --budget=0.3 --seed_size=1000 --random_seed=1410 --verbose=0
    python main.py --method="max_disagreement" --base_model="mlp" --prediction_threshold=0.00001 --dataset_name="$1" --budget=0.3 --seed_size=1000 --random_seed=1410 --verbose=0
    # python plots.py --dataset_name="$1" --budget=0.3 --seed_size=1000 --random_seed=1410
}


do_experiments "accelerometer"
do_experiments "adult"
do_experiments "bank_marketing"
do_experiments "firewall"
do_experiments "chess"
