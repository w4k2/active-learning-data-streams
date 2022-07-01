#! /bin/bash

# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=5000 --budget=0.5 --random_seed=1410
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=5000 --budget=0.5 --random_seed=1410
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=5000 --budget=0.5 --random_seed=1410
# python main.py --method="random" --base_model="mlp"  --stream_len=5000 --budget=0.5 --random_seed=1410
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.5 --random_seed=1410
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.5 --random_seed=1410
# python plots.py --stream_len=5000 --budget=0.5 --random_seed=1410

# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=5000 --budget=0.4 --random_seed=1410
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=5000 --budget=0.4 --random_seed=1410
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=5000 --budget=0.4 --random_seed=1410
# python main.py --method="random" --base_model="mlp"  --stream_len=5000 --budget=0.4 --random_seed=1410
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.4 --random_seed=1410
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.4 --random_seed=1410
# python plots.py --stream_len=5000 --budget=0.4 --random_seed=1410

python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=5000 --budget=0.3 --random_seed=1410
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=5000 --budget=0.3 --random_seed=1410
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=5000 --budget=0.3 --random_seed=1410
# python main.py --method="random" --base_model="mlp"  --stream_len=5000 --budget=0.3 --random_seed=1410
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.3 --random_seed=1410
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.3 --random_seed=1410
python plots.py --stream_len=5000 --budget=0.3 --random_seed=1410

# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=5000 --budget=0.2 --random_seed=1410
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=5000 --budget=0.2 --random_seed=1410
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=5000 --budget=0.2 --random_seed=1410
# python main.py --method="random" --base_model="mlp"  --stream_len=5000 --budget=0.2 --random_seed=1410
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.2 --random_seed=1410
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.2 --random_seed=1410
# python plots.py --stream_len=5000 --budget=0.2 --random_seed=1410

# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=5000 --budget=0.1 --random_seed=1410
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=5000 --budget=0.1 --random_seed=1410
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=5000 --budget=0.1 --random_seed=1410
# python main.py --method="random" --base_model="mlp"  --stream_len=5000 --budget=0.1 --random_seed=1410
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.1 --random_seed=1410
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.1 --random_seed=1410
# python plots.py --stream_len=5000 --budget=0.1 --random_seed=1410

# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=10000 --budget=0.3 --random_seed=1410
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=10000 --budget=0.3 --random_seed=1410
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=10000 --budget=0.3 --random_seed=1410
# python main.py --method="random" --base_model="mlp"  --stream_len=10000 --budget=0.3 --random_seed=1410
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=10000 --budget=0.3 --random_seed=1410
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=10000 --budget=0.3 --random_seed=1410
# python plots.py --stream_len=10000 --budget=0.3 --random_seed=1410

# python main.py --method="ours" --base_model="ng" --beta1=0.9 --prediction_threshold=0.8 --stream_len=5000 --budget=0.5 --random_seed=45
# python main.py --method="all_labeled" --base_model="ng" --stream_len=5000 --budget=0.5 --random_seed=45
# python main.py --method="all_labeled_ensemble" --base_model="ng" --stream_len=5000 --budget=0.5 --random_seed=45
# python main.py --method="random" --base_model="ng"  --stream_len=5000 --budget=0.5 --random_seed=45
# python main.py --method="fixed_uncertainty" --base_model="ng" --prediction_threshold=0.95 --stream_len=5000 --budget=0.5 --random_seed=45
# python main.py --method="variable_uncertainty" --base_model="ng" --prediction_threshold=0.95 --stream_len=5000 --budget=0.5 --random_seed=45
# python plots.py --stream_len=5000 --budget=0.5 --base_model="ng" --random_seed=45

# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=5000 --budget=0.3 --seed_size=100 --random_seed=1410
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=5000 --budget=0.3 --seed_size=100 --random_seed=1410
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=5000 --budget=0.3 --seed_size=100 --random_seed=1410
# python main.py --method="random" --base_model="mlp"  --stream_len=5000 --budget=0.3 --seed_size=100 --random_seed=1410
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.3 --seed_size=100 --random_seed=1410
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.3 --seed_size=100 --random_seed=1410
# python plots.py --stream_len=5000 --budget=0.3 --seed_size=100 --random_seed=1410

# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=5000 --budget=0.3 --seed_size=500 --random_seed=1410
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=5000 --budget=0.3 --seed_size=500 --random_seed=1410
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=5000 --budget=0.3 --seed_size=500 --random_seed=1410
# python main.py --method="random" --base_model="mlp"  --stream_len=5000 --budget=0.3 --seed_size=500 --random_seed=1410
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.3 --seed_size=500 --random_seed=1410
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.3 --seed_size=500 --random_seed=1410
# python plots.py --stream_len=5000 --budget=0.3 --seed_size=500 --random_seed=1410

# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=5000 --budget=0.3 --seed_size=1000 --random_seed=1410
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=5000 --budget=0.3 --seed_size=1000 --random_seed=1410
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=5000 --budget=0.3 --seed_size=1000 --random_seed=1410
# python main.py --method="random" --base_model="mlp"  --stream_len=5000 --budget=0.3 --seed_size=1000 --random_seed=1410
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.3 --seed_size=1000 --random_seed=1410
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.3 --seed_size=1000 --random_seed=1410
# python plots.py --stream_len=5000 --budget=0.3 --seed_size=1000 --random_seed=1410
