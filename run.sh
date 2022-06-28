#! /bin/bash

# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=5000 --budget=0.5
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=5000 --budget=0.5
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=5000 --budget=0.5
# python main.py --method="random" --base_model="mlp"  --stream_len=5000 --budget=0.5
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.5
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.5
# python plots.py --stream_len=5000 --budget=0.5

# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=5000 --budget=0.4
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=5000 --budget=0.4
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=5000 --budget=0.4
# python main.py --method="random" --base_model="mlp"  --stream_len=5000 --budget=0.4
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.4
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.4
# python plots.py --stream_len=5000 --budget=0.4

# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=5000 --budget=0.3
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=5000 --budget=0.3
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=5000 --budget=0.3
# python main.py --method="random" --base_model="mlp"  --stream_len=5000 --budget=0.3
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.3
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.3
# python plots.py --stream_len=5000 --budget=0.3

# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=5000 --budget=0.2
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=5000 --budget=0.2
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=5000 --budget=0.2
# python main.py --method="random" --base_model="mlp"  --stream_len=5000 --budget=0.2
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.2
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.2
# python plots.py --stream_len=5000 --budget=0.2

# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=5000 --budget=0.1
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=5000 --budget=0.1
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=5000 --budget=0.1
# python main.py --method="random" --base_model="mlp"  --stream_len=5000 --budget=0.1
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.1
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=5000 --budget=0.1
# python plots.py --stream_len=5000 --budget=0.1

# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --prediction_threshold=0.6 --stream_len=10000 --budget=0.3
# python main.py --method="all_labeled" --base_model="mlp" --stream_len=10000 --budget=0.3
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --stream_len=10000 --budget=0.3
# python main.py --method="random" --base_model="mlp"  --stream_len=10000 --budget=0.3
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=10000 --budget=0.3
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95 --stream_len=10000 --budget=0.3
# python plots.py --stream_len=10000 --budget=0.3

python main.py --method="ours" --base_model="ng" --beta1=0.9 --prediction_threshold=0.6 --stream_len=5000 --budget=0.5
python main.py --method="all_labeled" --base_model="ng" --stream_len=5000 --budget=0.5
python main.py --method="all_labeled_ensemble" --base_model="ng" --stream_len=5000 --budget=0.5
python main.py --method="random" --base_model="ng"  --stream_len=5000 --budget=0.5
python main.py --method="fixed_uncertainty" --base_model="ng" --prediction_threshold=0.95 --stream_len=5000 --budget=0.5
python main.py --method="variable_uncertainty" --base_model="ng" --prediction_threshold=0.95 --stream_len=5000 --budget=0.5
python plots.py --stream_len=5000 --budget=0.5 --base_model="ng"