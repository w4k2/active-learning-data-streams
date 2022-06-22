#! /bin/bash

# python main.py --method="ours" --base_model="mlp" --beta1=0.99999 --prediction_threshold=0.3

# python main.py --method="all_labeled" --base_model="mlp"
# python main.py --method="all_labeled_ensemble" --base_model="mlp"

# python main.py --method="random" --base_model="mlp" 
# python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.95
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.95
# python main.py --method="variable_randomized_uncertainty" --base_model="mlp" --prediction_threshold=0.95


# python main.py --method="ours" --base_model="mlp" --beta1=0.9 --num_drifts=2 --prediction_threshold=0.6

# python main.py --method="all_labeled" --base_model="mlp" --num_drifts=2
# python main.py --method="all_labeled_ensemble" --base_model="mlp" --num_drifts=2

# python main.py --method="random" --base_model="mlp" --num_drifts=2
python main.py --method="fixed_uncertainty" --base_model="mlp" --prediction_threshold=0.9 --num_drifts=2
# python main.py --method="variable_uncertainty" --base_model="mlp" --prediction_threshold=0.8 --num_drifts=2
# python main.py --method="variable_randomized_uncertainty" --base_model="mlp" --prediction_threshold=0.8 --num_drifts=2

# python main.py --base_model="ng"
# python main.py --base_model="ng"
# python main.py --ensemble --base_model="ng"
# python main.py --method="confidence" --base_model="ng"

# python plots.py