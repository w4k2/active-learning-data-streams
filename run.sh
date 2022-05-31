#! /bin/bash

python main.py --method="ours" --base_model="mlp"
python main.py --method="all_labeled" --base_model="mlp"
python main.py --method="all_labeled_ensemble" --base_model="mlp"
python main.py --method="confidence" --base_model="mlp"

# python main.py --base_model="ng"
# python main.py --base_model="ng"
# python main.py --ensemble --base_model="ng"
# python main.py --method="confidence" --base_model="ng"

# python plots.py