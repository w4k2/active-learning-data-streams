#! /bin/bash

python main.py --base_model="mlp"
python baselines.py --base_model="mlp"
python baselines.py --ensemble --base_model="mlp"
python baselines.py --method="confidence" --base_model="mlp"

# python main.py --base_model="ng"
# python baselines.py --base_model="ng"
# python baselines.py --ensemble --base_model="ng"
# python baselines.py --method="confidence" --base_model="ng"

# python plots.py