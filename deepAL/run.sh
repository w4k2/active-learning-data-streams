#! /bin/bash

method=$1
device=$2
if [ -z "$method" ];
then
    echo "wrong method"
    exit
fi
if [ -z "$device" ];
then
    echo "wrong device"
    exit
fi


for budget in 0.1 0.2 0.3;
do
    if [ "$method" == "RandomSampling" ];
    then
        for seed in 0 1 2 3 4 5 6 7 8 9;
        do
            python main.py --strategy_name="$method" --n_init_labeled=1000 --dataset_name="MNIST" --budget="$budget" --threshold=0.1 --device="$device" --seed="$seed"
            python main.py --strategy_name="$method" --n_init_labeled=2000 --dataset_name="CIFAR10" --budget="$budget" --threshold=0.1 --device="$device" --seed="$seed"
            python main.py --strategy_name="$method" --n_init_labeled=2000 --dataset_name="SVHN" --budget="$budget" --threshold=0.1 --device="$device" --seed="$seed"
        done
    else
        python tune_hyperparameters.py --strategy_name="$method" --n_init_labeled=1000 --dataset_name="MNIST" --budget="$budget" --threshold=0.1 --device="$device"
        python tune_hyperparameters.py --strategy_name="$method" --n_init_labeled=2000 --dataset_name="CIFAR10" --budget="$budget" --threshold=0.1 --device="$device"
        python tune_hyperparameters.py --strategy_name="$method" --n_init_labeled=2000 --dataset_name="SVHN" --budget="$budget" --threshold=0.1 --device="$device"
    fi
done
