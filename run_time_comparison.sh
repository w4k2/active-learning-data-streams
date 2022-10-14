#! /bin/bash


# DATASET="wine"
for DATASET in "abalone" "nursery" "firewall";
do

for METHOD in "random" "fixed_uncertainty" "variable_uncertainty" "classification_margin" "vote_entropy" "ours";
do
    echo -e "\n\ndataset $DATASET method $METHOD\n\n"
    for BATCH_SIZE in 100 200 500;
    do
        echo -e "\nbatch size $BATCH_SIZE\n"
        time python main.py --method=$METHOD --base_model="mlp"  --dataset_name=$DATASET --budget=0.3 --random_seed=42 --seed_size=1000 --verbose=0 --batch_mode --batch_size=$BATCH_SIZE
    done
    echo -e "\nfull training\n"
    time python main.py --method=$METHOD --base_model="mlp"  --dataset_name=$DATASET --budget=0.3 --random_seed=42 --seed_size=1000 --verbose=0
done

done