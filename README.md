# Self-labeling Selective Sampling

This repo contains code for paper Self-labeling Selective Sampling.
Repository is organised as follow:

* folder data contains script for loading data and stores datasets.
* folder plots contains images generated for paper.
* folder results contains .npy files with results of experiments.
* utils contains various scripts with some code used in experiments.
* main directory contains code with experimenets or some helping scripts for generating images, generating tables, runing preliminary studies etc.

All of experiments are implemented in Python. 

## Preliminaries

To run experiments first download datasets by running script:

```
bash download_data.sh
```

Next create and activate conda environment:

```
conda env create -f environment.yml
conda activate active-learning
```

## Runing experiments

To perform experiments with smaller datasets with various seed size run script:

```
bash run.sh
```

To perform experimetns with bigger datasets run:

```
bash run_big_datasets.sh
```

Both of these scripts use python script `main.py`. It is a script with basic code for our experiments: namely it reads proper dataset, split the data, perform initial training with seed datasets and the runs selective sampling on stream. Code from `main.py` is used in `tune_hyperpams.py` - the script for hyperparameter tuning. We run experiments 3 times for each random seed and we evaluate multiple hyperparameter values. After that best value is determined automatically and we run experiments 10 times with selected hyperaparamters and for different random seeds (not the same as the ones used for tuning). 

Results are stored in the form of the numpy files in results folder. Due to high number of results from our experiments, we split results for different active learning algorithms in different folders. We store in two separate files for each experiment the balanced accuracy obtained in experiments and the iteration number where budget have ended. The name of each file contains information about experiment it was obtained from, namely we use following naming convention: `{acc/budget_end}_{base mode name}_{dataset}_seed_{seed size}_budget_{size of budget in experiment}_random_seed_{random seed used in experiment}.npy`. So for example file with accuracy obtained with MLP model for wine dataset, seed size 1000, budget 0.3 and random seed 4 will be named: `acc_mlp_wine_seed_1000_budget_0.3_random_seed_4.npy`.

The latex tables with results for our experiments can be generated with `tables.py` script. 