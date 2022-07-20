import argparse
import math
import os
import tqdm
import torch
import random
import numpy as np
import distutils.util

from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import active_learning_strategies
import self_labeling_strategies
import utils.ensemble
import data.load_data
import utils.diversity
import utils.mlp_pytorch
import utils.new_model
from utils.online_bagging import OnlineBagging


def main():
    args = parse_args()
    seed_everything(args.random_seed)

    seed_data, seed_target, test_data, test_target, train_stream, num_classes = data.load_data.get_data(
        args.dataset_name, args.seed_size, args.random_seed)
    if args.method == 'online_bagging':
        base_model = get_base_model(args)
        model = OnlineBagging(base_estimator=base_model, n_estimators=args.num_classifiers)
    elif args.method in ('all_labeled_ensemble', 'ours', 'vote_entropy', 'consensus_entropy', 'max_disagreement'):
        models = [get_base_model(args) for _ in range(args.num_classifiers)]
        model = utils.ensemble.Ensemble(models, diversify=True)
    else:
        model = get_base_model(args)

    acc, budget_end = training(
        train_stream, seed_data, seed_target, test_data, test_target, model, args, num_classes)

    os.makedirs(f'results/{args.method}', exist_ok=True)
    experiment_parameters = f'{args.base_model}_{args.dataset_name}_seed_{args.seed_size}_budget_{args.budget}'
    np.save(f'results/{args.method}/acc_{experiment_parameters}.npy', acc)
    np.save(
        f'results/{args.method}/budget_end_{experiment_parameters}.npy', budget_end)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', choices=[
        'accelerometer', 'adult', 'bank_marketing',
        'firewall', 'chess', 'nursery',
        'poker', 'mushroom', 'wine', 'abalone'
    ], required=True)
    parser.add_argument('--seed_size', type=int, default=200, help='seed size for model training')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--budget', type=float, default=0.3)

    parser.add_argument('--method', choices=(
                        'ours', 'all_labeled', 'all_labeled_ensemble', 'online_bagging',
                        'random', 'fixed_uncertainty', 'variable_uncertainty', 'classification_margin',
                        'vote_entropy', 'consensus_entropy', 'max_disagreement'),
                        default='ours')
    parser.add_argument('--base_model', choices=('mlp', 'ng', 'online_bagging'), default='mlp')
    parser.add_argument('--prediction_threshold', type=float, default=0.6)
    parser.add_argument('--ensemble_diversify', action='store_true')
    parser.add_argument('--num_classifiers', type=int, default=9)
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', type=distutils.util.strtobool, default=True)

    args = parser.parse_args()
    return args


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_base_model(args):
    if args.base_model == 'ng':
        model = GaussianNB()
    elif args.base_model == 'mlp':
        # model = utils.mlp_pytorch.MLPClassifierPytorch(hidden_layer_sizes=(  # TODO why it is working better for pytorch implementation of MLP? it is not the case of changing optimizer between fit and partial fit
        #     100, 100), learning_rate_init=0.001, max_iter=500, beta_1=args.beta1)
        model = MLPClassifier(hidden_layer_sizes=(
            100, 100), learning_rate_init=0.001, max_iter=1000, beta_1=args.beta1)
    else:
        raise ValueError("Invalid base classifier")
    return model


def training(train_stream, seed_data, seed_target, test_data, test_target, model, args, num_classes):
    model.fit(seed_data, seed_target)
    test_pred = model.predict(test_data)
    acc = balanced_accuracy_score(test_target, test_pred)
    print(f'accuracy after training with seed = {acc}')

    acc_list = list()
    budget_end = -1
    current_budget = math.floor(len(train_stream) * args.budget)

    if args.method == 'ours':
        strategy = self_labeling_strategies.Ours(model, num_classes, args.prediction_threshold)
    elif args.method == 'random':
        strategy = active_learning_strategies.RandomSampling(model)
    elif args.method == 'fixed_uncertainty':
        strategy = active_learning_strategies.FixedUncertainty(
            model, args.prediction_threshold)
    elif args.method == 'variable_uncertainty':
        strategy = active_learning_strategies.VariableUncertainty(
            model, args.prediction_threshold)
    elif args.method == 'classification_margin':
        strategy = active_learning_strategies.ClassificationMargin(model, args.prediction_threshold)
    elif args.method == 'vote_entropy':
        strategy = active_learning_strategies.VoteEntropy(model, args.prediction_threshold)
    elif args.method == 'consensus_entropy':
        strategy = active_learning_strategies.ConsensusEntropy(model, args.prediction_threshold)
    elif args.method == 'max_disagreement':
        strategy = active_learning_strategies.MaxDisagreement(model, args.prediction_threshold)

    if args.verbose:
        train_stream = tqdm.tqdm(train_stream, total=len(train_stream))

    for i, (obj, target) in enumerate(train_stream):
        test_pred = model.predict(test_data)
        acc = balanced_accuracy_score(test_target, test_pred)
        # print(acc)
        acc_list.append(acc)
        obj = np.expand_dims(obj, 0)

        if args.method in ('all_labeled', 'all_labeled_ensemble', 'online_bagging'):
            model.partial_fit(obj, target)
        elif args.method in ('ours', ):
            if current_budget > 0 and strategy.request_label(obj, current_budget, args.budget):
                model.partial_fit(obj, target)
                current_budget -= 1
                if args.method == 'ours':
                    strategy.last_predictions.append(int(target))
            else:
                train, label, train_kwargs = strategy.use_self_labeling(obj, current_budget, args.budget)
                if train:
                    model.partial_fit(obj, label, **train_kwargs)
        else:  # active learning strategy
            if current_budget > 0 and strategy.request_label(obj, current_budget, args.budget):
                model.partial_fit(obj, target)
                current_budget -= 1

        if current_budget == 0:
            current_budget = -1
            budget_end = i
            print(f'budget ended at {i}')

    print(f'budget after training = {current_budget}')
    print(f'final acc = {acc_list[-1]}')
    return acc_list, budget_end


if __name__ == '__main__':
    main()
