import argparse
import distutils.util
import math
import os
import random

import mkl
import numpy as np
import sklearn.model_selection
import torch
import tqdm
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import active_learning_strategies
import data.load_data
import self_labeling_strategies
import utils.diversity
import utils.ensemble
import utils.mlp_pytorch
import utils.new_model
import utils.stream
from utils.online_bagging import OnlineBagging


def main():
    args = parse_args()
    acc, budget_end, _ = do_experiment(args)
    save_results(args, acc, budget_end)


def do_experiment(args):
    mkl.set_num_threads(20)
    seed_everything(args.random_seed)

    train_data, train_target, test_data, test_target, num_classes = data.load_data.get_data(args.dataset_name, args.random_seed)

    if args.method == 'online_bagging':
        base_model = get_base_model(args)
        model = OnlineBagging(base_estimator=base_model, n_estimators=args.num_classifiers)
    elif args.method in ('all_labeled_ensemble', 'ours', 'vote_entropy', 'consensus_entropy', 'max_disagreement', 'min_margin'):
        models = [get_base_model(args) for _ in range(args.num_classifiers)]
        diversify = args.method == 'ours'
        model = utils.ensemble.Ensemble(models, diversify=diversify)
    else:
        model = get_base_model(args)

    if args.method in ('all_labeled', 'all_labeled_ensemble'):
        acc = training_full_dataset(model, train_data, train_target, test_data, test_target)
        budget_end = -1
        budget_after = 0
    else:
        X_stream, seed_data, y_stream, seed_target = sklearn.model_selection.train_test_split(train_data, train_target,
                                                                                              test_size=args.seed_size, random_state=args.random_seed, stratify=train_target)
        train_stream = utils.stream.Stream(X_stream, y_stream)
        acc, budget_end, budget_after = training_stream(
            train_stream, seed_data, seed_target, test_data, test_target, model, args, num_classes)
    return acc, budget_end, budget_after


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
                        'vote_entropy', 'consensus_entropy', 'max_disagreement', 'min_margin'),
                        default='ours')
    parser.add_argument('--base_model', choices=('mlp', 'ng', 'online_bagging'), default='mlp')
    parser.add_argument('--prediction_threshold', type=float, default=0.6)
    parser.add_argument('--ensemble_diversify', action='store_true')
    parser.add_argument('--num_classifiers', type=int, default=9)
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for MLP')
    parser.add_argument('--batch_mode', action='store_true')
    parser.add_argument('--batch_size', default=50, type=int)

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
            100, 100), learning_rate_init=args.lr, max_iter=5000, beta_1=args.beta1)
    else:
        raise ValueError("Invalid base classifier")
    return model


def training_full_dataset(model, train_data, train_target, test_data, test_target):
    model.fit(train_data, train_target)
    test_pred = model.predict(test_data)
    acc = balanced_accuracy_score(test_target, test_pred)
    print(f'final acc = {acc}')
    return [acc]


def training_stream(train_stream, seed_data, seed_target, test_data, test_target, model, args, num_classes):
    seed_target = np.squeeze(seed_target, axis=1)
    model.fit(seed_data, seed_target)
    test_pred = model.predict(test_data)
    acc = balanced_accuracy_score(test_target, test_pred)
    print(f'accuracy after training with seed = {acc}')

    acc_list = list()
    budget_end = -1
    current_budget = math.floor(len(train_stream) * args.budget)
    strategy = get_strategy(model, args, num_classes)

    if args.method == 'ours':
        lambdas = np.ones_like(seed_target, dtype=float)

    if args.verbose:
        train_stream = tqdm.tqdm(train_stream, total=len(train_stream))

    for i, (obj, target) in enumerate(train_stream):
        test_pred = model.predict(test_data)
        acc = balanced_accuracy_score(test_target, test_pred)
        acc_list.append(acc)
        obj = np.expand_dims(obj, 0)

        if args.method in ('ours', ):
            if current_budget > 0 and strategy.request_label(obj, current_budget, args.budget):
                seed_data, seed_target = update_training_data(seed_data, seed_target, obj, target)
                lambdas = np.concatenate((lambdas, [1.0]), axis=0)
                seed_data, seed_target, lambdas = partial_fit(seed_data, seed_target, model, args, lambdas)
                current_budget -= 1
                if args.method == 'ours':
                    strategy.last_predictions.append(int(target))
            else:
                train, label, poisson_lambda = strategy.use_self_labeling(obj, current_budget, args.budget)
                if train:
                    seed_data, seed_target = update_training_data(seed_data, seed_target, obj, label)
                    lambdas = np.concatenate((lambdas, [poisson_lambda]), axis=0)
                    seed_data, seed_target, lambdas = partial_fit(seed_data, seed_target, model, args, lambdas)
        else:  # active learning strategy
            if current_budget > 0 and strategy.request_label(obj, current_budget, args.budget):
                seed_data, seed_target = update_training_data(seed_data, seed_target, obj, target)
                seed_data, seed_target, lambdas = partial_fit(seed_data, seed_target, model, args)
                current_budget -= 1

        if current_budget == 0:
            current_budget = -1
            budget_end = i
            print(f'budget ended at {i}')

    print(f'budget after training = {current_budget}')
    print(f'final acc = {acc_list[-1]}')
    return acc_list, budget_end, current_budget


def partial_fit(seed_data, seed_target, model, args, lambdas=None):
    if args.batch_mode and len(seed_data) % args.batch_size == 0:
        if lambdas is not None:
            model.partial_fit(seed_data, seed_target, lambdas)
        else:
            model.partial_fit(seed_data, seed_target)
    elif not args.batch_mode:
        if lambdas is not None:
            model.partial_fit(seed_data, seed_target, lambdas)
        else:
            model.partial_fit(seed_data, seed_target)
    return seed_data, seed_target, lambdas


def update_training_data(seed_data, seed_target, obj, target):
    seed_data = np.concatenate((seed_data, obj), axis=0)
    seed_target = np.concatenate((seed_target, target), axis=0)
    return seed_data, seed_target


def get_strategy(model, args, num_classes):
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
        strategy = active_learning_strategies.ConsensusEntropy(model, args.prediction_threshold, num_classes)
    elif args.method == 'max_disagreement':
        strategy = active_learning_strategies.MaxDisagreement(model, args.prediction_threshold)
    return strategy


def save_results(args, acc, budget_end):
    os.makedirs(f'results/{args.method}', exist_ok=True)
    experiment_parameters = f'{args.base_model}_{args.dataset_name}_seed_{args.seed_size}_budget_{args.budget}_random_seed_{args.random_seed}'
    np.save(f'results/{args.method}/acc_{experiment_parameters}.npy', acc)
    np.save(
        f'results/{args.method}/budget_end_{experiment_parameters}.npy', budget_end)


if __name__ == '__main__':
    main()
