import argparse
import math
import os
import tqdm
import torch
import random
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import active_learning_strategies
import self_labeling_strategies
import utils.ensemble
import utils.data
import utils.diversity
import utils.mlp_pytorch
import utils.new_model
from utils.online_bagging import OnlineBagging


def main():
    args = parse_args()
    seed_everything(args.random_seed)

    seed_data, seed_target, train_stream = utils.data.get_data(
        args.stream_len, args.seed_size, args.random_seed, args.num_classes)
    if args.method == 'all_labeled_ensemble':
        models = [get_base_model(args) for _ in range(args.num_classifiers)]
        model = utils.ensemble.Ensemble(
            models, diversify=args.ensemble_diversify)
    elif args.method == 'ours':
        models = [get_base_model(args) for _ in range(args.num_classifiers)]
        model = utils.ensemble.Ensemble(models, diversify=True)
    elif args.method == 'ours_new':
        base_model = get_base_model(args)
        model = utils.new_model.Model(base_model)
    else:
        model = get_base_model(args)
    model.fit(seed_data, seed_target)

    acc, budget_end = stream_learning(
        train_stream, seed_data, seed_target, model, args)

    os.makedirs(f'results/{args.method}', exist_ok=True)
    experiment_parameters = f'{args.base_model}_{args.stream_len}_seed_{args.seed_size}_budget_{args.budget}'
    np.save(f'results/{args.method}/acc_{experiment_parameters}.npy', acc)
    np.save(
        f'results/{args.method}/budget_end_{experiment_parameters}.npy', budget_end)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--stream_len', type=int, default=10000)
    parser.add_argument('--seed_size', type=int, default=200,
                        help='seed size for model training')
    parser.add_argument('--random_seed', type=int, default=2042)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--budget', type=int, default=0.5)

    parser.add_argument('--method', choices=('ours', 'all_labeled',
                        'all_labeled_ensemble', 'random', 'fixed_uncertainty',
                                             'variable_uncertainty', 'variable_randomized_uncertainty'), default='ours')
    parser.add_argument('--base_model', choices=('mlp',
                        'ng', 'online_bagging'), default='mlp')
    parser.add_argument('--prediction_threshold', type=float, default=0.6)
    parser.add_argument('--ensemble_diversify', action='store_true')
    parser.add_argument('--num_classifiers', type=int, default=9)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 for Adam optimizer')

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
        model = utils.mlp_pytorch.MLPClassifierPytorch(hidden_layer_sizes=(
            100, 100), learning_rate_init=0.001, max_iter=500, beta_1=args.beta1)
    elif args.base_model == 'online_bagging':
        model = OnlineBagging(base_estimator=MLPClassifier(
            learning_rate_init=0.01, max_iter=500), n_estimators=5)
    else:
        raise ValueError("Invalid base classifier")
    return model


def stream_learning(train_stream, seed_data, seed_target, model, args):
    predictions_list = list()
    targets_list = list()
    budget_end = -1
    current_budget = math.floor(len(train_stream) * args.budget)

    if args.method == 'ours':
        strategy = self_labeling_strategies.Ours(model, args.num_classes, args.prediction_threshold)
    elif args.method == 'random':
        strategy = active_learning_strategies.RandomSampling(model)
    elif args.method == 'fixed_uncertainty':
        strategy = active_learning_strategies.FixedUncertainty(
            model, args.prediction_threshold)
    elif args.method == 'variable_uncertainty':
        strategy = active_learning_strategies.VariableUncertainty(
            model, args.prediction_threshold)
    elif args.method == 'variable_randomized_uncertainty':
        strategy = active_learning_strategies.VariableRandomizedUncertainty(
            model, args.prediction_threshold)

    train_stream = tqdm.tqdm(train_stream, total=len(train_stream))

    for i, (obj, target) in enumerate(train_stream):
        predictions_list.append(model.predict(obj))
        targets_list.append(target)

        if args.method in ('all_labeled', 'all_labeled_ensemble'):
            model.partial_fit(obj, target)
        elif args.method in ('ours', ):
            if current_budget > 0 and strategy.request_label(obj, current_budget, args.budget):
                model.partial_fit(obj, target)
                current_budget -= 1
                if args.method == 'ours':
                    strategy.last_predictions.append(int(target))
            else:
                train, label = strategy.use_self_labeling(obj, current_budget, args.budget)
                if train:
                    model.partial_fit(obj, label)
        elif args.method in ('random', 'fixed_uncertainty', 'variable_uncertainty', 'variable_randomized_uncertainty') and current_budget > 0:
            if strategy.request_label(obj, current_budget, args.budget):
                model.partial_fit(obj, target)
                current_budget -= 1

        if current_budget == 0:
            current_budget = -1
            budget_end = i
            print(f'budget ended at {i}')

    print(f'budget after training = {current_budget}')
    acc = compute_acc(predictions_list, targets_list)
    return acc, budget_end


def compute_acc(predictions_list, targets_list, batch_size=100):
    acc_stream = list()
    for i in range(30, len(predictions_list)):
        preds = predictions_list[:i]
        preds = np.stack(preds)
        targets = targets_list[:i]
        targets = np.stack(targets)
        acc = accuracy_score(targets, preds)
        acc_stream.append(acc)
    return acc_stream


if __name__ == '__main__':
    main()
