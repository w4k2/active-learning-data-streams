import collections
import argparse
import math
import os
import tqdm

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import utils.ensemble
import utils.data
import utils.diversity
import utils.mlp_pytorch
import utils.new_model
from utils.online_bagging import OnlineBagging


def main():
    np.random.seed(42)
    args = parse_args()

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

    parser.add_argument('--method', choices=('ours', 'ours_new', 'all_labeled',
                        'all_labeled_ensemble', 'confidence'), default='ours')
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
    prediction_threshold = args.prediction_threshold
    labeling_budget = math.floor(len(train_stream) * args.budget)

    if not args.debug:
        train_stream = tqdm.tqdm(train_stream, total=len(train_stream))

    num_classes = args.num_classes
    last_predictions = collections.deque([], maxlen=500)
    train_with_selflabeling = True

    for i, (obj, target) in enumerate(train_stream):
        predictions_list.append(model.predict(obj))
        targets_list.append(target)

        if args.method == 'ours':
            supports = model.predict_proba_separate(obj)
            predictions = np.argmax(supports, axis=2)

            confident_supports = []
            confident_preds = []
            for supp, pred in zip(supports, predictions):
                if np.max(supp) > prediction_threshold:
                    max_supp = np.max(supp)
                    confident_supports.append(max_supp)
                    confident_preds.append(pred)

            if len(confident_supports) > 0:
                if all(pred == confident_preds[0] for pred in confident_preds):
                    max_supp = max(confident_supports)
                    if labeling_budget > 0:
                        poisson_lambda = max_supp / prediction_threshold
                    else:
                        poisson_lambda = abs(
                            prediction_threshold - max_supp) / prediction_threshold
                    label = confident_preds[0]
                    label = np.expand_dims(label, 0)

                    if len(last_predictions) >= min(last_predictions.maxlen, 30):
                        _, current_dist = np.unique(
                            list(last_predictions), return_counts=True)
                        current_dist = current_dist / len(last_predictions)
                        delta_p = current_dist[label] - (1.0 / num_classes)
                        if delta_p <= 0:
                            train_with_selflabeling = True
                        else:
                            train_with_selflabeling = False
                            if args.debug:
                                print(
                                    f'{i} label = {label}, current_dist = {current_dist} delta_p = {delta_p}')

                    if train_with_selflabeling:  # label == target and
                        model.partial_fit(
                            obj, label, poisson_lambda=poisson_lambda)
                        last_predictions.append(int(label))

                    if args.debug and label != target:
                        print(
                            f'sample {i} training with wrong target - consistent confident supports')
                        print('max_support = ', supports)
                        print('predictions = ', predictions)
                        print('target = ', target)
                        print('poisson_lambda = ', poisson_lambda)
                        print('\n\n')
                else:
                    if labeling_budget > 0:
                        poisson_lambda = max_supp / prediction_threshold
                        model.partial_fit(obj, target, poisson_lambda=1.0)
                        last_predictions.append(int(target))
                        labeling_budget -= 1
                    else:
                        # train_unconfident(model, prediction_threshold, obj, target, supports, predictions, last_predictions, debug=args.debug)
                        pass
            else:
                # traning when there are no confident supports seem to be worse
                if labeling_budget > 0:
                    poisson_lambda = max_supp / prediction_threshold
                    model.partial_fit(obj, target, poisson_lambda=1.0)
                    last_predictions.append(int(target))
                    labeling_budget -= 1
                else:
                    # train_unconfident(model, prediction_threshold, obj, target, supports, predictions, last_predictions, debug=args.debug)
                    pass
        elif args.method in ('all_labeled', 'all_labeled_ensemble'):
            model.partial_fit(obj, target)
        elif args.method == 'confidence' and labeling_budget > 0:
            pred_prob = model.predict_proba(obj)
            max_prob = np.max(pred_prob, axis=1)[0]
            if max_prob < prediction_threshold:
                model.partial_fit(obj, target)
                labeling_budget -= 1

        if labeling_budget == 0:
            labeling_budget = -1
            budget_end = i
            print(f'budget ended at {i}')

    acc = compute_acc(predictions_list, targets_list)
    print(f'budget after training = {labeling_budget}')
    return acc, budget_end


def compute_acc(predictions_list, targets_list, batch_size=1000):
    acc_stream = list()
    for i in range(0, len(predictions_list) - batch_size + 1):
        preds = predictions_list[i:i+batch_size]
        preds = np.stack(preds)
        targets = targets_list[i:i+batch_size]
        targets = np.stack(targets)
        acc = accuracy_score(targets, preds)
        acc_stream.append(acc)
    return acc_stream


def train_unconfident(model, prediction_threshold, obj, target, supports, predictions, last_predictions, debug=False):
    idx = np.argmax([np.max(s) for s in supports])
    label = predictions[idx]
    label = np.expand_dims(label, 0)
    last_predictions.append(int(label))
    max_supp = np.max(supports[idx])
    poisson_lambda = abs(prediction_threshold -
                         max_supp) / prediction_threshold
    if debug and label != target:
        print('\ntraining with wrong target - incosistant or unconfident labels')
        print('max_support = ', supports)
        print('predictions = ', predictions)
        print('target = ', target)
        print('poisson_lambda = ', poisson_lambda)
    model.partial_fit(obj, label, poisson_lambda)


if __name__ == '__main__':
    main()
