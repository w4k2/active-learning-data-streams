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
import utils.utils
import utils.diversity
import utils.mlp_pytorch
from utils.utils import OnlineBagging


def main():
    np.random.seed(42)
    args = parse_args()

    seed_data, seed_target, train_stream, test_X, test_y = utils.utils.get_data(args.stream_len, args.seed_size)
    if args.method == 'all_labeled_ensemble':
        models = [get_base_model(args.base_model) for _ in range(args.num_classifiers)]
        model = utils.ensemble.Ensemble(models, diversify=args.ensemble_diversify)
    elif args.method == 'ours':
        models = [get_base_model(args.base_model) for _ in range(args.num_classifiers)]
        model = utils.ensemble.Ensemble(models, diversify=True)
    else:
        model = get_base_model(args.base_model)
    model.fit(seed_data, seed_target)

    acc, budget_end, all_ensemble_pred = stream_learning(train_stream, test_X, test_y, seed_data, seed_target, model, args,
                                                         budget=args.budget, prediction_threshold=args.prediction_threshold)

    os.makedirs(f'results/{args.method}', exist_ok=True)
    experiment_parameters = f'{args.base_model}_{args.stream_len}_seed_{args.seed_size}_budget_{args.budget}'
    np.save(f'results/{args.method}/acc_{experiment_parameters}.npy', acc)
    np.save(f'results/{args.method}/budget_end_{experiment_parameters}.npy', budget_end)

    if args.method == 'ours':
        np.save(f'results/ours/targets_{experiment_parameters}.npy', test_y)
        np.save(f'results/ours/all_ensemble_pred_{experiment_parameters}.npy', all_ensemble_pred)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--stream_len', type=int, default=5000)
    parser.add_argument('--seed_size', type=int, default=200)
    parser.add_argument('--base_model', choices=('mlp', 'ng', 'online_bagging'), default='mlp')
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--method', choices=('ours', 'all_labeled', 'all_labeled_ensemble', 'confidence'), default='ours')
    parser.add_argument('--prediction_threshold', type=float, default=0.6)
    parser.add_argument('--ensemble_diversify', action='store_true')
    parser.add_argument('--num_classifiers', type=int, default=9)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    return args


def get_base_model(model_name):
    if model_name == 'ng':
        model = GaussianNB()
    elif model_name == 'mlp':
        # model = MLPClassifier(learning_rate_init=0.008, max_iter=1000)
        model = utils.mlp_pytorch.MLPClassifierPytorch(learning_rate_init=0.001, max_iter=500, beta_1=0.99999999)
    elif model_name == 'online_bagging':
        model = OnlineBagging(base_estimator=MLPClassifier(learning_rate_init=0.01, max_iter=500), n_estimators=5)
    else:
        raise ValueError("Invalid base classifier")
    return model


def stream_learning(train_stream, test_X, test_y, seed_data, seed_target, model, args, budget=100, prediction_threshold=0.5):
    all_ensemble_pred = []
    acc = []
    budget_end = -1

    print('test y counts = ', np.unique(test_y, return_counts=True))

    if not args.debug:
        train_stream = tqdm.tqdm(train_stream, total=len(train_stream))

    num_classes = np.unique(test_y).size
    last_predictions = collections.deque([], maxlen=200)
    train_with_selflabeling = True

    for i, (obj, target) in enumerate(train_stream):
        if args.method == 'ours':
            if i % 100 == 0:
                preds = model.predict(test_X)
                print('preds count: ', np.unique(preds, return_counts=True))

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
                    if budget > 0:
                        poisson_lambda = max_supp / prediction_threshold
                    else:
                        poisson_lambda = abs(prediction_threshold - max_supp) / prediction_threshold
                    label = confident_preds[0]
                    label = np.expand_dims(label, 0)

                    if len(last_predictions) == last_predictions.maxlen:
                        _, current_dist = np.unique(list(last_predictions), return_counts=True)
                        current_dist = current_dist / len(last_predictions)
                        delta_p = current_dist[label] - (1.0 / num_classes)
                        if delta_p <= 0:
                            train_with_selflabeling = True
                        else:
                            train_with_selflabeling = False
                            if args.debug:
                                print(f'label = {label}, current_dist = {current_dist} delta_p = {delta_p}')

                    if train_with_selflabeling:
                        model.partial_fit(obj, label, poisson_lambda)
                        last_predictions.append(label)

                    if args.debug and label != target:
                        print(f'sample {i} training with wrong target - consistent confident supports')
                        print('max_support = ', supports)
                        print('predictions = ', predictions)
                        print('target = ', target)
                        print('poisson_lambda = ', poisson_lambda)
                        print('\n\n')
                else:
                    if budget > 0:
                        most_confident_idx = np.argmax([np.max(supp) for supp in supports])
                        most_confident_pred = predictions[most_confident_idx]
                        min_supp = np.min(supports[:, :, most_confident_pred])

                        # poisson_lambda = max_supp / min_supp
                        poisson_lambda = max_supp / prediction_threshold
                        model.partial_fit(obj, target, poisson_lambda=poisson_lambda)
                        last_predictions.append(label)
                        budget -= 1
                    else:
                        # train_unconfident(model, prediction_threshold, obj, target, supports, predictions, last_predictions, debug=args.debug)
                        pass
            else:
                # traning when there are no confident supports seem to be worse
                if budget > 0:
                    # poisson_lambda = prediction_threshold / max_supp
                    poisson_lambda = max_supp / prediction_threshold
                    model.partial_fit(obj, target, poisson_lambda=poisson_lambda)
                    last_predictions.append(label)
                    budget -= 1
                else:
                    # train_unconfident(model, prediction_threshold, obj, target, supports, predictions, last_predictions, debug=args.debug)
                    pass
        elif args.method == 'all_labeled' or args.method == 'all_labeled_ensemble':
            model.partial_fit(obj, target)
        elif args.method == 'confidence' and budget > 0:
            pred_prob = model.predict_proba(obj)
            max_prob = np.max(pred_prob, axis=1)[0]
            if max_prob < prediction_threshold:
                model.partial_fit(obj, target)
                budget -= 1

        if budget == 0:
            budget = -1
            budget_end = i
            print(f'budget ended at {i}')

        test_pred = model.predict(test_X)
        acc.append(accuracy_score(test_y, test_pred))
        if args.method == 'ours':
            test_supports = model.predict_proba_separate(test_X)
            test_predictions = np.argmax(test_supports, axis=2)
            all_ensemble_pred.append(test_predictions)

    preds = model.predict(test_X)
    print('preds count: ', np.unique(preds, return_counts=True))

    print(f'budget after training = {budget}')
    return acc, budget_end, all_ensemble_pred


def train_unconfident(model, prediction_threshold, obj, target, supports, predictions, last_predictions, debug=False):
    idx = np.argmax([np.max(s) for s in supports])
    label = predictions[idx]
    label = np.expand_dims(label, 0)
    last_predictions.append(label)
    max_supp = np.max(supports[idx])
    poisson_lambda = abs(prediction_threshold - max_supp) / prediction_threshold
    if debug and label != target:
        print('\ntraining with wrong target - incosistant or unconfident labels')
        print('max_support = ', supports)
        print('predictions = ', predictions)
        print('target = ', target)
        print('poisson_lambda = ', poisson_lambda)
    model.partial_fit(obj, label, poisson_lambda)


def plot_confidence(pool, data, target):
    for i, model in enumerate(pool):
        correct_predictions = [0 for _ in range(10)]
        all_predictions = [0 for _ in range(10)]
        for x, y in zip(data, target):
            y_pred = model.predict_proba(np.expand_dims(x, axis=0))

            interval_idx = math.floor(np.max(y_pred)*10)
            if interval_idx == 10:
                interval_idx = 9
            all_predictions[interval_idx] += 1
            if np.argmax(y_pred, axis=1) == y:
                correct_predictions[interval_idx] += 1

        confidence_plot = [correct/num_all if num_all != 0 else 0 for correct, num_all in zip(correct_predictions, all_predictions)]

        plt.subplot(len(pool), 2, i*2 + 1)
        plt.bar(list(range(10)), confidence_plot)
        plt.xlabel('confidence intervals')
        plt.ylabel('accuracy')

        plt.subplot(len(pool), 2, i*2 + 2)
        plt.bar(list(range(10)), all_predictions)
        plt.xlabel('confidence intervals')
        plt.ylabel('number of predictions')
    plt.show()


if __name__ == '__main__':
    main()
