import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import utils.ensemble
import utils.utils
from utils.utils import OnlineBagging


def main():
    np.random.seed(42)
    args = parse_args()

    seed_data, seed_target, train_stream, test_X, test_y = utils.utils.get_data(args.stream_len, args.seed_percentage)
    if args.method == 'all_labeled_ensemble':
        models = [get_base_model(args.base_model) for _ in range(args.num_classifiers)]
        model = utils.ensemble.Ensemble(models, diversify=args.ensemble_diversify)
    elif args.method == 'ours':
        models = [get_base_model(args.base_model) for _ in range(args.num_classifiers)]
        model = utils.ensemble.Ensemble(models, diversify=True)
    else:
        model = get_base_model(args.base_model)
    model.fit(seed_data, seed_target)

    acc, budget_end, all_ensemble_pred = stream_learning_baseline(train_stream, test_X, test_y, model, args,
                                                                  budget=args.budget, prediction_threshold=args.prediction_threshold)

    os.makedirs(f'results/{args.method}', exist_ok=True)
    experiment_parameters = f'{args.base_model}_{args.stream_len}_seed_{args.seed_percentage}_budget_{args.budget}'
    np.save(f'results/{args.method}/acc_{experiment_parameters}.npy', acc)
    np.save(f'results/{args.method}/budget_end_{experiment_parameters}.npy', budget_end)

    if args.method == 'ours':
        np.save(f'results/ours/targets_{experiment_parameters}.npy', test_y)
        np.save(f'results/ours/all_ensemble_pred_{experiment_parameters}.npy', all_ensemble_pred)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--stream_len', type=int, default=5000)
    parser.add_argument('--seed_percentage', type=float, default=0.1)
    parser.add_argument('--base_model', choices=('mlp', 'ng', 'online_bagging'), default='mlp')
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--method', choices=('ours', 'all_labeled', 'all_labeled_ensemble', 'confidence'), default='ours')
    parser.add_argument('--prediction_threshold', type=float, default=0.6)
    parser.add_argument('--ensemble_diversify', action='store_true')
    parser.add_argument('--num_classifiers', type=int, default=9)

    args = parser.parse_args()
    return args


def get_base_model(model_name):
    if model_name == 'ng':
        model = GaussianNB()
    elif model_name == 'mlp':
        model = MLPClassifier(learning_rate_init=0.008, max_iter=500)
    elif model_name == 'online_bagging':
        model = OnlineBagging(base_estimator=MLPClassifier(learning_rate_init=0.01, max_iter=500), n_estimators=5)
    else:
        raise ValueError("Invalid base classifier")
    return model


def stream_learning_baseline(train_stream, test_X, test_y, model, args, budget=100, prediction_threshold=0.5):
    all_ensemble_pred = []
    acc = []
    budget_end = -1

    for i, (obj, target) in enumerate(train_stream):
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

            if len(confident_supports) > 0 and all(pred == confident_preds[0] for pred in confident_preds):
                max_supp = max(confident_supports)
                poisson_lambda = max_supp / prediction_threshold
                label = confident_preds[0]
                label = np.expand_dims(label, 0)
                # if i < 1500:
                model.partial_fit(obj, label, poisson_lambda)
                # if label != target:
                #     print('training with wrong target')
                #     print('max_support = ', supports)
                #     print('predictions = ', predictions)
                #     print('target = ', target)
                #     print('poisson_lambda = ', poisson_lambda)
            else:
                # if i < 1500:
                budget = training_on_budget(model, prediction_threshold, budget, obj, target, supports, predictions)
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

    print(f'budget after training = {budget}')
    return acc, budget_end, all_ensemble_pred


def training_on_budget(model_pool, prediction_threshold, budget, obj, target, supports, predictions):
    if budget > 0:
        model_pool.partial_fit(obj, target, poisson_lambda=1.0)
        budget -= 1
    else:
        idx = np.argmax([np.max(s) for s in supports])
        max_supp = np.max(supports[idx])
        label = predictions[idx]
        label = np.expand_dims(label, 0)
        poisson_lambda = abs(prediction_threshold - max_supp) / prediction_threshold
        # if label != target:
        #     print('training with wrong target')
        #     print('max_support = ', supports)
        #     print('predictions = ', predictions)
        #     print('target = ', target)
        #     print('poisson_lambda = ', poisson_lambda)
        model_pool.partial_fit(obj, label, poisson_lambda)
    return budget


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
