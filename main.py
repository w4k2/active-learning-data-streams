import numpy as np
import argparse
import os
import math
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils.utils import get_data, OnlineBagging
from utils.ensemble import Ensemble


def main():
    args = parse_args()

    seed_data, seed_target, train_stream, test_X, test_y = get_data(args.stream_len, args.seed_percentage)
    pool = Ensemble([get_base_model(args.base_model) for _ in range(args.num_classifiers)], diversify=True)
    pool.fit(seed_data, seed_target)
    # plot_confidence(pool, data, target)

    acc, budget_end, all_ensemble_pred = stream_learning(train_stream, test_X, test_y, pool, prediction_threshold=args.prediction_threshold, budget=args.budget)

    os.makedirs('results/ours', exist_ok=True)
    experiment_parameters = f'{args.base_model}_{args.stream_len}_seed_{args.seed_percentage}_budget_{args.budget}_prediction_threshold_{args.prediction_threshold}'
    np.save(f'results/ours/acc_{experiment_parameters}.npy', acc)
    np.save(f'results/ours/targets_{experiment_parameters}.npy', test_y)
    np.save(f'results/ours/budget_end_{experiment_parameters}.npy', budget_end)
    np.save(f'results/ours/all_ensemble_pred_{experiment_parameters}.npy', all_ensemble_pred)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream_len', type=int, default=5000)
    parser.add_argument('--seed_percentage', type=float, default=0.1)
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--prediction_threshold', type=float, default=0.6)
    parser.add_argument('--base_model', choices=('mlp', 'ng'), default='mlp')
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


def stream_learning(train_stream, test_X, test_y, model_pool, prediction_threshold=0.7, budget=100):
    all_ensemble_pred = []
    acc = []
    budget_end = -1

    for i, (obj, target) in enumerate(train_stream):
        supports = model_pool.predict_proba_separate(obj)
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
            retrain(model_pool, obj, label, poisson_lambda)
            # if label != target:
            #     print('training with wrong target')
            #     print('max_support = ', supports)
            #     print('predictions = ', predictions)
            #     print('target = ', target)
            #     print('poisson_lambda = ', poisson_lambda)
        else:
            # if i < 1500:
            budget = training_on_budget(model_pool, prediction_threshold, budget, obj, target, supports, predictions)

        if budget == 0:
            budget = -1
            budget_end = i
            print(f'budget ended at {i}')

        test_supports = model_pool.predict_proba_separate(test_X)
        test_predictions = np.argmax(test_supports, axis=2)
        all_ensemble_pred.append(test_predictions)
        test_avrg_pred = np.mean(test_supports, axis=0)
        test_pred = np.argmax(test_avrg_pred, axis=1)
        acc.append(accuracy_score(test_y, test_pred))

    print(f'budget after training = {budget}')
    all_ensemble_pred = np.array(all_ensemble_pred)
    # print(all_ensemble_pred.shape)  # (3375, 9, 1125)
    return acc, budget_end, all_ensemble_pred


def training_on_budget(model_pool, prediction_threshold, budget, obj, target, supports, predictions):
    if budget > 0:
        retrain(model_pool, obj, target, 1.0)
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
        retrain(model_pool, obj, label, poisson_lambda)
    return budget


def retrain(model_pool, obj, label, poisson_lambda):
    for model in model_pool.models:
        num_repeats = np.random.poisson(lam=poisson_lambda)
        for _ in range(num_repeats):
            label = np.ravel(label)
            model.partial_fit(obj, label)


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
