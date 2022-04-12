import numpy as np
import argparse
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from strlearn.streams import StreamGenerator
from utils import select_al_seed, OnlineBagging


def main():
    args = parse_args()

    stream = StreamGenerator(
        n_chunks=args.stream_len,
        chunk_size=1,
        n_drifts=0,
        random_state=2042,
    )

    data, target, stream = select_al_seed(stream, args.seed_percentage)
    pool = generate_classifier_pool(data, target, base_classifier=args.base_model)

    all_preds, all_targets = stream_learning(stream, pool, prediction_threshold=args.prediction_threshold, budget=args.budget)

    os.makedirs('results/ours', exist_ok=True)
    np.save(f'results/ours/preds_{args.base_model}_{args.stream_len}_seed_{args.seed_percentage}_budget_{args.budget}_prediction_threshold_{args.prediction_threshold}.npy', all_preds)
    np.save(f'results/ours/targets_{args.base_model}_{args.stream_len}_seed_{args.seed_percentage}_budget_{args.budget}_prediction_threshold_{args.prediction_threshold}.npy', all_targets)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream_len', type=int, default=5000)
    parser.add_argument('--seed_percentage', type=float, default=0.1)
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--prediction_threshold', type=float, default=0.6)
    parser.add_argument('--base_model', choices=('mlp', 'ng'), default='mlp')

    args = parser.parse_args()
    return args


def generate_classifier_pool(data, target, num_classifiers=5, base_classifier='mlp'):
    classifier_pool = []
    for _ in range(num_classifiers):
        model = get_base_model(base_classifier)
        model_data, model_target = get_model_dataset(data, target)
        model.fit(model_data, model_target)
        classifier_pool.append(model)
    return classifier_pool


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


def get_model_dataset(data, target):
    selected_data = []
    selected_target = []
    for i in range(data.shape[0]):
        num_repeats = np.random.poisson(lam=1.0)
        if num_repeats == 0:
            continue

        for _ in range(num_repeats):
            selected_data.append(data[i])
            selected_target.append(target[i])

    selected_data = np.stack(selected_data, axis=0)
    selected_target = np.stack(selected_target, axis=0)

    return selected_data, selected_target


def stream_learning(stream, model_pool, prediction_threshold=0.7, budget=100):
    all_predictions = []
    all_targets = []

    for i, (obj, target) in enumerate(stream):
        supports, predictions = pool_inference(model_pool, obj)

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
                poisson_lambda = max_supp / prediction_threshold
                label = confident_preds[0]
                label = np.expand_dims(label, 0)
                model_pool = retrain(model_pool, obj, label, poisson_lambda)
                # if label != target:
                #     print('training with wrong target')
                #     print('max_support = ', supports)
                #     print('predictions = ', predictions)
                #     print('target = ', target)
                #     print('poisson_lambda = ', poisson_lambda)
                # pass
            else:
                model_pool, budget = training_on_budget(model_pool, prediction_threshold, budget, obj, target, supports, predictions)
        else:
            model_pool, budget = training_on_budget(model_pool, prediction_threshold, budget, obj, target, supports, predictions)

        if budget == 0:
            budget = -1
            print(f'budget ended at {i}')

        avrg_pred = np.mean(supports, axis=0)
        pred = np.argmax(avrg_pred)
        all_predictions.append(pred)
        all_targets.append(target)

    print(f'budget after training = {budget}')
    return all_predictions, all_targets


def pool_inference(model_pool, obj):
    supports = []
    predictions = []
    for model in model_pool:
        probs = model.predict_proba(obj)
        supports.append(probs)
        pred = np.argmax(probs, axis=1)[0]
        predictions.append(pred)
    return supports, predictions


def training_on_budget(model_pool, prediction_threshold, budget, obj, target, supports, predictions):
    if budget > 0:
        model_pool = retrain(model_pool, obj, target, 1.0)
        budget -= 1
    else:
        idx = np.argmax([np.max(s) for s in supports])
        max_supp = np.max(supports[idx])
        label = predictions[idx]
        label = np.expand_dims(label, 0)
        poisson_lambda = abs(prediction_threshold - max_supp) / prediction_threshold
        if label != target:
            print('training with wrong target')
            print('max_support = ', supports)
            print('predictions = ', predictions)
            print('target = ', target)
            print('poisson_lambda = ', poisson_lambda)
        model_pool = retrain(model_pool, obj, label, poisson_lambda)
        # pass
    return model_pool, budget


def retrain(model_pool, obj, label, poisson_lambda):
    new_pool = []

    for model in model_pool:
        num_repeats = np.random.poisson(lam=poisson_lambda)

        for _ in range(num_repeats):
            model.partial_fit(obj, label)

        new_pool.append(model)

    return new_pool


if __name__ == '__main__':
    main()
