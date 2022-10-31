import random

from main import *


def main():
    args = parse_args()

    best_acc = 0.0
    best_threshold = 0.0

    prediction_threshold_inteval = get_prediction_threshold_inteval(args.method)
    thresholds = [sample_prediction_threshold(prediction_threshold_inteval) for _ in range(20)]

    for prediction_threshold in thresholds:
        args.prediction_threshold = prediction_threshold

        avrg_acc = 0.0
        print(f'\nnew experiment with prediction threshold = {prediction_threshold}')
        for random_seed in [42, 43, 44]:
            args.random_seed = random_seed
            acc, budget_end, budget_after = do_experiment(args)
            avrg_acc += acc[-1]
        avrg_acc /= 3

        if avrg_acc > best_acc:
            best_acc = avrg_acc
            best_threshold = prediction_threshold

    args.prediction_threshold = best_threshold
    print(f'dataset_name = {args.dataset_name} method = {args.method} random_seed = {args.random_seed} base model = {args.base_model} seed size = {args.seed_size} budget = {args.budget} best prediction threshold = {best_threshold}')

    for random_seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        args.random_seed = random_seed
        acc, budget_end, budget_after = do_experiment(args)
        save_results(args, acc, budget_end)


def get_prediction_threshold_inteval(method_name):
    if method_name == 'ours':
        return (0.5, 1.0)
    elif method_name == 'fixed_uncertainty':
        return (0.5, 1.0)
    elif method_name == 'variable_uncertainty':
        return (0.5, 1.0)
    elif method_name == 'classification_margin':
        return (0.0, 0.8)
    elif method_name == 'vote_entropy':
        return (1.0, 50.0)
    elif method_name == 'consensus_entropy':
        return (0.1, 1.0)
    elif method_name == 'max_disagreement':
        return (1.0, 20.0)
    elif method_name == 'min_margin':
        return (0.0, 0.5)
    else:
        raise ValueError(f'Undefined hyperparameter interval for {method_name} algorithm')


def sample_prediction_threshold(prediction_threshold_interval):
    b, e = prediction_threshold_interval
    interval = abs(e - b)
    threshold = random.random() * interval + b
    return threshold


if __name__ == '__main__':
    main()
