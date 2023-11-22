from main import *


def main():
    args = parse_args()

    best_acc = 0.0
    best_threshold = 0.0

    prediction_threshold_inteval = get_prediction_threshold_inteval(
        args.strategy_name)
    thresholds = [sample_prediction_threshold(
        prediction_threshold_inteval) for _ in range(20)]

    args.use_validation_set = True
    for prediction_threshold in thresholds:
        args.prediction_threshold = prediction_threshold

        avrg_acc = 0.0
        print(
            f'\nnew experiment with prediction threshold = {prediction_threshold}')
        for random_seed in [42, 43, 44]:
            args.seed = random_seed
            acc, budget_end = do_experiment(args)
            avrg_acc += acc[-1]
        avrg_acc /= 3

        if avrg_acc > best_acc:
            best_acc = avrg_acc
            best_threshold = prediction_threshold

    args.prediction_threshold = best_threshold
    args.use_validation_set = False
    print(f'dataset_name = {args.dataset_name} method = {args.strategy_name} random_seed = {args.random_seed} seed size = {args.n_init_labeled} budget = {args.budget} best prediction threshold = {best_threshold}')

    for random_seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        args.seed = random_seed
        acc, budget_end = do_experiment(args)
        save_results(args, acc, budget_end)


def get_prediction_threshold_inteval(method_name):
    if method_name == 'Ours':
        return (0.5, 1.0)
    elif method_name == 'LeastConfidence':
        return (0.9, 1.0)
    elif method_name == 'MarginSampling':
        return (0.0, 0.8)
    elif method_name == 'EntropySampling':
        return (0.1, 1.0)
    elif method_name == 'BALDDropout':
        return (0.0, 1.0)
    elif method_name == 'ConsensusEntropy':
        return (0.1, 1.0)
    else:
        raise ValueError(
            f'Undefined hyperparameter interval for {method_name} algorithm')


def sample_prediction_threshold(prediction_threshold_interval):
    b, e = prediction_threshold_interval
    interval = abs(e - b)
    threshold = random.random() * interval + b
    return threshold


if __name__ == '__main__':
    main()
