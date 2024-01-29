import argparse
import random
import os
import numpy as np
import torch
import math
import tqdm
from utils import get_dataset, get_net, get_strategy


def main():
    args = parse_args()
    acc_list, budget_end = do_experiment(args)
    save_results(args, acc_list, budget_end)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--n_init_labeled', type=int,
                        default=1000, help="number of init labeled samples")
    parser.add_argument('--dataset_name', type=str, default="MNIST",
                        choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10"], help="dataset")
    parser.add_argument('--budget', type=float, default=0.3)
    parser.add_argument('--update_size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=None,
                        help='sampling threshold for AL method')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use_validation_set', action='store_true')

    parser.add_argument('--strategy_name', type=str, default="RandomSampling",
                        choices=[
                            "RandomSampling",
                            "LeastConfidence",
                            "MarginSampling",
                            "EntropySampling",
                            "LeastConfidenceDropout",
                            "MarginSamplingDropout",
                            "EntropySamplingDropout",
                            "KMeansSampling",
                            "KCenterGreedy",
                            "BALDDropout",
                            "AdversarialBIM",
                            "AdversarialDeepFool",
                            'ConsensusEntropy',
                            'Ours',
                        ], help="query strategy")
    args = parser.parse_args()
    return args


def do_experiment(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    dataset = get_dataset(args.dataset_name, args.use_validation_set)
    net = get_net(args.dataset_name, device)
    strategy_class = get_strategy(args.strategy_name)
    if strategy_class.__name__ == 'SelfLabelingSelectiveSampling':
        strategy = strategy_class(
            dataset, net, args.threshold, args.n_init_labeled)
    else:
        strategy = strategy_class(dataset, net, args.threshold)

    acc_list, budget_end = training_stream(args, dataset, strategy)
    return acc_list, budget_end


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def training_stream(args, dataset, strategy):
    dataset.initialize_labels(args.n_init_labeled)
    print(f"number of labeled pool: {args.n_init_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
    print(f"number of testing pool: {dataset.n_test}")
    print()

    print("Initial training")
    strategy.train()
    preds = strategy.predict(dataset.get_test_data())
    init_acc = dataset.cal_test_acc(preds)
    print(f"Initial training testing accuracy: {init_acc}")

    unlabeled_len = len(dataset.get_unlabeled_data()[0])
    current_budget = math.floor(unlabeled_len * args.budget)

    num_new_samples = 0
    last_acc = init_acc
    acc_list = list()
    budget_end = -1

    print('current_budget = ', current_budget)
    # pbar = tqdm.tqdm(dataset, total=unlabeled_len)
    pbar = dataset
    for i, (X_train, _, idx) in enumerate(pbar):
        if current_budget > 0 and strategy.should_label(X_train, args.budget):
            current_budget -= 1
            strategy.update([idx])
            num_new_samples += 1
        elif hasattr(strategy, 'use_self_labeling'):
            use_sl, label, poisson_lambda = strategy.use_self_labeling(
                X_train, current_budget)
            if use_sl:
                label = torch.from_numpy(label)
                strategy.update_self_labeling([idx], label, poisson_lambda)
                num_new_samples += 1

        if num_new_samples == args.update_size:
            num_new_samples = 0
            strategy.train()

            preds = strategy.predict(dataset.get_test_data())
            last_acc = dataset.cal_test_acc(preds)
            # pbar.set_description(
            #     "test acc {:.4f}, remaning budget {}".format(last_acc, current_budget))

        acc_list.append(last_acc)
        if current_budget == 0:
            budget_end = i

    if num_new_samples > 0:
        strategy.train()
        preds = strategy.predict(dataset.get_test_data())
        last_acc = dataset.cal_test_acc(preds)
        acc_list.append(last_acc)

    print('final acc = ', last_acc)
    print('remaning budget after training = ', current_budget)

    return acc_list, budget_end


def save_results(args, acc, budget_end, incorrect_fraction=None):
    os.makedirs(f'results/{args.strategy_name}', exist_ok=True)
    experiment_parameters = f'{args.dataset_name}_n_init_labeled_{args.n_init_labeled}_budget_{args.budget}_random_seed_{args.seed}_update_size_{args.update_size}'
    np.save(f'results/{args.strategy_name}/acc_{experiment_parameters}.npy', acc)
    np.save(f'results/{args.strategy_name}/budget_end_{experiment_parameters}.npy', budget_end)
    # np.save(f'results/{args.strategy_name}/incorrect_fraction_{experiment_parameters}.npy', incorrect_fraction)


if __name__ == '__main__':
    main()
