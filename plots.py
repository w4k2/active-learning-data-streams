import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

import utils.diversity
from main import parse_args


def main():
    args = parse_args()
    plot_model('mlp', args)
    # plt.figure()
    # plot_model('ng', args)
    plt.show()


def plot_model(model_name, args):
    results = [
        'results/ours/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/all_labeled/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/all_labeled_ensemble/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/random/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/fixed_uncertainty/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/variable_uncertainty/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/variable_randomized_uncertainty/{}_{}_{}_seed_{}_budget_{}.npy',
    ]
    plot_labels = ['ours', 'all labeled', 'all labeled ensemble',
                   'random', 'fixed_uncertainty', 'variable_uncertainty', 'variable_randomized_uncertainty']

    diversity = None
    diversity_unsupervised = None

    # plt.subplot(3, 1, 1)

    for i, (filepath, result_label) in enumerate(zip(results, plot_labels)):
        acc = np.load(filepath.format('acc', model_name,
                      args.stream_len, args.seed_size, args.budget))
        acc = gaussian_filter1d(acc, sigma=1)
        budget_end = np.load(filepath.format(
            'budget_end', model_name, args.stream_len, args.seed_size, args.budget))
        print('budget_end = ', budget_end)

        # if result_label == 'ours':
        #     classifier_preds = np.load(filepath.format('all_ensemble_pred', model_name, args.stream_len, args.seed_size, args.budget))
        #     targets = np.load(filepath.format('targets', model_name, args.stream_len, args.seed_size, args.budget))
        #     diversity = utils.diversity.q_statistic_sequence(classifier_preds, targets)
        #     diversity_unsupervised = utils.diversity.q_statistic_sequence(classifier_preds, targets, unsupervised=True)

        plt.plot(acc, color=f"C{i}", label=result_label)
        if budget_end > -1:
            plt.axvline(x=budget_end, color=f"C{i}", linestyle="--")
        plt.title(
            f'{model_name} strlen {args.stream_len} seed size {args.seed_size} budget {args.budget}')
        plt.xlabel('samples')
        plt.ylabel('Accuracy')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.subplot(3, 1, 2)
    # plt.plot(diversity)
    # plt.xlabel('samples')
    # plt.ylabel('Q statistic')

    # plt.subplot(3, 1, 3)
    # plt.plot(diversity_unsupervised)
    # plt.xlabel('samples')
    # plt.ylabel('Q statistic unsupervised')


if __name__ == '__main__':
    main()
