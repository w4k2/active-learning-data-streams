import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

from main import parse_args


def main():
    args = parse_args()
    plot_model(args)


def plot_model(args):
    results = [
        # 'results/ours/{}_{}_{}_seed_{}_budget_{}.npy',
        # 'results/all_labeled/{}_{}_{}_seed_{}_budget_{}.npy',
        # 'results/all_labeled_ensemble/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/random/{}_{}_{}_seed_{}_budget_{}_random_seed_{}.npy',
        # 'results/fixed_uncertainty/{}_{}_{}_seed_{}_budget_{}.npy',
        # 'results/variable_uncertainty/{}_{}_{}_seed_{}_budget_{}.npy',
        # 'results/classification_margin/{}_{}_{}_seed_{}_budget_{}.npy',
        # 'results/vote_entropy/{}_{}_{}_seed_{}_budget_{}.npy',
        # 'results/consensus_entropy/{}_{}_{}_seed_{}_budget_{}.npy',
        # 'results/max_disagreement/{}_{}_{}_seed_{}_budget_{}.npy',
    ]
    plot_labels = [
        # 'ours', 'all labeled', 'all labeled ensemble',
        # 'random', 'fixed_uncertainty', 'variable_uncertainty', 'classification_margin',
        # 'vote_entropy', 'consensus_entropy', 'max_disagreement'
        'random'
    ]

    for i, (filepath, result_label) in enumerate(zip(results, plot_labels)):
        acc = np.load(filepath.format('acc', args.base_model, args.dataset_name, args.seed_size, args.budget, args.random_seed))
        acc = gaussian_filter1d(acc, sigma=1)
        budget_end = np.load(filepath.format(
            'budget_end', args.base_model, args.dataset_name, args.seed_size, args.budget, args.random_seed))
        print('budget_end = ', budget_end)

        plt.plot(acc, color=f"C{i}", label=result_label)
        if budget_end > -1:
            plt.axvline(x=budget_end, color=f"C{i}", linestyle="--")
        plt.title(
            f'{args.base_model} strlen {args.dataset_name} seed size {args.seed_size} budget {args.budget}')
        plt.xlabel('samples')
        plt.ylabel('Accuracy')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
    # plt.savefig('plots/new_results/{}_dataset_name_{}_seed_size_{}_budget_{}_random_seed_{}.png'.format(args.base_model,
    #             args.dataset_name, args.seed_size, args.budget, args.random_seed), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
