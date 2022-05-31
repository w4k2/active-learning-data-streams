import matplotlib.pyplot as plt
import numpy as np

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
        'results/ours/{}_{}_{}_seed_{}_budget_{}_prediction_threshold_0.6.npy',
        'results/all_labeled/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/all_labeled_ensemble/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/confidence/{}_{}_{}_seed_{}_budget_{}.npy',
    ]
    plot_labels = ['ours', 'all labeled', 'all labeled ensemble', 'confidence']

    diversity = None

    plt.subplot(2, 1, 1)

    for i, (filepath, result_label) in enumerate(zip(results, plot_labels)):
        acc = np.load(filepath.format('acc', model_name, args.stream_len, args.seed_percentage, args.budget))
        budget_end = np.load(filepath.format('budget_end', model_name, args.stream_len, args.seed_percentage, args.budget))
        print('budget_end = ', budget_end)

        if result_label == 'ours':
            classifier_preds = np.load(filepath.format('all_ensemble_pred', model_name, args.stream_len, args.seed_percentage, args.budget))
            targets = np.load(filepath.format('targets', model_name, args.stream_len, args.seed_percentage, args.budget))
            diversity = utils.diversity.cumulative_q_statistic(classifier_preds, targets)

        plt.plot(acc, color=f"C{i}", label=result_label)
        if budget_end > -1:
            plt.axvline(x=budget_end, color=f"C{i}", linestyle="--")
        plt.title(f'{model_name} strlen {args.stream_len} seed percentage {args.seed_percentage} budget {args.budget}')

    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(diversity)


if __name__ == '__main__':
    main()
