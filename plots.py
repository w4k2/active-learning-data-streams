import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


def main():
    results = [
        'results/ours/{}_ng_5000_seed_0.1_budget_100_prediction_threshold_0.6.npy',
        'results/all_labeled/{}_ng_5000_seed_0.1_budget_100.npy',
        'results/all_labeled_ensemble/{}_ng_5000_seed_0.1_budget_100.npy',
        'results/confidence/{}_ng_5000_seed_0.1_budget_100.npy',
    ]
    plot_labels = ['ours', 'all labeled', 'all labeled ensemble', 'confidence']

    for filepath, result_label in zip(results, plot_labels):
        preds = np.load(filepath.format('preds'))
        targets = np.load(filepath.format('targets'))

        acc = accumulative_acc(targets, preds)
        plt.plot(acc, label=result_label)

    plt.legend()
    plt.show()


def accumulative_acc(targets, preds, start_at=50):
    acc = []
    for i in range(start_at, len(preds)):
        acc.append(accuracy_score(targets[:i], preds[:i]))
    return acc


if __name__ == '__main__':
    main()
