import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


def main():
    plot_model('mlp')
    plt.figure()
    plot_model('ng')
    plt.show()


def plot_model(model_name, stream_len=5000, seed_size=0.1, budget=100):
    results = [
        'results/ours/{}_{}_{}_seed_{}_budget_{}_prediction_threshold_0.6.npy',
        'results/all_labeled/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/all_labeled_ensemble/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/confidence/{}_{}_{}_seed_{}_budget_{}.npy',
    ]
    plot_labels = ['ours', 'all labeled', 'all labeled ensemble', 'confidence']

    for filepath, result_label in zip(results, plot_labels):
        preds = np.load(filepath.format('preds', model_name, stream_len, seed_size, budget))
        targets = np.load(filepath.format('targets', model_name, stream_len, seed_size, budget))

        acc = accumulative_acc(targets, preds)
        plt.plot(acc, label=result_label)
        plt.title(f'{model_name} strlen {stream_len} seed size {seed_size} budget {budget}')

    plt.legend()


def accumulative_acc(targets, preds, start_at=50):
    acc = []
    for i in range(start_at, len(preds)):
        acc.append(accuracy_score(targets[:i], preds[:i]))
    return acc


if __name__ == '__main__':
    main()
