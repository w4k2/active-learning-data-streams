import matplotlib.pyplot as plt
import numpy as np


def main():
    for i, begin_acc in enumerate([0.5, 0.55, 0.6, 0.63]):
        print('begin_acc = ', begin_acc)
        acc = np.load(f'results/begin_acc_{begin_acc}_acc.npy')
        plt.plot(acc, label=f'{begin_acc}', color=f"C{i}")

        budget_end = np.load(f'results/begin_acc_{begin_acc}_budget_end.npy')
        print('budget_end = ', budget_end)
        if budget_end > -1:
            plt.axvline(x=budget_end, color=f"C{i}", linestyle="--")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
