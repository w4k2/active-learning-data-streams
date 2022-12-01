import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    colors = sns.color_palette("husl", 4)
    with sns.axes_style("darkgrid"):
        for i, begin_acc in enumerate([0.5, 0.55, 0.6, 0.63]):
            print('begin_acc = ', begin_acc)
            acc = np.load(f'results/begin_acc_{begin_acc}_acc.npy')
            plt.plot(acc, label=f'{begin_acc}', color=colors[i])

            budget_end = np.load(f'results/begin_acc_{begin_acc}_budget_end.npy')
            print('budget_end = ', budget_end)
            if budget_end > -1:
                plt.axvline(x=budget_end, color=colors[i], linestyle="--")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
