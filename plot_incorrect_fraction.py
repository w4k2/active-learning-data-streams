import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    budget = 0.3
    random_seed = 43
    results = [
        'results/ours/{}_mlp_{}_seed_{}_budget_{}_random_seed_{}.npy',
        'results/ours/{}_mlp_{}_seed_{}_budget_{}_random_seed_{}_no_filter.npy',
    ]
    labels = ['SL2S', 'no prior filter']

    colors = sns.color_palette("husl", 6)
    with sns.axes_style("darkgrid"):
        for i, seed_size in enumerate([100, 500, 1000]):
            for j, filepath in enumerate(results):
                acc = np.load(filepath.format('acc', 'wine', seed_size, budget, random_seed))
                plt.subplot(3, 2, 2*i+1)
                plt.plot(acc, label=labels[j], color=colors[j])
                incorrect_frac = np.load(filepath.format('incorrect_fraction', 'wine', seed_size, budget, random_seed))
                plt.subplot(3, 2, 2*i+2)
                plt.plot(incorrect_frac, label=labels[j], color=colors[j])

        seed_sizes = ('seed size 100', 'seed size 500', 'seed size 1000')
        for i in range(1, 7):
            plt.subplot(3, 2, i)
            from matplotlib.ticker import StrMethodFormatter
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
            plt.legend()
            if i % 2 == 1:
                plt.ylabel('balanced accuracy')
            else:
                plt.ylabel('fraction of incorrect labels')

        plt.subplot(3, 2, 5)
        plt.xlabel('iterations')
        plt.subplot(3, 2, 6)
        plt.xlabel('iterations')
        plt.show()


if __name__ == '__main__':
    main()
