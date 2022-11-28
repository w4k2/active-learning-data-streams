import pathlib
from main import *
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    seed_everything(315)
    train_data, train_target, test_data, test_target, num_classes = data.load_data.get_data('wine', 315)
    colors = sns.color_palette("husl", 9)

    model_base = utils.ensemble.Ensemble([], diversify=True)
    model_base.load(pathlib.Path('./wine_model_base'))

    model_no_filter = utils.ensemble.Ensemble([], diversify=True)
    model_no_filter.load(pathlib.Path('./wine_model_no_filter'))

    with sns.axes_style("darkgrid"):
        plt.subplot(2, 3, 1)
        y_pred = model_base.predict(test_data)
        labels, counts = np.unique(y_pred, return_counts=True)
        print(labels, counts)

        bar_colors = [colors[l] for l in labels]
        plt.bar(labels, counts, color=bar_colors, width=0.5)
        plt.ylabel('samples count')
        plt.xlabel('class')
        plt.ylim(0, 640)
        plt.title('base')

        plt.subplot(2, 3, 2)
        y_pred = model_no_filter.predict(test_data)
        labels, counts = np.unique(y_pred, return_counts=True)
        print(labels, counts)

        bar_colors = [colors[l] for l in labels]
        plt.bar(labels, counts, color=bar_colors, width=0.5)
        plt.xlabel('class')
        plt.ylim(0, 640)
        plt.title('no filter')

        plt.subplot(2, 3, 3)
        labels, counts = np.unique(test_target, return_counts=True)
        print(labels, counts)
        plt.bar(labels, counts, color=bar_colors, width=0.5)
        plt.xlabel('class')
        plt.ylim(0, 640)
        plt.title('dataset')

    model_base = utils.ensemble.Ensemble([], diversify=True)
    model_base.load(pathlib.Path('./wine_model_base_balanced'))

    model_no_filter = utils.ensemble.Ensemble([], diversify=True)
    model_no_filter.load(pathlib.Path('./wine_model_no_filter_balanced'))

    with sns.axes_style("darkgrid"):
        plt.subplot(2, 3, 4)
        y_pred = model_base.predict(test_data)
        labels, counts = np.unique(y_pred, return_counts=True)
        print(labels, counts)

        bar_colors = [colors[l+1] for l in labels]
        plt.bar(labels+1, counts, color=bar_colors, width=0.5)
        plt.xlim(-0.5, 4.5)
        plt.ylabel('samples count')
        plt.xlabel('class')
        plt.ylim(0, 560)

        plt.subplot(2, 3, 5)
        y_pred = model_no_filter.predict(test_data)
        labels, counts = np.unique(y_pred, return_counts=True)

        bar_colors = [colors[l+1] for l in labels]
        plt.bar(labels+1, counts, color=bar_colors, width=0.5)
        plt.xlabel('class')
        plt.xlim(-0.5, 4.5)
        plt.ylim(0, 560)

        plt.subplot(2, 3, 6)
        labels, counts = np.unique(test_target, return_counts=True)
        bar_colors = [colors[int(l)] for l in labels]
        plt.bar(labels, counts, color=bar_colors, width=0.5)
        plt.xlabel('class')
        plt.xlim(-0.5, 4.5)
        plt.ylim(0, 560)

    plt.show()


if __name__ == '__main__':
    main()
