import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import utils.stream

import data.load_data
from main import parse_args


def main():
    np.random.seed(42)
    args = parse_args()

    train_data, train_target, _, _, _ = data.load_data.get_data(args.dataset_name, args.random_seed)

    X_stream, _, y_stream, _ = sklearn.model_selection.train_test_split(train_data, train_target,
                                                                        test_size=args.seed_size, random_state=args.random_seed, stratify=train_target)

    classes, counts = np.unique(y_stream, return_counts=True)
    print('count = ', classes, counts)
    percentage_whole = get_percentage({c: cc for c, cc in zip(classes, counts)})
    print('stream counts whole dataset: ', percentage_whole)
    print()
    count_dict = {class_idx: count for class_idx, count in zip(classes, counts)}

    colors = ['r', 'g', 'b', 'orange', 'm', 'black', 'brown', 'linen', 'lime', 'crimson', 'slategrey', 'tab:olive', 'tab:purple', 'aquamarine', 'rosybrown', 'yellow', 'khaki', 'mediumseagreen']
    colors_mapping = {class_idx: color for class_idx, color in zip(classes, colors)}

    diff_standard = []
    labeled_colors = set()
    # plt.subplot(2, 1, 1)
    # plt.axis('off')
    x_pos = range(len(y_stream))
    stream_counts = {class_idx: 0 for class_idx in classes}
    for i, (x, label) in enumerate(zip(x_pos, y_stream)):
        color = colors_mapping[label[0]]
        class_count = None
        if color not in labeled_colors:
            class_count = count_dict[int(label)]
            labeled_colors.add(color)

        stream_counts[int(label)] += 1
        perc = get_percentage(stream_counts)
        diff = get_diff(perc, percentage_whole)
        diff_standard.append(diff)
        if i in (100, 200, 500, 1000, 2000, 3000):
            print(f'stream counts at sample {i}: ', perc, ' diff = ', diff)

        # plt.axvline(x=x, c=color, label=class_count)
    print()
    # plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.axis('off')
    diff_stream = []
    x_pos = range(len(y_stream))
    stream = utils.stream.Stream(X_stream, y_stream)
    stream_counts = {class_idx: 0 for class_idx in classes}
    for i, (x, (_, label)) in enumerate(zip(x_pos, stream)):
        color = colors_mapping[label[0]]
        # plt.axvline(x=x, c=color)

        stream_counts[int(label)] += 1
        perc = get_percentage(stream_counts)
        diff = get_diff(perc, percentage_whole)
        diff_stream.append(diff)
        if i in (100, 200, 500, 1000, 2000, 3000):
            print(f'stream counts at sample {i}: ', perc, ' diff = ', diff)

    plt.plot(diff_standard)
    plt.plot(diff_stream)
    plt.show()


def get_percentage(counts):
    count_all = sum(counts.values())
    percentage_dict = {}
    for label, count in counts.items():
        percentage = count / count_all
        percentage_dict[label] = percentage

    return percentage_dict


def get_diff(counts, counts_whole):
    diff = 0.0
    weights_sum = sum(1.0 / np.array(list(counts_whole.values())))
    for class_idx in counts.keys():
        c = counts[class_idx]
        cw = counts_whole[class_idx]
        diff += 1 / cw / weights_sum * abs(c - cw)
    return diff


if __name__ == '__main__':
    main()
