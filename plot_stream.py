import numpy as np
import matplotlib.pyplot as plt

import utils.utils
from main import parse_args


def main():
    np.random.seed(42)
    args = parse_args()

    _, _, train_stream, _, _ = utils.utils.get_data(args.stream_len, args.seed_size, args.random_seed)

    stream_labels = list(zip(*train_stream))[1]
    stream_labels = np.array(stream_labels)
    print('count = ', np.unique(stream_labels, return_counts=True))
    assert len(stream_labels) == len(train_stream)

    x_pos = list(range(len(stream_labels)))

    colors_mapping = {
        0: 'r',
        1: 'g',
        2: 'b',
    }

    for x, label in zip(x_pos, stream_labels):
        color = colors_mapping[label[0]]
        plt.axvline(x=x, c=color)

    # plt.scatter(x, stream_labels, s=1)
    plt.show()


if __name__ == '__main__':
    main()
