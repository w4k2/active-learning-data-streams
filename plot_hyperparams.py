import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize


def main():
    dataset_sizes = {
        'accelerometer': 5000,
        'adult': 48842,
        'bank_marketing': 45211,
        'firewall': 65478,
        'chess': 27870,
        'nursery': 12958,
        'mushroom': 8124,
        'wine': 4873,
        'abalone': 4098,
    }
    plot_points = []

    with open('hyperparams_found.txt', 'r') as f:
        for line in f.readlines():
            splitted = line.split('=')[1:]
            values = [s.split(' ')[1] for s in splitted]
            dataset_name = values[0]
            method_name = values[1]
            budget = float(values[-2])
            hyperparam_value = float(values[-1][:-2])

            if method_name == 'classification_margin':
                plot_points.append((dataset_name, dataset_sizes[dataset_name], hyperparam_value))

    colors = {
        'nursery': 'r',
        'mushroom': 'g',
        'wine': 'b',
        'abalone': 'm',
    }
    for dataset_name, size, value in plot_points:
        plt.plot(size, value, 'o', color=colors[dataset_name])

    # x = [p[1] for p in plot_points]
    # y = [p[2] for p in plot_points]

    # coeff, _ = scipy.optimize.curve_fit(lambda t, a, b: a+b*np.log(t), x, y)
    # a, b = coeff

    # x_new = np.linspace(x[0]-1000, 50000, 50)
    # y_new = a + b * np.log(x_new)
    # plt.plot(x_new, y_new)

    plt.show()


if __name__ == '__main__':
    main()
