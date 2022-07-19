from ctypes import alignment
import numpy as np
import tabulate


def main():
    parameters_to_load = [
        ('mlp', 'accelerometer', 1000, 0.1),
        ('mlp', 'accelerometer', 1000, 0.2),
        ('mlp', 'accelerometer', 1000, 0.3),
        ('mlp', 'accelerometer', 1000, 0.4),
        ('mlp', 'accelerometer', 1000, 0.5)
    ]
    column_names = ['budget', '0.1', '0.2', '0.3', '0.4', '0.5']
    table = generate_table(parameters_to_load, column_names)
    print('\n\nbase model mlp, stream len 5000, seed size 200, variable budget')
    text_table = tabulate.tabulate(table)
    print(text_table)
    latex_table = tabulate.tabulate(table, tablefmt='latex', numalign='center')
    print()
    print(latex_table)

    # parameters_to_load = [('mlp', 'accelerometer', 200, 0.3), ('mlp', 'accelerometer', 200, 0.3), ('mlp', 'accelerometer', 200, 0.3), ('mlp', 'accelerometer', 200, 0.3)]
    # column_names = ['stream len', '5000', '10000', '20000', '50000']
    # table = generate_table(parameters_to_load, column_names)
    # print('\n\nbase model mlp, variable stream len, seed size 200, budget 0.3')
    # text_table = tabulate.tabulate(table)
    # print(text_table)
    # latex_table = tabulate.tabulate(table, tablefmt='latex')
    # print()
    # print(latex_table)

    parameters_to_load = [('mlp', 'accelerometer', 100, 0.3), ('mlp', 'accelerometer', 500, 0.3), ('mlp', 'accelerometer', 1000, 0.3)]
    column_names = ['seed size', '100', '500', '1000']
    table = generate_table(parameters_to_load, column_names)
    print('\n\nbase model mlp, stream len 5000, variable seed size, budget 0.3')
    text_table = tabulate.tabulate(table)
    print(text_table)
    latex_table = tabulate.tabulate(table, tablefmt='latex', numalign='center')
    print()
    print(latex_table)


def generate_table(paramters_to_load, column_names):
    results = [
        'results/all_labeled/acc_{}_{}_seed_{}_budget_{}.npy',
        'results/all_labeled_ensemble/acc_{}_{}_seed_{}_budget_{}.npy',
        'results/random/acc_{}_{}_seed_{}_budget_{}.npy',
        'results/fixed_uncertainty/acc_{}_{}_seed_{}_budget_{}.npy',
        'results/variable_uncertainty/acc_{}_{}_seed_{}_budget_{}.npy',
        'results/classification_margin/acc_{}_{}_seed_{}_budget_{}.npy',
        'results/vote_entropy/acc_{}_{}_seed_{}_budget_{}.npy',
        'results/consensus_entropy/acc_{}_{}_seed_{}_budget_{}.npy',
        'results/max_disagreement/acc_{}_{}_seed_{}_budget_{}.npy',
        'results/ours/acc_{}_{}_seed_{}_budget_{}.npy',
    ]
    method_names = [
        'all labeled', 'all labeled ensemble',
        'random', 'fixed_uncertainty', 'variable_uncertainty', 'classification_margin',
        'vote_entropy', 'consensus_entropy', 'max_disagreement', 'ours',
    ]

    table = [column_names]

    for filepath, method_name in zip(results, method_names):
        table.append([method_name])
        for params in paramters_to_load:
            acc_training = np.load(filepath.format(*params))
            acc_final = acc_training[-1]
            acc_final = '{:.4f}'.format(acc_final)
            table[-1].append(acc_final)

    return table


if __name__ == '__main__':
    main()
