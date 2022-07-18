import numpy as np
import tabulate


def main():
    parameters_to_load = [('mlp', 5000, 200, 0.1), ('mlp', 5000, 200, 0.2), ('mlp', 5000, 200, 0.3), ('mlp', 5000, 200, 0.4), ('mlp', 5000, 200, 0.5)]
    column_names = ['budget', '0.1', '0.2', '0.3', '0.4', '0.5']
    table = generate_table(parameters_to_load, column_names)
    print('\n\nbase model mlp, stream len 5000, seed size 200, variable budget')
    text_table = tabulate.tabulate(table)
    print(text_table)
    latex_table = tabulate.tabulate(table, tablefmt='latex')
    print()
    print(latex_table)

    parameters_to_load = [('mlp', 5000, 200, 0.3), ('mlp', 10000, 200, 0.3), ('mlp', 20000, 200, 0.3), ('mlp', 50000, 200, 0.3)]
    column_names = ['stream len', '5000', '10000', '20000', '50000']
    table = generate_table(parameters_to_load, column_names)
    print('\n\nbase model mlp, variable stream len, seed size 200, budget 0.3')
    text_table = tabulate.tabulate(table)
    print(text_table)
    latex_table = tabulate.tabulate(table, tablefmt='latex')
    print()
    print(latex_table)

    parameters_to_load = [('mlp', 5000, 100, 0.3), ('mlp', 5000, 200, 0.3), ('mlp', 5000, 500, 0.3), ('mlp', 5000, 1000, 0.3)]
    column_names = ['seed size', '100', '200', '500', '1000']
    table = generate_table(parameters_to_load, column_names)
    print('\n\nbase model mlp, stream len 5000, variable seed size, budget 0.3')
    text_table = tabulate.tabulate(table)
    print(text_table)
    latex_table = tabulate.tabulate(table, tablefmt='latex')
    print()
    print(latex_table)


def generate_table(paramters_to_load, column_names):
    results = [
        'results/ours/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/all_labeled/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/all_labeled_ensemble/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/online_bagging/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/random/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/fixed_uncertainty/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/variable_uncertainty/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/classification_margin/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/vote_entropy/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/consensus_entropy/{}_{}_{}_seed_{}_budget_{}.npy',
        'results/max_disagreement/{}_{}_{}_seed_{}_budget_{}.npy',
    ]
    method_names = [
        'ours', 'all labeled', 'all labeled ensemble', 'online bagging',
        'random', 'fixed_uncertainty', 'variable_uncertainty', 'classification_margin',
        'vote_entropy', 'consensus_entropy', 'max_disagreement'
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
