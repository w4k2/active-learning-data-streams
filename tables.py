import numpy as np
import tabulate


def main():
    dataset_list = ('adult', 'bank_marketing', 'firewall', 'chess', 'nursery', 'mushroom', 'wine', 'abalone',)
    for dataset_name in dataset_list:
        parameters_to_load = [
            ('mlp', dataset_name, 1000, 0.1),
            ('mlp', dataset_name, 1000, 0.2),
            ('mlp', dataset_name, 1000, 0.3),
            ('mlp', dataset_name, 1000, 0.4),
            ('mlp', dataset_name, 1000, 0.5)
        ]
        column_names = ['budget', '0.1', '0.2', '0.3', '0.4', '0.5']
        table = generate_table(parameters_to_load, column_names)
        best_idx = find_best(table)
        print(f'\n\nbase model mlp, dataset {dataset_name}, seed size 200, variable budget')
        text_table = tabulate.tabulate(table)
        print(text_table)
        latex_table = tabulate.tabulate(table, tablefmt='latex', numalign='center')
        latex_table = bold_best_results(latex_table, best_idx)
        print()
        print(latex_table)

    print('\n\n')
    for dataset_name in dataset_list:
        parameters_to_load = [
            ('mlp', dataset_name, 100, 0.3),
            ('mlp', dataset_name, 500, 0.3),
            ('mlp', dataset_name, 1000, 0.3)
        ]
        column_names = ['seed size', '100', '500', '1000']
        table = generate_table(parameters_to_load, column_names)
        best_idx = find_best(table)
        print(f'\n\nbase model mlp, dataset {dataset_name}, variable seed size, budget 0.3')
        text_table = tabulate.tabulate(table)
        print(text_table)
        latex_table = tabulate.tabulate(table, tablefmt='latex', numalign='center')
        latex_table = bold_best_results(latex_table, best_idx)
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
            try:
                acc_training = np.load(filepath.format(*params))
                acc_final = acc_training[-1]
            except FileNotFoundError:
                acc_final = np.nan

            acc_final = '{:.4f}'.format(acc_final)
            table[-1].append(acc_final)

    return table


def find_best(table):
    table = np.array(table)
    table = table[1:, 1:]
    table.astype(float)
    best_indexes = np.argmax(table, axis=0)
    best_indexes = set((best_idx, column_idx) for column_idx, best_idx in enumerate(best_indexes))
    return best_indexes


def bold_best_results(latex_table, best_indexes):
    """
    latex_table - str
    best_indexes - set of table indexes with best results (indexes counted with row and columns labels excluded, zero-based)
    """
    new_table = ""

    rows_list = latex_table.split('\n')
    new_table += rows_list[0]
    new_table += "\n"
    new_table += rows_list[1]
    new_table += "\n"
    new_table += rows_list[2]
    new_table += "\n"

    line_end = repr(" \\ ")[1:]

    for row_idx, row in enumerate(rows_list[3:-2]):
        new_row = ""
        cell_list = row.split('&')

        new_row += cell_list[0]
        new_row += " & "
        for column_idx, cell in enumerate(cell_list[1:]):
            if column_idx == len(cell_list) - 2:
                cell = cell[:-3]  # remove the latex line ending \\
            if (row_idx, column_idx) in best_indexes:
                cell = "\\textbf{} {} {}".format('{', cell, '}')
            new_row += cell
            new_row += line_end if column_idx == len(cell_list) - 2 else " & "

        new_row = new_row[:-2]
        new_table += new_row
        new_table += "\n"

    new_table += rows_list[-2]
    new_table += "\n"
    new_table += rows_list[-1]
    new_table += "\n"

    return new_table


if __name__ == '__main__':
    main()
