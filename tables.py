import numpy as np
import tabulate


def main():
    dataset_list_part1 = ('adult', 'bank_marketing', 'firewall', 'chess')
    results_list = [
        ('mlp', 1000, 0.1),
        ('mlp', 1000, 0.2),
        ('mlp', 1000, 0.3),
        ('mlp', 1000, 0.4),
        ('mlp', 1000, 0.5),
    ]
    table_from_results(dataset_list_part1, results_list, 6, 'budget & 0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\\\ \n')
    print('\n\n')
    dataset_list_part2 = ('nursery', 'mushroom', 'wine', 'abalone')
    table_from_results(dataset_list_part2, results_list, 6, 'budget & 0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\\\ \n')

    print('\n\n\n\n')
    results_list = [
        ('mlp', 100, 0.3),
        ('mlp', 500, 0.3),
        ('mlp', 1000, 0.3)
    ]
    table_from_results(dataset_list_part1, results_list, 4, 'seed size & 100 & 500 & 1000 \\\\ \n')
    print('\n\n')
    table_from_results(dataset_list_part2, results_list, 4, 'seed size & 100 & 500 & 1000 \\\\ \n')


def table_from_results(dataset_list, results_list, num_columns, custom_line):
    whole_table = "\\begin{tabular}{l|" + "c" * (num_columns - 1) + "}\n"
    for dataset_name in dataset_list:
        whole_table = add_header(whole_table, dataset_name, num_columns=num_columns, custom_line=custom_line)
        table = generate_table(results_list, dataset_name)
        best_idx = find_best(table)
        latex_table = tabulate.tabulate(table, tablefmt='latex', numalign='center')
        latex_table = bold_best_results(latex_table, best_idx)

        whole_table = add_table_section(whole_table, latex_table)
    whole_table += '\end{tabular}'
    print(whole_table)


def add_header(table_str, dataset_name, num_columns, custom_line=None):
    table_str += '\n\\hline \n'
    table_str += '\\multicolumn{}{}{}{}c{}{}dataset {}{} \\\\ \n'.format('{', num_columns, '}', '{', '}', '{', dataset_name.replace("_", " "), '}')
    table_str += '\\hline \n'
    if custom_line:
        table_str += custom_line
        table_str += '\\hline \n'

    return table_str


def add_table_section(table_str, latex_table):
    table_lines = latex_table.split('\n')
    for line in table_lines:
        if line in ('\\hline', '\\end{tabular}') or line.startswith('\\begin{tabular}'):
            continue
        table_str += line
        table_str += '\n'

    return table_str


def generate_table(paramters_to_load, dataset_name):
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

    table = []

    for filepath, method_name in zip(results, method_names):
        table.append([method_name.replace("_", " ")])
        for params in paramters_to_load:
            try:
                model_name, seed_size, budget = params
                acc_training = np.load(filepath.format(model_name, dataset_name, seed_size, budget))
                acc_final = acc_training[-1]
            except FileNotFoundError:
                acc_final = np.nan

            acc_final = '{:.4f}'.format(acc_final)
            table[-1].append(acc_final)

    return table


def find_best(table):
    table = np.array(table)
    table = table[3:, 1:]
    table.astype(float)
    best_indexes = np.argmax(table, axis=0) + 2
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
