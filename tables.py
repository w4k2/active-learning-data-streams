import numpy as np
import tabulate


def main():
    results_list = [
        ('mlp', 1000, 0.1),
        ('mlp', 1000, 0.2),
        ('mlp', 1000, 0.3),
        ('mlp', 1000, 0.4),
        ('mlp', 1000, 0.5),
    ]
    dataset_list_part1 = ('adult', 'bank_marketing', 'firewall', 'chess')
    table_from_results(dataset_list_part1, results_list, 6, 'budget & 0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\\\ \n')
    # print('\n\n')
    dataset_list_part2 = ('nursery', 'mushroom', 'wine', 'abalone')
    # table_from_results(dataset_list_part2, results_list, 6, 'budget & 0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\\\ \n')

    # print('\n\n\n\n')
    results_list = [
        ('mlp', 100, 0.3),
        ('mlp', 500, 0.3),
        ('mlp', 1000, 0.3)
    ]
    # table_from_results(dataset_list_part1, results_list, 4, 'seed size & 100 & 500 & 1000 \\\\ \n')
    # print('\n\n')
    table_from_results(dataset_list_part2, results_list, 4, 'seed size & 100 & 500 & 1000 \\\\ \n')


def table_from_results(dataset_list, results_list, num_columns, custom_line):
    method_names = [
        'all_labeled',
        'all_labeled_ensemble',
        'random',
        'fixed_uncertainty',
        'variable_uncertainty',
        'classification_margin',
        'vote_entropy',
        'consensus_entropy',
        'max_disagreement',
        'ours',
    ]

    whole_table = "\\begin{tabular}{l|" + "c" * (num_columns - 1) + "}\n"
    for dataset_name in dataset_list:
        whole_table = add_header(whole_table, dataset_name, num_columns=num_columns, custom_line=custom_line)
        table, table_std = generate_averaged_table(results_list, dataset_name, method_names)
        best_idx = find_best(table)
        table = add_method_names(method_names, table, table_std)
        latex_table = tabulate.tabulate(table, tablefmt='latex', numalign='center')
        latex_table = bold_best_results(latex_table, best_idx)

        whole_table = add_table_section(whole_table, latex_table)
    whole_table += '\end{tabular}'
    print(whole_table)


def add_method_names(method_names, table, table_std):
    new_table = []
    for method_name, row, row_std in zip(method_names, table, table_std):
        new_table.append([])
        new_table[-1].append(method_name.replace("_", " "))
        new_table[-1].extend(['{:.3f}±{:.3f}'.format(acc, std) for (acc, std) in zip(row, row_std)])
    return new_table


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


def generate_averaged_table(paramters_to_load, dataset_name, method_names):
    table = []
    table_std = []

    for method_name in method_names:
        results = []
        for random_seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            row = read_row(paramters_to_load, method_name, dataset_name, random_seed)
            results.append(row)
        avrg_row = np.mean(results, axis=0, keepdims=False)
        table.append(avrg_row)
        row_std = np.std(results, axis=0, keepdims=False)
        table_std.append(row_std)

    return table, table_std


def read_row(paramters_to_load, method_name, dataset_name, random_seed):
    filepath = 'results/{}/acc_{}_{}_seed_{}_budget_{}_random_seed_{}.npy'

    row = []
    for params in paramters_to_load:
        try:
            model_name, seed_size, budget = params
            acc_training = np.load(filepath.format(method_name, model_name, dataset_name, seed_size, budget, random_seed))
            acc_final = acc_training[-1]
        except FileNotFoundError:
            acc_final = np.nan

        row.append(acc_final)

    return row


def find_best(table):
    table = table[2:]
    table = np.nan_to_num(table)
    best_indexes = np.argmax(table, axis=0) + 1
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
