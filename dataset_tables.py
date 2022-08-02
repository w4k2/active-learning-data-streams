import data.load_data
import numpy as np
import tabulate


def main():
    dataset_names = ["accelerometer", "adult", "bank_marketing", "firewall", "chess", "nursery", "mushroom", "wine", "abalone"]
    dataset_citations = ["\\cite{accelerometer_dataset}", "\cite{adult}", "\cite{bank}", "\cite{firewall}", "\cite{uci}", "\cite{nursery}", "\cite{uci}", "\cite{wine}", "\cite{abalone}"]
    dataset_attributes = [5, 14, 17, 12, 6, 8, 22, 12, 8]
    table = [['name', '\\#classes', '\\#samples', '\\#attributes', 'highest IR \\']]
    for name, citation, num_attributes in zip(dataset_names, dataset_citations, dataset_attributes):
        row = list()
        row.append('{} {}'.format(name.replace('_', ' '), citation))
        X_seed, y_seed, X_test, y_test, stream, num_classes = data.load_data.get_data(name, seed_size=100, random_seed=42)
        row.append(num_classes)
        num_samples = len(stream) + X_seed.shape[0] + X_test.shape[0]
        row.append(num_samples)
        row.append(num_attributes)
        _, y_train = zip(*stream)
        y_train = np.concatenate((y_train, y_seed))
        _, class_counts = np.unique(y_train, return_counts=True)
        highest_IR = max(class_counts) / min(class_counts)
        row.append('{:.4f}'.format(highest_IR))
        table.append(row)

    latex_table = tabulate.tabulate(table, tablefmt='latex', numalign='center')
    latex_table = latex_table.replace('\\textbackslash{}', '').replace('\\{', '{').replace('\\}', '}').replace(' cite', ' \\cite').replace('\\_', '_')
    print(latex_table)


if __name__ == '__main__':
    main()
