
import utils.stream
import utils.new_model
import utils.mlp_pytorch
import utils.ensemble
import utils.diversity
import self_labeling_strategies
import data.load_data
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import sklearn.model_selection
import numpy as np
import mkl
import math
from main import seed_everything, update_training_data, partial_fit
from utils.ensemble import sample_data
from sklearn.neural_network import MLPClassifier


class args:
    budget = 0.3
    prediction_threshold = 0.95
    batch_mode = False
    num_classifiers = 9
    base_model = 'mlp'
    beta1 = 0.9
    lr = 0.001


def main(begin_acc):
    random_seed = 42
    mkl.set_num_threads(20)
    seed_everything(random_seed)
    dataset_name = 'wine'
    seed_size = 1000

    train_data, train_target, test_data, test_target, num_classes = data.load_data.get_data(dataset_name, random_seed)
    X_stream, seed_data, y_stream, seed_target = sklearn.model_selection.train_test_split(train_data, train_target,
                                                                                          test_size=seed_size, random_state=random_seed, stratify=train_target)

    models = [MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate_init=args.lr, max_iter=5000, beta_1=args.beta1)
              for _ in range(args.num_classifiers)]
    model = utils.ensemble.Ensemble(models, diversify=True)
    classes = np.unique(seed_target)

    # using reflection here is ugly as hell, but it was done in order to not interrupt already running experiments
    def partial_fit(data, target, poisson_lambdas=None):
        for model in models:
            num_repeats = np.random.poisson(lam=poisson_lambdas)
            num_repeats = np.minimum(num_repeats, 4)
            model_data, model_target = sample_data(data, target, num_repeats)
            model_target = np.ravel(model_target)
            model.partial_fit(model_data, model_target, classes=classes)
    model.partial_fit = partial_fit

    train_stream = utils.stream.Stream(X_stream, y_stream)
    acc, budget_end, budget_after = training_stream(
        train_stream, seed_data, seed_target, test_data, test_target, model, num_classes, begin_acc)
    np.save(f'results/begin_acc_{begin_acc}_dataset_{dataset_name}_acc.npy', acc)
    np.save(f'results/begin_acc_{begin_acc}_dataset_{dataset_name}_budget_end.npy', budget_end)


def training_stream(train_stream, seed_data, seed_target, test_data, test_target, model, num_classes, begin_acc):
    seed_target = np.squeeze(seed_target, axis=1)
    model = fit_until_accuracy(model, seed_data, seed_target, test_data, test_target, acc=begin_acc)
    test_pred = model.predict(test_data)
    acc = balanced_accuracy_score(test_target, test_pred)
    print(f'accuracy after training with seed = {acc}')

    acc_list = list()
    budget_end = -1
    current_budget = math.floor(len(train_stream) * args.budget)
    strategy = self_labeling_strategies.Ours(model, num_classes, args.prediction_threshold)

    lambdas = np.ones_like(seed_target, dtype=float)

    for i, (obj, target) in enumerate(train_stream):
        test_pred = model.predict(test_data)
        acc = balanced_accuracy_score(test_target, test_pred)
        acc_list.append(acc)
        obj = np.expand_dims(obj, 0)

        if current_budget > 0 and strategy.request_label(obj, current_budget, args.budget):
            seed_data, seed_target = update_training_data(seed_data, seed_target, obj, target)
            lambdas = np.concatenate((lambdas, [1.0]), axis=0)
            seed_data, seed_target, lambdas = partial_fit(seed_data, seed_target, model, args, lambdas)
            current_budget -= 1
            strategy.last_predictions.append(int(target))
        else:
            train, label, poisson_lambda = strategy.use_self_labeling(obj, current_budget, args.budget)
            if train:
                seed_data, seed_target = update_training_data(seed_data, seed_target, obj, label)
                lambdas = np.concatenate((lambdas, [poisson_lambda]), axis=0)
                seed_data, seed_target, lambdas = partial_fit(seed_data, seed_target, model, args, lambdas)

        if current_budget == 0:
            current_budget = -1
            budget_end = i
            print(f'budget ended at {i}')

    print(f'budget after training = {current_budget}')
    print(f'final acc = {acc_list[-1]}')
    return acc_list, budget_end, current_budget


def fit_until_accuracy(model, seed_data, seed_target, test_data, test_target, acc=0.9):
    current_acc = 0.0
    classes = np.unique(seed_target)
    print(classes)

    while current_acc < acc:
        sizes = list(range(10, 50, 10)) + list(range(50, 200, 50)) + list(range(200, len(seed_target), 100))
        for i in sizes:
            models = [MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate_init=args.lr, max_iter=5000, beta_1=args.beta1)
                      for _ in range(args.num_classifiers)]
            model = utils.ensemble.Ensemble(models, diversify=True)

            def partial_fit(data, target, poisson_lambdas=None):
                for model in models:
                    num_repeats = np.random.poisson(lam=poisson_lambdas)
                    num_repeats = np.minimum(num_repeats, 4)
                    model_data, model_target = sample_data(data, target, num_repeats)
                    model_target = np.ravel(model_target)
                    model.partial_fit(model_data, model_target, classes=classes)
            model.partial_fit = partial_fit

            batch_data = seed_data[:i, :]
            batch_target = seed_target[:i]
            lambdas = np.ones_like(batch_target)
            if len(batch_data) < 100:
                for _ in range(100):
                    model.partial_fit(batch_data, batch_target, lambdas)
            else:
                model.fit(batch_data, batch_target)

            y_pred = model.predict(test_data)
            current_acc = balanced_accuracy_score(test_target, y_pred)
            print('current_acc = ', current_acc)
            if current_acc >= acc:
                break
    return model


def get_model(classes):
    models = [MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate_init=args.lr, max_iter=5000, beta_1=args.beta1)
              for _ in range(args.num_classifiers)]
    model = utils.ensemble.Ensemble(models, diversify=True)

    def partial_fit(data, target, poisson_lambdas=None):
        for model in models:
            num_repeats = np.random.poisson(lam=poisson_lambdas)
            num_repeats = np.minimum(num_repeats, 4)
            model_data, model_target = sample_data(data, target, num_repeats)
            model_target = np.ravel(model_target)
            model.partial_fit(model_data, model_target, classes=classes)
    model.partial_fit = partial_fit
    return model


if __name__ == '__main__':
    for acc in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        print('experiments for begin acc = ', acc)
        main(acc)

    # main(0.99)
