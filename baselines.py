import argparse
import os
import numpy as np

from strlearn.streams import StreamGenerator
from utils import select_seed
from main import get_base_model, get_model_dataset


class Ensemble:
    def __init__(self, models, diversify=False) -> None:
        self.models = models
        self.diversify = diversify

    def fit(self, data, target):
        for model in self.models:
            if self.diversify:
                train_data, train_target = get_model_dataset(data, target)
                model.fit(train_data, train_target)
            else:
                model.fit(data, target)

    def predict(self, data):
        predictions = []
        for model in self.models:
            pred = model.predict_proba(data)
            predictions.append(pred)
        predictions = np.stack(predictions, axis=0)
        pred_avrg = np.mean(predictions, axis=0)
        pred_label = np.argmax(pred_avrg, axis=-1)
        return pred_label

    def partial_fit(self, data, target):
        for model in self.models:
            model.partial_fit(data, target)


def main():
    args = parse_args()

    stream = StreamGenerator(
        n_chunks=args.stream_len,
        chunk_size=1,
        n_drifts=0,
        random_state=2042,
    )

    data, target, stream = select_seed(stream, args.seed_percentage)
    if args.ensemble:
        model = Ensemble([get_base_model(args.base_model) for _ in range(5)], diversify=args.ensemble_diversify)
    else:
        model = get_base_model(args.base_model)
    model.fit(data, target)

    all_preds, all_targets, budget_end = stream_learning_baseline(stream, model, args, budget=args.budget, prediction_threshold=args.prediction_threshold)

    method_name = f'{args.method}'
    if args.ensemble:
        method_name += '_ensemble'
    os.makedirs(f'results/{method_name}', exist_ok=True)
    experiment_parameters = f'{args.base_model}_{args.stream_len}_seed_{args.seed_percentage}_budget_{args.budget}'
    np.save(f'results/{method_name}/preds_{experiment_parameters}.npy', all_preds)
    np.save(f'results/{method_name}/targets_{experiment_parameters}.npy', all_targets)
    np.save(f'results/{method_name}/budget_end_{experiment_parameters}.npy', budget_end)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--stream_len', type=int, default=5000)
    parser.add_argument('--seed_percentage', type=float, default=0.1)
    parser.add_argument('--base_model', choices=('mlp', 'ng', 'online_bagging'), default='mlp')
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--method', choices=('all_labeled', 'confidence'), default='all_labeled')
    parser.add_argument('--prediction_threshold', type=float, default=0.6)
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--ensemble_diversify', action='store_true')

    args = parser.parse_args()
    return args


def stream_learning_baseline(stream, model, args, budget=100, prediction_threshold=0.5):
    all_predictions = []
    all_targets = []
    budget_end = -1

    for i, (obj, target) in enumerate(stream):
        all_targets.append(target)
        if args.method == 'all_labeled':
            pred = model.predict(obj)
            all_predictions.append(pred)
            model.partial_fit(obj, target)
        elif args.method == 'confidence':
            pred_prob = model.predict_proba(obj)
            pred = np.argmax(pred_prob, axis=1)
            all_predictions.append(pred)
            if budget > 0:
                max_prob = np.max(pred_prob, axis=1)[0]
                if max_prob < prediction_threshold:
                    model.partial_fit(obj, target)
                    budget -= 1

        if budget == 0:
            budget = -1
            budget_end = i
            print(f'budget ended at {i}')

    print(f'budget after training = {budget}')
    return all_predictions, all_targets, budget_end


if __name__ == '__main__':
    main()
