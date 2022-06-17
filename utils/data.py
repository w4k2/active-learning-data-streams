import numpy as np

from strlearn.streams import StreamGenerator
from sklearn.model_selection import train_test_split


def get_data(stream_len, seed_size, chunk_size, random_seed):
    stream = StreamGenerator(
        n_chunks=stream_len,
        chunk_size=chunk_size,
        n_drifts=0,
        random_state=random_seed,
        y_flip=0.0,
        n_classes=3,
        n_features=5,
        n_informative=5,
        n_redundant=0,
    )

    seed_data, seed_target, stream = select_from_seed(stream, seed_size)
    test_size = int(0.25 * stream_len * chunk_size)
    X_test, y_test, stream = select_from_seed(stream, test_size)

    def iterable_stream_generator(datastream):
        while not datastream.is_dry():
            yield datastream.get_chunk()

    return seed_data, seed_target, iterable_stream_generator(stream), X_test, y_test


def select_from_seed(stream, seed_size):
    seed_data = []
    seed_target = []
    current_size = 0

    while current_size < seed_size and not stream.is_dry():
        X, y = stream.get_chunk()
        seed_data.append(X)
        seed_target.append(y)
        current_size += len(y)

    if stream.is_dry():
        raise ValueError('Seed size excede stream size')

    seed_data = np.concatenate(seed_data, axis=0)
    seed_target = np.concatenate(seed_target, axis=0)

    return seed_data, seed_target, stream
