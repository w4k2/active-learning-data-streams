import numpy as np
from strlearn.streams import StreamGenerator


def get_data(stream_len, seed_size, random_seed, num_classes, num_drifts):
    stream = StreamGenerator(
        n_chunks=stream_len,
        chunk_size=1,
        n_drifts=num_drifts,
        random_state=random_seed,
        y_flip=0.0,
        n_classes=num_classes,
        n_features=5,
        n_informative=5,
        n_redundant=0,
    )

    seed_data, seed_target, stream = select_seed(stream, seed_size)
    return seed_data, seed_target, stream


def select_seed(stream, seed_size):
    data = []
    target = []
    new_stream = []

    while len(data) < seed_size:
        X, y = stream.get_chunk()
        data.append(X)
        target.append(y)

    while not stream.is_dry():
        batch = stream.get_chunk()
        new_stream.append(batch)

    data = np.concatenate(data, axis=0)
    target = np.concatenate(target, axis=0)

    return data, target, new_stream
