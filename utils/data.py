import numpy as np
from strlearn.streams import StreamGenerator


def get_data(stream_len, seed_size, random_seed, num_classes, test_size):
    stream = StreamGenerator(
        n_chunks=stream_len,
        chunk_size=1,
        n_drifts=0,
        random_state=random_seed,
        y_flip=0.0,
        n_classes=num_classes,
        n_features=5,
        n_informative=5,
        n_redundant=0,
    )

    stream = iter(iterable_stream(stream))
    seed_data, seed_target, stream = sample_set(stream, seed_size)
    test_size = int(test_size * len(stream))
    test_data, test_target, stream = sample_set(stream, test_size)
    return seed_data, seed_target, test_data, test_target, stream


def iterable_stream(datastream):
    while not datastream.is_dry():
        yield datastream.get_chunk()


def sample_set(stream, seed_size):
    data = []
    target = []
    new_stream = []

    for batch in stream:
        if len(data) < seed_size:
            X, y = batch
            data.append(X)
            target.append(y)
        else:
            new_stream.append(batch)

    data = np.concatenate(data, axis=0)
    target = np.concatenate(target, axis=0)

    return data, target, new_stream
