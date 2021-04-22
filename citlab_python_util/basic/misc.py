import numpy as np


def create_sequence_mask(lengths, maxlen=None, dtype=np.bool):
    """ Returns a mask array representing the first N positions of each cell. This function is a numpy
    equivalent to the TensorFlow function tf.sequence_mask(...)

    Args:
        lengths: integer array, all its values <= maxlen.
        maxlen: scalar integer, size of last dimension of returned array. Default is the maximum value in `lengths`.
        dtype: output type of the resulting array.

    Returns:
        A mask array of shape `lengths.shape + (maxlen,)`, cast to specified dtype.
    Raises:
        ValueError: if `maxlen` is not a scalar
    """
    # default 'maxlen' is maximum value in 'lengths'
    if maxlen is None:
        maxlen = np.max(lengths)
    if maxlen.shape is not None and len(maxlen.shape) != 0:
        raise ValueError("maxlen must be scalar for sequence_mask")

    # The basic idea is to compare a range row vector of size maxlen:
    # [0, 1, 2, 3, 4]
    # to length as a matrix with 1 column: [[1], [3], [2]].
    # Because of broadcasting on both arguments this comparison results
    # in a matrix of size (len(lengths), maxlen)
    row_vector = np.arange(maxlen, dtype=maxlen.dtype)
    matrix = np.expand_dims(lengths, axis=-1).astype(maxlen.dtype)
    result = row_vector < matrix

    if dtype is None or result.dtype == dtype:
        return result
    else:
        return result.astype(dtype)
