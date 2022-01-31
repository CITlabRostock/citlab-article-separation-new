import tensorflow as tf
from tensorflow.python.ops import variable_scope


def f_measure(precision, recall):
    """ Computes the F1-score for given precision and recall values.

    :param precision: the precision value
    :type precision: float
    :param recall: the recall value
    :type recall: float
    :return: F1-score (or 0.0 if both precision and recall are 0.0)
    """

    if precision == 0 and recall == 0:
        return 0.0
    else:
        return 2.0 * precision * recall / (precision + recall)


def f1_score(precision, recall, name=None):
    """
    Computes F1-Score based on precision and recall values. Makes sure to check for division by zero.
    :param precision: A real `Tensor`
    :param recall: A real `Tensor`
    :param name: Optional name for the variable_scope
    :return: F1-Score and identity update-op
    """
    with variable_scope.variable_scope(name, 'f1-score', (precision, recall)):
        f1 = (2 * precision * recall) / (precision + recall)
        # check for NaN
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        # we are working with scalar values and don't need to update any variables
        f1_update = tf.identity(f1)
    return f1, f1_update
