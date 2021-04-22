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
