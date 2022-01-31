import tensorflow as tf
import collections
import os
import operator
import logging


def stop_if_no_metric_improvement(eval_dir, metric_name, higher_is_better, max_steps, min_steps=0):
    """Returns True if `metric_name` in summary files contained in `eval_dir` does not improve within
    `max_steps`. Improvement is determined by `higher_is_better`. Always returns False for the first `min_steps`."""
    is_lhs_better = operator.gt if higher_is_better else operator.lt
    increase_or_decrease = 'increase' if higher_is_better else 'decrease'

    eval_results = read_eval_metrics(eval_dir)

    best_val = None
    best_val_step = None
    for step, metrics in eval_results.items():
        if step < min_steps:
            continue
        val = metrics[metric_name]
        if best_val is None or is_lhs_better(val, best_val):
            best_val = val
            best_val_step = step
        if step - best_val_step >= max_steps:
            logging.info(f"Early stopping triggered at step {step}!")
            logging.info(f'No {increase_or_decrease} in metric "{metric_name}" for {step - best_val_step} steps, '
                         f'which is greater than or equal to configured max steps ({max_steps}).')
            return True
    logging.info(f"Early stopping not triggered with metric '{metric_name}' at step {step}. Best value = {best_val} "
                 f"(at step {best_val_step}).")
    return False


def read_eval_metrics(eval_dir):
    """
    Helper to read eval metrics from eval summary files.
    :param eval_dir: directory containing summary files with eval metrics
    :return: `dict` with global steps mapping to `dict` of metric names and values
    """
    eval_metrics_dict = collections.defaultdict(dict)
    for event in yield_summaries(eval_dir):
        if not event.HasField('summary'):
            continue
        metrics = dict()
        for value in event.summary.value:
            if value.HasField('simple_value'):
                metrics[value.tag] = value.simple_value
        if metrics:
            eval_metrics_dict[event.step].update(metrics)
    return collections.OrderedDict(sorted(eval_metrics_dict.items(), key=lambda t: t[0]))


def yield_summaries(eval_dir):
    """
    Yields `tensorflow.Event` protos from event files in the eval directory.
    :param eval_dir: directory containing summary files with eval metrics
    :return: `tensorflow.Event` object read from the event files
    """
    if tf.compat.v1.gfile.Exists(eval_dir):
        for event_file in tf.compat.v1.gfile.Glob(os.path.join(eval_dir, 'events.out.tfevents.*')):
            for event in tf.compat.v1.train.summary_iterator(event_file):
                yield event
