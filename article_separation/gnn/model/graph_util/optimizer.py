import logging
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from utils.flags import update_params


class DecayOptimizer(object):
    def __init__(self, params):
        super(DecayOptimizer, self).__init__()
        self._name = "DecayOptimizer"
        self._params = params
        self._flags = params['flags']
        self._optimizer_params = dict()
        # Default params for the decay scenario
        self._optimizer_params["optimizer"] = 'adam'  # learning rate decay, 1.0 means no decay
        self._optimizer_params["learning_rate"] = 0.001  # initial learning rate
        self._optimizer_params["lr_decay_rate"] = 0.99  # learning rate decay, 1.0 means no decay
        self._optimizer_params["learning_circle"] = 3  # number of epochs with same learning rate

    def update_params(self):
        # Updating of the default params if provided via flags as a dict
        self._optimizer_params = update_params(self._optimizer_params, self._flags.optimizer_params, "Optimizer")

    def print_params(self):
        logging.info(f"optimizer_params for {self._name}:")
        sorted_dict = sorted(self._optimizer_params.items(), key=lambda kv: kv[0])
        for a in sorted_dict:
            logging.info(f"  {a[0]}: {a[1]}")

    def _get_lr(self):
        global_step = tf.math.floor(
            tf.cast(tf.compat.v1.train.get_or_create_global_step(), dtype=tf.float32) / self._params['steps_per_epoch'])
        lr = tf.compat.v1.train.exponential_decay(self._optimizer_params["learning_rate"],
                                                  global_step,
                                                  self._optimizer_params["learning_circle"],
                                                  self._optimizer_params["lr_decay_rate"],
                                                  True,
                                                  "LearningRate")
        return lr

    def get_opt(self):
        lr = self._get_lr()
        opt = None
        if self._optimizer_params["optimizer"] == 'nadam':
            opt = tf.contrib.opt.NadamOptimizer(lr)
        if self._optimizer_params["optimizer"] == 'adam':
            opt = tf.compat.v1.train.AdamOptimizer(lr)
        if self._optimizer_params["optimizer"] == 'rmsprop':
            opt = tf.compat.v1.train.RMSPropOptimizer(lr)
        if self._optimizer_params["optimizer"] == 'sgd':
            opt = tf.compat.v1.train.GradientDescentOptimizer(lr)
        return opt, lr


class FinalDecayOptimizer(DecayOptimizer):
    def __init__(self, params):
        super(FinalDecayOptimizer, self).__init__(params)
        self._name = "FinalDecayOptimizer"
        self._params = params
        self._flags = params['flags']
        # Default params for the decay scenario
        self._optimizer_params["final_epochs"] = 50  # number epochs with reducing learning rate
        self._optimizer_params["decay_fraction"] = 0.1  # reduce to this fraction of LR

    def _get_lr(self):
        global_step = tf.math.floor(
            tf.cast(tf.compat.v1.train.get_or_create_global_step(), dtype=tf.float32) / self._params['steps_per_epoch'])
        lr = cosine_decay(self._optimizer_params["learning_rate"],
                          global_step,
                          self._optimizer_params["learning_circle"],
                          self._optimizer_params["lr_decay_rate"],
                          self._optimizer_params["decay_fraction"],
                          self._flags.epochs,
                          self._optimizer_params["final_epochs"],
                          name="LearningRate")
        return lr


class WarmupFinalDecayOptimizer(FinalDecayOptimizer):
    def __init__(self, params):
        super(WarmupFinalDecayOptimizer, self).__init__(params)
        self._name = "WarmupFinalDecayOptimizer"
        self._params = params
        self._flags = params['flags']
        # Default params for the decay scenario
        self._optimizer_params["warmup_epochs"] = 10  # number epochs with linear increasing
        # starts with an initial LR of LR/warmup_factor, to reach peak (LR) after warmup_epochs
        self._optimizer_params["warmup_factor"] = 10

    def _get_lr(self):
        global_step = tf.math.floor(
            tf.cast(tf.compat.v1.train.get_or_create_global_step(), dtype=tf.float32) / self._params['steps_per_epoch'])
        lr = warmup_cosine_decay(self._optimizer_params["learning_rate"], global_step,
                                 self._optimizer_params["learning_circle"], self._optimizer_params["lr_decay_rate"],
                                 self._optimizer_params["decay_fraction"],
                                 self._flags.epochs, self._optimizer_params["final_epochs"],
                                 self._optimizer_params["warmup_epochs"], self._optimizer_params["warmup_factor"],
                                 name="LearningRate")
        return lr


def cosine_decay(learn_rate,  # learning rate
                 epoch,  # epoch
                 batch,  # batch epoch
                 decay,  # decay
                 alpha,  # alpha
                 epochs,
                 final_epochs,  # finalepoch
                 delay=0,
                 name=None):
    with ops.name_scope(name, "LR_Finetune", [learn_rate, epoch]) as name:
        # learning_rate = ops.convert_to_tensor(
        #     learning_rate, name="initial_learning_rate")
        learn_rate = ops.convert_to_tensor(
            learn_rate, name="initial_learning_rate")
        dtype = tf.float32
        learn_rate = math_ops.cast(learn_rate, dtype)
        batch = math_ops.cast(batch, dtype)
        final_epochs = math_ops.cast(final_epochs, dtype)
        alpha = math_ops.cast(alpha, dtype)
        decay = math_ops.cast(decay, dtype)
        epoch = math_ops.cast(epoch, dtype)
        completed_fraction = (epoch - delay) / batch
        lam = control_flow_ops.cond(
            math_ops.less_equal(epoch, delay),
            lambda: learn_rate,
            lambda: learn_rate * (decay ** math_ops.floor(completed_fraction)))
        return control_flow_ops.cond(
            math_ops.less_equal(epoch, epochs - final_epochs),
            lambda: lam,
            lambda: lam * (alpha + (1 - alpha) * (0.5 + 0.5 * math_ops.cos(
                (epoch - epochs + final_epochs) / final_epochs * 3.14159))))


def warmup_cosine_decay(learn_rate,  # learning rate
                        epoch,  # epoch
                        batch,  # batch epoch
                        decay,  # decay
                        alpha,  # alpha
                        epochs,
                        final_epochs,  # finalepoch
                        warmup_epochs,
                        warmup_factor,
                        name=None):
    """
    piecewise defined function:

    from 0 to warmup_epoch: linear increas from learningrate to warmup_factor*learningrate

    from warmup_epoch to epochs - final_epochs: decay using alpha and learning circle

    from epochs - final_epochs to end: cosine cooldown like in adam final/cosine_decay
    """

    start = learn_rate / warmup_factor
    peak = learn_rate
    learn_rate = control_flow_ops.cond(
        math_ops.less(epoch, warmup_epochs),
        lambda: start + (peak - start) / warmup_epochs * epoch,
        lambda: peak)

    return cosine_decay(learn_rate=learn_rate, epoch=epoch,
                        batch=batch,
                        decay=decay,
                        alpha=alpha,
                        epochs=epochs,
                        final_epochs=final_epochs,
                        delay=warmup_epochs,
                        name=name)
