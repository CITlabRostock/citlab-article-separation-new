import logging
import re
import tensorflow as tf
from tensorflow.python.training import checkpoint_utils
from utils.flags import update_params


class WarmStarter(object):
    def __init__(self, flags):
        super(WarmStarter, self).__init__()
        self._name = "WarmStarter"
        self._flags = flags
        self._starter_params = dict()
        # Directory with checkpoints file or path to checkpoint.
        self._starter_params["dir"] = ""
        # exclude weights to load (e.g. '.*/logit/.*' or 'graph/block[34].*', regex has to match the full weight name.
        self._starter_params["exclude"] = None
        # if true, for weights set EMA-weights (if available). Otherwise [EMA]-weights are mapped to [EMA]-weights
        self._starter_params["w_as_ema"] = False
        # if true, no ema-weights are set (also, if calc_ema = true)
        self._starter_params["ignore_ema"] = False
        # map old prefix to new prefix - so graph/<old_graph>:graph/<new_graph>
        self._starter_params["map"] = None

    def update_params(self):
        # Updating of the default params if provided via flags as a dict
        self._starter_params = update_params(self._starter_params, self._flags.warmstart_params, "warmstart")

    def get_warmstart_settings(self):
        mapdict = {} if \
            not self._starter_params["map"] or len(self._starter_params["map"]) == 0 \
            else {self._starter_params["map"].split(":")[0]: self._starter_params["map"].split(":")[1]}
        return get_warmstart_from_ckpt(self._starter_params["dir"],
                                       calc_ema=self._flags.calc_ema,
                                       reset_global_steps=self._flags.reset_global_step,
                                       re_exclude=self._starter_params["exclude"],
                                       warmstart_ema=self._starter_params["w_as_ema"],
                                       ignore_ema=self._starter_params["ignore_ema"],
                                       prefix_old_to_new=mapdict)


def get_warmstart_from_ckpt(warmstart_dir,
                            re_exclude: str = None,
                            prefix_old_to_new: dict = None,
                            calc_ema=False,
                            warmstart_ema=True,
                            ignore_ema=False,
                            reset_global_steps=False):
    logging.info(f"Warmstart from: {warmstart_dir}")
    cpk_vars = checkpoint_utils.list_variables(warmstart_dir)
    exclude = ["applyGradients", "backupVariables", "Adam", "RMSProp", "beta1_power", "beta2_power", "optimizer_scope"]
    if reset_global_steps:
        logging.info("Reset global steps to 0")
        exclude.append('global_step')
    var_name_to_prev_var_name = {}
    for v in cpk_vars:
        if all([excl not in v[0] for excl in exclude]) and not (re_exclude and re.fullmatch(re_exclude, v[0])):
            var_name_to_prev_var_name[v[0]] = v[0]
        else:
            logging.info(f"exclude: {v[0]}")
    if prefix_old_to_new:
        for old, new in prefix_old_to_new.items():
            var_name_to_prev_var_name = {new + k[len(old):] if k[:len(old)] == old else k: v for k, v in
                                         var_name_to_prev_var_name.items()}
            logging.info(f"map variables of prefix '{old}' to new variables '{new}'")
    map_dft = {k: v for k, v in var_name_to_prev_var_name.items() if not k.endswith('/ExponentialMovingAverage')}
    map_ema = {k: v for k, v in var_name_to_prev_var_name.items() if k.endswith('/ExponentialMovingAverage')}
    if warmstart_ema:
        map_dft = {k: v + '/ExponentialMovingAverage' if v + '/ExponentialMovingAverage' in map_ema.values() else v for
                   k, v in map_dft.items()}

    # expand each weight as ema
    map_ema.update({k + '/ExponentialMovingAverage': v for k, v in map_dft.items() if
                    "batchnorm/moving_variance" not in k and
                    "batchnorm/moving_mean" not in k and
                    k + '/ExponentialMovingAverage' not in map_ema.keys()})

    # if calc_ema = true and ignore_ema = false: add ema-variables to final map
    logging.info(f"ignore_ema = {ignore_ema} and calc_ema = {calc_ema}")
    if calc_ema and not ignore_ema:
        logging.info("add emas to map")
        map_dft.update({k: v for k, v in map_ema.items() if k not in map_dft.keys()})

    if 'global_step/ExponentialMovingAverage' in map_dft:
        del map_dft['global_step/ExponentialMovingAverage']

    for k, v in map_dft.items():
        logging.info(f"set weights from '{v}' to '{k}'")

    if any(['culstm' in k and k.endswith('ExponentialMovingAverage') for k in map_dft.keys()]):
        logging.info("at least one EMA-weight is used for culstm -"
                     " maybe results in an error. Use 'ignore_ema'=True to avoid this.")

    var_to_load = [e + "[^/]" for e in map_dft.keys()]
    return tf.estimator.WarmStartSettings(ckpt_to_initialize_from=warmstart_dir,
                                          vars_to_warm_start=var_to_load,
                                          var_name_to_prev_var_name=map_dft)
