import logging
import os
import glob
import time
import tensorflow as tf
import python_util.basic.flags as flags
from article_separation.gnn.io import copy_model
from article_separation.util.early_stopping import stop_if_no_metric_improvement
from article_separation.util.warmstart import WarmStarter


# How to train
# ============
flags.define_integer('epochs', 200, 'Epochs to train.')
flags.define_integer('samples_per_epoch', 8192,
                     'Samples to show per epoch. 0 leads to a pure validation run (given a trained checkpoint).')
flags.define_integer('batch_size', 16, 'number of elements in a training batch (default: %(default)s).'
                                       'samples between optimizer steps.')
flags.define_integer('train_accum_steps', 1,
                     'Reduce on device batchSize by gradient accumulation (default: %(default)s).'
                     'Train_batch_size is divided by this factor BUT the gradient is accumulated'
                     'this many times, until an optimization step is performed. This allows for HIGH'
                     'batchSizes even with limited memory and huge models.')
flags.define_boolean('calc_ema', False, 'Choose whether you want to use EMA weights or not,')
flags.define_float('clip_grad', 0.0, 'gradient clipping value: for positive values GLOBAL norm clipping is performed,'
                                     ' for negative values LOCAL norm clipping is performed (default: %(default)s)')
flags.define_integer('early_stopping_max_steps', 0, 'If the model-defined metric does not increase for max_steps, the '
                                                    'training is stopped early (0 deactivates early stopping)')
flags.define_integer('early_stopping_min_steps', 0, 'Early stopping is not done for the first min_steps')
flags.define_string('optimizer', 'FinalDecayOptimizer', 'the optimizer used to compute and apply gradients.')
flags.define_dict('optimizer_params', {}, "dict of key=value pairs defining the configuration of the optimizer.")

# What to train
# ==============
flags.define_string('train_list', None, '.lst-file specifying the dataset used for training')
flags.define_string('train_scopes', '', 'Change only variables in this scope during training')
flags.define_string('not_train_scopes', '', 'Explicitly exclude variables of these scopes from being trained.')

# Validation
# ==========
flags.define_string('eval_list', None, '.lst-file specifying the dataset used for validation')
flags.define_integer('eval_every_n', 1, "Evaluate/Validate every 'n' epochs")

# Checkpointing / Loading
# =======================
flags.define_string('checkpoint_dir', '', 'Checkpoint to save model information in.')
flags.define_boolean('warmstart', False, 'load pretrained model (if checkpoint_dir already exists, throws exception).')
flags.define_dict('warmstart_params', {}, 'see parameter of warmstart tf_aip.util.warmstart.WarmStarter')
flags.define_boolean('reset_global_step', False, 'resets global_step, this restarts the learning_rate decay,'
                                                 'only works with load from warmstart_dir')
flags.define_boolean('delete_event_files', True,
                     'If True all event files necessary for tensorboard visualization are deleted.')
flags.define_string('export_best', None, 'if given value is a validation metric, save best (can be multiple)')
flags.define_boolean('export_best_max', True, '(with export_best) better == {True: higher, False: lower}')
flags.define_string('profile_dir', '', 'save profile file from training process')

# Hardware
# ========
flags.define_boolean('xla', False, 'Disable in case of XLA related errors or performance issues (default: %(default)s)')
flags.define_list('gpu_devices', int, 'space seperated list of GPU indices to use. ', " ", [])
flags.define_string('dist_strategy', 'mirror',
                    'DistributionStrategy in MultiGPU scenario. mirror - MirroredStrategy, ps - ParameterServerStrategy')
flags.define_boolean('gpu_auto_tune', False, 'GPU auto tune (default: %(default)s)')
flags.define_boolean('ensure_cpu_export', True, "if True, an extra cpu compatible .pb-file is saved"
                                                "has NO effect in cpu-training/mode or if gpu-graph is cpu-compatible")
flags.define_float('gpu_memory_fraction', 1.0, 'TF1.10+ required. Set between 0.1 and 1, value - 0.09 is passed to '
                                               'session_config, to take overhead in account, smaller val_batch_size '
                                               'may needed (default: %(default)s)')


class TrainerBase(object):
    def __init__(self):
        self._flags = flags.FLAGS
        flags.print_flags()
        self._input_fn_generator = None
        self._model = None
        self._run_config = None
        self._current_epoch = 0
        self._train_collection = None
        self._train_op = None
        self._best = {}
        self._params = {'num_gpus': len(self._flags.gpu_devices),
                        'steps_per_epoch': int(self._flags.samples_per_epoch / self._flags.batch_size)}
        if self._flags.train_accum_steps > 1:
            if self._flags.train_accum_steps > self._flags.batch_size:
                logging.warning("Accumulation step number is greater than batch_size. Ignoring it.")
            else:
                self._flags.batch_size = int(self._flags.batch_size / self._flags.train_accum_steps)
                logging.info(f"Due to accumulation strategy, per call batch_size is "
                             f"reduced to {self._flags.batch_size}")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def train(self):
        # Setup warmstart
        if self._flags.warmstart:
            if os.path.exists(self._flags.checkpoint_dir):
                raise Exception(f"warmstart set and model in checkpoint_dir '{self._flags.checkpoint_dir}' already "
                                f"exists. Delete '{self._flags.checkpoint_dir}', or unset warmstart.")
            warmstarter = WarmStarter(self._flags)
            warmstarter.update_params()
            ws = warmstarter.get_warmstart_settings()
        else:
            if self._flags.reset_global_step:
                logging.warning("--reset_global_step has no effect without --warmstart_dir!")
            ws = None
        start_epoch = self.get_current_epoch_from_file() + 1
        # Setup Estimator
        self._create_run_config()
        estimator = tf.estimator.Estimator(
            model_fn=self._model.model_fn,
            model_dir=self._flags.checkpoint_dir,
            params=self._params,
            config=self._run_config,
            warm_start_from=ws)

        hooks = []
        if len(self._flags.profile_dir) > 0:
            hooks.append(
                tf.estimator.ProfilerHook(
                    output_dir=self._flags.profile_dir,
                    save_steps=10
                )
            )
        # train the model given by the estimator
        self._train(estimator, start_epoch, train_hooks=hooks)
        # always export please
        self._model.export(ckpt_dir=self._flags.checkpoint_dir, info="final")
        # export best models
        if self._flags.export_best:
            # val_lists = self._flags.val_lists if self._flags.val_lists else [self._flags.val_list]
            # for val_list in val_lists:
            val_list = self._flags.eval_list
            for key in self._flags.export_best.split(","):
                key_list = os.path.join(key, os.path.basename(val_list)[:-4])
                self._model.export(ckpt_dir=os.path.join(self._flags.checkpoint_dir, "best", key_list),
                                   info="best_" + key + "_" + os.path.basename(val_list)[:-4])

    def _train(self, estimator, start_epoch=0, train_hooks=None):
        for self._current_epoch in range(start_epoch, self._flags.epochs):
            t1 = time.time()
            if self._params['steps_per_epoch'] > 0:
                logging.info('Start Training in epoch ' + str(self._current_epoch + 1))
                estimator.train(input_fn=self._input_fn_generator.get_train_dataset,
                                steps=self._params['steps_per_epoch'],
                                hooks=train_hooks)
                self.save_current_epoch2file()
            t2 = max(time.time(), t1 + 1)
            if self._current_epoch == start_epoch or self._current_epoch == self._flags.epochs - 1 or (
                    self._current_epoch + 1) % self._flags.eval_every_n == 0:
                logging.info('Start Evaluation in epoch ' + str(self._current_epoch + 1))
                val_list = self._flags.eval_list
                eval_name = os.path.basename(val_list)[:-4]
                eval_results = estimator.evaluate(input_fn=self._input_fn_generator.get_eval_dataset,
                                                  name=eval_name)
                logging.info(f'Evaluation results after epoch {self._current_epoch + 1}:')
                # if PR-Curve is in results, this destroys the prints - substitute byte output with <bytes>
                for key, val in eval_results.items():
                    if isinstance(val, bytes):
                        eval_results[key] = "<bytes>"
                sample_per_sec = float(self._params['steps_per_epoch'] * self._flags.batch_size *
                                       self._flags.train_accum_steps) / (t2 - t1)
                logging.info(f"\t{eval_results}")
                logging.info(f"\t| time train:{t2 - t1:6.1f} | val:{time.time() - t2:6.1f} | "
                             f"TSamplePs{sample_per_sec:7.1f} |")
                if self._params['steps_per_epoch'] > 0 and self._current_epoch + 1 != float(
                        eval_results['global_step']) / self._params['steps_per_epoch']:
                    logging.warning("Current epoch does not match global_step/steps_per_epoch!, "
                                    "learn rate calculation may be wrong.")
                if self._flags.export_best:
                    for key_ in self._flags.export_best.split(","):
                        key_list = os.path.join(key_, os.path.basename(val_list)[:-4])
                        if key_ not in eval_results:
                            logging.warning(f"cannot save best model because key {key_} is not in "
                                            f"evaluation measures {eval_results.keys()}. Skip saving.")
                        else:
                            current = eval_results[key_]
                            logging.info(f"current value of '{key_}': {current} (best = "
                                         f"{self._best[key_list] if key_list in self._best else '?'})")
                            if key_list not in self._best or \
                                    (self._flags.export_best_max and current > self._best[key_list]) or \
                                    (not self._flags.export_best_max and current < self._best[key_list]):
                                # save new best model
                                logging.info(f"new best model: from "
                                             f"{self._best[key_list] if key_list in self._best else '?'} to {current} ")
                                self._best[key_list] = current
                                os.makedirs(os.path.join(self._flags.checkpoint_dir, "best", key_list),
                                            exist_ok=True)
                                copy_model(self._flags.checkpoint_dir,
                                           os.path.join(self._flags.checkpoint_dir, "best", key_list))

                # Optionally clear event files
                if self._flags.delete_event_files:
                    event_paths = glob.glob(os.path.join(estimator.model_dir, 'events.out.tfevents*'))
                    for event_path in event_paths:
                        os.remove(event_path)
                # Early stopping
                if self._flags.early_stopping_max_steps:
                    if not self._model.get_early_stopping_params():
                        raise NotImplementedError('Early stopping is activated by the trainer, but not implemented'
                                                  'by the given model.')
                    metric_name, higher_is_better = self._model.get_early_stopping_params()
                    if stop_if_no_metric_improvement(estimator.eval_dir(name=eval_name),
                                                     metric_name, higher_is_better,
                                                     self._flags.early_stopping_max_steps,
                                                     self._flags.early_stopping_min_steps):
                        break

    def _create_run_config(self):
        if self._params['num_gpus'] > 0:
            gpu_list = ','.join(str(x) for x in self._flags.gpu_devices)
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
            if self._flags.gpu_auto_tune:
                os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'
            else:
                os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
            gpu_options = tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=self._flags.gpu_memory_fraction - 0.09,
                allow_growth=False)  # - 0.09 for memory overhead
            session_config = tf.compat.v1.ConfigProto(
                gpu_options=gpu_options,
                allow_soft_placement=True)
            if self._flags.xla:
                session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            distribution_strategy = None
            if self._params['num_gpus'] > 1:
                distribution_strategy = self._get_distribution_strategy()

            self._run_config = tf.estimator.RunConfig(train_distribute=distribution_strategy,
                                                      session_config=session_config,
                                                      save_checkpoints_steps=self._params['steps_per_epoch'],
                                                      log_step_count_steps=self._params['steps_per_epoch'],
                                                      keep_checkpoint_max=2)
        else:
            self._run_config = tf.estimator.RunConfig(train_distribute=None,
                                                      save_checkpoints_steps=self._params['steps_per_epoch'],
                                                      log_step_count_steps=self._params['steps_per_epoch'],
                                                      keep_checkpoint_max=2)

    def _get_distribution_strategy(self):
        devices = []
        for idx in range(self._params['num_gpus']):
            devices.append('/gpu:' + str(idx))
        if self._flags.dist_strategy == 'mirror':
            logging.info("Using MirroredStrategy")
            strategy = tf.distribute.MirroredStrategy(devices=devices)
        elif self._flags.dist_strategy == 'cs':
            logging.info("Using CentralStorageStrategy")
            strategy = tf.distribute.experimental.CentralStorageStrategy(compute_devices=devices)
        else:
            logging.info("NKNOWN STRATEGY. Defaulting to MirroredStrategy.")
            strategy = tf.distribute.MirroredStrategy(devices=devices)
        return strategy

    def save_current_epoch2file(self):
        with open(os.path.join(self._flags.checkpoint_dir, "current_epoch.info"), "w") as f:
            f.write(str(self._current_epoch))

    def get_current_epoch_from_file(self):
        if os.path.isfile(os.path.join(self._flags.checkpoint_dir, "current_epoch.info")):
            with open(os.path.join(self._flags.checkpoint_dir, "current_epoch.info"), "r") as f:
                current_epoch = int(f.read())
        else:
            current_epoch = int(-1)
        return current_epoch
