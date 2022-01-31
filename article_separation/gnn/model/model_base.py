import copy
import datetime
import logging
import os
import tensorflow as tf
import gnn.model.graph_util.optimizer as optimizers
import utils.flags as flags
from utils.io_utils import get_export_list


class GraphBase(object):
    def __init__(self, params):
        self.graph_params = dict()
        self._flags = params['flags']

    def infer(self, inputs, is_training):
        pass

    def print_params(self):
        sorted_dict = sorted(self.graph_params.items(), key=lambda kv: kv[0])
        if len(sorted_dict) > 0:
            logging.info("graph_params:")
            for a in sorted_dict:
                logging.info(f"  {a[0]}: {a[1]}")


class ModelBase(object):
    def __init__(self, params):
        self._params = copy.deepcopy(params)
        self._params["flags"] = flags.FLAGS
        self._net_id = None
        self._model_params=dict()

        self._graph = None

        self._inputs = None
        self._targets = None

        self._graph_out = None
        self._predictions = None

        self._export_outputs = None
        self._placeholder_names = None

        self._loss = None
        self._metrics = None

        self._train_collection = None
        self._train_op = None

        self._eval_hooks = None
        self._current_epoch = 0

        self._is_training = False

        # Load optimizer
        try:
            get_optimizer = getattr(optimizers, self._params['flags'].optimizer)
            self._optimizer = get_optimizer(self._params)
            self._optimizer.update_params()
        except Exception:
            logging.warning("No Optimizer set. Maybe in LAV mode?")

    def get_graph(self):
        """
        Model specific definition of the graph
        :return:
        """

    def get_loss(self):
        """
        Model specific calculation of the loss
        :return:
        """

    def get_predictions(self):
        """
        Model specific calculation of the prediction
        :return:
        """

    def get_metrics(self):
        """
        Model specific calculation of the metrics
        :return:
        """

    def print_params(self):
        logging.info("MODEL:")
        try:
            self.get_graph().print_params()
        except AttributeError as ex:
            logging.warning("Can not call self.get_graph().print_params(). Please debug it!")
        try:
            self._optimizer.print_params()
        except AttributeError as ex:
            logging.warning("No Optimizer set. Maybe in LAV mode?")

    def get_placeholder(self):
        """
        Model specific inputs as placeholder (dict)
        e.g.:
                return {"img": tf.compat.v1.placeholder(tf.float32, [None, self._flags.image_height, None], name="inImg"),
                        "imgLen": tf.compat.v1.placeholder(tf.int32, [None], name="inSeqLen")}
        :return:
        """
        pass

    def serving_input_receiver_fn(self):
        """
        Similar to get_placeholder but puts it into a estimator expected form
        :return:
        """
        inputs = self.get_placeholder()
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    def get_output_nodes(self, has_graph=True):
        """
        Model specific output names
        e.g.:
        if has_graph:
            logits3d = tf.transpose(self._graph_out['logits'], [1, 0, 2])
            tf.identity(logits3d, name="outConfMat")  # name to grab from java
            tf.identity(self._graph_out['outLen'], name="outSeqLen")  # name to grab from java
        return "outConfMat" + "," + "outSeqLen"  # return names as comma separated string without spaces
        :return:
        """
        pass

    def get_hooks(self):
        """get Model specific hooks to 'self._eval_hooks ...
        :return:
        """
        pass

    def get_export_outputs(self):
        """
        Model specific calculation of the export outputs
        :return:
        """
        pass

    def set_tensorboard(self, mode):
        """
        Model specific tesnorboard setup
        :return:
        """
        pass

    def get_target_keys(self):
        """
        needed for lav (load_and_validate)
        names(str) of the target information which the loss function receives from the input generator during training.
        e.g.    def calc_loss(self):
                    self._targets_fin = self._ctc_label_dense_to_sparse(self._targets['tgt'], self._targets['tgtLen'])
                    ...
                in this case the return would be "tgt,tgtLen"
        :return:
        """
        pass

    def export_helper(self):
        """
        Model specific function which is run at the end of export. Purpose e.g. copy preproc etc. to export dir
        :return:
        """
        pass

    def print_evaluate(self, output_nodes_dict, target_dict):
        """is called in lav(load_and_validate) in each batch"""
        pass
        return 1, 1

    def print_evaluate_summary(self):
        """is called at end of lav(load_and_validate), can use model variables or plot something"""
        pass

    def get_early_stopping_params(self):
        """
        needed for early stopping
        return the name of the metric to evaluate and a boolean that indicates if a higher metric is better or worse
        """
        pass

    def model_fn(self, features, labels, mode, params):
        if mode == tf.estimator.ModeKeys.TRAIN:
            self._current_epoch += 1
            self._is_training = True
        else:
            self._is_training = False

        self._params = copy.deepcopy(params)
        self._params["flags"] = flags.FLAGS

        self._inputs, self._targets = features, labels
        self._graph = self.get_graph()

        with tf.compat.v1.variable_scope('graph', reuse=False, custom_getter=None):
            self._graph_out = self._graph.infer(self._inputs, mode == tf.estimator.ModeKeys.TRAIN)
            self._train_collection = self.get_train_collection()
        # If you want an EMA model...
        if self._params['flags'].calc_ema:
            logging.debug('Setup EMA and Backup weights.')
            emadecay = 0.75 ** (
                    float(self._params['flags'].batch_size) / max(self._params['flags'].samples_per_epoch, 1))
            ema = tf.train.ExponentialMovingAverage(emadecay)
            ema_op = ema.apply(self._train_collection)
            # In Case of Prediction and Eval the EMA output is used as prediction and for metric calculations
            if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
                with tf.compat.v1.variable_scope('graph', reuse=True, custom_getter=self.ema_getter(ema)):
                    self._graph_out = self._graph.infer(self._inputs, mode == tf.estimator.ModeKeys.TRAIN)

        self._predictions = self.get_predictions()

        if mode == tf.estimator.ModeKeys.TRAIN and self._current_epoch == 1:
            # Cnt Trainable Variables
            total_parameters = 0
            for variable in self._train_collection:
                if 'opaque_kernel' in variable.name:
                    logging.info('Skipping opaque_kernel for counting parameters.')
                    continue
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                logging.info(f'{variable_parameters} trainable parameter in {variable.name}')
                total_parameters += variable_parameters
            logging.info(f"Total number of Trainable Parameters: {total_parameters}")

        if mode == tf.estimator.ModeKeys.PREDICT:
            self._export_outputs = self.get_export_outputs()
            return tf.estimator.EstimatorSpec(
                tf.estimator.ModeKeys.PREDICT,
                predictions=self._predictions,
                export_outputs=self._export_outputs)

        self._loss = self.get_loss()
        self.set_tensorboard(mode)

        if mode == tf.estimator.ModeKeys.EVAL:
            self._metrics = self.get_metrics()
            self._eval_hooks = self.get_hooks()
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=self._loss, predictions=self._predictions,
                eval_metric_ops=self._metrics, evaluation_hooks=self._eval_hooks)
        if mode == tf.estimator.ModeKeys.TRAIN:
            self._train_op, learning_rate = self.get_train_op()
            tf.compat.v1.summary.scalar('LR', learning_rate)
            if self._params['flags'].calc_ema:
                self._train_op = tf.group([self._train_op, ema_op])
            # logging_hook = tf.train.LoggingTensorHook({"loss2" : self._loss}, every_n_iter=1000)
            return tf.estimator.EstimatorSpec(mode=mode, loss=self._loss, train_op=self._train_op)

    def get_train_collection(self):
        train_collection = []
        exclude_collection = []
        if self._params['flags'].train_scopes:
            for train_scope in self._params['flags'].train_scopes.split(","):
                logging.debug(f'train_scopes is {train_scope.strip()}')
                train_collection.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                                    scope=train_scope.strip()))
        else:
            train_collection = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        if self._params['flags'].not_train_scopes:
            train_collection_fin = []
            for v in train_collection:
                keepVar = True
                for not_train_scope in self._params['flags'].not_train_scopes.split(","):
                    if not_train_scope in v.name:
                        keepVar = False
                        break
                if keepVar:
                    train_collection_fin.append(v)
                else:
                    exclude_collection.append(v)
            train_collection = train_collection_fin
        logging.debug('Variables to be trained:')
        for var in train_collection:
            logging.debug(var.name)
        logging.debug('Variables explicitely excluded from training:')
        for var in exclude_collection:
            logging.debug(var.name)
        return train_collection

    def ema_getter(self, ema):
        def _ema_getter(getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            ema_var = ema.average(var)
            return ema_var if ema_var is not None else var

        return _ema_getter

    def ema_getter_export(self, shortcut=None):
        def _ema_getter(getter, name, *args, **kwargs):
            # Perform just for trainscope variables (others do not have any EMA)
            if self._params['flags'].train_scopes:
                in_train_scope = False
                for aTS in self._params['flags'].train_scopes.split(","):
                    if aTS in name:
                        in_train_scope = True
                        break
                if not in_train_scope:
                    var = getter(name, *args, **kwargs)
                    return var
            if self._params['flags'].not_train_scopes:
                for aTS in self._params['flags'].not_train_scopes.split(","):
                    if aTS in name:
                        var = getter(name, *args, **kwargs)
                        return var

            # No EMA for batch_norm moving averages...
            if '/moving_mean' in name or '/moving_variance' in name:
                var = getter(name, *args, **kwargs)
                return var
            use_short_cut = False
            if shortcut is not None:
                for aSC in shortcut:
                    if aSC in name:
                        use_short_cut = True
            if use_short_cut:
                ema_var = getter(name, *args, **kwargs)
            else:
                ema_var = getter(name + '/ExponentialMovingAverage', *args, **kwargs)
            if ema_var is None:
                var = getter(name, *args, **kwargs)
            return ema_var if ema_var is not None else var

        return _ema_getter

    def get_gradient(self, optimizer):
        gradients = optimizer.compute_gradients(self._loss, self._train_collection)
        if self._params['flags'].clip_grad > 0:
            # gradients = [(tf.clip_by_value(grad, -self._params['flags'].clip_grad, self._params['flags'].clip_grad), var)
            #              for grad, var in gradients if grad is not None]
            grads, vars = zip(*gradients)
            grads, _ = tf.clip_by_global_norm(grads, self._params['flags'].clip_grad)
            gradients = list(zip(grads, vars))
        if self._params['flags'].clip_grad < 0:
            gradients = [(tf.clip_by_norm(grad, -self._params['flags'].clip_grad), var)
                         for grad, var in gradients if grad is not None]
        return gradients

    def accum_op(self, optimizer, accum_vars):
        gradients = self.get_gradient(optimizer)
        ## Adds to each element from the list you initialized earlier with zeros its gradient
        accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gradients)]
        return gradients, tf.group(accum_ops)

    def minimize_op(self, optimizer, accum_vars):
        gradients, accum_ops = self.accum_op(optimizer, accum_vars)
        with tf.control_dependencies([accum_ops]):
            global_step = tf.compat.v1.train.get_or_create_global_step()
            minimize_op = optimizer.apply_gradients(
                [(accum_vars[i] / self._params['flags'].train_accum_steps, gv[1]) for i, gv in enumerate(gradients)],
                global_step=global_step, name="train")
        with tf.control_dependencies([minimize_op]):
            zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        return tf.group(accum_ops, minimize_op, zero_ops)

    def get_train_op(self):
        with tf.compat.v1.variable_scope("optimizer_scope"):
            optimizer, learning_rate = self._optimizer.get_opt()

            if self._params['flags'].train_accum_steps > 1:
                accum_step = tf.Variable(0, name='accum_step', trainable=False)
                # Creation of a list of variables with the same shape as the trainable ones initialized with 0s
                accum_vars = []
                for tv in self._train_collection:
                    if 'opaque_kernel' in tv.name:
                        accum_vars.append(tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False, validate_shape=False))
                    else:
                        accum_vars.append(tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False))
                cond = tf.equal(tf.math.floormod(accum_step + 1, self._params['flags'].train_accum_steps), 0)
                train_op = tf.cond(cond, lambda: self.minimize_op(optimizer, accum_vars),
                                   lambda: self.accum_op(optimizer, accum_vars)[1])
                with tf.control_dependencies([train_op]):
                    increment_global_step_op = tf.compat.v1.assign(accum_step, accum_step + 1)
                    train_op = tf.group(train_op, increment_global_step_op)
            else:
                gradients = self.get_gradient(optimizer)
                global_step = tf.compat.v1.train.get_or_create_global_step()
                minimize_op = optimizer.apply_gradients(
                    gradients, global_step=global_step, name="train")
                train_op = minimize_op
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            train_op = tf.group(train_op, update_ops)
            return train_op, learning_rate

    # If opaque_vars is NOT None we are in the GPU fix run (we create a CPU usable PB)
    def export(self, ckpt_dir, info=None, opaque_vars=None):
        logging.info("Export model as .pb")
        enforced_part_of_name = []
        shortcut = None
        if opaque_vars:
            # In GPU fix run, the ckpt loaded does not contain any cuLSTMs, hence we skip them in EMA getter and
            # directly load them. This is valid, since the helper ckpt just contains the EMA weights for the CURNN cells
            shortcut = []
            for o_var in opaque_vars:
                o_var = o_var.replace("/ExponentialMovingAverage", "")
                o_var = o_var.replace(":0", "")
                o_var = o_var.replace("cudnn_lstm/opaque_kernel", "rnn/multi_rnn_cell/cell_0/cudnn_compatible")
                shortcut.append(o_var)
                enforced_part_of_name.append(o_var)

        graph = tf.Graph()
        self._graph = self.get_graph()
        with graph.as_default():
            self._inputs = self.get_placeholder()
            self._placeholder_names = []  # save place_holder_names for later use e.g write_netconfig
            for inp in sorted(self._inputs.items()):
                if isinstance(inp[1], dict):
                    for inp_inner in inp[1].items():
                        self._placeholder_names.append(inp_inner[1].name[:-2])
                else:
                    self._placeholder_names.append(inp[1].name[:-2])
            logging.info(f'PlaceHolderNames used for export: {str(self._placeholder_names)}')
            if not self._params['flags'].calc_ema:
                with tf.compat.v1.variable_scope('graph', reuse=False, custom_getter=None):
                    self._graph_out = self._graph.infer(self._inputs, is_training=False)
            else:
                # To be on the save side we only load the EMA weights.
                enforced_part_of_name.append("/ExponentialMovingAverage")
                # If you want an EMA model you have to setup/load the EMA variables
                with tf.compat.v1.variable_scope('graph', reuse=False, custom_getter=self.ema_getter_export(shortcut)):
                    self._graph_out = self._graph.infer(self._inputs, is_training=False)

            self._train_collection = self.get_train_collection()
            output_nodes = self.get_output_nodes()

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=
                                              self._params["flags"].gpu_memory_fraction - 0.09,
                                              allow_growth=False)  # - 0.09 for memory overhead
        session_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options,
                                                  allow_soft_placement=True,
                                                  log_device_placement=False)

        if self._params["flags"].samples_per_epoch > 0 or not self._params["flags"].warmstart:
            load_checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
        else:
            load_checkpoint = tf.train.get_checkpoint_state(self._params["flags"].warmstart_params['dir'])
        logging.info(f"Load checkpoint for export: {load_checkpoint.model_checkpoint_path}")
        with tf.compat.v1.Session(graph=graph, config=session_config) as session:
            to_restore = get_export_list(enforced_part_of_name, self._train_collection)
            logging.info("Vars to restore for export:")
            if opaque_vars:
                logging.info("EMA IMPLICITLY used for opaque variables.")
            for cur_var in to_restore:
                logging.info(cur_var)

            # add '_gpu' to filename if model can only be run on gpu due to cudnnLSTM variables
            add_gpu = ""
            saver = tf.compat.v1.train.Saver(var_list=to_restore)
            saver.restore(session, load_checkpoint.model_checkpoint_path)

            perform_gpu_fix_run = False
            is_cudnn, opaque_vars_n = self._is_cudnn()
            if is_cudnn:
                logging.info("Graph contains gpu-only cudnnLSTM variables!")
                add_gpu = "_gpu"
                logging.info('Creating additional checkpoint for cuDNN-CPU-Net using variables:')
                global_vars = tf.compat.v1.global_variables()
                for var in global_vars:
                    logging.info(var.name)
                ckpt_dir_sec = ckpt_dir + '/cpu_ckpt/'
                if not os.path.isdir(ckpt_dir_sec):
                    os.mkdir(ckpt_dir_sec)
                saver = tf.compat.v1.train.Saver()
                saver.save(session, ckpt_dir_sec + 'model')
                logging.info(f'at: {ckpt_dir_sec}')
                perform_gpu_fix_run = True

            input_graph_def = session.graph.as_graph_def()
            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_nodes.split(","))

        net_info = "" if info is None else "_" + info
        self._net_id = os.path.basename(self._params['flags'].checkpoint_dir) + \
            net_info + "_" + datetime.datetime.today().strftime('%Y-%m-%d')
        if not os.path.exists(os.path.join(self._params["flags"].checkpoint_dir, "export")):
            os.makedirs(os.path.join(self._params["flags"].checkpoint_dir, "export"))
        pb_path = os.path.join(os.path.join(self._params["flags"].checkpoint_dir, "export"),
                               self._net_id + add_gpu + ".pb")
        logging.info(f"Save as: {pb_path}")
        with tf.io.gfile.GFile(pb_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        logging.info(f"{len(output_graph_def.node)} ops in the final graph.")
        self.export_helper()
        if self._params["num_gpus"] != 0 and perform_gpu_fix_run:
            temp_gpus = self._params["num_gpus"]
            self._params["num_gpus"] = 0
            self._graph = self.get_graph()
            self.export(ckpt_dir=ckpt_dir_sec, opaque_vars=opaque_vars_n)
            self._params["num_gpus"] = temp_gpus
            self._graph = self.get_graph()

    def _is_cudnn(self):
        is_cudnn = False
        opaque_vars = []
        for var in tf.compat.v1.global_variables():
            if 'opaque_kernel' in var.name:
                is_cudnn = True
                opaque_vars.append(var.name)
        return is_cudnn, opaque_vars

    def batch2debug_dir(self, input_batch=None, image_fn=None):
        """only called in lav_xx with 'debug_dir' set"""
        pass

    def outnodes2debug_dir(self, output_dict=None, image_fn=None):
        """only called in lav_xx with 'debug_dir' set"""
        pass
