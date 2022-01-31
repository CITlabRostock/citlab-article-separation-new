import os
import logging
import warnings
import time
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # set tf log_level to warning(2), default: info(1)
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'  # no tune necessary, short running time
import tensorflow as tf
import utils.flags as flags
from gnn.input.input_dataset import InputGNN
import gnn.model.model_relation as models
from utils.io_utils import get_path_from_exportdir, load_graph

# General
# =======
flags.define_string('model_dir', '', 'Checkpoint containing the exported model')
flags.define_string('model_type', 'ModelRelation', "Type of model (currently only 'ModelRelation')")
flags.define_string('eval_list', '', '.lst-file specifying the dataset used for validation')

# Model parameter
# ===============
flags.define_integer('num_classes',             2, 'number of classes (including garbage class)')
flags.define_integer('num_relation_components', 2, 'number of components of the associated relations')
flags.define_integer('sample_num_relations_to_consider', 100,
                     'number of sampled relations to be tested (half positive, half negative)')
flags.define_boolean('sample_relations', False, 'sample relations to consider or use full graph')

# Visual features
# ===============
flags.define_boolean('image_input', False, 'use image as additional input for GNN (visual features are '
                                           'calculated from image and regions)')
flags.define_boolean('assign_visual_features_to_nodes', True, 'use visual node features')
flags.define_boolean('assign_visual_features_to_edges', False, 'use visual edge features')
flags.define_string('backbone', 'ARU_v1', 'Backbone graph to use.')
flags.define_boolean('mvn', True, 'MVN on the input.')
flags.define_dict('graph_backbone_params', {}, "key=value pairs defining the configuration of the backbone."
                                               "Backbone parametrization")
# E.g. train script: --feature_map_generation_params from_layer=[Mixed_5d,,Mixed_6e,Mixed_7c] layer_depth=[-1,128,-1,-1]
flags.define_dict('feature_map_generation_params',
                  {'layer_depth': [-1, -1, -1]},
                  "key=value pairs defining the configuration of the feature map generation."
                  "FeatureMap Generator parametrization, see main graph, e.g. model_fn.model_fn_objdet.graphs.main.SSD")

# Input function
# ===============
flags.define_dict('input_params', {}, "dict of key=value pairs defining the configuration of the input.")

# Misc
# ====
flags.define_integer('num_p_r_thresholds', 20, 'number of thresholds used for precision-recall-curve')
flags.define_list('gpu_devices', int, 'INT', 'list of GPU indices to use. ', [])
flags.define_float('gpu_memory_fraction', 0.95, 'set between 0.1 and 1, value - 0.09 is passed to session_config, to '
                                                'take overhead in account, smaller val_batch_size may needed, '
                                                '(default: %(default)s)')
flags.define_integer("batch_limiter", -1, "set to positiv value to stop validation after this number of batches")
flags.FLAGS.parse_flags()
flags.define_boolean("try_gpu", True if flags.FLAGS.gpu_devices != [] else False,
                     "try to load '<model>_gpu.pb' if possible")
flags.FLAGS.parse_flags()


class LavGNN(object):
    def __init__(self):
        self._flags = flags.FLAGS
        flags.print_flags()
        self._params = {'num_gpus': len(self._flags.gpu_devices)}
        self._val_dataset = None
        self._val_dataset_iterator = None
        self._next_batch = None
        self._pb_path = None
        if self._flags.try_gpu:
            try:
                self._pb_path = os.path.join(get_path_from_exportdir(self._flags.model_dir,
                                                                     pattern="*_gpu.pb",
                                                                     not_pattern=None))
            except IOError:
                logging.warning("Could not find gpu-model-pb-file, continue with cpu-model-pb-file.")
        if not self._pb_path:
            self._pb_path = os.path.join(get_path_from_exportdir(self._flags.model_dir,
                                                                 pattern="*best*.pb",
                                                                 not_pattern="_gpu.pb"))
        logging.info(f"Using pb_path: {self._pb_path}")
        self._input_fn_generator = InputGNN(self._flags)
        self._model = getattr(models, self._flags.model_type)(self._params)

    def evaluate(self):
        logging.info("Start evaluation...")
        graph = load_graph(self._pb_path)
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self._flags.gpu_devices)
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=self._flags.gpu_memory_fraction - 0.09,
                                              allow_growth=False)  # - 0.09 for memory overhead
        session_config = tf.compat.v1.ConfigProto(
            gpu_options=gpu_options,
            allow_soft_placement=True)
        sess = tf.compat.v1.Session(graph=graph, config=session_config)

        with sess.graph.as_default() as graph:
            with tf.Graph().as_default():  # write dummy placeholder in another graph
                placeholders = self._model.get_placeholder()
            # Placeholders:
            # 'num_nodes'             [batch_size]
            # 'num_interacting_nodes' [batch_size]
            # 'interacting_nodes'     [batch_size, max_num_interacting_nodes, 2]
            # 'node_features'         [batch_size, max_num_nodes, node_feature_dim]
            # 'edge_features'         [batch_size, max_num_interacting_nodes, edge_feature_dim]
            # 'image'                 [batch_size, pad_height, pad_width, channels]
            # 'image_shape'           [batch_size, 3]
            # 'visual_regions_nodes'  [batch_size, max_num_nodes, 2, max_num_points_visual_regions_nodes]
            # 'num_points_visual_regions_nodes' [batch_size, max_num_nodes]
            # 'visual_regions_edges'  [batch_size, max_num_nodes, 2, max_num_points_visual_regions_edges]
            # 'num_points_visual_regions_edges' [batch_size, max_num_nodes]
            # 'relations_to_consider' = dict()
            # for each classifier_name:
            #   'relations_to_consider[classifier_name]' = 'relations_to_consider_' + classifier_name
            #   [batch_size, max_num_relations, num_relation_components]
                if not self._flags.assign_visual_features_to_nodes:
                    if 'visual_regions_nodes' in placeholders:
                        del placeholders['visual_regions_nodes']
                    if 'num_points_visual_regions_nodes' in placeholders:
                        del placeholders['num_points_visual_regions_nodes']
                if not self._flags.assign_visual_features_to_edges:
                    if 'visual_regions_edges' in placeholders:
                        del placeholders['visual_regions_edges']
                    if 'num_points_visual_regions_edges' in placeholders:
                        del placeholders['num_points_visual_regions_edges']
                if not self._flags.assign_visual_features_to_nodes and not self._flags.assign_visual_features_to_edges:
                    if 'image' in placeholders:
                        del placeholders['image']
                    if 'image_shape' in placeholders:
                        del placeholders['image_shape']

            self._val_dataset = self._input_fn_generator.get_eval_dataset()
            self._val_dataset_iterator = tf.compat.v1.data.make_one_shot_iterator(self._val_dataset)
            self._next_batch = self._val_dataset_iterator.get_next()

            output_nodes_names = self._model.get_output_nodes(has_graph=False).split(",")
            output_nodes = [graph.get_tensor_by_name(x + ":0") for x in output_nodes_names]
            output_nodes_dict = {}

            target_keys = self._model.get_target_keys().split(",")
            target_dict = {}
            feed_dict = {}

            targets = []
            probs = []

            batch_counter = 0
            start_timer = time.time()
            while True:  # Loop until val_dataset is empty
                if self._flags.batch_limiter != -1 and self._flags.batch_limiter <= batch_counter:
                    logging.info(f"Stop validation after {batch_counter} batches with")
                    break
                try:
                    # get one batch (input_dict, target_dict) from generator
                    next_batch = sess.run([self._next_batch])[0]
                    batch_counter += 1

                    # assign placeholder_dict to feed_dict
                    for key in placeholders:
                        if type(placeholders[key]) == dict:
                            for i in placeholders[key]:
                                input_name = graph.get_tensor_by_name(placeholders[key][i].name)
                                feed_dict[input_name] = next_batch[0][key][i]
                        else:
                            input_name = graph.get_tensor_by_name(placeholders[key].name)
                            feed_dict[input_name] = next_batch[0][key]

                    # run model with feed_dict
                    output_nodes_res = sess.run(output_nodes, feed_dict=feed_dict)
                    # contains a list of output tensors in order as in the return string of model.get_output_nodes

                    # assign targets and model outputs to dicts for the models print_evaluate function
                    for key in target_keys:
                        target_dict[key] = next_batch[1][key]
                    for key, value in zip(output_nodes_names, output_nodes_res):
                        output_nodes_dict[key] = value

                    target = target_dict['relations_to_consider_gt']
                    output = output_nodes_dict['output_belong_to_same_instance']
                    targets.append(target)
                    probs.append(output[:, :, -1])

                # break as soon as val_dataset is empty
                except tf.errors.OutOfRangeError:
                    break

            # Compute Precision, Recall, F1
            full_targets = np.squeeze(np.concatenate(targets, axis=-1))
            full_probs = np.squeeze(np.concatenate(probs, axis=-1))
            prec, rec, thresholds = precision_recall_curve(full_targets, full_probs)
            f_score = (2 * prec * rec) / (prec + rec)  # element-wise (broadcast)
            f_score[np.isnan(f_score)] = 0  # remove NaN (0.0/0.0)

            # P, R, F at relative thresholds
            logging.info("Relative Thresholds:")
            logging.info(f" |{'Threshold':>10}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}")
            logging.info(" | " + "-" * 45)
            for j in range(self._flags.num_p_r_thresholds + 1):
                i = j * ((len(thresholds) - 1) // self._flags.num_p_r_thresholds)
                logging.info(f" |{thresholds[i]:10f}{prec[i]:12f}{rec[i]:12f}{f_score[i]:12f}")

            # P, R, F at fixed thresholds
            logging.info("Fixed Thresholds:")
            logging.info(f" |{'Threshold':>10}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}")
            logging.info(" | " + "-" * 45)
            step = 1 / self._flags.num_p_r_thresholds
            j = 0
            for i in range(len(thresholds)):
                if thresholds[i] >= j * step:
                    logging.info(f" |{thresholds[i]:10f}{prec[i]:12f}{rec[i]:12f}{f_score[i]:12f}")
                    j += 1
                    if j * step >= 1.0:
                        break

            # Best F1-Score
            i_f = np.argmax(f_score)
            logging.info("Best F1-Score:")
            logging.info(f" |{'Threshold':>10}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}")
            logging.info(" | " + "-" * 45)
            logging.info(f" |{thresholds[i_f]:10f}{prec[i_f]:12f}{rec[i_f]:12f}{f_score[i_f]:12f}")

            # Compute AUC-ROC
            auc_roc = roc_auc_score(full_targets, full_probs)
            logging.info(f"AUC-ROC: {auc_roc:12f}")

            # Compute Accuracy
            acc = accuracy_score(full_targets, full_probs > 0.5)
            logging.info(f"Accuracy: {acc:12f}")

            logging.info(f"Time: {time.time() - start_timer:.2f} seconds")
            self._model.print_evaluate_summary()
        logging.info("Evaluation finished.")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')  # for TF "Converting sparse IndexedSlices to a dense Tensor of unknown shape"
    logging.getLogger().setLevel('INFO')
    tf.get_logger().propagate = False
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.info("Running Evaluation.")
    eval_rel = LavGNN()
    eval_rel.evaluate()
