import logging
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import placeholder as ph
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import citlab_python_util.basic.flags as flags
from citlab_article_separation.gnn.input.input_dataset import InputGNN
from citlab_article_separation.gnn.clustering.textblock_clustering import TextblockClustering
from citlab_article_separation.gnn.io import plot_graph_clustering_and_page, save_clustering_to_page, \
    save_conf_to_json, build_thresholded_relation_graph
from citlab_python_util.io.path_util import *
from citlab_python_util.parser.xml.page.page import Page
from citlab_article_separation.gnn.input.feature_generation import discard_text_regions_and_lines as discard_regions


# General
# =======
flags.define_string('model_dir',  '', 'Checkpoint containing the exported model')
flags.define_string('eval_list',   '', '.lst-file specifying the dataset used for evaluation')
flags.define_integer('batch_size', 1, 'number of elements to be evaluated in each batch (default: %(default)s).')

# Model parameter
# ===============
flags.define_integer('num_classes',             2, 'number of classes (including garbage class)')
flags.define_integer('num_relation_components', 2, 'number of components of the associated relations')
flags.define_boolean('sample_relations',    False, 'sample relations to consider or use full graph')
flags.define_integer('sample_num_relations_to_consider', 100,
                     'number of sampled relations to be tested (half pos, half neg)')

# Visual input
# ===============
flags.define_boolean('image_input', False,
                     'use image as additional input for GNN (visual input is calculated from image and regions)')
flags.define_boolean('assign_visual_features_to_nodes', True, 'use visual node features, only if image_input is True')
flags.define_boolean('assign_visual_features_to_edges', False, 'use visual edge features, only if image_input is True')
flags.define_boolean('mvn', True, 'MVN on the image input')

# Input function
# ===============
flags.define_boolean('create_data_from_pagexml', False,
                     'generate input data on the fly from the pagexml or load it directly from json')
flags.define_choices('interaction_from_pagexml', ['fully', 'delaunay'], 'fully', str, "('fully', 'delaunay')",
                     'determines the setup of the interacting_nodes when loading from pagexml.')
flags.define_dict('input_params', {}, "dict of key=value pairs defining the input configuration")

# Confidences & Clustering
# ========================
flags.define_choices('clustering_method', ['dbscan', 'linkage', 'greedy', 'dbscan_std'], 'dbscan', str,
                     "('dbscan', 'linkage', 'greedy', 'dbscan_std')", 'clustering method to be used')
flags.define_dict('clustering_params', {}, "dict of key=value pairs defining the clustering configuration")
flags.define_string("out_dir", "", "directory to save graph confidences jsons and clustering pageXMLs. It retains the "
                                   "folder structure of the input data. Use an empty 'out_dir' for the original folder")
flags.define_boolean("only_save_conf", False, "Only save the graph confidences and skip the clustering process")

# Misc
# ====
flags.define_integer('num_p_r_thresholds', 20, 'number of thresholds used for precision-recall-curve')
flags.define_list('gpu_devices', int, 'INT', 'list of GPU indices to use. ', [])
flags.define_float('gpu_memory_fraction', 0.95, 'set between 0.1 and 1, value - 0.09 is passed to session_config, to '
                                                'take overhead in account, smaller val_batch_size may needed, '
                                                '(default: %(default)s)')
flags.define_string("debug_dir", "", "directory to save debug outputs")
flags.define_integer("batch_limiter", -1, "set to positiv value to stop validation after this number of batches")
flags.FLAGS.parse_flags()
flags.define_boolean("try_gpu", True if flags.FLAGS.gpu_devices != [] else False,
                     "try to load '<model>_gpu.pb' if possible")
flags.FLAGS.parse_flags()


class EvaluateRelation(object):
    def __init__(self):
        self._flags = flags.FLAGS
        self._params = {'num_gpus': len(self._flags.gpu_devices)}
        self._page_paths = None
        self._json_paths = None
        self._dataset = None
        self._dataset_iterator = None
        self._next_batch = None
        self._pb_path = None
        if self._flags.try_gpu:
            try:
                self._pb_path = os.path.join(get_path_from_exportdir(self._flags.model_dir, "*_gpu.pb", "cpu"))
            except IOError:
                logging.warning("Could not find gpu-model-pb-file, continue with cpu-model-pb-file")
        if not self._pb_path:
            self._pb_path = os.path.join(get_path_from_exportdir(self._flags.model_dir, "*best*.pb", "_gpu.pb"))
        logging.info(f"pb_path is {self._pb_path}")
        self._input_fn = InputGNN(self._flags)
        self._tb_clustering = TextblockClustering(self._flags)
        # Print params
        flags.print_flags()
        self._input_fn.print_params()
        self._tb_clustering.print_params()

    def _load_graph(self):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.io.gfile.GFile(self._pb_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we can use again a convenient built-in function to import a graph_def into the
        # current default Graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="",
                                producer_op_list=None)
        return graph

    def _get_placeholder(self):
        input_fn_params = self._input_fn.input_params

        ph_dict = dict()
        ph_dict['num_nodes'] = ph(tf.int32, [None], name='num_nodes')  # [batch_size]
        ph_dict['num_interacting_nodes'] = ph(tf.int32, [None], name='num_interacting_nodes')  # [batch_size]
        ph_dict['interacting_nodes'] = ph(tf.int32, [None, None, 2], name='interacting_nodes')  # [batch_size, max_num_interacting_nodes, 2]

        # add node features if present
        if 'node_feature_dim' in input_fn_params and input_fn_params["node_feature_dim"] > 0:
            # feature dim by masking
            if 'node_input_feature_mask' in input_fn_params:
                node_feature_dim = input_fn_params["node_input_feature_mask"].count(True) if \
                    input_fn_params["node_input_feature_mask"] else input_fn_params["node_feature_dim"]
            else:
                node_feature_dim = input_fn_params["node_feature_dim"]
            # [batch_size, max_num_nodes, node_feature_dim]
            ph_dict['node_features'] = ph(tf.float32, [None, None, node_feature_dim], name='node_features')

        # add edge features if present
        if 'edge_feature_dim' in input_fn_params and input_fn_params["edge_feature_dim"] > 0:
            # feature dim by masking
            if 'edge_input_feature_mask' in input_fn_params:
                edge_feature_dim = input_fn_params["edge_input_feature_mask"].count(True) if \
                    input_fn_params["edge_input_feature_mask"] else input_fn_params["edge_feature_dim"]
            else:
                edge_feature_dim = input_fn_params["edge_feature_dim"]
            # [batch_size, max_num_interacting_nodes, edge_feature_dim]
            ph_dict['edge_features'] = ph(tf.float32, [None, None, edge_feature_dim], name='edge_features')

        # add visual features
        if self._flags.image_input:
            if self._flags.assign_visual_features_to_nodes or self._flags.assign_visual_features_to_edges:
                img_channels = 1
                if 'load_mode' in input_fn_params and input_fn_params['load_mode'] == 'RGB':
                    img_channels = 3
                # [batch_size, pad_height, pad_width, channels] float
                ph_dict['image'] = ph(tf.float32, [None, None, None, img_channels], name="image")
                # [batch_size, 3] int
                ph_dict['image_shape'] = ph(tf.int32, [None, 3], name="image_shape")
                if self._flags.assign_visual_features_to_nodes:
                    # [batch_size, max_num_nodes, 2, max_num_points_visual_regions_nodes] float
                    ph_dict['visual_regions_nodes'] = ph(tf.float32, [None, None, 2, None], name="visual_regions_nodes")
                    # [batch_size, max_num_nodes] int
                    ph_dict['num_points_visual_regions_nodes'] = ph(tf.int32, [None, None],
                                                                    name="num_points_visual_regions_nodes")
                if self._flags.assign_visual_features_to_edges:
                    # [batch_size, max_num_nodes, 2, max_num_points_visual_regions_edges] float
                    ph_dict['visual_regions_edges'] = ph(tf.float32, [None, None, 2, None], name="visual_regions_edges")
                    # [batch_size, max_num_nodes] int
                    ph_dict['num_points_visual_regions_edges'] = ph(tf.int32, [None, None],
                                                                    name="num_points_visual_regions_edges")
            else:
                logging.warning(f"Image_input was set to 'True', but no visual features were assigned. Specify flags.")

        # relations for evaluation
        # [batch_size, max_num_relations, num_relation_components]
        ph_dict['relations_to_consider_belong_to_same_instance'] = \
            ph(tf.int32, [None, None, self._flags.num_relation_components],
               name='relations_to_consider_belong_to_same_instance')
        # # [batch_size]
        # ph_dict['num_relations_to_consider_belong_to_same_instance'] = \
        #     ph(tf.int32, [None], name='num_relations_to_consider_belong_to_same_instance')
        return ph_dict

    def evaluate(self):
        logging.info("Start evaluation...")
        graph = self._load_graph()
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self._flags.gpu_devices)
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=self._flags.gpu_memory_fraction - 0.09,
                                              allow_growth=False)  # - 0.09 for memory overhead
        session_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        sess = tf.compat.v1.Session(graph=graph, config=session_config)

        with sess.graph.as_default() as graph:
            # for i in [n.name for n in tf.get_default_graph().as_graph_def().node if "graph" not in n.name]:
            #     logging.debug(i)

            with tf.Graph().as_default():  # write dummy placeholder in another graph
                placeholders = self._get_placeholder()
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
                # 'relations_to_consider_belong_to_same_instance' [batch_size, max_num_relations, num_relation_components]

            self._dataset = self._input_fn.get_dataset()
            self._dataset_iterator = tf.compat.v1.data.make_one_shot_iterator(self._dataset)
            self._next_batch = self._dataset_iterator.get_next()
            with open(self._flags.eval_list, 'r') as eval_list_file:
                self._json_paths = [line.rstrip() for line in eval_list_file.readlines()]
                if not self._flags.create_data_from_pagexml:
                    self._page_paths = [get_page_from_json_path(json_path) for json_path in self._json_paths]

            output_node = graph.get_tensor_by_name("output_belong_to_same_instance:0")
            target_key = "relations_to_consider_gt"
            targets = []  # gather targets for precision/recall evaluation
            probs = []  # gather probabilities for precision/recall evaluation
            feed_dict = {}

            batch_counter = 0
            start_timer = time.time()
            while True:  # Loop until dataset is empty
                if self._flags.batch_limiter != -1 and self._flags.batch_limiter <= batch_counter:
                    logging.info(f"stop validation after {batch_counter} batches with "
                                 f"{self._flags.batch_size} samples each.")
                    break
                try:
                    page_path = self._page_paths.pop(0)
                    json_path = self._json_paths.pop(0)

                    logging.info(f"Processing... {page_path}")
                    # Skip files where json is missing (e.g. when there are less than 2 text regions)
                    if not os.path.isfile(json_path):
                        logging.warning(f"No json file found to given pageXML {page_path}. Skipping.")
                        continue

                    # get one batch (input_dict, target_dict) from generator
                    next_batch = sess.run([self._next_batch])[0]
                    batch_counter += 1
                    target = next_batch[1][target_key]
                    targets.append(target)
                    # num_relations_to_consider = next_batch[0]["num_relations_to_consider_belong_to_same_instance"]

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
                    output = sess.run(output_node, feed_dict=feed_dict)

                    # evaluate the output
                    class_probabilities = output[0, :, 1]
                    probs.append(class_probabilities)

                    # TODO: manually set class_probabilities of edges over horizontal separators to zero?!

                    if 'node_features' in placeholders:
                        node_features_node = graph.get_tensor_by_name('node_features:0')
                        node_features = feed_dict[node_features_node][0]  # assume batch_size = 1
                    if 'edge_features' in placeholders:
                        edge_features_node = graph.get_tensor_by_name('edge_features:0')
                        edge_features = feed_dict[edge_features_node][0]  # assume batch_size = 1

                    # clustering of confidence graph
                    confidences = np.reshape(class_probabilities, [node_features.shape[0], -1])

                    if self._flags.only_save_conf:
                        # save confidences
                        save_conf_to_json(confidences=confidences,
                                          page_path=page_path,
                                          save_dir=self._flags.out_dir)
                        # skip clustering
                        continue

                    self._tb_clustering.set_confs(confidences)
                    self._tb_clustering.calc(method=self._flags.clustering_method)

                    # save pageXMLs with new clusterings
                    cluster_path = save_clustering_to_page(clustering=self._tb_clustering.tb_labels,
                                                           page_path=page_path,
                                                           save_dir=self._flags.out_dir,
                                                           info=self._tb_clustering.get_info(self._flags.clustering_method))
                    # info = self._tb_clustering.get_info(self._flags.clustering_method)
                    # save_name = re.sub(r'\.xml$', '_clustering.xml', os.path.basename(os.path.relpath(page_path)))
                    # page_dir = re.sub(r'page$', 'clustering', os.path.dirname(os.path.relpath(page_path)))
                    # save_dir = self._flags.out_dir
                    # if info:
                    #     save_dir = os.path.join(save_dir, page_dir, info)
                    # else:
                    #     save_dir = os.path.join(save_dir, page_dir)
                    # cluster_path = os.path.join(save_dir, save_name)

                    # debug output
                    # TODO: add more debug images for (corrects/falses/targets/predictions etc.)
                    if self._flags.debug_dir:
                        if not os.path.isdir(self._flags.debug_dir):
                            os.makedirs(self._flags.debug_dir)

                        relations_node = graph.get_tensor_by_name('relations_to_consider_belong_to_same_instance:0')
                        relations = feed_dict[relations_node][0]  # assume batch_size = 1

                        # if 'edge_features' in placeholders:
                        #     feature_dicts = [{'separated': bool(e)} for e in edge_features[:, :1].flatten()]
                        #     graph_full = build_weighted_relation_graph(relations.tolist(),
                        #                                                class_probabilities.tolist(),
                        #                                                feature_dicts)
                        # else:
                        nx_graph = build_thresholded_relation_graph(relations, class_probabilities,
                                                                    self._tb_clustering.clustering_params["confidence_threshold"])

                        # # full confidence graph
                        # edge_colors = []
                        # for u, v, d in graph_full.edges(data='weight'):
                        #     edge_colors.append(d)
                        # plot_graph_and_page(page_path=page_path,
                        #                     graph=graph_full,
                        #                     node_features=node_features,
                        #                     save_dir=self._flags.debug_dir,
                        #                     with_edges=True,
                        #                     with_labels=True,
                        #                     desc='confidences',
                        #                     edge_color=edge_colors,
                        #                     edge_cmap=plt.get_cmap('jet'),
                        #                     edge_vmin=0.0,
                        #                     edge_vmax=1.0)
                        # # confidence histogram
                        # plot_confidence_histogram(class_probabilities, 10, page_path, self._flags.debug_dir,
                        #                           desc='conf_hist')

                        # clustered graph
                        edge_colors = []
                        for u, v, d in nx_graph.edges(data='weight'):
                            edge_colors.append(d)
                        plot_graph_clustering_and_page(graph=nx_graph,
                                                       node_features=node_features,
                                                       page_path=page_path,
                                                       cluster_path=cluster_path,
                                                       save_dir=self._flags.debug_dir,
                                                       threshold=self._tb_clustering.clustering_params["confidence_threshold"],
                                                       info=self._tb_clustering.get_info(self._flags.clustering_method),
                                                       with_edges=True,
                                                       with_labels=True,
                                                       edge_color=edge_colors,
                                                       edge_cmap=plt.get_cmap('jet'))

                # break as soon as dataset is empty
                # (IndexError for empty page_paths list, OutOfRangeError for empty tf dataset)
                except (tf.errors.OutOfRangeError, IndexError):
                    break

            # # Compute Precision, Recall, F1
            # full_targets = np.squeeze(np.concatenate(targets, axis=-1))
            # full_probs = np.squeeze(np.concatenate(probs, axis=-1))
            # prec, rec, thresholds = precision_recall_curve(full_targets, full_probs)
            # f_score = (2 * prec * rec) / (prec + rec)  # element-wise (broadcast)
            #
            # # P, R, F at relative thresholds
            # print("\n Relative Thresholds:")
            # print(f" |{'Threshold':>10}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}")
            # print(" | " + "-" * 45)
            # for j in range(self._flags.num_p_r_thresholds + 1):
            #     i = j * ((len(thresholds) - 1) // self._flags.num_p_r_thresholds)
            #     print(f" |{thresholds[i]:10f}{prec[i]:12f}{rec[i]:12f}{f_score[i]:12f}")
            #
            # # P, R, F at fixed thresholds
            # print("\n Fixed Thresholds:")
            # print(f" |{'Threshold':>10}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}")
            # print(" | " + "-" * 45)
            # step = 1 / self._flags.num_p_r_thresholds
            # j = 0
            # for i in range(len(thresholds)):
            #     if thresholds[i] >= j * step:
            #         print(f" |{thresholds[i]:10f}{prec[i]:12f}{rec[i]:12f}{f_score[i]:12f}")
            #         j += 1
            #         if j * step >= 1.0:
            #             break
            #
            # # Best F1-Score
            # i_f = np.argmax(f_score)
            # print("\n Best F1-Score:")
            # print(f" |{'Threshold':>10}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}")
            # print(" | " + "-" * 45)
            # print(f" |{thresholds[i_f]:10f}{prec[i_f]:12f}{rec[i_f]:12f}{f_score[i_f]:12f}")

            logging.info("Time: {:.2f} seconds".format(time.time() - start_timer))
        logging.info("Evaluation finished.")


if __name__ == "__main__":
    logging.getLogger().setLevel('INFO')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    eval_rel = EvaluateRelation()
    eval_rel.evaluate()
