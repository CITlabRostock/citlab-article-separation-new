import logging
import time
import os
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from tensorflow.compat.v1 import placeholder as ph
import python_util.basic.flags as flags
from python_util.io.path_util import get_path_from_exportdir, get_page_from_json_path
from python_util.basic.misc import split_list
from python_util.parser.xml.page.page import Page
from article_separation.gnn.input.input_dataset import InputGNN
from article_separation.gnn.input.feature_generation import is_aligned_horizontally_separated, is_aligned_heading_separated
from article_separation.gnn.clustering.textblock_clustering import TextblockClustering
from article_separation.gnn.io import save_clustering_to_page, save_conf_to_json, load_graph

# General
# =======
flags.define_string('model_dir', '', 'Checkpoint containing the exported model')
flags.define_string('eval_list', '', '.lst-file specifying the dataset used for evaluation')
flags.define_integer('batch_size', 1, 'number of elements to be evaluated in each batch (default: %(default)s).')

# Model parameter
# ===============
flags.define_integer('num_classes', 2, 'number of classes (including garbage class)')
flags.define_integer('num_relation_components', 2, 'number of components of the associated relations')
flags.define_integer('sample_num_relations_to_consider', 100,
                     'number of sampled relations to be tested (half positive, half negative)')
flags.define_boolean('sample_relations', False, 'sample relations to consider or use full graph')

# Visual input
# ===============
flags.define_boolean('image_input', False,
                     'use image as additional input for GNN (visual input is calculated from image and regions)')
flags.define_boolean('assign_visual_features_to_nodes', True, 'use visual node features, only if image_input is True')
flags.define_boolean('assign_visual_features_to_edges', False, 'use visual edge features, only if image_input is True')
flags.define_boolean('mvn', True, 'MVN on the image input')

# Input function
# ===============
flags.define_dict('input_params', {}, "dict of key=value pairs defining the input configuration")

# Confidences & Clustering
# ========================
flags.define_choices('clustering_method', ['dbscan', 'linkage', 'greedy', 'dbscan_std'], 'dbscan', str,
                     "('dbscan', 'linkage', 'greedy', 'dbscan_std')", 'clustering method to be used')
flags.define_boolean('mask_horizontally_separated_confs', False,
                     'set confidences of edges over horizontal separators, '
                     'whose nodes are also vertically and horizontally aligned, to zero.')
flags.define_boolean('mask_heading_separated_confs', False,
                     'set confidences of edges from regions (upper) to headings (lower), '
                     'whose nodes are also vertically and horizontally aligned, to zero.')
flags.define_dict('clustering_params', {}, "dict of key=value pairs defining the clustering configuration. Have a look "
                                           "at the __init__ method of the 'TextblockClustering' class for details. ")
flags.define_string("out_dir", "", "directory to save graph confidences jsons and clustering pageXMLs. It retains the "
                                   "folder structure of the input data. Use an empty 'out_dir' for the original folder")
flags.define_choices('save_conf', ['no_conf', 'with_conf', 'only_conf'], 'no_conf', str,
                     "('no_conf', 'with_conf', 'only_conf')", 'handles the saving of the graph confidences.')

# Misc
# ====
flags.define_integer('num_workers', 1, 'number of partitions to create from original list file and to compute in '
                                       'parallel. Only works when no external jsons are used.')
flags.define_list('gpu_devices', int, 'INT', 'list of GPU indices to use. ', [])
flags.define_float('gpu_memory_fraction', 0.95, 'set between 0.1 and 1, value - 0.09 is passed to session_config, to '
                                                'take overhead in account, smaller val_batch_size may needed, '
                                                '(default: %(default)s)')
flags.define_integer("batch_limiter", -1, "set to positive value to stop validation after this number of batches")
flags.FLAGS.parse_flags()
flags.define_boolean("try_gpu", True if flags.FLAGS.gpu_devices != [] else False,
                     "try to load '<model>_gpu.pb' if possible")
flags.FLAGS.parse_flags()
FLAGS = flags.FLAGS


def get_placeholder(input_fn_params):
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
    ph_dict = dict()
    ph_dict['num_nodes'] = ph(tf.int32, [None], name='num_nodes')  # [batch_size]
    ph_dict['num_interacting_nodes'] = ph(tf.int32, [None], name='num_interacting_nodes')  # [batch_size]
    ph_dict['interacting_nodes'] = ph(tf.int32, [None, None, 2],
                                      name='interacting_nodes')  # [batch_size, max_num_interacting_nodes, 2]

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
    if FLAGS.image_input:
        if FLAGS.assign_visual_features_to_nodes or FLAGS.assign_visual_features_to_edges:
            img_channels = 1
            if 'load_mode' in input_fn_params and input_fn_params['load_mode'] == 'RGB':
                img_channels = 3
            # [batch_size, pad_height, pad_width, channels] float
            ph_dict['image'] = ph(tf.float32, [None, None, None, img_channels], name="image")
            # [batch_size, 3] int
            ph_dict['image_shape'] = ph(tf.int32, [None, 3], name="image_shape")
            if FLAGS.assign_visual_features_to_nodes:
                # [batch_size, max_num_nodes, 2, max_num_points_visual_regions_nodes] float
                ph_dict['visual_regions_nodes'] = ph(tf.float32, [None, None, 2, None], name="visual_regions_nodes")
                # [batch_size, max_num_nodes] int
                ph_dict['num_points_visual_regions_nodes'] = ph(tf.int32, [None, None],
                                                                name="num_points_visual_regions_nodes")
            if FLAGS.assign_visual_features_to_edges:
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
        ph(tf.int32, [None, None, FLAGS.num_relation_components],
           name='relations_to_consider_belong_to_same_instance')
    return ph_dict


def mask_horizontally_separated_confs(confs, page_path):
    timer = time.time()
    # get page information
    page = Page(page_path)
    regions = page.get_regions()
    if FLAGS.mask_horizontally_separated_confs and 'SeparatorRegion' not in regions:
        logging.warning(f"No separators found for confidence masking.")
        return confs
    text_regions = regions['TextRegion']
    separator_regions = regions['SeparatorRegion']
    num_text_regions = len(text_regions)
    # compute mask
    masked = np.ones_like(confs, dtype=np.int32)
    for i in range(num_text_regions):
        for j in range(i + 1, num_text_regions):
            tr_i = text_regions[i]
            tr_j = text_regions[j]
            # mask edges over text regions which are vertically AND horizontally aligned (i.e. same column)
            # and where the lower text region is a heading
            if FLAGS.mask_heading_separated_confs:
                if is_aligned_heading_separated(tr_i, tr_j):
                    logging.debug(f"Pair ({i}, {j}) separated by heading. "
                                  f"Previous confs: ({i}, {j}) = {confs[i, j]:.4f}, ({j}, {i}) = {confs[j, i]:.4f}")
                    masked[i, j] = 0
                    masked[j, i] = 0
                    continue
            # mask edges over text regions which are vertically AND horizontally aligned (i.e. same column)
            # and separated by a horizontal separator region
            if FLAGS.mask_horizontally_separated_confs:
                if is_aligned_horizontally_separated(tr_i, tr_j, separator_regions):
                    logging.debug(f"Pair ({i}, {j}) horizontally separated. "
                                  f"Previous confs: ({i}, {j}) = {confs[i, j]:.4f}, ({j}, {i}) = {confs[j, i]:.4f}")
                    masked[i, j] = 0
                    masked[j, i] = 0
    logging.info(f"Time for masking horizontally separated confidences = {time.time() - timer}.")
    return masked * confs


def gnn_clustering(json_paths):
    # Get pb path
    pb_path = None
    if os.path.isfile(FLAGS.model_dir):
        if not os.path.splitext(os.path.basename(FLAGS.model_dir))[1] == ".pb":
            raise IOError(f"Given model path {FLAGS.model_dir} is not a .pb")
        pb_path = FLAGS.model_dir
    else:
        if FLAGS.try_gpu:
            try:
                pb_path = os.path.join(get_path_from_exportdir(FLAGS.model_dir, "*_gpu.pb", "cpu"))
            except IOError:
                logging.warning("Could not find gpu-model-pb-file, continue with cpu-model-pb-file")
        if not pb_path:
            pb_path = os.path.join(get_path_from_exportdir(FLAGS.model_dir, "*best*.pb", "_gpu.pb"))
    logging.info(f"pb_path is {pb_path}")

    # Build input function
    input_fn = InputGNN(FLAGS)

    # Get corresponding pageXML files
    page_paths = [get_page_from_json_path(json_path) for json_path in json_paths]

    # Build Textblock clustering
    tb_clustering = TextblockClustering(FLAGS)

    # Load graph
    graph = load_graph(pb_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in FLAGS.gpu_devices)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction - 0.09,
                                          allow_growth=False)  # - 0.09 for memory overhead
    session_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.compat.v1.Session(graph=graph, config=session_config)

    with sess.graph.as_default() as graph:
        with tf.Graph().as_default():  # write dummy placeholder in another graph
            placeholders = get_placeholder(input_fn.input_params)

        # Build dataset iterator
        dataset = input_fn.get_dataset_from_file_paths(json_paths, is_training=False)
        dataset_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        next_batch = dataset_iterator.get_next()

        output_node = graph.get_tensor_by_name("output_belong_to_same_instance:0")
        feed_dict = {}

        batch_counter = 0
        start_timer = time.time()
        while True:  # Loop until dataset is empty
            if FLAGS.batch_limiter != -1 and FLAGS.batch_limiter <= batch_counter:
                logging.info(f"stop validation after {batch_counter} batches with "
                             f"{FLAGS.batch_size} samples each.")
                break
            try:
                page_path = page_paths.pop(0)
                json_path = json_paths.pop(0)
                logging.info(f"Page: {page_path}")
                logging.info(f"Json: {json_path}")

                logging.info(f"Processing... {page_path}")
                # Skip files where json is missing (e.g. when there are less than 2 text regions)
                if not os.path.isfile(json_path):
                    logging.warning(f"No json file found to given pageXML {page_path}. Skipping.")
                    continue

                # get one batch (input_dict, target_dict) from generator
                batch = sess.run([next_batch])[0]
                batch_counter += 1

                # assign placeholder_dict to feed_dict
                for key in placeholders:
                    if type(placeholders[key]) == dict:
                        for i in placeholders[key]:
                            input_name = graph.get_tensor_by_name(placeholders[key][i].name)
                            feed_dict[input_name] = batch[0][key][i]
                    else:
                        input_name = graph.get_tensor_by_name(placeholders[key].name)
                        feed_dict[input_name] = batch[0][key]

                # run model with feed_dict
                output = sess.run(output_node, feed_dict=feed_dict)

                # evaluate the output
                class_probabilities = output[0, :, 1]

                if 'node_features' in placeholders:
                    node_features_node = graph.get_tensor_by_name('node_features:0')
                    node_features = feed_dict[node_features_node][0]  # assume batch_size = 1

                # clustering of confidence graph
                confidences = np.reshape(class_probabilities, [node_features.shape[0], -1])
                # Manually set confidences of edges over horizontal separators and headings to zero
                if FLAGS.mask_heading_separated_confs or FLAGS.mask_horizontally_separated_confs:
                    confidences = mask_horizontally_separated_confs(confidences, page_path)

                if FLAGS.save_conf != 'no_conf':
                    # save confidences
                    save_conf_to_json(confidences=confidences,
                                      page_path=page_path,
                                      save_dir=FLAGS.out_dir)
                    if FLAGS.save_conf == 'only_conf':
                        # skip clustering
                        continue

                tb_clustering.set_confs(confidences)
                tb_clustering.calc(method=FLAGS.clustering_method)

                # save pageXMLs with new clusterings
                cluster_path = save_clustering_to_page(clustering=tb_clustering.tb_labels,
                                                       page_path=page_path,
                                                       save_dir=FLAGS.out_dir,
                                                       info=tb_clustering.get_info(FLAGS.clustering_method))

            # break as soon as dataset is empty
            # (IndexError for empty page_paths list, OutOfRangeError for empty tf dataset)
            except (tf.errors.OutOfRangeError, IndexError):
                break
        logging.info(f"Time: {time.time() - start_timer:.2f} seconds")
        logging.info("Evaluation finished.")


if __name__ == "__main__":
    logging.getLogger().setLevel('INFO')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    flags.print_flags()
    # dummy InputFn to print flags
    dummy = InputGNN(FLAGS)
    dummy.print_params()
    # dummy TextblockClustering to print flags
    dummy = TextblockClustering(FLAGS)
    dummy.print_params()

    # load jsons
    jsons = [line.rstrip() for line in open(FLAGS.eval_list, "r")]
    n = FLAGS.num_workers

    # parallel over n workers (regarding the input list)
    if n > 1:
        processes = []
        for index, sublist in enumerate(split_list(jsons, n)):
            # start worker
            p = mp.Process(target=gnn_clustering, args=(sublist,))
            p.start()
            logging.info(f"Started worker {index}")
            processes.append(p)
        for p in processes:
            p.join()
        logging.info("All workers done.")
    # single threaded
    else:
        gnn_clustering(jsons)
