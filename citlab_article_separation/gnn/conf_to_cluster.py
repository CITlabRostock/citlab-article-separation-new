import json
import logging
import time
import numpy as np

import citlab_python_util.basic.flags as flags
from citlab_article_separation.gnn.clustering.textblock_clustering import TextblockClustering
from citlab_article_separation.gnn.io import save_clustering_to_page
from citlab_python_util.io.path_util import get_page_from_conf_path
from citlab_python_util.parser.xml.page.page import Page
from citlab_article_separation.gnn.input.feature_generation import discard_text_regions_and_lines as discard_regions

flags.define_string('eval_list',   '', '.lst-file specifying the confidence json files used for clustering')
flags.define_choices('clustering_method', ['dbscan', 'linkage', 'greedy', 'dbscan_std'], 'dbscan', str,
                     "('dbscan', 'linkage', 'greedy', 'dbscan_std')", 'clustering method to be used')
flags.define_dict('clustering_params', {}, "dict of key=value pairs defining the clustering configuration")
flags.define_string("out_dir", "", "directory to save clustering pageXMLs. It retains the folder structure of "
                                   "the input data. Use an empty 'out_dir' for the original folder")
flags.FLAGS.parse_flags()
FLAGS = flags.FLAGS


if __name__ == "__main__":
    logging.getLogger().setLevel('INFO')

    # Initialize TextblockClustering
    tb_clustering = TextblockClustering(FLAGS)
    # print flags
    flags.print_flags()
    tb_clustering.print_params()

    # load file paths
    with open(FLAGS.eval_list, "r") as eval_list_file:
        json_paths = [line.rstrip() for line in eval_list_file.readlines()]
        page_paths = [get_page_from_conf_path(json_path) for json_path in json_paths]

    # iterate over dataset
    start_timer = time.time()
    for json_path, page_path in zip(json_paths, page_paths):
        # load page
        page = Page(page_path)
        # Get text regions and discard degenerate ones
        text_regions = page.get_text_regions()
        text_regions, _ = discard_regions(text_regions)
        num_nodes = len(text_regions)
        # load confidence json
        logging.info(f"Processing... {json_path}")
        with open(json_path, "r") as json_file:
            data = json.load(json_file)["confidences"]
            assert(len(data) == num_nodes), f"Mismatch: Number of TextRegions in page ({num_nodes}), Number of " \
                                            f"TextRegions in confidence json ({len(data)})."
            # fill confidence array
            confidences = np.empty((num_nodes, num_nodes))
            i = 0
            for tb in data:
                confidences[i] = list(data[tb].values())
                i += 1

        # set confidences (should already be symmetric)
        tb_clustering.set_confs(confidences, symmetry_fn=None)
        # do clustering
        tb_clustering.calc(method=FLAGS.clustering_method)
        # save pageXMLs with new clusterings
        cluster_path = save_clustering_to_page(clustering=tb_clustering.tb_labels,
                                               page_path=page_path,
                                               save_dir=FLAGS.out_dir,
                                               info=tb_clustering.get_info(FLAGS.clustering_method))
    logging.info("Time: {:.2f} seconds".format(time.time() - start_timer))
    logging.info("Clustering process finished.")
