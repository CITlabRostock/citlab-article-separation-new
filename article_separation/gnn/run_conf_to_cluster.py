import json
import logging
import time
import numpy as np
import multiprocessing as mp
import python_util.basic.flags as flags
from python_util.parser.xml.page.page import Page
from python_util.basic.misc import split_list
from python_util.io.path_util import get_page_from_conf_path
from article_separation.gnn.io import save_clustering_to_page
from article_separation.gnn.clustering.textblock_clustering import TextblockClustering


flags.define_string('eval_list',   '', 'input list with paths to confidence json files')
flags.define_choices('clustering_method', ['dbscan', 'linkage', 'greedy', 'dbscan_std'], 'dbscan', str,
                     "('dbscan', 'linkage', 'greedy', 'dbscan_std')", 'clustering method to be used')
flags.define_dict('clustering_params', {}, "dict of key=value pairs defining the clustering configuration")
flags.define_string("out_dir", "", "directory to save clustering pageXMLs. It retains the folder structure of "
                                   "the input data. Use an empty 'out_dir' for the original folder")
flags.define_integer('num_workers', 1, 'number of partitions to create from original list file and to compute in '
                                       'parallel. Only works when no external jsons are used.')
flags.FLAGS.parse_flags()
FLAGS = flags.FLAGS


def conf_to_cluster(json_paths):
    # Get corresponding pageXML files
    page_paths = [get_page_from_conf_path(json_path) for json_path in json_paths]
    # Initialize Textblock clustering
    tb_clustering = TextblockClustering(FLAGS)
    # iterate over dataset
    start_timer = time.time()
    for json_path, page_path in zip(json_paths, page_paths):
        # load page
        page = Page(page_path)
        # Get text regions
        text_regions = page.get_text_regions()
        num_nodes = len(text_regions)
        # load confidence json
        logging.info(f"Processing... {json_path}")
        with open(json_path, "r") as json_file:
            data = json.load(json_file)["confidences"]
            assert (len(data) == num_nodes), f"Mismatch: Number of TextRegions in page ({num_nodes}), Number of " \
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
    logging.info(f"Time: {time.time() - start_timer:.2f} seconds")
    logging.info("Clustering process finished.")


if __name__ == "__main__":
    logging.getLogger().setLevel('INFO')

    flags.print_flags()
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
            p = mp.Process(target=conf_to_cluster, args=(sublist,))
            p.start()
            logging.info(f"Started worker {index}")
            processes.append(p)
        for p in processes:
            p.join()
        logging.info("All workers done.")
    # single threaded
    else:
        conf_to_cluster(jsons)
