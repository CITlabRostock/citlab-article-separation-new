import numpy as np
import os
import json
import logging
import re
import shutil
import tensorflow as tf
from scipy.stats import gmean
from python_util.parser.xml.page.page import Page


def load_graph(pb_path):
    """Loads TensorFlow graph from a protobuf"""
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="",
                            producer_op_list=None)
    return graph


def get_export_list(enforced_part_of_name, train_collection):
    global_vars = tf.compat.v1.global_variables()
    to_export = []
    exclude = ["applyGradients", "backupVariables"]
    for v in global_vars:
        if all([excl not in v.name for excl in exclude]):
            # Just regard variables containing a given keyword
            if len(enforced_part_of_name) != 0 and v.name in train_collection:
                for part in enforced_part_of_name:
                    if part in v.name:
                        to_export.append(v)
                        continue
            else:
                to_export.append(v)
    return to_export


def copy_model(dir_src, dir_dest):
    """Copies a TensorFlow model from `dir_src` to `dir_dest`."""
    logging.info(f"Copying model from '{dir_src}' to '{dir_dest}'.")
    with open(os.path.join(dir_src, "checkpoint"), "r") as file_checkpoint:
        first_line = file_checkpoint.readlines()[0]
        name = re.search('"(.*)"', first_line, re.IGNORECASE).group(1)
        logging.debug(f"Checkpoint name = {name}")
        if os.path.exists(dir_dest):
            logging.debug("Destination folder found.")
            for f in os.listdir(dir_dest):
                logging.debug(f"Try to delete {f} in folder {dir_dest}")
                os.remove(os.path.join(dir_dest, f))
        else:
            os.mkdir(dir_dest)
        files = os.listdir(dir_src)
        logging.debug(f"Source files are {files}")
        for file in files:
            if file.startswith(name):
                logging.debug(f"Copy {os.path.join(dir_src, file)} to {os.path.join(dir_dest, file)}")
                shutil.copyfile(os.path.join(dir_src, file), os.path.join(dir_dest, file))
        with open(os.path.join(dir_dest, "checkpoint"), "w+") as file_checkpoint_best:
            file_checkpoint_best.write(f'model_checkpoint_path: "{name}"\nall_model_checkpoint_paths: "{name}"\n')


def save_conf_to_json(confidences, page_path, save_dir, symmetry_fn=gmean):
    """
    Saves graph confidences to a json file.

    It loads the text regions given in the `page_path` pageXML file and creates an entry for each possible
    combination of two text regions. The order of the entries in the `confidences` array is expected to match
    the order of the text regions given in the pageXMl file.

    Since the output of the graph neural network is not symmetric, a `symmetry_fn` is used to make the
    confidences symmetric.

    The json file is saved to the `save_dir` directory, matching the name of the given pageXML file.

    :param confidences: square array containing the confidences
    :param page_path: file path to corresponding pageXML file
    :param save_dir: directory to save the json file
    :param symmetry_fn: function that averages opposing entries (i,j) and (j,i) in the confidence array
    :return: None
    """
    page = Page(page_path)
    text_regions = page.get_regions()['TextRegion']
    assert len(confidences) == len(text_regions), f"Number of nodes in confidences ({len(confidences)}) does not " \
                                                  f"match number of text regions ({len(text_regions)}) in {page_path}."

    # make confidences symmetric
    if symmetry_fn:
        conf_transpose = confidences.transpose()
        temp_mat = np.stack([confidences, conf_transpose], axis=-1)
        confidences = symmetry_fn(temp_mat, axis=-1)

    # Build confidence dict
    conf_dict = dict()
    for i in range(len(text_regions)):
        conf_dict[text_regions[i].id] = dict()
        for j in range(len(text_regions)):
            conf_dict[text_regions[i].id][text_regions[j].id] = str(confidences[i, j])
    out_dict = dict()
    out_dict["confidences"] = conf_dict

    # Dump json
    save_name = os.path.splitext(os.path.basename(page_path))[0] + "_confidences.json"
    page_dir = re.sub(r'page$', 'confidences', os.path.dirname(page_path))
    save_dir = os.path.join(save_dir, page_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, "w") as out_file:
        json.dump(out_dict, out_file)
        logging.info(f"Saved json with graph confidences '{save_path}'")


def save_clustering_to_page(clustering, page_path, save_dir, info=""):
    """
    Saves a text region clustering to a new pageXML file.

    It loads the text regions given in the `page_path` pageXML file and overwrites the custom article tag of
    each text line in those text regions with the clustering id given in the `clustering` list.

    The new pageXML file is saved to the `save_dir` directory, matching the name of the given pageXML file and
    including an optional `info` string.

    :param clustering: list of clustering ids for the text regions
    :param page_path: file path to corresponding pageXML file
    :param save_dir: directory to save the new pageXML file
    :param info: (optional) string to be attached to the file name
    :return: path to the saved pageXML file
    """
    page = Page(page_path)
    text_regions = page.get_regions()['TextRegion']
    assert len(clustering) == len(text_regions), f"Number of nodes in clustering ({len(clustering)}) does not " \
                                                 f"match number of text regions ({len(text_regions)}) in {page_path}."

    # Set textline article ids based on clustering
    for index, text_region in enumerate(text_regions):
        article_id = clustering[index]
        for text_line in text_region.text_lines:
            text_line.set_article_id(f"a{article_id}")
    # overwrite text regions (and text lines)
    page.set_text_regions(text_regions, overwrite=True)

    # Write pagexml
    page_path = os.path.relpath(page_path)
    save_name = re.sub(r'\.xml$', '_clustering.xml', os.path.basename(page_path))
    page_dir = re.sub(r'page$', 'clustering', os.path.dirname(page_path))
    if info:
        save_dir = os.path.join(save_dir, page_dir, info)
    else:
        save_dir = os.path.join(save_dir, page_dir)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    page.write_page_xml(save_path)
    logging.info(f"Saved pageXML with graph clustering '{os.path.abspath(save_path)}'")
    return save_path
