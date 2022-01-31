import numpy as np
import os
import json
import logging
import re
from scipy.stats import gmean
from python_util.parser.xml.page.page import Page


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
    logging.info(f"Saved pageXML with graph clustering '{save_path}'")
    return save_path
