import numpy as np
import os
import functools
import json
import logging
import re
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gmean
from copy import deepcopy
import networkx as nx

from citlab_python_util.parser.xml.page.page import Page
import citlab_python_util.parser.xml.page.plot as plot_util
from citlab_article_separation.gnn.input.feature_generation import discard_text_regions_and_lines as discard_regions
from citlab_python_util.io.path_util import *


def build_weighted_relation_graph(edges, weights, feature_dicts=None):
    if type(edges) == np.ndarray:
        assert edges.ndim == 2, f"Expected 'relations' to be 2d, got {edges.ndim}d."
        edges = edges.tolist()
    if type(weights) == np.ndarray:
        assert weights.ndim == 1, f"Expected 'weights' to be 1d, got {weights.ndim}d."
        weights = weights.tolist()
    assert (len(edges) == len(weights)), f"Number of elements in 'relations' {len(edges)} and" \
                                             f" 'weights' {len(weights)} has to match."

    graph = nx.DiGraph()
    for i in range(len(edges)):
        if feature_dicts is not None:
            graph.add_edge(*edges[i], weight=weights[i], **feature_dicts[i])
        else:
            graph.add_edge(*edges[i], weight=weights[i])
    return graph


def create_undirected_graph(digraph, symmetry_fn=gmean, reciprocal=False):
    graph = nx.Graph()
    graph.graph.update(deepcopy(digraph.graph))
    graph.add_nodes_from((n, deepcopy(d)) for n, d in digraph._node.items())
    for u, successcors in digraph.succ.items():
        for v, data in successcors.items():
            u_v_data = deepcopy(data)
            if v in digraph.pred[u]:  # reciprocal (both edges present)
                # edge data handling
                v_u_data = digraph.pred[u][v]
                if symmetry_fn:
                    u_v_data['weight'] = symmetry_fn([u_v_data['weight'], v_u_data['weight']])
                graph.add_edge(u, v, **u_v_data)
            elif not reciprocal:
                graph.add_edge(u, v, **u_v_data)
    return graph


def build_thresholded_relation_graph(edges, weights, threshold, reciprocal=False):
    graph_full = build_weighted_relation_graph(edges, weights)
    graph = create_undirected_graph(graph_full, reciprocal=reciprocal)
    edges_below_threshold = [(u, v) for u, v, w in graph.edges.data('weight') if w < threshold]
    graph.remove_edges_from(edges_below_threshold)
    return graph


def build_confidence_graph_dict(graph, page_path):
    page = Page(page_path)
    text_regions = page.get_regions()['TextRegion']
    text_regions, _ = discard_regions(text_regions)
    assert graph.number_of_nodes() == len(text_regions), \
        f"Number of nodes in graph ({graph.number_of_nodes()}) does not match number of text regions " \
        f"({len(text_regions)}) in {page_path}."

    out_dict = dict()
    page_name = os.path.basename(page_path)
    out_dict[page_name] = dict()

    for text_region in text_regions:
        out_dict[page_name][text_region.id] = {'article_id': None, 'confidences': dict()}

    for i, j, w in graph.edges.data('weight'):
        out_dict[page_name][text_regions[i].id]['confidences'][text_regions[j].id] = w

    return out_dict


def save_conf_to_json(confidences, page_path, save_dir, symmetry_fn=gmean):
    page = Page(page_path)
    text_regions = page.get_regions()['TextRegion']
    text_regions, _ = discard_regions(text_regions)
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
    page = Page(page_path)
    text_regions = page.get_regions()['TextRegion']
    # discard text regions
    text_regions, _ = discard_regions(text_regions)
    assert len(clustering) == len(text_regions), f"Number of nodes in clustering ({len(clustering)}) does not " \
                                                 f"match number of text regions ({len(text_regions)}) in {page_path}."

    # Set textline article ids based on clustering
    for index, text_region in enumerate(text_regions):
        article_id = clustering[index]
        for text_line in text_region.text_lines:
            text_line.set_article_id("a" + str(article_id))
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


def graph_edge_conf_histogram(graph, num_bins):
    conf = [w for u, v, w in graph.edges.data('weight')]
    c = np.array(conf)
    hist, bin_edges = np.histogram(conf, bins=num_bins, range=(0.0, 1.0))
    for i in range(num_bins):
        logging.debug(f"Edges with conf [{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}): {hist[i]}")
    plt.hist(conf, num_bins, range=(0.0, 1.0), rwidth=0.98)
    plt.xticks(np.arange(0, 1.01, 0.1))
    plt.show()


def plot_confidence_histogram(confidences, bins, page_path, save_dir, desc=None):
    # Get img path
    img_path = get_img_from_page_path(page_path)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.canvas.set_window_title(img_path)
    ax.set_xticks(np.arange(0, 1.01, 0.1))

    # Plot histogram
    counts, bins, _ = ax.hist(confidences, bins=bins, range=(0.0, 1.0), rwidth=0.98)

    # Save image
    desc = '' if desc is None else f'_{desc}'
    save_name = re.sub(r'\.xml$', f'{desc}.jpg', os.path.basename(page_path))
    page_dir = os.path.dirname(page_path)
    save_dir = os.path.join(save_dir, page_dir, f"{desc}")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=1000)
    logging.info(f"Saved debug image '{save_path}'")
    plt.close(plt.gcf())


def plot_graph_clustering_and_page(graph, node_features, page_path, cluster_path, save_dir,
                                   threshold, info, with_edges=True, with_labels=True, **kwds):
    # Get pagexml and image file
    original_page = Page(page_path)
    img_path = get_img_from_page_path(page_path)
    cluster_page = Page(cluster_path)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.canvas.set_window_title(img_path)
    axes[0].set_title(f'GT_with_graph_conf_threshold{threshold}')
    axes[1].set_title(f'cluster_{info}')

    # Plot Cluster page
    plot_util.plot_pagexml(cluster_page, img_path, ax=axes[1], plot_article=True, plot_legend=False)
    for ax in axes:
        ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Add graph to subplot
    # Get node positions from input
    page_resolution = original_page.get_image_resolution()
    region_centers = node_features[:, 2:4] * page_resolution
    positions = dict()
    for n in range(node_features.shape[0]):
        positions[n] = region_centers[n]
        # node_positions[n][1] = page_resolution[1] - node_positions[n][1]  # reverse y-values (for plotting)

    # Get article colors according to baselines
    article_dict = original_page.get_article_dict()
    unique_ids = sorted(set(article_dict.keys()), key=functools.cmp_to_key(compare_article_ids))
    if None in unique_ids:
        article_colors = dict(zip(unique_ids, plot_util.COLORS[:len(unique_ids) - 1] + [plot_util.DEFAULT_COLOR]))
    else:
        article_colors = dict(zip(unique_ids, plot_util.COLORS[:len(unique_ids)]))
    # Build node colors (region article_id matching baselines article_id)
    region_article_ids = get_region_article_ids(original_page)
    node_colors = [article_colors[a_id] for a_id in region_article_ids]
    node_colors = [node_colors[i] for i in list(graph.nodes)]  # reorder coloring according to node_list

    # Draw nodes
    graph_views = dict()
    node_collection = nx.draw_networkx_nodes(graph, positions, ax=axes[0], node_color=node_colors, node_size=50)
    node_collection.set_zorder(3)
    graph_views['nodes'] = [node_collection]
    # Draw edges
    if with_edges:
        edge_collection = nx.draw_networkx_edges(graph, positions, ax=axes[0], width=0.5, arrows=False, **kwds)
        if edge_collection is not None:
            edge_collection.set_zorder(2)
            graph_views['edges'] = [edge_collection]
        # optional colorbar
        if 'edge_cmap' in kwds and 'edge_color' in kwds:
            # add colorbar to confidence graph
            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            # norm = Normalize(vmin=min(kwds['edge_color']), vmax=max(kwds['edge_color']))
            norm = Normalize(vmin=threshold, vmax=1.0)
            fig.colorbar(ScalarMappable(norm=norm, cmap=kwds['edge_cmap']), cax=cax, format="%.2f")
            # hacky way to get same size adjustment on other two images
            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes("left", size="5%", pad=0.05)
            cax.axis("off")
    # Draw labels
    if with_labels:
        label_collection = nx.draw_networkx_labels(graph, positions, ax=axes[0], font_size=5)
        graph_views['labels'] = [label_collection]
    plt.connect('key_press_event', lambda event: toggle_graph_view(event, graph_views))
    # Draw page underneath
    plot_util.plot_pagexml(original_page, img_path, ax=axes[0], plot_article=True, plot_legend=False)

    # Save image
    page_path = os.path.relpath(page_path)
    save_name = re.sub(r'\.xml$', f'_clustering_debug.jpg', os.path.basename(page_path))
    page_dir = os.path.dirname(page_path)
    if info:
        save_dir = os.path.join(save_dir, page_dir, info)
    else:
        save_dir = os.path.join(save_dir, page_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    logging.info(f"Saved debug image '{save_path}'")
    plt.close(plt.gcf())


# def plot_graph_clustering_and_page_debug(graph, node_features, page_path, cluster_path, save_dir,
#                                          threshold, info, with_edges=True, with_labels=True, **kwds):
#     # Get pagexml and image file
#     original_page = Page(page_path)
#     img_path = get_img_from_page_path(page_path)
#     cluster_page = Page(cluster_path)
#
#     from citlab_article_separation.gnn.input.feature_generation import delaunay_edges, get_data_from_pagexml
#     num_nodes = len(graph.nodes)
#     _, _, _, _, resolution = get_data_from_pagexml(page_path)
#     norm_x, norm_y = float(resolution[0]), float(resolution[1])
#     interacting_nodes = delaunay_edges(num_nodes, node_features, norm_x, norm_y)
#
#     delaunay_graph = nx.Graph()
#     delaunay_graph.add_nodes_from(graph.nodes)
#     delaunay_graph.add_edges_from(interacting_nodes)
#     # graph = delaunay_graph
#
#     print(delaunay_graph.nodes)
#     print(delaunay_graph.edges)
#
#     fig, axes = plt.subplots(1, 1, figsize=(16, 9))
#     plot_util.plot_pagexml(original_page, img_path, ax=axes, plot_article=True, plot_legend=False)
#     axes.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
#
#     # Add graph to subplot
#     # Get node positions from input
#     page_resolution = original_page.get_image_resolution()
#     region_centers = node_features[:, 2:4] * page_resolution
#     positions = dict()
#     for n in range(node_features.shape[0]):
#         positions[n] = region_centers[n]
#         # node_positions[n][1] = page_resolution[1] - node_positions[n][1]  # reverse y-values (for plotting)
#
#     # Get article colors according to baselines
#     article_dict = original_page.get_article_dict()
#     unique_ids = sorted(set(article_dict.keys()), key=functools.cmp_to_key(compare_article_ids))
#     if None in unique_ids:
#         article_colors = dict(zip(unique_ids, plot_util.COLORS[:len(unique_ids) - 1] + [plot_util.DEFAULT_COLOR]))
#     else:
#         article_colors = dict(zip(unique_ids, plot_util.COLORS[:len(unique_ids)]))
#     # Build node colors (region article_id matching baselines article_id)
#     region_article_ids = get_region_article_ids(original_page)
#     node_colors = [article_colors[a_id] for a_id in region_article_ids]
#     node_colors = [node_colors[i] for i in list(graph.nodes)]  # reorder coloring according to node_list
#
#     # Draw nodes
#     node_collection = nx.draw_networkx_nodes(graph, positions, ax=axes, node_color="darkgreen", node_size=50, **kwds)
#     node_collection.set_zorder(3)
#     # Draw labels
#     label_collection = nx.draw_networkx_labels(graph, positions, ax=axes, font_size=5, **kwds)
#     # # Draw edges
#     # edge_collection = nx.draw_networkx_edges(delaunay_graph, positions, ax=axes, width=0.5, arrows=False, **kwds)
#     # # edge_collection.set_zorder(2)
#
#     edge_collection = nx.draw_networkx_edges(graph, positions, ax=axes, width=1.5, arrows=False, **kwds)
#     edge_collection.set_zorder(2)
#     # optional colorbar
#     if 'edge_cmap' in kwds and 'edge_color' in kwds:
#         # add colorbar to confidence graph
#         divider = make_axes_locatable(axes)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         # norm = Normalize(vmin=min(kwds['edge_color']), vmax=max(kwds['edge_color']))
#         norm = Normalize(vmin=threshold, vmax=1.0)
#         fig.colorbar(ScalarMappable(norm=norm, cmap=kwds['edge_cmap']), cax=cax, format="%.2f")
#
#     # Save image
#     save_name = re.sub(r'\.xml$', f'_2.jpg', os.path.basename(page_path))
#     page_dir = os.path.dirname(page_path)
#     if info:
#         save_dir = os.path.join(save_dir, page_dir, info)
#     else:
#         save_dir = os.path.join(save_dir, page_dir)
#     if not os.path.isdir(save_dir):
#         os.makedirs(save_dir)
#     save_path = os.path.join(save_dir, save_name)
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=1000)
#     logging.info(f"Saved debug image '{save_path}'")
#     plt.close(plt.gcf())


def plot_graph_and_page(page_path, graph, node_features, save_dir, img_path=None,
                        with_edges=True, with_labels=True, desc=None, **kwds):
    # Get pagexml and image file
    page = Page(page_path)
    if img_path is None:
        img_path = get_img_from_page_path(page_path)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.canvas.set_window_title(img_path)
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Add graph to subplot
    # Get node positions from input
    page_resolution = page.get_image_resolution()
    region_centers = node_features[:, 2:4] * page_resolution
    positions = dict()
    for n in range(node_features.shape[0]):
        positions[n] = region_centers[n]
        # node_positions[n][1] = page_resolution[1] - node_positions[n][1]  # reverse y-values (for plotting)

    # Get article colors according to baselines
    article_dict = page.get_article_dict()
    unique_ids = sorted(set(article_dict.keys()), key=functools.cmp_to_key(compare_article_ids))
    if None in unique_ids:
        article_colors = dict(zip(unique_ids, plot_util.COLORS[:len(unique_ids) - 1] + [plot_util.DEFAULT_COLOR]))
    else:
        article_colors = dict(zip(unique_ids, plot_util.COLORS[:len(unique_ids)]))
    # Build node colors (region article_id matching baselines article_id)
    region_article_ids = get_region_article_ids(page)
    node_colors = [article_colors[a_id] for a_id in region_article_ids]
    node_colors = [node_colors[i] for i in list(graph.nodes)]  # reorder coloring according to node_list

    # Draw nodes
    graph_views = dict()
    node_collection = nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=node_colors, node_size=50)
    node_collection.set_zorder(3)
    graph_views['nodes'] = [node_collection]
    # Draw edges
    if with_edges:
        edge_collection = nx.draw_networkx_edges(graph, positions, ax=ax, width=0.5, arrows=False, **kwds)
        if edge_collection is not None:
            edge_collection.set_zorder(2)
            graph_views['edges'] = [edge_collection]
        # optional colorbar
        if 'edge_cmap' in kwds and 'edge_vmin' in kwds and 'edge_vmax' in kwds:
            norm = Normalize(vmin=kwds['edge_vmin'], vmax=kwds['edge_vmax'])
            fig.colorbar(ScalarMappable(norm=norm, cmap=kwds['edge_cmap']), ax=ax, fraction=0.046, pad=0.04)
    # Draw labels
    if with_labels:
        label_collection = nx.draw_networkx_labels(graph, positions, ax=ax, font_size=5)
        graph_views['labels'] = [label_collection]
        # Draw page underneath
        plot_util.plot_pagexml(page, img_path, ax=ax, plot_article=True, plot_legend=False)

    plt.connect('key_press_event', lambda event: toggle_graph_view(event, graph_views))

    # Save image
    desc = '' if desc is None else f'_{desc}'
    save_name = re.sub(r'\.xml$', f'{desc}.jpg', os.path.basename(page_path))
    page_dir = os.path.dirname(page_path)
    save_dir = os.path.join(save_dir, page_dir, f"{desc}")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=1000)
    logging.info(f"Saved debug image '{save_path}'")
    plt.close(plt.gcf())


def compare_article_ids(a, b):
    # assume article IDs are named like "a<INTEGER>"
    if a is None and b is None:
        return 0
    elif a is None:
        return 1
    elif b is None:
        return -1
    elif int(a[1:]) < int(b[1:]):
        return -1
    elif int(a[1:]) == int(b[1:]):
        return 0
    else:
        return 1


def toggle_graph_view(event, views):
    """Switch between different views in the current plot by pressing the ``event`` key.

    :param event: the key event given by the user, various options available, e.g. to toggle the edges
    :param views: dictionary of different views given by name:object pairs
    :type event: matplotlib.backend_bases.KeyEvent
    :return: None
    """
    # Toggle nodes with optional labels
    if event.key == 'c' and "nodes" in views:
        for node_collection in views["nodes"]:
            is_visible = node_collection.get_visible()
            node_collection.set_visible(not is_visible)
        if "labels" in views:
            for label_collection in views["labels"]:
                for label in label_collection.values():
                    is_visible = label.get_visible()
                    label.set_visible(not is_visible)
        plt.draw()
    # Toggle edges
    if event.key == 'e' and "edges" in views:
        for edge_collection in views["edges"]:
            is_visible = edge_collection.get_visible()
            edge_collection.set_visible(not is_visible)
        plt.draw()
    if event.key == 'h':
        print("\tc: toggle graph nodes\n"
              "\te: toggle graph edges\n")
    else:
        return


def get_region_article_ids(page):
    assert type(page) == Page, f"Expected object of type 'Page', got {type(page)}."
    text_regions = page.get_regions()['TextRegion']
    text_regions_article_id = []
    for text_region in text_regions:
        # get all article_ids for textlines in this region
        tr_article_ids = []
        for text_line in text_region.text_lines:
            tr_article_ids.append(text_line.get_article_id())
        # count article_id occurences
        unique_article_ids = list(set(tr_article_ids))
        article_id_occurences = np.array([tr_article_ids.count(a_id) for a_id in unique_article_ids], dtype=np.int32)
        # assign article_id by majority vote
        if article_id_occurences.shape[0] > 1:
            assign_index = np.argmax(article_id_occurences)
            assign_article_id = unique_article_ids[int(assign_index)]
            text_regions_article_id.append(assign_article_id)
        else:
            text_regions_article_id.append(unique_article_ids[0])
    return text_regions_article_id
