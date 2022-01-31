import logging
import numpy as np


def augment_geometric_features(node_features, config, desc=''):
    logging.info(f"applying augment_geometric_features{' - ' if desc else ''}{desc}: {config}")
    # apply feature augmentation at random, with a 50% probability for each module
    if "scaling" in config:
        if np.random.uniform(0, 1) < 0.5:
            node_features = scaling_noise(node_features)
            logging.debug("geometric augmentation - scaling")
    if "rotation" in config:
        if np.random.uniform(0, 1) < 0.5:
            node_features = rotation_noise(node_features)
            logging.debug("geometric augmentation - rotation")
    if "translation" in config:
        if np.random.uniform(0, 1) < 0.5:
            node_features = translation_noise(node_features)
            logging.debug("geometric augmentation - translation")
    return node_features


def scaling_noise(node_features, mean=1.0, std=0.04):
    num_nodes = node_features.shape[0]
    # random scalings
    horizontal_scalings = np.ones(num_nodes) * np.random.normal(loc=mean, scale=std)
    vertical_scalings = np.ones(num_nodes) * np.random.normal(loc=mean, scale=std)
    # feature augmentation
    node_features = horizontal_scaling(node_features, horizontal_scalings)
    node_features = vertical_scaling(node_features, vertical_scalings)
    return node_features


def horizontal_scaling(node_features, scaling):
    scaling = np.expand_dims(scaling, axis=1)
    # region x values
    node_features[:, (0, 2)] *= scaling
    # baseline x values
    if node_features.shape[1] >= 12:
        node_features[:, (4, 6, 8, 10)] *= scaling
    return node_features


def vertical_scaling(node_features, scaling):
    scaling = np.expand_dims(scaling, axis=1)
    # region y values
    node_features[:, (1, 3)] *= scaling
    # baseline y values
    if node_features.shape[1] >= 12:
        node_features[:, (5, 7, 9, 11)] *= scaling
        # text height
        if node_features.shape[1] >= 16:
            node_features[:, 15] *= np.squeeze(scaling)
    return node_features


def rotation_noise(node_features, mean_coherent=0.0, std_coherent=0.052):  # mean_incoherent=0.0, std_incoherent=0.017):
    # 0.052 ~= Pi / 60 (3°)
    # 0.017 ~= Pi / 180 (1°)
    # coherent part
    angle = np.random.normal(loc=mean_coherent, scale=std_coherent)
    node_features = coherent_rotation(node_features, angle)
    # # incoherent part
    # num_nodes = node_features.shape[0]
    # angles = np.random.normal(loc=mean_incoherent, scale=std_incoherent, size=num_nodes)
    # node_features = incoherent_rotation(node_features, angles)
    return node_features


def coherent_rotation(node_features, angle):
    # NOTE: for small angles we don't need to compute new sizes for the features
    # center for rotation
    regions_center_x = np.mean(node_features[:, 2])
    regions_center_y = np.mean(node_features[:, 3])
    rotation_center = np.array([regions_center_x, regions_center_y])
    # region centers
    # move to origin
    region_centers_x = node_features[:, 2] - rotation_center[0]
    region_centers_y = node_features[:, 3] - rotation_center[1]
    # coherent rotation
    rotated_region_centers_x = np.cos(angle) * region_centers_x - np.sin(angle) * region_centers_y
    rotated_region_centers_y = np.sin(angle) * region_centers_x + np.cos(angle) * region_centers_y
    node_features[:, 2] = rotated_region_centers_x + rotation_center[0]
    node_features[:, 3] = rotated_region_centers_y + rotation_center[1]
    # baseline centers
    if node_features.shape[1] >= 12:
        # move to origin
        baseline_centers_x = node_features[:, (6, 10)] - rotation_center[0]
        baseline_centers_y = node_features[:, (7, 11)] - rotation_center[1]
        # coherent rotation
        rotated_baseline_centers_x = np.cos(angle) * baseline_centers_x - np.sin(angle) * baseline_centers_y
        rotated_baseline_centers_y = np.sin(angle) * baseline_centers_x + np.cos(angle) * baseline_centers_y
        node_features[:, (6, 10)] = rotated_baseline_centers_x + rotation_center[0]
        node_features[:, (7, 11)] = rotated_baseline_centers_y + rotation_center[1]
    return node_features


# def incoherent_rotation(node_features, angles):
#     # NOTE: for small angles we don't need to compute new sizes for the features
#     # feature augmentation
#     # region centers
#     rotated_x = np.cos(angles) * (node_features[:, 2]) - np.sin(angles) * (node_features[:, 3])
#     rotated_y = np.sin(angles) * (node_features[:, 2]) + np.cos(angles) * (node_features[:, 3])
#     node_features[:, 2] = rotated_x
#     node_features[:, 3] = rotated_y
#     # baseline centers
#     if node_features.shape[1] >= 12:
#         angles = np.expand_dims(angles, axis=1)  # for broadcasting
#         rotated_x = np.cos(angles) * (node_features[:, (6, 10)]) - np.sin(angles) * (node_features[:, (7, 11)])
#         rotated_y = np.sin(angles) * (node_features[:, (6, 10)]) + np.cos(angles) * (node_features[:, (7, 11)])
#         node_features[:, (6, 10)] = rotated_x
#         node_features[:, (7, 11)] = rotated_y
#     return node_features


def translation_noise(node_features, mean_coherent=0.0, std_coherent=0.01, mean_incoherent=0.0, std_incoherent=0.005):
    num_nodes = node_features.shape[0]
    # random incoherent part
    horizontal_displacements = np.random.normal(loc=mean_incoherent, scale=std_incoherent, size=num_nodes)
    vertical_displacements = np.random.normal(loc=mean_incoherent, scale=std_incoherent, size=num_nodes)
    # add random coherent part (gets broadcasted)
    horizontal_displacements += np.random.normal(loc=mean_coherent, scale=std_coherent)
    vertical_displacements += np.random.normal(loc=mean_coherent, scale=std_coherent)
    # feature augmentation
    # region centers
    node_features[:, 2] += horizontal_displacements
    node_features[:, 3] += vertical_displacements
    # baseline centers
    if node_features.shape[1] >= 12:
        horizontal_displacements = np.expand_dims(horizontal_displacements, axis=1)
        vertical_displacements = np.expand_dims(vertical_displacements, axis=1)
        node_features[:, (6, 10)] += horizontal_displacements
        node_features[:, (7, 11)] += vertical_displacements
    return node_features
