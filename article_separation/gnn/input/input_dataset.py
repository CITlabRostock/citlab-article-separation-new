import numpy as np
import tensorflow as tf
import logging
from random import Random
from PIL import Image

from citlab_python_util.io.path_util import *
from citlab_article_separation.gnn.input.create_input_and_target import get_input_and_target_from_json,\
    get_input_and_target_from_xml
from citlab_python_util.image_processing.image_resizer import resize_image_ratio


class InputGNN(object):
    """Input Function Generator for article separation"""

    def __init__(self, flags):
        self.input_params = dict()
        self._flags = flags
        self._eval_list = self._flags.eval_list

        # Default params which are scenario dependent
        self.input_params["node_feature_dim"] = 4  # number of node input
        self.input_params["edge_feature_dim"] = 0  # number of edge input
        self.input_params["node_input_feature_mask"] = []  # list of booleans indicating if a node/edge feature is of
        self.input_params["edge_input_feature_mask"] = []  # interest or not (empty or of length '_feature_dim')
        self.input_params["num_parallel_load"] = 4  # Parallel calls for loading
        self.input_params["prefetch_load"] = 4  # Prefetch number for loading. Note that this prefetches entire batches

        # Optional params when loading from pagexml instead of json
        self.input_params['interaction_from_pagexml'] = 'delaunay'  # 'delaunay' or 'fully'
        self.input_params['external_jsons'] = []  # list of json files including addtional input

        # Params for visual feature input
        self.input_params["load_mode"] = 'L'
        self.input_params["resize_max_dim"] = 1024
        self.input_params["resize_min_dim"] = 256
        self.input_params["pad_to_max_dim"] = False

        # Updating of the default params if provided via flags as a dict
        for i in flags.input_params:
            if i not in self.input_params:
                logging.critical(f"Given input_params-key '{i}' is not used by class 'InputGNN'!")
        self.input_params.update(flags.input_params)

        # Image loading and resizing parameters
        self._img_load_mode = self.input_params["load_mode"]

        self._img_channels = 1
        if self._img_load_mode == 'RGB':
            self._img_channels = 3

        self._original_image_shape = [None, None, self._img_channels]
        self._expected_image_shape = [None, None, self._img_channels]
        # Ratio-aware resizing
        if self.input_params["resize_max_dim"] > 0 and self.input_params["resize_min_dim"] > 0:
            if self.input_params["pad_to_max_dim"]:
                self._expected_image_shape = [self.input_params["resize_max_dim"],
                                              self.input_params["resize_max_dim"],
                                              self._img_channels]
        else:
            logging.error("Error in resizing parameters for input image.")
            exit(1)

        self._rnd = Random()

    def print_params(self):
        print("##### {}:".format("INPUT"))
        sorted_dict = sorted(self.input_params.items(), key=lambda kv: kv[0])
        for a in sorted_dict:
            print("  {}: {}".format(a[0], a[1]))

    def get_dataset(self):
        dataset = tf.data.TextLineDataset(self._eval_list)
        dataset = self._process_dataset(dataset)
        return dataset

    def _process_dataset(self, dataset):
        dataset = dataset.map(lambda file: self._map_element(file),
                              num_parallel_calls=self.input_params['num_parallel_load'])
        dataset = dataset.apply(tf.data.experimental.ignore_errors())

        # input dict and padding
        input_dict_shape = {'num_nodes': [],
                            'interacting_nodes': [None, 2],
                            'num_interacting_nodes': [],
                            'relations_to_consider_belong_to_same_instance': [None, self._flags.num_classes],
                            'num_relations_to_consider_belong_to_same_instance': []}
        input_dict_pad = {'num_nodes': tf.constant(0, dtype=tf.int32),
                          'interacting_nodes': tf.constant(0, dtype=tf.int32),
                          'num_interacting_nodes': tf.constant(0, dtype=tf.int32),
                          'relations_to_consider_belong_to_same_instance': tf.constant(0, dtype=tf.int32),
                          'num_relations_to_consider_belong_to_same_instance': tf.constant(0, dtype=tf.int32)}

        # add node input if present
        if self.input_params["node_feature_dim"] > 0:
            node_feature_dim = self.input_params["node_input_feature_mask"].count(True) if \
                self.input_params["node_input_feature_mask"] else self.input_params["node_feature_dim"]
            input_dict_shape['node_features'] = [None, node_feature_dim]
            input_dict_pad['node_features'] = tf.constant(0, dtype=tf.float32)

        # add edge input if present
        if self.input_params["edge_feature_dim"] > 0:
            edge_feature_dim = self.input_params["edge_input_feature_mask"].count(True) if \
                self.input_params["edge_input_feature_mask"] else self.input_params["edge_feature_dim"]
            input_dict_shape['edge_features'] = [None, edge_feature_dim]
            input_dict_pad['edge_features'] = tf.constant(0, dtype=tf.float32)

        # add image input if present
        if self._flags.image_input:
            input_dict_shape['image'] = self._expected_image_shape
            input_dict_shape['image_shape'] = [3]
            # TODO: generate visual regions as [None, None, 2] (note changes in other classes, when working with it)
            input_dict_shape['visual_regions_nodes'] = [None, 2, None]
            input_dict_shape['num_points_visual_regions_nodes'] = [None]
            input_dict_shape['visual_regions_edges'] = [None, 2, None]
            input_dict_shape['num_points_visual_regions_edges'] = [None]
            input_dict_pad['image'] = tf.constant(0, dtype=tf.float32)
            input_dict_pad['image_shape'] = tf.constant(0, dtype=tf.int32)
            input_dict_pad['visual_regions_nodes'] = tf.constant(0, dtype=tf.float32)
            input_dict_pad['num_points_visual_regions_nodes'] = tf.constant(0, dtype=tf.int32)
            input_dict_pad['visual_regions_edges'] = tf.constant(0, dtype=tf.float32)
            input_dict_pad['num_points_visual_regions_edges'] = tf.constant(0, dtype=tf.int32)

        # target dict and padding
        target_dict_shape = {'relations_to_consider_gt': [None]}
        target_dict_pad = {'relations_to_consider_gt': tf.constant(0, dtype=tf.int32)}

        dataset = dataset.padded_batch(self._flags.batch_size,
                                       (input_dict_shape, target_dict_shape),
                                       (input_dict_pad, target_dict_pad))

        return dataset.prefetch(self.input_params['prefetch_load'])

    def _map_element(self, file_path):
        Tout_list = [tf.int32, tf.int32, tf.int32]
        if self.input_params["node_feature_dim"] > 0:
            Tout_list.append(tf.float32)
        if self.input_params["edge_feature_dim"] > 0:
            Tout_list.append(tf.float32)
        if self._flags.image_input:
            # image, image_shape
            Tout_list.extend([tf.float32, tf.int32])
            # visual_regions_nodes, num_points_visual_regions_nodes
            Tout_list.extend([tf.float32, tf.int32])
            # visual_regions_edges, num_points_visual_regions_edges
            Tout_list.extend([tf.float32, tf.int32])
        Tout_list.extend([tf.int32, tf.int32, tf.int32])

        return_list = tf.py_func(func=self._parse_function, inp=[file_path], Tout=Tout_list, stateful=False)

        num_nodes = return_list[0]
        interacting_nodes = return_list[1]
        num_interacting_nodes = return_list[2]
        index = 3

        # input dict
        return_dict_inputs = {'num_nodes': num_nodes,
                              'interacting_nodes': interacting_nodes,
                              'num_interacting_nodes': num_interacting_nodes}

        # add node input if present
        if self.input_params["node_feature_dim"] > 0:
            node_features = return_list[index]
            index += 1
            # optionally mask node_features
            if len(self.input_params['node_input_feature_mask']) > 0:
                if len(self.input_params['node_input_feature_mask']) == self.input_params["node_feature_dim"]:
                    # include only the input given in the node input feature mask
                    # inputs['node_features']: [batch_size, max_num_nodes, node_feature_dim] float
                    node_features = self._mask_features(node_features, self.input_params['node_input_feature_mask'])
                    # inputs['node_features']: [batch_size, max_num_nodes, node_feature_dim (new)] float
                else:
                    logging.error(f"Length of node feature mask ({len(self.input_params['node_input_feature_mask'])}) "
                                  f"does not match provided node feature dim ({self.input_params['node_feature_dim']}).")
                    exit(-1)
            return_dict_inputs['node_features'] = node_features

        # add edge input if present
        if self.input_params["edge_feature_dim"] > 0:
            edge_features = return_list[index]
            index += 1
            # optionally mask edge_features
            if len(self.input_params['edge_input_feature_mask']) > 0:
                if len(self.input_params['edge_input_feature_mask']) == self.input_params["edge_feature_dim"]:
                    # include only the input given in the edge input feature mask
                    # inputs['edge_features']: [batch_size, max_num_nodes, edge_feature_dim] float
                    edge_features = self._mask_features(edge_features, self.input_params['edge_input_feature_mask'])
                    # inputs['edge_features']: [batch_size, max_num_nodes, edge_feature_dim (new)] float
                else:
                    logging.error(f"Length of edge feature mask ({len(self.input_params['edge_input_feature_mask'])}) "
                                  f"does not match provided edge feature dim ({self.input_params['edge_feature_dim']}).")
                    exit(-1)
            return_dict_inputs['edge_features'] = edge_features

        # add image input if present
        if self._flags.image_input:
            image = return_list[index]
            image_shape = return_list[index + 1]
            visual_regions_nodes = return_list[index + 2]
            num_points_visual_regions_nodes = return_list[index + 3]
            visual_regions_edges = return_list[index + 4]
            num_points_visual_regions_edges = return_list[index + 5]
            index += 6

            image.set_shape(self._original_image_shape)
            tensor_dict = {'image': image, 'image_shape': image_shape}

            # Ratio aware min/max resizing
            if self.input_params["resize_max_dim"] > 0 and self.input_params["resize_min_dim"] > 0:
                tensor_dict = resize_image_ratio(tensor_dict,
                                                 min_dimension=self.input_params["resize_min_dim"],
                                                 max_dimension=self.input_params["resize_max_dim"],
                                                 pad_to_max_dimension=self.input_params["pad_to_max_dim"])
            return_dict_inputs['image'] = tensor_dict['image']
            return_dict_inputs['image_shape'] = tensor_dict['image_shape']
            return_dict_inputs['visual_regions_nodes'] = visual_regions_nodes
            return_dict_inputs['num_points_visual_regions_nodes'] = num_points_visual_regions_nodes
            return_dict_inputs['visual_regions_edges'] = visual_regions_edges
            return_dict_inputs['num_points_visual_regions_edges'] = num_points_visual_regions_edges

        # relations
        return_dict_inputs['relations_to_consider_belong_to_same_instance'] = return_list[index]
        return_dict_inputs['num_relations_to_consider_belong_to_same_instance'] = return_list[index + 1]

        # target dict
        return_dict_targets = dict()
        return_dict_targets['relations_to_consider_gt'] = return_list[index + 2]
        return return_dict_inputs, return_dict_targets

    @staticmethod
    def _mask_features(features, mask):
        # input: [..., feature_dim] float
        # mask: list of type bool and length feature_dim
        new_features = tf.gather(features, tf.reshape(tf.where(mask), [-1]), axis=-1)
        # new_features: [batch_size, ..., new_feature_dim] float
        return new_features

    def _parse_function(self, file_path):
        file_path = file_path.decode("utf-8")
        if not self._flags.create_data_from_pagexml:
            data_dict = get_input_and_target_from_json(file_path, False)
        else:
            data_dict = get_input_and_target_from_xml(file_path, False,
                                                      self.input_params['external_jsons'],
                                                      self.input_params['interaction_from_pagexml'],
                                                      self._flags.image_input)

        num_nodes = data_dict['num_nodes']
        interacting_nodes = data_dict['interacting_nodes']
        num_interacting_nodes = data_dict['num_interacting_nodes']
        return_list = [num_nodes, interacting_nodes, num_interacting_nodes]

        # add node input if present
        if 'node_features' in data_dict:
            if self.input_params["node_feature_dim"] == 0:
                logging.error(f"Error in parsing {file_path}. Node features present, but node_feature_dim set to 0.")
                exit(1)
            node_features = data_dict['node_features']
            return_list.append(node_features)

        # add edge input if present
        if 'edge_features' in data_dict:
            if self.input_params["edge_feature_dim"] == 0:
                logging.error(f"Error in parsing {file_path}. Edge features present, but edge_feature_dim set to 0.")
                exit(1)
            edge_features = data_dict['edge_features']
            return_list.append(edge_features)

        # add visual regions if present
        if self._flags.image_input:
            if not ('visual_regions_nodes' in data_dict and 'num_points_visual_regions_nodes' in data_dict) and \
                    ('visual_regions_edges' in data_dict and 'num_points_visual_regions_edges' in data_dict):
                logging.error(f"Error in parsing {file_path}. Image_input set to True, but visual regions missing.")
                exit(1)
            # load image
            if not self._flags.create_data_from_pagexml:
                image_fn = get_img_from_json_path(file_path)
            else:
                image_fn = get_img_from_page_path(file_path)
            # load image
            image = Image.open(fp=image_fn).convert(self._img_load_mode)
            image = np.array(image, dtype=np.float32)
            # If we load a grayscale image with 2 dims, we add a dummy third dim
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            image_shape = np.asarray([image.shape[0], image.shape[1], image.shape[2]], dtype=np.int32)
            return_list.extend([image, image_shape])
            # add node regions
            visual_regions_nodes = data_dict['visual_regions_nodes']
            num_points_visual_regions_nodes = data_dict['num_points_visual_regions_nodes']
            return_list.extend([visual_regions_nodes, num_points_visual_regions_nodes])
            # add edge regions
            visual_regions_edges = data_dict['visual_regions_edges']
            num_points_visual_regions_edges = data_dict['num_points_visual_regions_edges']
            return_list.extend([visual_regions_edges, num_points_visual_regions_edges])

        gt_relations = data_dict['gt_relations']
        # sample relations
        if self._flags.sample_relations:
            relations_to_consider, num_relations_to_consider, relations_to_consider_gt = \
                self._sample_relations(num_nodes, gt_relations,
                                       self._flags.sample_num_relations_to_consider,
                                       self._flags.num_classes,
                                       self._flags.num_relation_components)
            return_list.extend([relations_to_consider, num_relations_to_consider, relations_to_consider_gt])
        # or use full graph relations
        else:
            relations_to_consider, num_relations_to_consider, relations_to_consider_gt = \
                self._build_full_relations(num_nodes, gt_relations)
            return_list.extend([relations_to_consider, num_relations_to_consider, relations_to_consider_gt])

        return return_list

    def _build_full_relations(self, num_nodes, gt_relations):
        # node relations
        node_indices = np.arange(num_nodes, dtype=np.int32)
        node_indices = np.tile(node_indices, [num_nodes, 1])
        node_indices_t = np.transpose(node_indices)
        relations_to_consider = np.stack([node_indices_t, node_indices], axis=2).reshape([-1, 2])
        # number of node relations
        num_relations_to_consider = np.array(relations_to_consider.shape[0], dtype=np.int32)
        # corresponding relation ground-truth
        gt_indices = np.split(gt_relations[:, 1:], indices_or_sections=2, axis=1)
        relations_to_consider_gt = np.zeros([num_nodes, num_nodes], dtype=np.int32)
        relations_to_consider_gt[tuple(gt_indices)] = 1
        relations_to_consider_gt = relations_to_consider_gt.reshape([-1])
        return relations_to_consider, num_relations_to_consider, relations_to_consider_gt

    def _sample_relations(self, num_nodes, gt_relations, sample_num_relations_to_consider, num_classes, rel_components):
        # num_nodes, int
        # gt_relations, [num_gt_relations, 1 + rel_dim], class as first element, followed by the relation
        # sample_num_relations_to_consider, int
        # num_classes, int

        relations_to_consider = []
        relations_to_consider_gt = []
        num_sample_false = sample_num_relations_to_consider // 2
        num_sample_true_per_class = sample_num_relations_to_consider // (2 * (num_classes - 1))

        pos_rel_set = set()

        if gt_relations is not None and gt_relations.shape[0] > 0:
            unique_relations = np.array(gt_relations)
            gt_class = unique_relations[:, 0]
            gt_rels = []
            for aRel in unique_relations[:, 1:]:
                gt_rels.append(tuple(aRel))
            pos_rel_set = set(gt_rels)
            rel_num = len(gt_rels)

            # Sample positive examples
            classContainer = [[] for _ in range(num_classes)]

            indices = list(range(rel_num))
            self._rnd.shuffle(indices)
            for idx in indices:
                aGT_class = gt_class[idx]
                aClassContainer = classContainer[aGT_class]
                if len(aClassContainer) < num_sample_true_per_class:
                    rel = gt_rels[idx]
                    aClassContainer.append(rel)

            for aClass in range(1, num_classes):
                aClassContainer = classContainer[aClass]
                relations_to_consider.extend(aClassContainer)
                relations_to_consider_gt.extend([aClass] * len(aClassContainer))

        # Sample negative examples
        neg_samples = 0
        negative_relations = []
        for _ in range(32 * num_sample_false):
            if neg_samples == num_sample_false:
                break
        # while neg_samples != num_sample_false:  # can get stuck if not enough samples are available
            aRel = tuple([self._rnd.randint(0, num_nodes - 1) for _ in range(rel_components)])
            if aRel not in negative_relations and aRel not in pos_rel_set:
                negative_relations.append(aRel)
                neg_samples += 1

        relations_to_consider.extend(negative_relations)
        relations_to_consider_gt.extend([0] * neg_samples)

        return np.array(relations_to_consider, dtype=np.int32), \
               np.array(len(relations_to_consider), dtype=np.int32), \
               np.array(relations_to_consider_gt, dtype=np.int32)
