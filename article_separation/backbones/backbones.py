import logging
from gnn.model.model_base import GraphBase
from backbones.ARU_cutted_v1 import ARU_cutted_v1_CNN
from backbones.ARU_v1 import ARU_v1_CNN
from backbones.RU_v2 import RU_v2_CNN
from backbones.Inception_v3 import Inception_v3_CNN


class Backbones(GraphBase):
    """
    ALL Backbones have to take [bS, H, W, C] as inputs and return 2 outputs lastFeatureMap, {dict containing whatever...}
    """

    def __init__(self, backbone, params):
        super(Backbones, self).__init__(params)
        self._backbone_graph = None
        if backbone == 'Inception_v3':
            self._backbone_graph = Inception_v3_CNN(params)
        if backbone == 'ARU_v1':
            self._backbone_graph = ARU_v1_CNN(params)
        if backbone == 'RU_v2':
            self._backbone_graph = RU_v2_CNN(params)
        if backbone == 'ARU_cutted_v1':
            self._backbone_graph = ARU_cutted_v1_CNN(params)

        if not self._backbone_graph:
            logging.warning("NO GRAPH SUITABLE BACKBONE CHOSEN!")

    def infer(self, inputs, is_training):
        # inputs = shape_utils.check_min_image_dim(33, inputs)
        # batch_norm = (self._batch_norm and is_training)
        return self._backbone_graph.infer(inputs, is_training=is_training)

    def print_params(self):
        sorted_dict = sorted(self._backbone_graph.graph_params.items(), key=lambda kv: kv[0])
        logging.info("backbone_params:")
        if len(sorted_dict) > 0:
            for a in sorted_dict:
                logging.info(f"  {a[0]}: {a[1]}")
