import os
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # set tf log_level to warning(2), default: info(1)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import python_util.basic.flags as flags
from article_separation.gnn.trainer.trainer_base import TrainerBase
from article_separation.gnn.input.input_dataset import InputGNN
from article_separation.gnn.model.model_relation import ModelRelation


# Model parameter
# ===============
flags.define_integer('num_classes', 2, 'number of classes (including garbage class)')
flags.define_integer('num_relation_components', 2, 'number of components of the associated relations')
flags.define_string('num_hidden_units', '64,32', "hidden layer config, e.g. '32,32' or '64,64'")
flags.define_integer('sample_num_relations_to_consider', 300,
                     "number of sampled relations to be tested (half positive, half negative)")
flags.define_boolean('sample_relations', True, 'sample relations to consider or use full graph')
flags.define_float('weight_decay', 0.0, 'L2 weight decay')

flags.define_dict('rel_params', {}, "dict of key=value pairs defining the configuration of the relation graph.")
flags.define_dict('gnn_params_1', {}, "dict of key=value pairs defining the configuration of the gnn.")
flags.define_dict('message_fn_params_1', {}, "dict of key=value pairs defining the configuration of the message_fn.")
flags.define_dict('update_fn_params_1', {}, "dict of key=value pairs defining the configuration of the update_fn.")

# Visual features
# ===============
flags.define_boolean('image_input', False, 'use image as additional input for GNN (visual features are '
                                           'calculated from image and regions)')
flags.define_string('backbone', 'ARU_v1', 'Backbone graph to use.')
flags.define_integer('channels', 1, "image channels for image input")
flags.define_integer('n_classes', 2, "number of classes for backbone net")
flags.define_boolean('mvn', False, 'MVN on the input.')
flags.define_dict('graph_backbone_params', {}, "key=value pairs defining the configuration of the backbone."
                                               "Backbone parametrization")
# E.g. train script: --feature_map_generation_params from_layer=[Mixed_5d,,Mixed_6e,Mixed_7c] layer_depth=[-1,128,-1,-1]
flags.define_dict('feature_map_generation_params',
                  {'layer_depth': [-1, -1, -1]},
                  "key=value pairs defining the configuration of the feature map generation."
                  "FeatureMap Generator parametrization, see main graph, e.g. model_fn.model_fn_objdet.graphs.main.SSD")

# Input function
# ===============
flags.define_dict('input_params', {}, "key=value pairs defining the configuration of the input function")
flags.define_list('augmentation_config', str, 'STR',
                  "names of the augmentation modules to use. Empty means no distortions. "
                  "Available distortions are: ['scaling', 'rotation', 'translation'].",
                  ["scaling", "rotation", "translation"])


class TrainerGNN(TrainerBase):
    def __init__(self):
        super(TrainerGNN, self).__init__()
        self._input_fn_generator = InputGNN(self._flags)
        self._input_fn_generator.print_params()
        self._model = ModelRelation(self._params)
        self._model.print_params()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')  # for TF "Converting sparse IndexedSlices to a dense Tensor of unknown shape"
    logging.getLogger().setLevel('INFO')
    tf.get_logger().propagate = False
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.info("Running Trainer.")
    trainer = TrainerGNN()
    trainer.train()
