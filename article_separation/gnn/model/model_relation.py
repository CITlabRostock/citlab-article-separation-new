import tensorflow as tf
from python_util.math.measure import f1_score
from python_util.math.rounding import safe_div
from article_separation.gnn.model.model_base import ModelBase
from article_separation.gnn.model.graph.graph_relation import GraphRelation
from article_separation.gnn.model.graph_util import layers
from article_separation.gnn.model.graph_util.misc import curve_streaming_op


class ModelRelation(ModelBase):
    def __init__(self, params):
        super(ModelRelation, self).__init__(params)
        self._flags = self._params['flags']

    def get_graph(self):
        return GraphRelation(self._params)

    def get_loss(self):
        with tf.compat.v1.variable_scope("loss_scope"):
            logits = self._graph_out['logits']
            # logits: [batch_size, max_num_relations, num_classes] float

            num_relations_to_consider = self._inputs['num_relations_to_consider_belong_to_same_instance']
            # num_relations_to_consider: [batch_size] int

            relations_to_consider_gt = self._targets['relations_to_consider_gt']
            # relations_to_consider_gt: [batch_size, max_num_relations] int

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=relations_to_consider_gt)
            # losses: [batch_size, max_num_relations] float

            # ===== mask out the losses we don't care about
            mask = tf.sequence_mask(num_relations_to_consider, dtype=tf.float32)
            # mask: [batch_size, max_num_relations] float
            losses = losses * mask
            # losses: [batch_size, max_num_relations] float

            total_num_relations = tf.reduce_sum(mask)
            # total_num_relations: scalar int

            loss = safe_div(tf.reduce_sum(losses), total_num_relations, 'divide_loss_by_num_relatiosn')
            # loss: scalar float

            # logits_per_classifier = self._graph_out['logits']
            # # logits_per_classifier: dictionary of [batch_size, max_num_relations, num_classes] float
            #
            # num_relations_to_consider = self._inputs['num_relations_to_consider']
            # # num_relations_to_consider: dictionary of [batch_size] int
            # relations_to_consider_gt = self._targets['relations_to_consider_gt']
            # # relations_to_consider_gt: dictionary of [batch_size, max_num_relations] int
            #
            # # losses_per_classifier = {}
            # # for each classifier ...
            # for idx, classifier_name in enumerate(self._flags.classifiers):
            #     losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #         logits=logits_per_classifier[classifier_name],
            #         labels=relations_to_consider_gt[classifier_name])
            #     # losses: [batch_size, max_num_relations] float
            #
            #
            #     # num_relations: dictionary of [batch_size]
            #     mask = tf.sequence_mask(num_relations_to_consider[classifier_name], dtype=tf.float32)
            #     # mask: [batch_size, max_num_relations] float
            #
            #     losses = losses * mask
            #     # losses: [batch_size, max_num_relations] float
            #
            #     total_num_relations = tf.reduce_sum(mask)
            #     # total_num_relations: scalar int
            #
            #     losses_per_classifier[classifier_name] = safe_div(tf.reduce_sum(losses), total_num_relations,
            #                                                       'divide_loss_by_num_relations')
            #     # losses_per_classifier: dictionary of scalars float
            #
            # loss = 0
            # for _, classifier_name in enumerate(self._flags.classifiers):
            #     loss += losses_per_classifier[classifier_name]
            # # loss: scalar float

            # weight decay
            if self._flags.weight_decay > 0.0:
                loss += self._flags.weight_decay * tf.add_n(
                    [layers.l2_loss(var) for var in tf.compat.v1.trainable_variables() if 'bias' not in var.op.name])

            return tf.identity(loss, name="total_loss")

    def get_metrics(self):
        logits = self._graph_out['logits']
        # logits: [batch_size, max_num_relations, num_classes] float

        num_relations_to_consider = self._inputs['num_relations_to_consider_belong_to_same_instance']
        # num_relations_to_consider: [batch_size] int

        relations_to_consider_gt = self._targets['relations_to_consider_gt']
        # relations_to_consider_gt: [batch_size, max_num_relations] int

        return_dict = dict()
        with tf.variable_scope("relation_classification_metrics"):
            labels = relations_to_consider_gt
            # labels: [batch_size, max_num_relations] int
            num_classes = self._flags.num_classes
            labels_one_hot = tf.cast(tf.one_hot(indices=labels, depth=num_classes), dtype=tf.bool)
            # labels_one_hot: [batch_size, max_num_relations, num_classes] int
            probabilities = layers.softmax(logits)
            # probabilities: [batch_size, max_num_relations, num_classes] float
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            # predictions: [batch_size, max_num_relations] int
            mask = tf.sequence_mask(num_relations_to_consider, dtype=tf.int32)
            # mask: [batch_size, max_num_relations] int
            update_collections = [tf.compat.v1.GraphKeys.UPDATE_OPS]

            # ACCURACY
            tf_acc, tf_acc_update_op = tf.compat.v1.metrics.accuracy(labels=labels,
                                                                     predictions=predictions,
                                                                     weights=mask,
                                                                     updates_collections=update_collections,
                                                                     name='ACCURACY')
            return_dict['ACCURACY'] = (tf_acc, tf_acc_update_op)

            # PRECISION
            tf_prec, tf_prec_update_op = tf.compat.v1.metrics.precision(labels=labels,
                                                                        predictions=predictions,
                                                                        weights=mask,
                                                                        updates_collections=update_collections,
                                                                        name='PRECISION')
            return_dict['PRECISION'] = (tf_prec, tf_prec_update_op)

            # RECALL
            tf_recall, tf_recall_update_op = tf.compat.v1.metrics.recall(labels=labels,
                                                                         predictions=predictions,
                                                                         weights=mask,
                                                                         updates_collections=update_collections,
                                                                         name='RECALL')
            return_dict['RECALL'] = (tf_recall, tf_recall_update_op)

            # F1_SCORE
            tf_f1, tf_f1_update_op = f1_score(tf_prec, tf_recall, name='F1_SCORE')
            return_dict['F1_SCORE'] = (tf_f1, tf_f1_update_op)

            # AUC_PR
            return_dict['AUC_PR'] = \
                tf.compat.v1.metrics.auc(labels=labels_one_hot,
                                         predictions=probabilities,
                                         weights=mask,
                                         summation_method='careful_interpolation',
                                         updates_collections=update_collections,
                                         name='AUC_PR',
                                         curve='PR')

            # AUC_ROC
            return_dict['AUC_ROC'] = \
                tf.compat.v1.metrics.auc(labels=labels_one_hot,
                                         predictions=probabilities,
                                         weights=mask,
                                         summation_method='careful_interpolation',
                                         updates_collections=update_collections,
                                         name='AUC_PR',
                                         curve='ROC')

            # PR/ROC_CURVE
            (pr_curve, pr_update_op), (roc_curve, roc_update_op) = \
                curve_streaming_op(labels=labels_one_hot,
                                   predictions=probabilities,
                                   num_thresholds=201,
                                   weights=mask,
                                   updates_collections=update_collections,
                                   name='PR_ROC_CURVE',
                                   curve='both')
            return_dict['PR_CURVE'] = (pr_curve, pr_update_op)
            return_dict['ROC_CURVE'] = (roc_curve, roc_update_op)

        # logits_per_classifier = self._graph_out['logits']
        # # logits_per_classifier: dictionary of [batch_size, max_num_relations, num_classes] float
        #
        # num_relations_to_consider = self._inputs['num_relations_to_consider']
        # # num_relations_to_consider: dictionary of [batch_size] int
        #
        # relations_to_consider_gt = self._targets['relations_to_consider_gt']
        # # relations_to_consider_gt: dictionary of [batch_size, max_num_relations] int
        #
        # return_dict = {}
        #
        # with tf.variable_scope("relation_classification_metrics"):
        #     # evaluate for each classifier
        #     for idx, classifier_name in enumerate(self._flags.classifiers):
        #         labels = relations_to_consider_gt[classifier_name]
        #         # labels: [batch_size, max_num_relations] int
        #         num_classes = self._flags.num_classes_per_classifier[idx]
        #         labels_one_hot = tf.cast(tf.one_hot(indices=labels, depth=num_classes), dtype=tf.bool)
        #         # labels_one_hot: [batch_size, max_num_relations, num_classes] int
        #         probabilities = layers.softmax(logits_per_classifier[classifier_name])
        #         # probabilities: [batch_size, max_num_relations, num_classes] float
        #         predictions = tf.argmax(logits_per_classifier[classifier_name], axis=-1, output_type=tf.int32)
        #         # predictions: [batch_size, max_num_relations] int
        #         mask = tf.sequence_mask(num_relations_to_consider[classifier_name], dtype=tf.int32)
        #         # mask: [batch_size, max_num_relations] int
        #         update_collections = [tf.compat.v1.GraphKeys.UPDATE_OPS]
        #
        #         # ACCURACY
        #         tf_acc, tf_acc_update_op = tf.metrics.accuracy(labels=labels,
        #                                                        predictions=predictions,
        #                                                        weights=mask,
        #                                                        updates_collections=update_collections,
        #                                                        name='ACCURACY_' + classifier_name)
        #         return_dict['ACCURACY_' + classifier_name] = (tf_acc, tf_acc_update_op)
        #
        #         # PRECISION
        #         tf_prec, tf_prec_update_op = tf.metrics.precision(labels=labels,
        #                                                           predictions=predictions,
        #                                                           weights=mask,
        #                                                           updates_collections=update_collections,
        #                                                           name='PRECISION_' + classifier_name)
        #         return_dict['PRECISION_' + classifier_name] = (tf_prec, tf_prec_update_op)
        #
        #         # RECALL
        #         tf_recall, tf_recall_update_op = tf.metrics.recall(labels=labels,
        #                                                            predictions=predictions,
        #                                                            weights=mask,
        #                                                            updates_collections=update_collections,
        #                                                            name='RECALL_' + classifier_name)
        #         return_dict['RECALL_' + classifier_name] = (tf_recall, tf_recall_update_op)
        #
        #         # F1_SCORE
        #         tf_f1, tf_f1_update_op = f1_score(tf_prec, tf_recall, name='F1_SCORE_' + classifier_name)
        #         return_dict['F1_SCORE_' + classifier_name] = (tf_f1, tf_f1_update_op)
        #
        #         # AUC_PR
        #         return_dict['AUC_PR_' + classifier_name] = \
        #             tf.compat.v1.metrics.auc(labels=labels_one_hot,
        #                                      predictions=probabilities,
        #                                      weights=mask,
        #                                      updates_collections=update_collections,
        #                                      name='AUC_PR_' + classifier_name,
        #                                      curve='PR')
        #
        #         # AUC_ROC
        #         return_dict['AUC_ROC_' + classifier_name] = \
        #             tf.compat.v1.metrics.auc(labels=labels_one_hot,
        #                                      predictions=probabilities,
        #                                      weights=mask,
        #                                      updates_collections=update_collections,
        #                                      name='AUC_PR_' + classifier_name,
        #                                      curve='ROC')
        #
        #         # PR/ROC_CURVE
        #         (pr_curve, pr_update_op), (roc_curve, roc_update_op) = \
        #             curve_streaming_op(labels=labels_one_hot,
        #                                predictions=probabilities,
        #                                num_thresholds=201,
        #                                weights=mask,
        #                                updates_collections=update_collections,
        #                                name='PR_ROC_CURVE_' + classifier_name,
        #                                curve='both')
        #         return_dict['PR_CURVE_' + classifier_name] = (pr_curve, pr_update_op)
        #         return_dict['ROC_CURVE_' + classifier_name] = (roc_curve, roc_update_op)
        return return_dict

    def get_placeholder(self):
        from tensorflow.compat.v1 import placeholder as ph
        ph_dict = dict()
        ph_dict['num_nodes'] = ph(tf.int32, [None], name='num_nodes')  # [batch_size]
        ph_dict['num_interacting_nodes'] = ph(tf.int32, [None], name='num_interacting_nodes')  # [batch_size]
        ph_dict['interacting_nodes'] = ph(tf.int32, [None, None, 2],
                                          name='interacting_nodes')  # [batch_size, max_num_interacting_nodes, 2]

        # add node features if present
        if 'node_feature_dim' in self._flags.input_params and self._flags.input_params["node_feature_dim"] > 0:
            # feature dim by masking
            if 'node_input_feature_mask' in self._flags.input_params:
                node_feature_dim = self._flags.input_params["node_input_feature_mask"].count(True) if \
                    self._flags.input_params["node_input_feature_mask"] else self._flags.input_params[
                    "node_feature_dim"]
            else:
                node_feature_dim = self._flags.input_params["node_feature_dim"]
            # [batch_size, max_num_nodes, node_feature_dim]
            ph_dict['node_features'] = ph(tf.float32, [None, None, node_feature_dim], name='node_features')

        # add edge features if present
        if 'edge_feature_dim' in self._flags.input_params and self._flags.input_params["edge_feature_dim"] > 0:
            # feature dim by masking
            if 'edge_input_feature_mask' in self._flags.input_params:
                edge_feature_dim = self._flags.input_params["edge_input_feature_mask"].count(True) if \
                    self._flags.input_params["edge_input_feature_mask"] else self._flags.input_params[
                    "edge_feature_dim"]
            else:
                edge_feature_dim = self._flags.input_params["edge_feature_dim"]
            # [batch_size, max_num_interacting_nodes, edge_feature_dim]
            ph_dict['edge_features'] = ph(tf.float32, [None, None, edge_feature_dim], name='edge_features')

        # add visual features
        if self._flags.image_input:
            img_channels = 1
            if 'load_mode' in self._flags.input_params and self._flags.input_params['load_mode'] == 'RGB':
                img_channels = 3
            # [batch_size, pad_height, pad_width, channels] float
            ph_dict['image'] = ph(tf.float32, [None, None, None, img_channels], name="image")
            # [batch_size, 3] int
            ph_dict['image_shape'] = ph(tf.int32, [None, 3], name="image_shape")
            # [batch_size, max_num_nodes, 2, max_num_points_visual_regions_nodes] float
            ph_dict['visual_regions_nodes'] = ph(tf.float32, [None, None, 2, None], name="visual_regions_nodes")
            # [batch_size, max_num_nodes] int
            ph_dict['num_points_visual_regions_nodes'] = ph(tf.int32, [None, None],
                                                            name="num_points_visual_regions_nodes")
            # [batch_size, max_num_nodes, 2, max_num_points_visual_regions_edges] float
            ph_dict['visual_regions_edges'] = ph(tf.float32, [None, None, 2, None], name="visual_regions_edges")
            # [batch_size, max_num_nodes] int
            ph_dict['num_points_visual_regions_edges'] = ph(tf.int32, [None, None],
                                                            name="num_points_visual_regions_edges")

        # relations for evaluation
        # [batch_size, max_num_relations, num_relation_components]
        ph_dict['relations_to_consider_belong_to_same_instance'] = \
            ph(tf.int32, [None, None, self._flags.num_relation_components],
               name='relations_to_consider_belong_to_same_instance')
        # ph_dict['relations_to_consider'] = dict()
        # for idx, classifier_name in enumerate(self._flags.classifiers):
        #     # [batch_size, max_num_relations, num_relation_components]
        #     ph_dict['relations_to_consider'][classifier_name] = \
        #         ph(tf.int32, [None, None, self._flags.num_relation_components_per_classifier[idx]],
        #            name='relations_to_consider_' + classifier_name)
        #     # # [batch_size]
        #     # ph_dict['num_relations_to_consider'][classifier_name] = \
        #     #     ph(tf.int32, [None], name='num_relations_to_consider_' + classifier_name)
        return ph_dict

    def _define_output(self):
        name_output = 'output_belong_to_same_instance'
        tf.identity(layers.softmax(self._graph_out['logits']), name=name_output)
        # for classifier_name in self._flags.classifiers:
        #     name_output = 'output_' + classifier_name
        #     tf.identity(layers.softmax(self._graph_out['logits'][classifier_name]), name=name_output)
        #     # output nodes are softmax'd logits per classifier

    def get_output_nodes(self, has_graph=True):
        if has_graph:
            self._define_output()
        return 'output_belong_to_same_instance'
        # names = []
        # for classifier_name in self._flags.classifiers:
        #     name_output = 'output_' + classifier_name
        #     names.append(name_output)
        # return ','.join(names)  # return names as comma separated string without spaces

    def get_predictions(self):
        return self._graph_out['logits']

    def get_target_keys(self):
        # is a dict() -> {classifier: gt_relations}
        return 'relations_to_consider_gt'

    # @staticmethod
    # def _create_sequence_mask(lengths, maxlen=None, dtype=np.bool):
    #     # default 'maxlen' is maximum value in 'lengths'
    #     if maxlen is None:
    #         maxlen = np.max(lengths)
    #     if maxlen.shape is not None and len(maxlen.shape) != 0:
    #         raise ValueError("maxlen must be scalar for sequence_mask")
    #
    #     # The basic idea is to compare a range row vector of size maxlen:
    #     # [0, 1, 2, 3, 4]
    #     # to length as a matrix with 1 column: [[1], [3], [2]].
    #     # Because of broadcasting on both arguments this comparison results
    #     # in a matrix of size (len(lengths), maxlen)
    #     row_vector = np.arange(maxlen, dtype=maxlen.dtype)
    #     matrix = np.expand_dims(lengths, axis=-1)
    #     result = row_vector < matrix
    #
    #     if dtype is None or result.dtype == dtype:
    #         return result
    #     else:
    #         return result.astype(dtype)

    def get_early_stopping_params(self):
        # return in that order:
        # - metric_name to evaluate
        # - boolean indicating if a higher metric is better
        return "loss", False

    def export_helper(self):
        pass
