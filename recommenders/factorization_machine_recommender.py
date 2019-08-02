#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : factorization_machine_recommender.py
# Author            : Wan Li
# Date              : 26.07.2019
# Last Modified Date: 26.07.2019
# Last Modified By  : Wan Li

import tensorflow as tf
import recsys.recommenders.recommender_base as recommender_base
import recsys.modules.interactions.fm_layer as fm_layer
import recsys.modules.extractions.fully_connected_layer as fully_connected_layer

def FactorizationMachineRecommender(feature_dim, factor_dim=5, init_model_dir=None, save_model_dir='FMRec', l2_reg=None, train=True, serve=False):
    """
        Vanilla FM recommender
        Model: F(X) -> score
    """
    rec = recommender_base.Recommender(init_model_dir=init_model_dir,
        save_model_dir=save_model_dir, train=train, serve=serve)
    
    @rec.traingraph.inputgraph(outs=['X1', 'X2', 'dy'])
    def train_input_graph(subgraph):
        subgraph['X1'] = tf.placeholder(tf.float32, shape=[None, feature_dim], name="X1")
        subgraph['X2'] = tf.placeholder(tf.float32, shape=[None, feature_dim], name="X2")
        subgraph['dy'] = tf.placeholder(tf.float32, shape=[None], name="dy")
        subgraph.register_global_input_mapping({'x1': subgraph['X1'],
                                                'x2': subgraph['X2'],
                                                'label': subgraph['dy']})

    @rec.servegraph.inputgraph(outs=['X'])
    def serve_input_graph(subgraph):
        subgraph['X'] = tf.placeholder(tf.float32, shape=[None, feature_dim], name="X")
        subgraph.register_global_input_mapping({'x': subgraph['X']})
    
    @rec.traingraph.interactiongraph(ins=['X1', 'X2', 'dy'])
    def train_fushion_graph(subgraph):
        linear1 = fully_connected_layer.apply(subgraph['X1'], [1], subgraph,
            relu_in=False, relu_mid=False, relu_out=False,
            dropout_in=None, dropout_mid=None, dropout_out=None,
            bias_in=True, bias_mid=True, bias_out=True, batch_norm=False,
            train=False, l2_reg=l2_reg, scope='LinearComponent')
        linear1 = tf.squeeze(linear1) # shaped [None, ]
        interactive1 = fm_layer.apply(subgraph['X1'], factor_dim, l2_weight=0.01, scope="InteractiveComponent")
        interactive1 = tf.squeeze(tf.math.reduce_sum(interactive1, axis=1))
        
        linear2 = fully_connected_layer.apply(subgraph['X2'], [1], subgraph,
            relu_in=False, relu_mid=False, relu_out=False,
            dropout_in=None, dropout_mid=None, dropout_out=None,
            bias_in=True, bias_mid=True, bias_out=True, batch_norm=False,
            train=False, l2_reg=l2_reg, scope='LinearComponent')
        linear2 = tf.squeeze(linear2)
        interactive2 = fm_layer.apply(subgraph['X2'], factor_dim, l2_weight=0.01, scope="InteractiveComponent")
        interactive2 = tf.squeeze(tf.math.reduce_sum(interactive2, axis=1))
        dy_tilde = (linear1 + interactive1) - (linear2 + interactive2)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=subgraph['dy'], logits=dy_tilde, name='loss')
        subgraph.register_global_loss(tf.reduce_mean(loss))
        subgraph.register_global_output(subgraph['dy'])
        subgraph.register_global_output(dy_tilde)
        tf.summary.scalar('loss', tf.reduce_mean(loss))
        summary = tf.summary.merge_all()
        subgraph.register_global_summary(summary)

    @rec.servegraph.interactiongraph(ins=['X'])
    def serve_fusion_graph(subgraph):
        linear = fully_connected_layer.apply(subgraph['X'], [1], subgraph,
            relu_in=False, relu_mid=False, relu_out=False,
            dropout_in=None, dropout_mid=None, dropout_out=None,
            bias_in=True, bias_mid=True, bias_out=True, batch_norm=False,
            train=False, l2_reg=l2_reg, scope='LinearComponent')
        linear = tf.squeeze(linear) # shaped [None, ]
        interactive = fm_layer.apply(subgraph['X'], factor_dim, l2_weight=0.01, scope="InteractiveComponent")
        interactive = tf.squeeze(tf.math.reduce_sum(interactive, axis=1))
        score = linear + interactive
        subgraph.register_global_output(score)

    @rec.traingraph.optimizergraph
    def optimizer_graph(subgraph):
        losses = tf.add_n(subgraph.get_global_losses())
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
        subgraph.register_global_operation(optimizer.minimize(losses))

    @rec.traingraph.connector
    def train_connect(graph):
        graph.interactiongraph['X1'] = graph.inputgraph['X1']
        graph.interactiongraph['X2'] = graph.inputgraph['X2']
        graph.interactiongraph['dy'] = graph.inputgraph['dy']

    @rec.servegraph.connector
    def serve_connect(graph):
        graph.interactiongraph['X'] = graph.inputgraph['X']

    return rec
