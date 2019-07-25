#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : linear_recommender.py
# Author            : Wan Li
# Date              : 23.07.2019
# Last Modified Date: 23.07.2019
# Last Modified By  : Wan Li

import tensorflow as tf
import recsys.recommenders.recommender_base as recommender_base
import recsys.modules.extractions.fully_connected_layer as fully_connected_layer

def LinearRankNetRec(feature_dim, init_model_dir=None, save_model_dir='LinearRankNetRec', l2_reg=None, train=True, serve=False):
    """
        Linear Represented RankNet Recommender
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
    
    @rec.traingraph.fusiongraph(ins=['X1', 'X2'], outs=['dy_tilde'])
    def train_fushion_graph(subgraph):
        logits_1 = fully_connected_layer.apply(subgraph['X1'], [1], subgraph,
            relu_in=False, relu_mid=False, relu_out=False,
            dropout_in=None, dropout_mid=None, dropout_out=None,
            bias_in=True, bias_mid=True, bias_out=True, batch_norm=False,
            train=False, l2_reg=l2_reg, scope='Weights1dTensor')
        logits_2 = fully_connected_layer.apply(subgraph['X1'], [1], subgraph,
            relu_in=False, relu_mid=False, relu_out=False,
            dropout_in=None, dropout_mid=None, dropout_out=None,
            bias_in=True, bias_mid=True, bias_out=True, batch_norm=False,
            train=False, l2_reg=l2_reg, scope='Weights1dTensor')
        subgraph['dy_tilde'] = tf.squeeze(logits_1 - logits_2)
    
    @rec.traingraph.interactiongraph(ins=['dy_tilde', 'dy'])
    def train_interaction_graph(subgraph):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=subgraph['dy'], logits=subgraph['dy_tilde'], name='loss')
        subgraph.register_global_loss(tf.reduce_mean(loss))

    @rec.servegraph.fusiongraph(ins=['X'])
    def serve_fusion_graph(subgraph):
        logit = fully_connected_layer.apply(subgraph['X'], [1], subgraph,
            relu_in=False, relu_mid=False, relu_out=False,
            dropout_in=None, dropout_mid=None, dropout_out=None,
            bias_in=True, bias_mid=True, bias_out=True, batch_norm=False,
            train=False, l2_reg=l2_reg, scope='Weights1dTensor')
        subgraph.register_global_output(logit)

    @rec.traingraph.optimizergraph
    def optimizer_graph(subgraph):
        losses = tf.add_n(subgraph.get_global_losses())
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        subgraph.register_global_operation(optimizer.minimize(losses))

    @rec.traingraph.connector
    def train_connect(graph):
        graph.fusiongraph['X1'] = graph.inputgraph['X1']
        graph.fusiongraph['X2'] = graph.inputgraph['X2']
        graph.interactiongraph['dy'] = graph.inputgraph['dy']
        graph.interactiongraph['dy_tilde'] = graph.fusiongraph['dy_tilde']

    @rec.servegraph.connector
    def serve_connect(graph):
        graph.fusiongraph['X'] = graph.inputgraph['X']

    return rec

if __name__ == "__main__":
    LinearRankNetRec(feature_dim=3)