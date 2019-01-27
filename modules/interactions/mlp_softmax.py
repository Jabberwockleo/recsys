#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : mlp_softmax.py
# Author            : Wan Li
# Date              : 26.01.2019
# Last Modified Date: 26.01.2019
# Last Modified By  : Wan Li

import tensorflow as tf
import recsys.modules.extractions.fully_connected_layer as fully_connected_layer

def apply(in_tensor, dims, subgraph, item_bias=None, extra=None,
               l2_reg=None, labels=None, dropout=None, train=None, scope=None):
    """
        MLP softmax layer
        final layer, registers loss for training and prediction for serving
    """
    with tf.variable_scope(scope, default_name='MLPSoftmax', reuse=tf.AUTO_REUSE):
        if train: 
            logits = fully_connected_layer.apply(in_tensor=in_tensor,
                dims=dims,
                subgraph=subgraph,
                bias_in=True,
                bias_mid=True,
                bias_out=False,
                dropout_mid=dropout,
                l2_reg=l2_reg,
                scope='mlp_reg')
        else:
            logits = fully_connected_layer.apply(in_tensor=in_tensor,
                dims=dims,
                subgraph=subgraph,
                bias_in=True,
                bias_mid=True,
                bias_out=False,
                l2_reg=l2_reg,
                scope='mlp_reg')

        if item_bias is not None:
            logits += item_bias
        
        if train:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            subgraph.register_global_loss(tf.reduce_mean(loss))
        else:
            subgraph.register_global_output(logits)
