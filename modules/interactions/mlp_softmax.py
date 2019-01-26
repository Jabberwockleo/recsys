#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : mlp_softmax.py
# Author            : Wan Li
# Date              : 26.01.2019
# Last Modified Date: 26.01.2019
# Last Modified By  : Wan Li

import tensorflow as tf
import recsys.modules.extractions.fully_connected_layer as fully_connected_layer

def mlp_softmax(user, item, seq_len, max_seq_len, dims, subgraph, item_bias=None, extra=None,
               l2_reg=None, labels=None, dropout=None, train=None, scope=None):
    """
        MLP softmax layer
        final layer, registers loss for training and prediction for serving
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        
        # average item vectors user interacted with  
        seq_mask = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
        item = tf.reduce_mean(item * tf.expand_dims(seq_mask, axis=2), axis=1)

        if user is not None:
            in_tensor = tf.concat([user, item], axis=1)
        else:
            in_tensor = tf.concat([item], axis=1)

        if extra is not None:
            in_tensor = tf.concat([in_tensor, extra], axis=1)
        
        if train: 
            logits = fully_connected_layer(in_tensor=in_tensor,
                                 dims=dims,
                                 subgraph=subgraph,
                                 bias_in=True,
                                 bias_mid=True,
                                 bias_out=False,
                                 dropout_mid=dropout,
                                 l2_reg=l2_reg,
                                 scope='mlp_reg')
        else:
            logits = fully_connected_layer(in_tensor=in_tensor,
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
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                           logits=logits)
            subgraph.register_global_loss(tf.reduce_mean(loss))
        else:
            subgraph.register_global_output(logits)
