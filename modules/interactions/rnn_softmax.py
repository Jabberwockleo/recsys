#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : rnn_softmax.py
# Author            : Wan Li
# Date              : 27.01.2019
# Last Modified Date: 27.01.2019
# Last Modified By  : Wan Li

import tensorflow as tf

def apply(sequence, seq_len, total_items, num_units, cell_type='gru', softmax_samples=None,
        label=None, train=True, subgraph=None, scope=None):
    """
        RNN softmax layer
        final layer, registers loss for training and prediction for serving
    """
    with tf.variable_scope(scope, default_name='RNNSoftmax', reuse=tf.AUTO_REUSE):
        if cell_type == 'gru':
            rnn_cell = tf.nn.rnn_cell.GRUCell(num_units)
        elif cell_type == 'lstm-c' or cell_type == 'lstm-h':
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units)
        else:
            assert False, "Invalid RNN cell type."

        _, rnn_state = tf.nn.dynamic_rnn(cell=rnn_cell, 
            inputs=sequence, 
            sequence_length=seq_len,
            dtype=tf.float32)
        weight = tf.get_variable('weights', shape=[total_items, num_units], trainable=True,
                                      initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('biases', shape=[total_items], trainable=True,
                                    initializer=tf.zeros_initializer())
        
        if cell_type == 'gru':
            rnn_tensor = rnn_state
        elif cell_type == 'lstm-c':
            rnn_tensor = rnn_state[0]
        elif cell_type == 'lstm-h':
            rnn_tensor = rnn_state[1]
        
        if train:
            if softmax_samples is not None:
                loss = tf.nn.sampled_sparse_softmax_loss(weight=weight, bias=bias, num_sampled=softmax_samples, 
                                                         num_classes=total_items, labels=label, inputs=rnn_tensor)
            else:
                logits = tf.matmul(rnn_tensor, tf.transpose(weight)) + bias
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
            subgraph.register_global_loss(tf.reduce_mean(loss))
        else:
            logits = tf.matmul(rnn_tensor, tf.transpose(weight)) + bias
            subgraph.register_global_output(logits)