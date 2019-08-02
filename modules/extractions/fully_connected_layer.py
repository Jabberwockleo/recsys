#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : fully_connected_layer.py
# Author            : Wan Li
# Date              : 26.01.2019
# Last Modified Date: 26.01.2019
# Last Modified By  : Wan Li

import tensorflow as tf

def apply(in_tensor, dims, subgraph, relu_in=False, relu_mid=True, relu_out=False,
    dropout_in=None, dropout_mid=None, dropout_out=None,
    bias_in=True, bias_mid=True, bias_out=True, batch_norm=False,
    train=True, l2_reg=None, scope=None):
    """
        Fully connected layers creator
        Params:
            in_tensor: input tensor
            dims: list of num_of_nodes for mid-layers, last layer is out_tensor
            subgraph: to whose optimizer losses add if l2_reg is not None
    """
    with tf.variable_scope(scope, default_name='fully_connected_layer', reuse=tf.AUTO_REUSE) as var_scope:

        _in = in_tensor

        if relu_in:
            _in = tf.nn.relu(_in)

        if dropout_in is not None:
            _in = tf.nn.dropout(_in, 1 - dropout_in) # Tensorflow uses keep_prob

        for index, _out_dim in enumerate(dims):
            mat = tf.get_variable('FC_' + '_' + str(index),
                shape=[_in.shape[1], _out_dim], trainable=True,
                initializer=tf.initializers.truncated_normal())
            tf.summary.histogram('FC' + '_' + str(index) + '/weights', mat)
            if index == 0:
                add_bias = bias_in
            elif index == len(dims) - 1:
                add_bias = bias_out
            else:
                add_bias = bias_mid

            if add_bias:
                _bias = tf.get_variable('bias_' + '_' + str(index), shape=[_out_dim], trainable=True,
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32))
                tf.summary.histogram('FC' + '_' + str(index) + '/bias', _bias)
                _out = tf.matmul(_in, mat) + _bias
            else:
                _out = tf.matmul(_in, mat)

            if index < len(dims) - 1:
                if relu_mid:
                    _out = tf.nn.relu(_out)
                if dropout_mid is not None:
                    _out = tf.nn.dropout(_out, 1 - dropout_mid)
            elif index == len(dims) - 1:
                if relu_out:
                    _out = tf.nn.relu(_out)
                if dropout_out is not None:
                    _out = tf.nn.dropout(_out, 1 - dropout_out)
            if batch_norm:
                _out = tf.contrib.layers.batch_norm(_out, fused=True, decay=0.95,
                                    center=True, scale=True, is_training=train,
                                    scope="bn_"+str(index), updates_collections=None)
            _in = _out

        if l2_reg is not None:
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope.name):
                # add to global loss,  accumulated by tf.add_n() for optimizer later
                subgraph.register_global_loss(l2_reg * tf.nn.l2_loss(var))

        tf.summary.histogram('FC' + '/out', _out)
        return _out
