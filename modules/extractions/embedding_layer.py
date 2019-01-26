#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : embedding_layer.py
# Author            : Wan Li
# Date              : 26.01.2019
# Last Modified Date: 26.01.2019
# Last Modified By  : Wan Li

import tensorflow as tf

def apply(shape, id_=None, l2_reg=None, init='normal', subgraph=None, scope=None):
    """
        Embedding layer (aka. latent factor layer)
        Params:
            shape: embedding shape
            id_: ids to be looked up
            subgraph: to whose optimizer losses add if l2_reg is not None
    """
    if init == 'normal':
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)
    elif init == 'zero':
        initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    else:
        initializer = tf.constant_initializer(value=init, dtype=tf.float32)

    with tf.variable_scope(scope, default_name='embedding_layer', reuse=tf.AUTO_REUSE):
        embedding = tf.get_variable('embedding', shape=shape, trainable=True,
                                      initializer=initializer)
        if id_ is None:
            output = None
        else:
            output = tf.nn.embedding_lookup(embedding, id_)
        
        if l2_reg is not None:
            subgraph.register_global_loss(l2_reg * tf.nn.l2_loss(output))

    return embedding, output