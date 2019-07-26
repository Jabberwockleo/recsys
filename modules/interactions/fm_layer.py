#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : fm_layer.py
# Author            : Wan Li
# Date              : 26.07.2019
# Last Modified Date: 26.07.2019
# Last Modified By  : Wan Li

import tensorflow as tf

def apply(in_tensor, factor_num=10, l2_weight=0.01, scope=None, seed=100):
    """
        Factorized interaction layer
        This layer change shapes as follows:
            in_tensor is shaped (None, dn), factor_num is fn, output is shaped (None, fn)
        To get merged score apply additional sum op with tf.math.reduce_sum(out_tensor, axis=1)
    """
    with tf.variable_scope(scope, default_name="FMLayer", reuse=tf.AUTO_REUSE):
        feature_num = in_tensor.shape[1]
        factor_mat = tf.get_variable(name='interaction_factors',
            shape=[feature_num, factor_num],
            initializer=tf.truncated_normal_initializer(
            stddev=0.001, seed=seed)) # shaped (feature_num, factor_num)
        # Compute point-wise product <vi, vj>xixj = <vixi, vjxj>,
        #     factor dimension is retained
        val_tensor = tf.reshape(in_tensor, shape=[-1, feature_num, 1]) # shaped [None, feature_num, factor_num]
        # Compute (sum(vixi))^2
        vx_sum_squared = tf.math.pow(
            tf.math.reduce_sum(
                tf.math.multiply(factor_mat, val_tensor),
            axis=1),
        2) # shaped [-1, factor_num]
        # Compute sum(vixi)^2
        vx_squared_sum = tf.math.reduce_sum(
            tf.math.multiply(tf.math.pow(factor_mat, 2), tf.math.pow(val_tensor, 2)),
        axis=1)
        # Compute sum(<vi,vj>xixj) with factor dim retained
        out_tensor = 0.5 * tf.math.subtract(vx_sum_squared, vx_squared_sum)

        return out_tensor

