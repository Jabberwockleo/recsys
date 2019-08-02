#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : variable_average.py
# Author            : Wan Li
# Date              : 27.01.2019
# Last Modified Date: 27.01.2019
# Last Modified By  : Wan Li

import tensorflow as tf

def apply(sequence, seq_len):
    """
        Average variable squence
        Params:
            sequence: tensor of user item squences, shaped (batch_size, sequence_max_len, embedding_size)
            seq_len: tensor of item sequence length, shaped (batch_size, )
        Return:
            tensor of averaged representation, shaped (batch_size, embedding_size)
    """
    seq_mask = tf.sequence_mask(seq_len, tf.shape(sequence)[1], dtype=tf.float32)
    seq_mask = tf.expand_dims(seq_mask, axis=2)
    sum_tensor = tf.reduce_sum(sequence * seq_mask, axis=1)
    avg_tensor = tf.math.divide(sum_tensor,
        tf.expand_dims(tf.dtypes.cast(seq_len, tf.float32), axis=1))
    return avg_tensor
    