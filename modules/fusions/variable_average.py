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
    seq_mask = tf.sequence_mask(seq_len, tf.shape(sequence)[1], dtype=tf.float32) # shaped [batch_size, sequence_max_len]
    seq_mask = tf.expand_dims(seq_mask, axis=2) # shaped [batch_size, sequence_max_len, 1]
    sum_tensor = tf.reduce_sum(sequence * seq_mask, axis=1) # shaped [batch_size, embedding_size]
    seq_len_tensor = tf.expand_dims(tf.dtypes.cast(seq_len, tf.float32), axis=1)
    # avoid divide zero
    safe_seq_len_tensor = tf.where(tf.equal(seq_len_tensor, 0),
            tf.add(seq_len_tensor, tf.constant(1, tf.float32)), seq_len_tensor)
    avg_tensor = tf.math.divide(sum_tensor, safe_seq_len_tensor)
    return avg_tensor
