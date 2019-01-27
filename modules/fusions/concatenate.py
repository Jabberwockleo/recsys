#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : concatenate.py
# Author            : Wan Li
# Date              : 27.01.2019
# Last Modified Date: 27.01.2019
# Last Modified By  : Wan Li

import tensorflow as tf

def apply(tensor_array):
    """
        Concatenate feature tensors
        Params:
            tensor_array: array of tensors shaped (batch_size, X)
        Return:
            tensor of concatenated representation, shaped (batch_size, sum(X))
    """
    return tf.concat(tensor_array, axis=1)
