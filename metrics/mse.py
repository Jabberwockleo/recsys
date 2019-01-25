#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : mse.py
# Author            : Wan Li
# Date              : 25.01.2019
# Last Modified Date: 25.01.2019
# Last Modified By  : Wan Li

import numpy as np
from recsys.metrics.metric_base import Metric

class MSE(Metric):
    """
        Mean squared error
    """
    def __init__(self, name='MSE'):
        """
            Initializer
        """
        super(MSE, self).__init__(etype='regression', name=name)

    def compute(self, predictions, labels):
        """
            Evaluation
        """
        return np.square(predictions - labels)

if __name__ == "__main__":
    ins = MSE()
    res = ins.compute(np.array([1, 0, 1]), np.array([1, 1, 0]))
    print(res)