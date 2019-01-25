#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : precision.py
# Author            : Wan Li
# Date              : 25.01.2019
# Last Modified Date: 25.01.2019
# Last Modified By  : Wan Li

import numpy as np
from recsys.metrics.metric_base import Metric

class Precision(Metric):
    """
        Precision
    """
    def __init__(self, precision_at, name='Precision'):
        """
            Initializer
            Params:
                precision_at: array of topKs to evaluate
        """
        self._precision_at = np.array(precision_at)

        super(Precision, self).__init__(etype='rank', name=name)

    def compute(self, rank_above, negative_num):
        """
            Evaluation
            Params:
                rank_above: list[positive_list_index]aboved_negative_item_count
                negative_num: total number of negative items
        """
        del negative_num
        results = np.zeros(len(self._precision_at))
        for negative_above_count in rank_above:
            results += (negative_above_count <= self._precision_at).astype(np.float32)

        return results / self._precision_at

if __name__ == "__main__":
    ins = Precision([10])
    res = ins.compute(list(range(100)), None)
    print(res)