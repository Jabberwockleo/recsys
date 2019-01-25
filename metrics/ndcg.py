#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : ndcg.py
# Author            : Wan Li
# Date              : 25.01.2019
# Last Modified Date: 25.01.2019
# Last Modified By  : Wan Li

import numpy as np
from math import log
from recsys.metrics.metric_base import Metric

class NDCG(Metric):
    """
        Normalized Discounted Cumulative Gain
    """
    def __init__(self, ndcg_at, name='NDCG'):
        """
            Initializer
            Params:
                ndcg_at: array of topKs to evaluate
        """
        self._ndcg_at = np.array(ndcg_at)

        super(NDCG, self).__init__(etype='rank', name=name)

    def compute(self, rank_above, negative_num):
        """
            Evaluation
            Params:
                rank_above: list[positive_list_index]aboved_negative_item_count
                negative_num: total number of negative items
        """
        del negative_num
        denominator = 0.0
        for i in range(len(rank_above)):
            denominator += 1.0 / log(i + 2, 2)
        
        results = np.zeros(len(self._ndcg_at))
        for negative_above_count in rank_above:
            tmp = 1.0 / log(negative_above_count + 2, 2)
            results[negative_above_count < self._ndcg_at] += tmp
        
        return results / denominator

if __name__ == "__main__":
    ins = NDCG([50])
    res = ins.compute(list(range(100)), None)
    print(res)