#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : recall.py
# Author            : Wan Li
# Date              : 25.01.2019
# Last Modified Date: 25.01.2019
# Last Modified By  : Wan Li

import numpy as np
from recsys.metrics.metric_base import Metric

class Recall(Metric):
    """
        Recall
    """
    def __init__(self, recall_at, name='Recall'):
        """
            Initializer
            Params:
                recall_at: array of topKs to evaluate
        """
        self._recall_at = np.array(recall_at)

        super(Recall, self).__init__(etype='rank', name=name)

    def compute(self, rank_above, negative_num):
        """
            Evaluation
            Params:
                rank_above: list[positive_list_index]aboved_negative_item_count
                negative_num: total number of negative items
        """
        del negative_num
        results = np.zeros(len(self._recall_at))
        for negative_above_count in rank_above:
            results += (negative_above_count <= self._recall_at).astype(np.float32)

        return results / len(rank_above)

if __name__ == "__main__":
    ins = Recall([10])
    res = ins.compute(list(range(100)), None)
    print(res)