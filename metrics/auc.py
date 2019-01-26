#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : auc.py
# Author            : Wan Li
# Date              : 25.01.2019
# Last Modified Date: 25.01.2019
# Last Modified By  : Wan Li

import numpy as np
from recsys.metrics.metric_base import Metric

class AUC(Metric):
    """
        Area Under Receiver Operating Characteristic Curve
    """
    def __init__(self, name='AUC'):
        """
            Initializer
        """
        super(AUC, self).__init__(etype='rank', name=name)

    def compute(self, rank_above, negative_num):
        """
            Evaluation
            Params:
                rank_above: list[positive_list_index]aboved_negative_item_count
                negative_num: total number of negative items
        """
        return np.mean(1 - rank_above / negative_num) # mean(1 - FPR), because TPR is smooth

if __name__ == "__main__":
    ins = AUC([50])
    res = ins.compute(np.array(list(range(100))), 100)
    print(res)