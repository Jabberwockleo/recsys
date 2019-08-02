#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : feeds_featurizer.py
# Author            : Wan Li
# Date              : 02.08.2019
# Last Modified Date: 02.08.2019
# Last Modified By  : Wan Li

class FeedsFeaturizer(object):
    """
        Featurizer is for combining side information of user/item to a vector
    """
    def fea_user_demography_dim(self):
        """
           Dimension
        """
        raise "Please override."
        return 0

    def fea_user_stat_dim(self):
        """
           Dimension
        """
        raise "Please override."
        return 0

    def fea_user_history_dim(self):
        """
           Dimension of maximun history length
        """
        raise "Please override."
        return 0

    def fea_item_meta_dim(self):
        """
           Dimension
        """
        raise "Please override."
        return 0

    def fea_item_stat_dim(self):
        """
           Dimension
        """
        raise "Please override."
        return 0

    def fea_context_hour_dim(self):
        """
           Dimension
        """
        raise "Please override."
        return 0

    def total_item_num(self):
        """
           Dimension
        """
        raise "Please override."
        return 0

    def featurize(self, user_id, item_id):
        """
            Feature engineering
            Params:
                user_id: current user
                item_id: reference item
            Returns:
                dict:
                    user_demography_vec: float[]
                    user_stat_vec: float[]
                    user_history_vec: int[] # tail with zero padded
                    user_history_len: int
                    item_meta_vec: float[]
                    item_stat_vec: float[]
                    item_id: int
                    context_hour: float[]
        """
        raise "Please override."
        return {}

if __name__ == "__main__":
    pass
