#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : featurizer.py
# Author            : Wan Li
# Date              : 23.07.2019
# Last Modified Date: 23.07.2019
# Last Modified By  : Wan Li

class FeaturizerBase(object):
    """
        Featurizer is for combining side information of user/item to a vector
    """
    def feature_dim(self):
        """
           Dimension of the 1D feature vector
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
                1D float array sized self.feature_dim()
        """
        raise "Please override."
        return []
