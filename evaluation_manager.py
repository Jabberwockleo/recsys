#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : evaluation_manager.py
# Author            : Wan Li
# Date              : 26.01.2019
# Last Modified Date: 26.01.2019
# Last Modified By  : Wan Li

import numpy as np

class EvaluationManager(object):
    """
        Evaluation Manager
        Performs evalutations during training
    """
    def __init__(self, evaluators=[]):
        """
            Initializer
            Params:
                evaluators: evaluator instances
        """
        self.evaluators = evaluators

    def _full_rank(self, pos_samples, excluded_positive_samples, predictions):
        """
            Full rank for all items (e.g. softmax)
            Params:
                pos_samples: list of positive item_ids
                excluded_positive_samples: excluded from evaluation
                predictions: list[item_id]score scores of all items
            Returns (intermidiate data for evaluator):
                list[positive_item_pseudo_index]number_of_negative_item_ranked_above_this
                negative item number
        """
        pos_samples_set = set(pos_samples)
        pos_samples = np.array(pos_samples, dtype=np.int32)
        pos_predictions = predictions[pos_samples]

        excl_pos_samples_set = set(excluded_positive_samples)
        rank_above = np.zeros(len(pos_samples))

        pos_samples_len = len(pos_samples)
        for ind in range(len(predictions)):
            if ind not in excl_pos_samples_set and ind not in pos_samples_set:
                for pos_ind in range(pos_samples_len):
                    if pos_predictions[pos_ind] < predictions[ind]:
                        rank_above[pos_ind] += 1

        return rank_above, len(predictions) - len(excluded_positive_samples) - len(pos_samples)

    def _partial_rank(self, pos_scores, neg_scores):
        """
            Partial rank for some items
            Params:
                pos_scores: positive scores (free of item_id)
                neg_scores: negative scores (free of item_id)
            Returns (intermidiate data for evaluator):
                list[positive_item_index]number_of_negative_item_ranked_above_this
                negative item number
        """
        pos_scores = np.array(pos_scores)
        rank_above = np.zeros(len(pos_scores))
        pos_scores_len = len(pos_scores)

        for score in neg_scores:
            for pos_ind in range(pos_scores_len):
                if pos_scores[pos_ind] < score:
                    rank_above[pos_ind] += 1
        return rank_above, len(neg_scores)

    def full_evaluate(self, pos_samples, excluded_positive_samples, predictions):
        """
            Fully evaluate all items
            Return:
                dict[evalutator_name]evalute_score
        """
        results = {}
        rank_above, negative_num = self._full_rank(
            pos_samples, excluded_positive_samples, predictions)
        for evaluator in self.evaluators:
            if evaluator.etype == 'rank':
                results[evaluator.name] = evaluator.compute(
                    rank_above=rank_above, negative_num=negative_num)

        return results

    def partial_evaluate(self, pos_scores, neg_scores):
        """
            Partial evaluate some items
            Return:
                dict[evalutator_name]evalute_score
        """
        results = {}
        rank_above, negative_num = self._partial_rank(pos_scores, neg_scores)
        for evaluator in self.evaluators:
            if evaluator.etype == 'rank':
                results[evaluator.name] = evaluator.compute(rank_above=rank_above, negative_num=negative_num)

        return results
