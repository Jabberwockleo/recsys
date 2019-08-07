#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : feeds_sampler.py
# Author            : Wan Li
# Date              : 01.08.2019
# Last Modified Date: 02.08.2019
# Last Modified By  : Wan Li

import random
import numpy as np

from recsys.samplers.sampler_base import Sampler

def create_training_sampler(dataset, featurizer, max_pos_neg_per_user=5, num_process=5, seed=100, batch_size=10):
    """
        Creates a ranknet sampler for training:
        Params:
            dataset: Dataset instance, sampling user and positive/negative items
            featurizer: Featurizer instance,
                contains featurizer.featurize(user_id, item_id)-> dict
                    user_demography_vec: float[]
                    user_stat_vec: float[]
                    user_history_vec: int[]
                    user_history_len: int
                    item_meta_vec: float[]
                    item_stat_vec: float[]
                    item_id: int
                    context_hour: float[]
                and contains
                    fea_user_demography_dim() -> int
                    fea_user_stat_dim() -> int
                    fea_user_history_dim() -> int
                    fea_item_meta_dim() -> int
                    fea_item_stat_dim() -> int
                    fea_context_hour_dim() -> int
                    total_item_num() -> int
            max_pos_neg_per_user: random sample at most k items from positive items and negative items each to generate pairs
            num_process: number of feeding processed
            seed: seed for randomizer
    """
    random.seed(seed)
    def batch(dataset, featurizer, max_pos_neg_per_user=max_pos_neg_per_user):
        """
            Batcher function
            Returns: ndarray shaped (2*batch_size,)
                typed [
                ('user_demography_vec', (np.float32, featurizer.fea_user_demography_dim())),
                ('user_stat_vec', (np.float32, featurizer.fea_user_stat_dim())),
                ('user_history_vec', (np.float32, featurizer.fea_user_history_dim())),
                ('user_history_len', np.int32)
                ('item_meta_vec_1', (np.float32, featurizer.fea_item_meta_dim())),
                ('item_stat_vec_1', (np.float32, featurizer.fea_item_stat_dim())),
                ('item_id_1', np.int32)
                ('item_meta_vec_2', (np.float32, featurizer.fea_item_meta_dim())),
                ('item_stat_vec_2', (np.float32, featurizer.fea_item_stat_dim())),
                ('item_id_2', np.int32)
                ('context_hour', (np.float32, featurizer.fea_context_hour_dim())),
                ('label', np.int32)]
            assume n positive samples, m negative samples for each user, 2mn samples per user, 2 samples per iteration
        """
        fea_user_demography_dim = featurizer.fea_user_demography_dim()
        fea_user_stat_dim = featurizer.fea_user_stat_dim()
        fea_user_history_dim = featurizer.fea_user_history_dim()
        fea_item_meta_dim = featurizer.fea_item_meta_dim()
        fea_item_stat_dim = featurizer.fea_item_stat_dim()
        fea_context_hour_dim = featurizer.fea_context_hour_dim()
        input_data = None
        sample_cnt = 0
        warm_users = dataset.warm_users(threshold=max_pos_neg_per_user)
        while True:
            # get a random user
            user_id = random.sample(warm_users, 1)[0]
            # get positive items
            pos_items = dataset.get_positive_items(user_id, sort=False)
            pos_sample_num = min(len(pos_items), max_pos_neg_per_user)
            pos_items = random.sample(pos_items, pos_sample_num)
            if len(pos_items) == 0:
                continue
            # get negative items
            neg_items = dataset.get_negative_items(user_id)
            neg_sample_num = min(len(neg_items), max_pos_neg_per_user)
            neg_items = random.sample(neg_items, neg_sample_num)
            if len(neg_items) == 0:
                continue
            for pos_item in pos_items:
                for neg_item in neg_items:
                    pos_dict = featurizer.featurize(user_id, pos_item)
                    neg_dict = featurizer.featurize(user_id, neg_item)
                    sample = np.zeros(2, dtype=[
                        ('user_demography_vec', (np.float32, fea_user_demography_dim)),
                        ('user_stat_vec', (np.float32, fea_user_stat_dim)),
                        ('user_history_vec', (np.float32, fea_user_history_dim)),
                        ('user_history_len', np.int32),
                        ('item_meta_vec_1', (np.float32, fea_item_meta_dim)),
                        ('item_stat_vec_1', (np.float32, fea_item_stat_dim)),
                        ('item_id_1', np.int32),
                        ('item_meta_vec_2', (np.float32, fea_item_meta_dim)),
                        ('item_stat_vec_2', (np.float32, fea_item_stat_dim)),
                        ('item_id_2', np.int32),
                        ('context_hour', (np.float32, fea_context_hour_dim)),
                        ('label', np.int32)])
                    sample[0] = (
                        pos_dict['user_demography_vec'],
                        pos_dict['user_stat_vec'],
                        pos_dict['user_history_vec'],
                        pos_dict['user_history_len'],
                        pos_dict['item_meta_vec'],
                        pos_dict['item_stat_vec'],
                        pos_dict['item_id'],
                        neg_dict['item_meta_vec'],
                        neg_dict['item_stat_vec'],
                        neg_dict['item_id'],
                        pos_dict['context_hour'],
                        1)
                    sample[1] = (
                        pos_dict['user_demography_vec'],
                        pos_dict['user_stat_vec'],
                        pos_dict['user_history_vec'],
                        pos_dict['user_history_len'],
                        neg_dict['item_meta_vec'],
                        neg_dict['item_stat_vec'],
                        neg_dict['item_id'],
                        pos_dict['item_meta_vec'],
                        pos_dict['item_stat_vec'],
                        pos_dict['item_id'],
                        pos_dict['context_hour'],
                        1)
                    if input_data is None:
                        input_data = sample
                    else:
                        input_data = np.hstack([input_data, sample])
                    sample_cnt += 1
                    if sample_cnt == batch_size:
                        yield input_data # (2*batch_size,) per iteration
                        input_data = None
                        sample_cnt = 0
        raise "unreachable!"
    s = Sampler(dataset=dataset, generate_batch=batch, evaluation_type="SAMPLED", featurizer=featurizer, num_process=num_process)

    return s


def create_evaluation_sampler(dataset, featurizer, max_pos_neg_per_user=20, seed=100):
    """
        Creates a ranknet sampler for evaluation:
        Params:
            dataset: Dataset instance, sampling user and positive/negative items
            featurizer: Featurizer instance,
                contains featurizer.featurize(user_id, item_id)-> dict
                    user_demography_vec: float[]
                    user_stat_vec: float[]
                    user_history_vec: int[]
                    user_history_len: int
                    item_meta_vec: float[]
                    item_stat_vec: float[]
                    item_id: int
                    context_hour: float[]
                and contains
                    fea_user_demography_dim() -> int
                    fea_user_stat_dim() -> int
                    fea_user_history_dim() -> int
                    fea_item_meta_dim() -> int
                    fea_item_stat_dim() -> int
                    fea_context_hour_dim() -> int
                    total_item_num() -> int
            max_pos_neg_per_user: random sample at most k items from positive items and negative items each to generate pairs
            seed: seed for randomizer
    """
    random.seed(seed)
    def batch(dataset, featurizer, max_pos_neg_per_user=max_pos_neg_per_user):
        """
            Batcher function
            Returns: (labels, feature_vecs)
                labels: array of labels, 1 for positive item, -1 for negative item. Marks for later rank score classification
                feature_vecs: ndarray shaped (2*batch_size,)
                typed [
                ('user_demography_vec', (np.float32, featurizer.fea_user_demography_dim())),
                ('user_stat_vec', (np.float32, featurizer.fea_user_stat_dim())),
                ('user_history_vec', (np.float32, featurizer.fea_user_history_dim())),
                ('user_history_len', np.int32)
                ('item_meta_vec', (np.float32, featurizer.fea_item_meta_dim())),
                ('item_stat_vec', (np.float32, featurizer.fea_item_stat_dim())),
                ('item_id', np.int32)
                ('context_hour', (np.float32, featurizer.fea_context_hour_dim()))]
            assume n positive samples, m negative samples for each user, ? = m+n per user, (1,) per iteration
        """
        fea_user_demography_dim = featurizer.fea_user_demography_dim()
        fea_user_stat_dim = featurizer.fea_user_stat_dim()
        fea_user_history_dim = featurizer.fea_user_history_dim()
        fea_item_meta_dim = featurizer.fea_item_meta_dim()
        fea_item_stat_dim = featurizer.fea_item_stat_dim()
        fea_context_hour_dim = featurizer.fea_context_hour_dim()

        while True:
            for user_id in dataset.warm_users(threshold=max_pos_neg_per_user):
                # get positive items
                pos_items = dataset.get_positive_items(user_id, sort=False)
                pos_sample_num = min(len(pos_items), max_pos_neg_per_user)
                pos_items = random.sample(pos_items, pos_sample_num)
                if len(pos_items) == 0:
                    continue
                # get negative items
                neg_items = dataset.get_negative_items(user_id)
                neg_sample_num = min(len(neg_items), max_pos_neg_per_user)
                neg_items = random.sample(neg_items, neg_sample_num)
                if len(neg_items) == 0:
                    continue
                labels = []
                input_data = None
                for pos_item in pos_items:
                    sample = np.zeros(1, dtype=[
                        ('user_demography_vec', (np.float32, fea_user_demography_dim)),
                        ('user_stat_vec', (np.float32, fea_user_stat_dim)),
                        ('user_history_vec', (np.float32, fea_user_history_dim)),
                        ('user_history_len', np.int32),
                        ('item_meta_vec', (np.float32, fea_item_meta_dim)),
                        ('item_stat_vec', (np.float32, fea_item_stat_dim)),
                        ('item_id', np.int32),
                        ('context_hour', (np.float32, fea_context_hour_dim))])
                    fea_dict = featurizer.featurize(user_id, pos_item)
                    sample[0] = (
                        fea_dict['user_demography_vec'],
                        fea_dict['user_stat_vec'],
                        fea_dict['user_history_vec'],
                        fea_dict['user_history_len'],
                        fea_dict['item_meta_vec'],
                        fea_dict['item_stat_vec'],
                        fea_dict['item_id'],
                        fea_dict['context_hour'])
                    if input_data is None:
                        input_data = sample
                    else:
                        input_data = np.hstack([input_data, sample])
                    labels.append(1)
                for neg_item in neg_items:
                    sample = np.zeros(1, dtype=[
                        ('user_demography_vec', (np.float32, fea_user_demography_dim)),
                        ('user_stat_vec', (np.float32, fea_user_stat_dim)),
                        ('user_history_vec', (np.float32, fea_user_history_dim)),
                        ('user_history_len', np.int32)
                        ('item_meta_vec', (np.float32, fea_item_meta_dim)),
                        ('item_stat_vec', (np.float32, fea_item_stat_dim)),
                        ('item_id', np.int32)
                        ('context_hour', (np.float32, fea_context_hour_dim))])
                    fea_dict = featurizer.featurize(user_id, neg_item)
                    sample[0] = (
                        fea_dict['user_demography_vec'],
                        fea_dict['user_stat_vec'],
                        fea_dict['user_history_vec'],
                        fea_dict['user_history_len'],
                        fea_dict['item_meta_vec'],
                        fea_dict['item_stat_vec'],
                        fea_dict['item_id'],
                        fea_dict['context_hour'])
                    if input_data is None:
                        input_data = sample
                    else:
                        input_data = np.hstack([input_data, sample])
                    labels.append(-1)
                yield labels, input_data # (m+n,) per iteration
                yield [], [] # signals end of one user after batches
            yield None, None # signal finish
    s = Sampler(dataset=dataset, generate_batch=batch, evaluation_type="SAMPLED", featurizer=featurizer, num_process=1)

    return s

if __name__ == "__main__":
    pass
