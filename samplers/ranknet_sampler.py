#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : ranknet_sampler.py
# Author            : Wan Li
# Date              : 23.07.2019
# Last Modified Date: 23.07.2019
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
                contains featurizer.featurize(user_id, item_id)->float array of featurizer.feature_dim()
            max_pos_neg_per_user: random sample at most k items from positive items and negative items each to generate pairs
            num_process: number of feeding processed
            seed: seed for randomizer
    """
    random.seed(seed)
    def batch(dataset, featurizer, max_pos_neg_per_user=max_pos_neg_per_user):
        """
            Batcher function
            Returns: ndarray shaped (2,)
                typed [
                ('x1', (np.float32,  featurizer.feature_dim())), 
                ('x2', (np.float32,  featurizer.feature_dim())),
                ('label', np.int32)]
            assume n positive samples, m negative samples for each user, 2mn samples per user, 2 samples per iteration
        """
        dim = featurizer.feature_dim()
        input_data = None
        sample_cnt = 0
        while True:
            # get a random user
            user_id = random.randint(0, dataset.total_users()-1)
            # get positive items
            pos_items = dataset.get_positive_items(user_id, sort=False)
            pos_items = random.sample(pos_items, max_pos_neg_per_user)
            if len(pos_items) == 0:
                continue
            # get negative items
            neg_items = dataset.get_negative_items(user_id)
            neg_items = random.sample(neg_items, max_pos_neg_per_user)
            if len(neg_items) == 0:
                continue
            for pos_item in pos_items:
                for neg_item in neg_items:
                    pos_x = featurizer.featurize(user_id, pos_item)
                    neg_x = featurizer.featurize(user_id, neg_item)
                    sample = np.zeros(2, dtype=[
                        ('x1', (np.float32, dim)),
                        ('x2', (np.float32, dim)),
                        ('label', np.int32)])
                    sample[0] = (pos_x, neg_x, 1)
                    sample[1] = (neg_x, pos_x, 0)
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
                contains featurizer.featurize(user_id, item_id)->float array of featurizer.feature_dim()
            max_pos_neg_per_user: random sample at most k items from positive items and negative items each to generate pairs
            seed: seed for randomizer
    """
    random.seed(seed)
    def batch(dataset, featurizer, max_pos_neg_per_user=max_pos_neg_per_user):
        """
            Batcher function
            Returns: (labels, feature_vecs)
                labels: array of labels, 1 for positive item, -1 for negative item. Marks for later rank score classification
                feature_vecs: ndarray shaped (1,)
                typed [
                ('x1', (np.float32,  featurizer.feature_dim()))]
            assume n positive samples, m negative samples for each user, ? = m+n per user, (1,) per iteration
        """
        dim = featurizer.feature_dim()
        while True:
            for user_id in dataset.warm_users(threshold=max_pos_neg_per_user):
                # get positive items
                pos_items = dataset.get_positive_items(user_id, sort=False)
                pos_items = random.sample(pos_items, max_pos_neg_per_user)
                if len(pos_items) == 0:
                    continue
                # get negative items
                neg_items = dataset.get_negative_items(user_id)
                neg_items = random.sample(neg_items, max_pos_neg_per_user)
                if len(neg_items) == 0:
                    continue
                labels = []
                input_data = None
                for pos_item in pos_items:
                    sample = np.zeros(1, dtype=[('x', (np.float32, dim))])
                    x = featurizer.featurize(user_id, pos_item)
                    sample[0] = tuple([x])
                    if input_data is None:
                        input_data = sample
                    else:
                        input_data = np.hstack([input_data, sample])
                    labels.append(1)
                for neg_item in neg_items:
                    sample = np.zeros(1, dtype=[('x', (np.float32, dim))])
                    x = featurizer.featurize(user_id, neg_item)
                    sample[0] = tuple([x])
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