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


def evaluation_type():
    """
        Evaluation type used for model evaluation
        Return:
            FULL/SAMPLED
    """
    return "SAMPLED"


def create_training_sampler(dataset, featurizer, batch_size, max_pos_neg_per_user=5, num_process=5, seed=100):
    """
        Creates a ranknet sampler for training:
        Params:
            dataset: Dataset instance, sampling user and positive/negative items
            featurizer: Featurizer instance,
                contains featurizer.featurize(user_id, item_id)->float array of featurizer.feature_dim()
            batch_size: number of users per batch, for any user with n positive samples and m negative samples, it will generate 2nm training samples labeled 0/1
            max_pos_neg_per_user: random sample at most k items from positive items and negative items each to generate pairs
            num_process: number of feeding processed
            seed: seed for randomizer
    """
    random.seed(seed)
    def batch(dataset, featurizer, users_per_batch=batch_size, max_pos_neg_per_user=max_pos_neg_per_user):
        """
            Batcher function
            Returns: ndarray shaped (?,)
                typed [
                ('x1', (np.float32,  featurizer.feature_dim())), 
                ('x2', (np.float32,  featurizer.feature_dim())),
                ('label', np.int32)]
            assume n positive samples, m negative samples for each user, ? = 2mn * users_per_batch
        """
        dim = featurizer.feature_dim()
        while True:
            input_data = None
            for _ in range(users_per_batch):
                # get a random user
                user_id = random.randint(0, dataset.total_users()-1)
                # get positive items
                pos_items = dataset.get_positive_items(user_id, sort=False)
                pos_items = random.sample(pos_items, max_pos_neg_per_user)
                if len(pos_items) == 0:
                    continue
                # get negative items
                neg_items = dataset.get_negative_items(user_id, sort=False)
                neg_items = random.sample(neg_items, max_pos_neg_per_user)
                if len(neg_items) == 0:
                    continue
                for pos_item in pos_items:
                    for neg_item in neg_items:
                        sample = np.zeros(2, dtype=[
                            ('x1', (np.float32, dim)),
                            ('x2', (np.float32, dim)),
                            ('label', np.int32)])
                        pos_x = featurizer.featurize(user_id, pos_item)
                        neg_x = featurizer.featurize(user_id, neg_item)
                        sample[0] = (pos_x, neg_x, 1)
                        sample[1] = (pos_x, neg_x, 0)
                        if input_data is None:
                            input_data = sample
                        else:
                            input_data = np.hstack(input_date, sample)
            yield input_data
    s = Sampler(dataset=dataset, generate_batch=batch, num_process=num_process)
    
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
                feature_vecs: ndarray shaped (?,)
                typed [
                ('x1', (np.float32,  featurizer.feature_dim()))]
            assume n positive samples, m negative samples for each user, ? = m+n
        """
        dim = featurizer.feature_dim()
        while True:
            for user_id in dataset.warm_users(threshold=max_pos_neg_per_user):
                labels = []
                input_data = None
                # get positive items
                pos_items = dataset.get_positive_items(user_id, sort=False)
                pos_items = random.sample(pos_items, max_pos_neg_per_user)
                if len(pos_items) == 0:
                    continue
                # get negative items
                neg_items = dataset.get_negative_items(user_id, sort=False)
                neg_items = random.sample(neg_items, max_pos_neg_per_user)
                if len(neg_items) == 0:
                    continue
                for pos_item in pos_items:
                    sample = np.zeros(1, dtype=[('x1', (np.float32, dim))])
                    x = featurizer.featurize(user_id, pos_item)
                    sample[0] = (x)
                    if input_data is None:
                        input_data = sample
                    else:
                        input_data = np.hstack(input_date, sample)
                    labels.append(1)
                for neg_item in neg_items:
                    sample = np.zeros(1, dtype=[('x1', (np.float32, dim))])
                    x = featurizer.featurize(user_id, neg_item)
                    sample[0] = (x)
                    if input_data is None:
                        input_data = sample
                    else:
                        input_data = np.hstack(input_date, sample)
                    labels.append(-1)
                yield labels, input_data
                yield [], [] # signals end of one user after batches
            yield None, None # signal finish
    s = Sampler(dataset=dataset, generate_batch=batch, num_process=num_process)
    
    return s