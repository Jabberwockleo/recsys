#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : temporal_sampler.py
# Author            : Wan Li
# Date              : 27.01.2019
# Last Modified Date: 27.01.2019
# Last Modified By  : Wan Li

import random
import numpy as np

from recsys.samplers.sampler_base import Sampler

def create_training_sampler(dataset, batch_size, max_seq_len, num_process=5, seed=100):
    """
        Creates a temporal sampler for training
    """
    random.seed(seed)
    def batch(dataset, max_seq_len=max_seq_len, batch_size=batch_size):
        """
            Batcher function
            Returns: ndarray shaped (batch_size,)
                typed [('seq_item_id', (np.int32,  max_seq_len)), ('seq_len', np.int32), ('label', np.int32)]
        """
        while True:
            input_npy = np.zeros(batch_size, dtype=[
                ('seq_item_id', (np.int32,  max_seq_len)),
                ('seq_len', np.int32),
                ('label', np.int32)])
            
            for ind in range(batch_size):
                # get random users and all their positive items
                user_id = random.randint(0, dataset.total_users()-1)
                item_list = dataset.get_positive_items(user_id, sort=True)
                while len(item_list) <= 1:
                    user_id = random.randint(0, dataset.total_users()-1)
                    item_list = dataset.get_positive_items(user_id, sort=True)
                predict_pos = random.randint(1, len(item_list) - 1) # random split
                train_items = item_list[max(0, predict_pos-max_seq_len):predict_pos]
                padded_train_items = np.zeros(max_seq_len, np.int32)
                padded_train_items[:len(train_items)] = train_items
                input_npy[ind] = (padded_train_items, len(train_items), item_list[predict_pos])
            yield input_npy
    
    s = Sampler(dataset=dataset, generate_batch=batch, evaluation_type="FULL", num_process=num_process)
    
    return s

def create_evaluation_sampler(dataset, max_seq_len, seed=100):
    """
        Creates a temporal sampler for evaluation
    """
    random.seed(seed)
    def batch(dataset, max_seq_len=max_seq_len):
        """
            Batcher
            Returns:
                [label_item], item_history shaped (1,)
                    typed [('seq_item_id', (np.int32,  max_seq_len)), ('seq_len', np.int32)]
        """
        while True:
            for user_id in dataset.warm_users(threshold=5):
                input_npy = np.zeros(1, dtype=[('seq_item_id', (np.int32,  max_seq_len)),
                                                ('seq_len', np.int32)])
                
                item_list = dataset.get_positive_items(user_id, sort=True)
                if len(item_list) <= 1:
                    continue
                train_items = item_list[-max_seq_len-1:-1]
                pad_train_items = np.zeros(max_seq_len, np.int32)
                pad_train_items[:len(train_items)] = train_items
                input_npy[0] = (pad_train_items, len(train_items)) # input_npy is 1 length array
                yield [item_list[-1]], input_npy
                yield [], [] # signals end of one user after batches
            yield None, None # signal finish
            
    s = Sampler(dataset=dataset, generate_batch=batch, evaluation_type="FULL", num_process=1)
    
    return s
