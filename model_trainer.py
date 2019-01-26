#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : model_trainer.py
# Author            : Wan Li
# Date              : 23.01.2019
# Last Modified Date: 24.01.2019
# Last Modified By  : Wan Li

from termcolor import colored
from recsys.utils.evaluators import EvalManager
import sys
import numpy as np


class ModelTrainer(object):
    """
        Model Trainer
    """
    def __init__(self, model, train_iter_func=None, eval_iter_func=None):
        """
           Initializer:
           Params:
               train_iter_func: compute batch loss
               eval_iter_func: compute batch output
        """
        self._model = model
        # self._serve_batch_size = serve_batch_size
        if not self._model.isbuilt():
            self._model.build()
        
        if train_iter_func is None:
            self._train_iter_func = self._default_train_iter_func
        else:
            self._train_iter_func = train_iter_func
        
        if eval_iter_func is None:
            self._eval_iter_func = self._default_eval_iter_func
        else:
            self._eval_iter_func = eval_iter_func
        
        self._trained_it = 0 # iteration number
    
    def _default_train_iter_func(self, model, batch_data):
        """
           Default training loss
        """
        return np.sum(model.train(batch_data)['losses'])
    
    def _default_eval_iter_func(self, model, batch_data):
        """
            Default evaluation outputs
        """
        return np.squeeze(model.serve(batch_data)['outputs'])