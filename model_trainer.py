#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : model_trainer.py
# Author            : Wan Li
# Date              : 23.01.2019
# Last Modified Date: 26.01.2019
# Last Modified By  : Wan Li

from termcolor import colored
from recsys.evaluation_manager import EvaluationManager
import sys
import numpy as np


class ModelTrainer(object):
    """
        Model Trainer
    """
    def __init__(self, model, train_loss_func=None, evaluate_predict_func=None):
        """
           Initializer:
           Params:
               train_loss_func: compute batch loss
               evaluate_predict_func: compute batch output
        """
        self._model = model
        # self._serve_batch_size = serve_batch_size
        if not self._model.isbuilt():
            self._model.build()
        
        if train_loss_func is None:
            self._train_loss_func = self._default_train_loss_func
        else:
            self._train_loss_func = train_loss_func
        
        if evaluate_predict_func is None:
            self._evaluate_predict_func = self._default_evaluate_predict_func
        else:
            self._evaluate_predict_func = evaluate_predict_func
        
        self._trained_it = 0 # iteration number
    
    def _default_train_loss_func(self, model, batch_data):
        """
           Default training loss
        """
        return np.sum(model.train(batch_data)['losses'])
    
    def _default_evaluate_predict_func(self, model, batch_data):
        """
            Default predict outputs
        """
        return np.squeeze(model.serve(batch_data)['outputs'])

    def _evaluate(self, eval_sampler):
        """
            Evaluate during training
            Params:
                eval_sampler: data sampler for evaluation
        """
        metric_results = {} # dict[evaluator_name]list(evaluation result for users)
        for evaluator in self._eval_manager.evaluators:
            metric_results[evaluator.name] = []
        
        completed_user_count = 0
        pos_items, batch_data = eval_sampler.next_batch()
        while batch_data is not None:
            all_scores = []
            all_pos_items = []
            while len(batch_data) > 0: # batch_data = [] indicates data of one user is all sampled
                all_scores.append(self._evaluate_predict_func(self._model, batch_data))
                all_pos_items += pos_items
                pos_items, batch_data = eval_sampler.next_batch()
            # invoke all evaluators
            result = self._eval_manager.full_evaluate(positive_samples=all_pos_items,
                excluded_positive_samples=[],
                predictions=np.concatenate(all_scores, axis=0))
            completed_user_count += 1
            print('...Evaluated %d users' % completed_user_count, end='\r')
            for key in result:
                metric_results[key].append(result[key])
            pos_items, batch_data = eval_sampler.next_batch()
            
        return metric_results

    def train(self, total_iter, eval_iter, save_iter, train_sampler, eval_samplers=[], evaluators=[]):
        """
            Trainer
            Params:
                total_iter: total iterations
                eval_iter: perform evalutation every X iters
                save_iter: perform save ever X iters
        """
        acc_loss = 0
        self._eval_manager = EvalManager(evaluators=evaluators)
        
        train_sampler.reset()
        for sampler in eval_samplers:
            sampler.reset()
        
        print(colored('[Training starts, total_iter: %d, eval_iter: %d, save_iter: %d]' \
                          % (total_iter, eval_iter, save_iter), 'blue'))
        
        for _iter in range(total_iter):
            batch_data = train_sampler.next_batch()
            loss = self._train_iter_func(self._model, batch_data)
            acc_loss += loss
            self._trained_it += 1
            print('..Trained for %d iterations.' % _iter, end='\r')
            if (_iter + 1) % save_iter == 0:
                self._model.save(global_step=self._trained_it)
                print(' '*len('..Trained for %d iterations.' % _iter), end='\r')
                print(colored('[iter %d]' % self._trained_it, 'red'), 'Model saved.')
            if (_iter + 1) % eval_iter == 0:
                print(' '*len('..Trained for %d iterations.' % _iter), end='\r')
                print(colored('[iter %d]' % self._trained_it, 'red'), 'loss: %f' % (acc_loss/eval_iter))
                for sampler in eval_samplers:
                    print(colored('..(dataset: %s) evaluation' % sampler.name, 'green'))
                    sys.stdout.flush()
                    eval_results = self._evaluate(sampler)
                    for key, result in eval_results.items():
                        average_result = np.mean(result, axis=0)
                        if type(average_result) is np.ndarray:
                            print(colored('..(dataset: %s)' % sampler.name, 'green'), \
                                key, ' '.join([str(s) for s in average_result]))
                        else:
                            print(colored('..(dataset: %s)' % sampler.name, 'green'), \
                                key, average_result)
                acc_loss = 0