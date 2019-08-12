#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : model_trainer.py
# Author            : Wan Li
# Date              : 23.01.2019
# Last Modified Date: 26.01.2019
# Last Modified By  : Wan Li

import sys
import numpy as np
from termcolor import colored
import tensorflow as tf
from recsys.evaluation_manager import EvaluationManager

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
        result_dict = model.train(batch_data)
        return np.sum(result_dict['losses']), result_dict['summarys']

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
        if eval_sampler.evaluation_type() == "FULL":
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
                result = self._eval_manager.full_evaluate(pos_samples=all_pos_items,
                    excluded_positive_samples=[],
                    predictions=np.concatenate(all_scores, axis=0))
                completed_user_count += 1
                print('...Evaluated %d users' % completed_user_count, end='\r')
                for key in result:
                    metric_results[key].append(result[key])
                pos_items, batch_data = eval_sampler.next_batch()
            return metric_results
        elif eval_sampler.evaluation_type() == "SAMPLED":
            metric_results = {} # dict[evaluator_name]list(evaluation result for users)
            for evaluator in self._eval_manager.evaluators:
                metric_results[evaluator.name] = []

            completed_user_count = 0
            # data_labels [1, 1, ... -1, -1] 1 for pos_item, -1 for neg_item r.s.t a user
            data_labels, batch_data = eval_sampler.next_batch()
            while batch_data is not None:
                pos_scores = []
                neg_scores = []
                all_scores = []
                all_labels = []
                while len(batch_data) > 0: # batch_data = [] indicates data of one user is all sampled
                    all_labels.extend(data_labels)
                    outputs = self._evaluate_predict_func(self._model, batch_data)
                    all_scores.extend(np.atleast_1d(outputs))
                    data_labels, batch_data = eval_sampler.next_batch()
                for idx in range(len(all_labels)):
                    if all_labels[idx] == 1:
                        pos_scores.append(all_scores[idx])
                    elif all_labels[idx] == -1:
                        neg_scores.append(all_scores[idx])
                # invoke all evaluators
                result = self._eval_manager.partial_evaluate(pos_scores, neg_scores)
                completed_user_count += 1
                print('...Evaluated %d users' % completed_user_count, end='\r')
                for key in result:
                    metric_results[key].append(result[key])
                data_labels, batch_data = eval_sampler.next_batch()
            return metric_results
        else:
            raise "Sampler's evaluation_type() is unrecognized."

    def train(self, total_iter, eval_iter, save_iter, train_sampler, eval_samplers=[], evaluators=[]):
        """
            Trainer
            Params:
                total_iter: total iterations
                eval_iter: perform evalutation every X iters
                save_iter: perform save ever X iters
        """
        accumulated_loss = 0
        self._eval_manager = EvaluationManager(evaluators=evaluators)

        train_sampler.reset()
        for sampler in eval_samplers:
            sampler.reset()

        print(colored('[Training starts, total_iter: %d, eval_iter: %d, save_iter: %d]' \
                          % (total_iter, eval_iter, save_iter), 'blue'))

        for _iter in range(total_iter):
            batch_data = train_sampler.next_batch()
            loss, train_summary = self._train_loss_func(self._model, batch_data)
            accumulated_loss += loss
            self._trained_it += 1
            print('..Trained for %d iterations.' % _iter, end='\r')
            if (_iter + 1) % save_iter == 0:
                self._model.train_writer().add_summary(train_summary[0], _iter)
                self._model.save(global_step=self._trained_it)
                print(' '*len('..Trained for %d iterations.' % _iter), end='\r')
                print(colored('[iter %d]' % self._trained_it, 'red'), 'Model saved.')
            if (_iter + 1) % eval_iter == 0:
                print(' '*len('..Trained for %d iterations.' % _iter), end='\r')
                print(colored('[iter %d]' % self._trained_it, 'red'), 'loss: %f' % (accumulated_loss/eval_iter))
                summary_loss = tf.Summary()
                summary_loss.value.add(tag="eva_loss_tag", simple_value=(accumulated_loss/eval_iter))
                self._model.train_writer().add_summary(summary_loss, _iter)
                for sampler in eval_samplers:
                    print(colored('..(dataset: %s) evaluation' % sampler.name, 'green'))
                    sys.stdout.flush()
                    eval_results = self._evaluate(sampler)
                    for key, result in eval_results.items():
                        average_result = np.mean(result, axis=0)
                        summary_eva = tf.Summary()
                        summary_eva.value.add(tag=key, simple_value=average_result)
                        self._model.train_writer().add_summary(summary_eva, _iter)
                        if type(average_result) is np.ndarray:
                            print(colored('..(dataset: %s)' % sampler.name, 'green'), \
                                key, ' '.join([str(s) for s in average_result]))
                        else:
                            print(colored('..(dataset: %s)' % sampler.name, 'green'), \
                                key, average_result)
                accumulated_loss = 0
