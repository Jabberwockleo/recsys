#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : sampler_base.py
# Author            : Wan Li
# Date              : 27.01.2019
# Last Modified Date: 27.01.2019
# Last Modified By  : Wan Li

from multiprocessing import Process, Queue

class _Sampler(Process):
    """
        Child process
    """
    def __init__(self, dataset, q, generate_batch):
        """
            Initializer
        """
        self._q = q
        self._generate_batch = generate_batch
        self._dataset = dataset
        super(_Sampler, self).__init__()

    def run(self):
        """
            Enqueue sampled data
        """
        for input_npy in self._generate_batch(self._dataset):
            self._q.put(input_npy, block=True)


class Sampler(object):
    """
        Multi-process sampler
    """
    def __init__(self, dataset=None, evaluation_type=None, generate_batch=None, num_process=5):
        """
            Initializer
        """
        assert generate_batch is not None, "Batch generation function is not specified"
        assert dataset is not None, "Dataset is not specified"
        self._q = None
        self._dataset = dataset
        self._runner_list = []
        self._start = False
        self._num_process = num_process
        self._generate_batch = generate_batch
        self.name = self._dataset.name
        self.evaluation_type = evaluation_type

    def evaluation_type(self):
        """
            Return evaluation type, FULL/SAMPLED
        """
        return self.evaluation_type
        
    def next_batch(self):
        """
            Get next batch which is previously enqueued by daemon threads
        """
        if not self._start:
            self.reset()
        
        return self._q.get(block=True)
        
    def reset(self):
        """
            Reset threads and restart
        """
        while len(self._runner_list) > 0:
            runner = self._runner_list.pop()
            runner.terminate()
            del runner
        
        if self._q is not None:
            del self._q
        self._q = Queue(maxsize=self._num_process)
            
        for i in range(self._num_process):
            runner = _Sampler(self._dataset, self._q, self._generate_batch)
            runner.daemon = True
            runner.start()
            self._runner_list.append(runner)
        self._start = True