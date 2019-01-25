#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : metric_base.py
# Author            : Wan Li
# Date              : 25.01.2019
# Last Modified Date: 25.01.2019
# Last Modified By  : Wan Li

class Metric(object):
    """
        Abstract class
    """
    def __init__(self, etype, name):
        """
            Initializer
        """
        self.etype = etype # evaluation type
        self.name = name

    def compute(self):
        """
        """
        assert False, "calling an abstract method"
        return None
