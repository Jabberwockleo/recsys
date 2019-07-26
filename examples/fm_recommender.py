#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : fm_recommender.py
# Author            : Wan Li
# Date              : 26.07.2019
# Last Modified Date: 26.07.2019
# Last Modified By  : Wan Li

import importlib as imp
import sys
import numpy as np
sys.path = list(set(sys.path + ['../../']))


import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"
import tensorflow as tf


from sklearn import datasets
dataset_iris = datasets.load_iris(return_X_y=True)


def generate_sample_dataset_and_feature_dict(dataset_iris):
    """
        Make dataset from iris, assuming a user likes class 0 but dislikes class 1 and class 2
    """
    npd = np.zeros(len(dataset_iris[0]), dtype=[
        ('user_id', np.int32), ('item_id', np.int32), ('label', np.int32)])
    fea_dict = dict()
    for idx in range(len(dataset_iris[0])):
        if dataset_iris[1][idx] == 0:
            npd[idx] = (0, idx, 1)
        else:
            npd[idx] = (0, idx, -1)
        fea_dict[idx] = dataset_iris[0][idx]
    return npd, fea_dict

data_all, fea_dict = generate_sample_dataset_and_feature_dict(dataset_iris)


from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(data_all)
print(len(data_train), len(data_test))


# create dataset instance from structured ndarray data
import recsys.dataset as dataset
dataset = imp.reload(dataset)
train_dataset = dataset.Dataset(data_train, total_users=1, total_items=len(fea_dict), implicit_negative=False, name='Train')
test_dataset = dataset.Dataset(data_test, total_users=1, total_items=len(fea_dict), implicit_negative=False, name='Test')


# create featurizer for mapping: (user, item) -> vector
import recsys.featurizer as featurizer
featurizer = imp.reload(featurizer)
class Featurizer(featurizer.FeaturizerBase):
    """
        Custom featurizer
    """
    def __init__(self, fea_dict):
        self.fea_dict = fea_dict

    def feature_dim(self):
        """
            Dim of X
        """
        return 4
    
    def featurize(self, user_id, item_id):
        if user_id != 0:
            raise "error"
        return self.fea_dict[item_id]

featurizer_iris = Featurizer(fea_dict)


# create data sampler for training and testing
import recsys.samplers.ranknet_sampler as ranknet_sampler
ranknet_sampler = imp.reload(ranknet_sampler)
train_sampler = ranknet_sampler.create_training_sampler(
    dataset=train_dataset, featurizer=featurizer_iris, max_pos_neg_per_user=5, batch_size=10, num_process=1, seed=100)
test_sampler = ranknet_sampler.create_evaluation_sampler(
    dataset=test_dataset, featurizer=featurizer_iris, max_pos_neg_per_user=5, seed=10)


# create evaluation metric
import recsys.metrics.auc as auc
auc = imp.reload(auc)
auc_evaluator = auc.AUC()


# create model
import recsys.recommenders.factorization_machine_recommender as factorization_machine_recommender
factorization_machine_recommender = imp.reload(factorization_machine_recommender)
model = factorization_machine_recommender.FactorizationMachineRecommender(feature_dim=featurizer_iris.feature_dim(),
    factor_dim=5, init_model_dir=None, save_model_dir='FMRec', l2_reg=None, train=True, serve=True)


# create model trainer with model
import recsys.model_trainer as model_trainer
model_trainer = imp.reload(model_trainer)
trainer = model_trainer.ModelTrainer(model=model)


# train
import datetime
print(datetime.datetime.now())

trainer.train(total_iter=100, 
                    eval_iter=1,
                    save_iter=5,
                    train_sampler=train_sampler,
                    eval_samplers=[test_sampler], 
                    evaluators=[auc_evaluator])
print(datetime.datetime.now())


# Export Protobuf model for online serving
model.export(export_model_dir="pbModel", as_text=False)


# inspect internal op tensor value of training
b = train_sampler.next_batch()
print("data:", b)
print("dy, dy_tilde:", model.train_inspect_ports(batch_data=b, ports=model.traingraph.get_outputs()))
print("loss:", model.train_inspect_ports(batch_data=b, ports=model.traingraph.get_losses()))


# inspect internal op tensor value of testing
b = test_sampler.next_batch()
print("data:", b)
print("score:", model.serve_inspect_ports(batch_data=b[1], ports=model.servegraph.get_outputs()))


# inspect internal graph weights
def inspect_weights(checkpoint_model_dir):
    reader = tf.train.NewCheckpointReader(checkpoint_model_dir)
    variables = reader.get_variable_to_shape_map()
    for var, shape in variables.items():
        trained_weight = reader.get_tensor(var)
        print("VarName: {}".format(var))
        print("VarShape: {}".format(shape))
        print("VarValue: {}".format(trained_weight))
inspect_weights("./FMRec/model.ckpt")