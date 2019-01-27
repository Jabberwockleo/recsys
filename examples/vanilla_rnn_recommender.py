#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : vanilla_rnn_recommender.py
# Author            : Wan Li
# Date              : 27.01.2019
# Last Modified Date: 27.01.2019
# Last Modified By  : Wan Li

# GPU setup
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# download datasets
#import urllib.request
#dataset_prefix = 'http://s3.amazonaws.com/cornell-tech-sdl-openrec'
#urllib.request.urlretrieve('%s/lastfm/lastfm_test.npy' % dataset_prefix, 
#                   'lastfm_test.npy')
#urllib.request.urlretrieve('%s/lastfm/lastfm_train.npy' % dataset_prefix, 
#                   'lastfm_train.npy')

import numpy as np
import tensorflow as tf
import recsys.recommenders.recommender_base as recommender_base
import recsys.modules.extractions.embedding_layer as embedding_layer
import recsys.modules.interactions.rnn_softmax as rnn_softmax

import imp
recommender_base = imp.reload(recommender_base)
embedding_layer = imp.reload(embedding_layer)
rnn_softmax = imp.reload(rnn_softmax)

def VanillaRnnRec(batch_size, dim_item_embed, max_seq_len, total_items, num_units,
        l2_reg_embed=None, init_model_dir=None,
        save_model_dir='VanillaRnnRec', train=True, serve=False):
    
    rec = recommender_base.Recommender(init_model_dir=init_model_dir,
                      save_model_dir=save_model_dir, train=train, serve=serve)
    
    @rec.traingraph.inputgraph(outs=['seq_item_id', 'seq_len', 'label'])
    def train_input_graph(subgraph):
        subgraph['seq_item_id'] = tf.placeholder(tf.int32, 
                                      shape=[batch_size, max_seq_len],
                                      name='seq_item_id')
        subgraph['seq_len'] = tf.placeholder(tf.int32, 
                                      shape=[batch_size], 
                                      name='seq_len')
        subgraph['label'] = tf.placeholder(tf.int32, 
                                      shape=[batch_size], 
                                      name='label')
        subgraph.register_global_input_mapping({'seq_item_id': subgraph['seq_item_id'],
                                                'seq_len': subgraph['seq_len'],
                                                'label': subgraph['label']})
        
    @rec.servegraph.inputgraph(outs=['seq_item_id', 'seq_len'])
    def serve_input_graph(subgraph):
        subgraph['seq_item_id'] = tf.placeholder(tf.int32, 
                                      shape=[None, max_seq_len],
                                      name='seq_item_id')
        subgraph['seq_len'] = tf.placeholder(tf.int32, 
                                      shape=[None],
                                      name='seq_len')
        subgraph.register_global_input_mapping({'seq_item_id': subgraph['seq_item_id'],
                                                'seq_len': subgraph['seq_len']})
    
    @rec.traingraph.itemgraph(ins=['seq_item_id'], outs=['seq_vec'])
    @rec.servegraph.itemgraph(ins=['seq_item_id'], outs=['seq_vec'])
    def item_graph(subgraph):
        _, subgraph['seq_vec']= embedding_layer.apply(l2_reg=l2_reg_embed,
                                      init='normal',
                                      id_=subgraph['seq_item_id'],
                                      shape=[total_items, dim_item_embed],
                                      subgraph=subgraph,
                                      scope='item')

    @rec.traingraph.interactiongraph(ins=['seq_vec', 'seq_len', 'label'])
    def train_interaction_graph(subgraph):
        rnn_softmax.apply(
            sequence=subgraph['seq_vec'], 
            seq_len=subgraph['seq_len'], 
            num_units=num_units, 
            cell_type='gru',
            total_items=total_items, 
            label=subgraph['label'], 
            train=True, 
            subgraph=subgraph, 
            scope='RNNSoftmax')

    @rec.servegraph.interactiongraph(ins=['seq_vec', 'seq_len'])
    def serve_interaction_graph(subgraph):
        rnn_softmax.apply(
            sequence=subgraph['seq_vec'], 
            seq_len=subgraph['seq_len'],
            num_units=num_units,
            cell_type='gru',
            total_items=total_items, 
            train=False, 
            subgraph=subgraph, 
            scope='RNNSoftmax')

    @rec.traingraph.optimizergraph
    def optimizer_graph(subgraph):
        losses = tf.add_n(subgraph.get_global_losses())
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        subgraph.register_global_operation(optimizer.minimize(losses))

    @rec.traingraph.connector
    @rec.servegraph.connector
    def connect(graph):
        graph.itemgraph['seq_item_id'] = graph.inputgraph['seq_item_id']
        graph.interactiongraph['seq_len'] = graph.inputgraph['seq_len']
        graph.interactiongraph['seq_vec'] = graph.itemgraph['seq_vec']

    @rec.traingraph.connector.extend
    def train_connect(graph):
        graph.interactiongraph['label'] = graph.inputgraph['label']

    return rec

# npy data
import recsys.dataset as dataset
train_data = np.load('lastfm_train.npy')
test_data = np.load('lastfm_test.npy')
total_users = max(set(list(train_data['user_id']) + list(test_data['user_id']))) + 1
total_items = max(set(list(train_data['item_id']) + list(test_data['item_id']))) + 1
print(total_users, total_items)
train_data[:2], test_data[:2]

# datasets
import recsys.dataset as dataset
dataset = imp.reload(dataset)
train_dataset = dataset.Dataset(train_data, total_users, total_items, 
                        sortby='ts', name='Train')
test_dataset = dataset.Dataset(test_data, total_users, total_items, 
                       sortby='ts', name='Test')

# hyperparamerters
dim_item_embed = 50     # dimension of item embedding
max_seq_len = 100       # the maxium length of user's listen history
num_units = 32          # Number of units in the RNN model
total_iter = int(1e3)   # iterations for training 
batch_size = 100        # training batch size
eval_iter = 200         # iteration of evaluation
save_iter = eval_iter   # iteration of saving model   

# model
model = VanillaRnnRec(batch_size=batch_size, 
    dim_item_embed=dim_item_embed, 
    max_seq_len=max_seq_len, 
    total_items=train_dataset.total_items(), 
    num_units=num_units, 
    save_model_dir='VanillaRnnRec', 
    train=True, serve=True)

# evaluators
import recsys.metrics.auc as auc
import recsys.metrics.ndcg as ndcg
import recsys.metrics.recall as recall
import recsys.metrics.precision as precision

auc_evaluator = auc.AUC()
ndcg_evaluator = ndcg.NDCG(ndcg_at=[100])
recall_evaluator = recall.Recall(recall_at=[100, 200, 300, 400, 500])
precision_evaluator = precision.Precision(precision_at=[100])

# sampler
import recsys.samplers.temporal_sampler as temporal_sampler
train_sampler = temporal_sampler.create_training_sampler(batch_size=batch_size, max_seq_len=max_seq_len, 
                                dataset=train_dataset, num_process=1)
test_sampler = temporal_sampler.create_evaluation_sampler(dataset=test_dataset, 
                                         max_seq_len=max_seq_len)

# trainer
import recsys.model_trainer as model_trainer
model_trainer = imp.reload(model_trainer)
trainer = model_trainer.ModelTrainer(model=model)

# train/test
trainer.train(total_iter=total_iter, 
    eval_iter=eval_iter,
    save_iter=save_iter,
    train_sampler=train_sampler,
    eval_samplers=[test_sampler], 
    evaluators=[auc_evaluator, ndcg_evaluator, recall_evaluator, precision_evaluator])

# serve
serve_sampler = temporal_sampler.create_evaluation_sampler(dataset=test_dataset, max_seq_len=max_seq_len)
lbl, input_data = serve_sampler.next_batch()
print(lbl, input_data)
output_dict = model.serve(batch_data=input_data)
print("outputs:", output_dict)
predict_proba = output_dict['outputs'][0].ravel()
ind_largest = np.argsort(predict_proba)[-20:]
print("indices:", ind_largest)
print("probs:", predict_proba[ind_largest])

# export
model.export()

# predict using pb model
output_dict = model.predict_pb(feed_name_dict={
    'seq_item_id': input_data['seq_item_id'],
    'seq_len': input_data['seq_len']
})
print(output_dict)
