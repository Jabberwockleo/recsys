# GPU setup
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
import recsys.recommenders.recommender_base as recommender_base
import recsys.modules.extractions.embedding_layer as embedding_layer
import recsys.modules.interactions.mlp_softmax as mlp_softmax
import recsys.modules.fusions.variable_average as variable_average
import recsys.modules.fusions.concatenate as concatenate

import imp
recommender_base = imp.reload(recommender_base)
embedding_layer = imp.reload(embedding_layer)
mlp_softmax = imp.reload(mlp_softmax)
variable_average = imp.reload(variable_average)
concatenate = imp.reload(concatenate)

def VanlillaMlpRec(batch_size, dim_item_embed, max_seq_len, total_items,
        l2_reg_embed=None, l2_reg_mlp=None, dropout=None, init_model_dir=None,
        save_model_dir='VanlillaMlpRec/', train=True, serve=False):
    
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
                                      shape=[total_items,dim_item_embed],
                                      subgraph=subgraph,
                                      scope='item')

    @rec.traingraph.fusiongraph(ins=['seq_vec', 'seq_len'], outs=['fusion_vec'])
    @rec.servegraph.fusiongraph(ins=['seq_vec', 'seq_len'], outs=['fusion_vec'])
    def fusion_graph(subgraph):
        item_repr = variable_average.apply(sequence=subgraph['seq_vec'], seq_len=subgraph['seq_len'])
        fusion_vec = concatenate.apply([item_repr])
        subgraph['fusion_vec'] = fusion_vec

    @rec.traingraph.interactiongraph(ins=['fusion_vec', 'label'])
    def train_interaction_graph(subgraph):
        mlp_softmax.apply(
            in_tensor=subgraph['fusion_vec'],
            dims=[dim_item_embed, total_items],
            l2_reg=l2_reg_mlp,
            labels=subgraph['label'],
            dropout=dropout,
            train=True,
            subgraph=subgraph,
            scope='MLPSoftmax')

    @rec.servegraph.interactiongraph(ins=['fusion_vec'])
    def serve_interaction_graph(subgraph):
        mlp_softmax.apply(
            in_tensor=subgraph['fusion_vec'],
            dims=[dim_item_embed, total_items],
            l2_reg=l2_reg_mlp,
            train=False,
            subgraph=subgraph,
            scope='MLPSoftmax')

    @rec.traingraph.optimizergraph
    def optimizer_graph(subgraph):
        losses = tf.add_n(subgraph.get_global_losses())
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        subgraph.register_global_operation(optimizer.minimize(losses))

    @rec.traingraph.connector
    @rec.servegraph.connector
    def connect(graph):
        graph.itemgraph['seq_item_id'] = graph.inputgraph['seq_item_id']
        graph.fusiongraph['seq_len'] = graph.inputgraph['seq_len']
        graph.fusiongraph['seq_vec'] = graph.itemgraph['seq_vec']
        graph.interactiongraph['fusion_vec'] = graph.fusiongraph['fusion_vec']

    @rec.traingraph.connector.extend
    def train_connect(graph):
        graph.interactiongraph['label'] = graph.inputgraph['label']

    return rec

# ndarray data
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
total_iter = int(1e3)   # iterations for training 
batch_size = 100        # training batch size
eval_iter = 200         # iteration of evaluation
save_iter = eval_iter   # iteration of saving model   

# model
model = VanlillaMlpRec(batch_size=batch_size,
    total_items=train_dataset.total_items(),
    max_seq_len=max_seq_len,
    dim_item_embed=dim_item_embed,
    save_model_dir='VanlillaMlpRec/',
    train=True, 
    serve=True)

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
