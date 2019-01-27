#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : vanilla_mlp_recommender.py
# Author            : Wan Li
# Date              : 27.01.2019
# Last Modified Date: 27.01.2019
# Last Modified By  : Wan Li

def VanillaMlpRec(batch_size, dim_item_embed, max_seq_len, total_items,
        l2_reg_embed=None, l2_reg_mlp=None, dropout=None, init_model_dir=None,
        save_model_dir='VanillaMlpRec', train=True, serve=False):
    
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
