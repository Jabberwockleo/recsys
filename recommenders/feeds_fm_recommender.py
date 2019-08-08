#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : feeds_fm_recommender.py
# Author            : Wan Li
# Date              : 01.08.2019
# Last Modified Date: 01.08.2019
# Last Modified By  : Wan Li

import tensorflow as tf
import recsys.recommenders.recommender_base as recommender_base
import recsys.modules.extractions.fully_connected_layer as fully_connected_layer
import recsys.modules.extractions.embedding_layer as embedding_layer
import recsys.modules.fusions.concatenate as concatenate
import recsys.modules.fusions.variable_average as variable_average
import recsys.modules.interactions.fm_layer as fm_layer

def FeedsFMRecommender(fea_user_demography_dim, fea_user_stat_dim, fea_user_history_dim,
    fea_item_meta_dim, fea_item_stat_dim, fea_context_hour_dim,
    total_item_num, embedding_dim=128,
    factor_dim=7, init_model_dir=None, save_model_dir="FeedsFMRec", l2_reg=None, train=True, serve=True):
    """
        Feeds recommender for industrial purposes
        Model: F(X) -> score
    """
    rec = recommender_base.Recommender(init_model_dir=init_model_dir,
        save_model_dir=save_model_dir, train=train, serve=serve)
    
    @rec.traingraph.inputgraph(outs=["user_demography_vec", "user_stat_vec",
        "user_history_vec", "user_history_len", "context_hour"])
    @rec.servegraph.inputgraph(outs=["user_demography_vec", "user_stat_vec",
        "user_history_vec", "user_history_len", "context_hour"])
    def inputgraph(subgraph):
        subgraph["user_demography_vec"] = tf.placeholder(tf.float32,
            shape=[None, fea_user_demography_dim], name="user_demography_vec")
        subgraph["user_stat_vec"] = tf.placeholder(tf.float32,
            shape=[None, fea_user_stat_dim], name="user_stat_vec")
        subgraph["user_history_vec"] = tf.placeholder(tf.int32,
            shape=[None, fea_user_history_dim], name="user_history_vec")
        subgraph["user_history_len"] = tf.placeholder(tf.int32,
            shape=[None], name="user_history_len")
        subgraph["context_hour"] = tf.placeholder(tf.float32,
            shape=[None, fea_context_hour_dim], name="context_hour")
        subgraph.register_global_input_mapping({'user_demography_vec': subgraph['user_demography_vec'],
                                                'user_stat_vec': subgraph['user_stat_vec'],
                                                'user_history_vec': subgraph['user_history_vec'],
                                                'user_history_len': subgraph['user_history_len'],
                                                'context_hour': subgraph['context_hour']})
        pass
    
    @rec.traingraph.inputgraph.extend(outs=["item_meta_vec_1", "item_stat_vec_1", "item_id_1",
        "item_meta_vec_2", "item_stat_vec_2", "item_id_2", "dy"])
    def train_inputgraph(subgraph):
        subgraph["item_meta_vec_1"] = tf.placeholder(tf.float32,
            shape=[None, fea_item_meta_dim], name="item_meta_vec_1")
        subgraph["item_stat_vec_1"] = tf.placeholder(tf.float32,
            shape=[None, fea_item_stat_dim], name="item_stat_vec_1")
        subgraph["item_id_1"] = tf.placeholder(tf.int32,
            shape=[None], name="item_id_1")
        subgraph["item_meta_vec_2"] = tf.placeholder(tf.float32,
            shape=[None, fea_item_meta_dim], name="item_meta_vec_2")
        subgraph["item_stat_vec_2"] = tf.placeholder(tf.float32,
            shape=[None, fea_item_stat_dim], name="item_stat_vec_2")
        subgraph["item_id_2"] = tf.placeholder(tf.int32,
            shape=[None], name="item_id_2")
        subgraph["dy"] = tf.placeholder(tf.float32,
            shape=[None], name="label")
        subgraph.update_global_input_mapping({'item_meta_vec_1': subgraph['item_meta_vec_1'],
                                                'item_stat_vec_1': subgraph['item_stat_vec_1'],
                                                'item_id_1': subgraph['item_id_1'],
                                                'item_meta_vec_2': subgraph['item_meta_vec_2'],
                                                'item_stat_vec_2': subgraph['item_stat_vec_2'],
                                                'item_id_2': subgraph['item_id_2'],
                                                'label': subgraph['dy']})
        pass

    @rec.servegraph.inputgraph.extend(outs=["item_meta_vec", "item_stat_vec", "item_id"])
    def serve_inputgraph(subgraph):
        subgraph["item_meta_vec"] = tf.placeholder(tf.float32,
            shape=[None, fea_item_meta_dim], name="item_meta_vec")
        subgraph["item_stat_vec"] = tf.placeholder(tf.float32,
            shape=[None, fea_item_stat_dim], name="item_stat_vec")
        subgraph["item_id"] = tf.placeholder(tf.int32,
            shape=[None], name="item_id")
        subgraph.update_global_input_mapping({'item_meta_vec': subgraph['item_meta_vec'],
                                                'item_stat_vec': subgraph['item_stat_vec'],
                                                'item_id': subgraph['item_id']})
        pass

    @rec.traingraph.usergraph(ins=["user_demography_vec", "user_stat_vec", "user_history_vec", "user_history_len"],
        outs=["user_vec"])
    @rec.servegraph.usergraph(ins=["user_demography_vec", "user_stat_vec", "user_history_vec", "user_history_len"],
        outs=["user_vec"])
    def usergraph(subgraph):
        _, item_embedded_tensor = embedding_layer.apply(l2_reg=l2_reg,
            init="normal",
            id_=subgraph["user_history_vec"],
            shape=[total_item_num, embedding_dim],
            subgraph=subgraph,
            scope="ItemEmbedding") # shaped [-1, fea_user_history_dim, embedding_dim]
        user_history_repr = variable_average.apply(
            sequence=item_embedded_tensor, seq_len=subgraph["user_history_len"]) # shaped [-1, 1, embedding_dim]
        subgraph["user_vec"] = concatenate.apply([
            subgraph["user_demography_vec"], subgraph["user_stat_vec"], user_history_repr])
        pass
    
    @rec.traingraph.contextgraph(ins=["context_hour"], outs=["context_vec"])
    @rec.servegraph.contextgraph(ins=["context_hour"], outs=["context_vec"])
    def contextgraph(subgraph):
        subgraph["context_vec"] = subgraph["context_hour"]
        pass

    @rec.traingraph.itemgraph(ins=["item_meta_vec_1", "item_stat_vec_1", "item_id_1",
        "item_meta_vec_2", "item_stat_vec_2", "item_id_2"], outs=["item_vec_1", "item_vec_2"])
    def train_itemgraph(subgraph):
        _, item_embedded_tensor_1 = embedding_layer.apply(l2_reg=l2_reg,
            init="normal",
            id_=subgraph["item_id_1"],
            shape=[total_item_num, embedding_dim],
            subgraph=subgraph,
            scope="ItemEmbedding")
        _, item_embedded_tensor_2 = embedding_layer.apply(l2_reg=l2_reg,
            init="normal",
            id_=subgraph["item_id_2"],
            shape=[total_item_num, embedding_dim],
            subgraph=subgraph,
            scope="ItemEmbedding")
        subgraph["item_vec_1"] = concatenate.apply([
            subgraph["item_meta_vec_1"], subgraph["item_stat_vec_1"], item_embedded_tensor_1])
        subgraph["item_vec_2"] = concatenate.apply([
            subgraph["item_meta_vec_2"], subgraph["item_stat_vec_2"], item_embedded_tensor_2])
        pass

    @rec.servegraph.itemgraph(ins=["item_meta_vec", "item_stat_vec", "item_id"], outs=["item_vec"])
    def serve_itemgraph(subgraph):
        _, item_embedded_tensor = embedding_layer.apply(l2_reg=l2_reg,
            init="normal",
            id_=subgraph["item_id"],
            shape=[total_item_num, embedding_dim],
            subgraph=subgraph,
            scope="ItemEmbedding")
        subgraph["item_vec"] = concatenate.apply([
            subgraph["item_meta_vec"], subgraph["item_stat_vec"], item_embedded_tensor])
        pass

    @rec.traingraph.fusiongraph(ins=["user_vec", "item_vec_1", "item_vec_2", "context_vec"], outs=["X_1", "X_2"])
    def train_fusiongraph(subgraph):
        subgraph["X_1"] = concatenate.apply([subgraph["user_vec"], subgraph["item_vec_1"], subgraph["context_vec"]])
        subgraph["X_2"] = concatenate.apply([subgraph["user_vec"], subgraph["item_vec_2"], subgraph["context_vec"]])
        pass

    @rec.servegraph.fusiongraph(ins=["user_vec", "item_vec", "context_vec"], outs=["X"])
    def serve_fusiongraph(subgraph):
        subgraph["X"] = concatenate.apply([subgraph["user_vec"], subgraph["item_vec"], subgraph["context_vec"]])
        pass

    @rec.traingraph.interactiongraph(ins=["X_1", "X_2", "dy"])
    def train_interactiongraph(subgraph):
        linear1 = fully_connected_layer.apply(subgraph["X_1"], [1], subgraph,
            relu_in=False, relu_mid=False, relu_out=False,
            dropout_in=None, dropout_mid=None, dropout_out=None,
            bias_in=True, bias_mid=True, bias_out=True, batch_norm=False,
            train=False, l2_reg=l2_reg, scope="LinearComponent")
        linear1 = tf.squeeze(linear1) # shaped [None, ]
        interactive1 = fm_layer.apply(subgraph["X_1"], factor_dim, l2_weight=0.01, scope="InteractiveComponent")
        interactive1 = tf.squeeze(tf.math.reduce_sum(interactive1, axis=1))
        
        linear2 = fully_connected_layer.apply(subgraph["X_2"], [1], subgraph,
            relu_in=False, relu_mid=False, relu_out=False,
            dropout_in=None, dropout_mid=None, dropout_out=None,
            bias_in=True, bias_mid=True, bias_out=True, batch_norm=False,
            train=False, l2_reg=l2_reg, scope="LinearComponent")
        linear2 = tf.squeeze(linear2)
        interactive2 = fm_layer.apply(subgraph["X_2"], factor_dim, l2_weight=0.01, scope="InteractiveComponent")
        interactive2 = tf.squeeze(tf.math.reduce_sum(interactive2, axis=1))
        dy_tilde = (linear1 + interactive1) - (linear2 + interactive2)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=subgraph["dy"], logits=dy_tilde, name="loss")
        subgraph.register_global_loss(tf.reduce_mean(loss))
        tf.summary.scalar('loss', tf.reduce_mean(loss))
        summary = tf.summary.merge_all()
        subgraph.register_global_summary(summary)
        subgraph.register_global_output(subgraph["dy"])
        subgraph.register_global_output(dy_tilde)
        pass

    @rec.servegraph.interactiongraph(ins=["X"])
    def serve_interactiongraph(subgraph):
        linear = fully_connected_layer.apply(subgraph["X"], [1], subgraph,
            relu_in=False, relu_mid=False, relu_out=False,
            dropout_in=None, dropout_mid=None, dropout_out=None,
            bias_in=True, bias_mid=True, bias_out=True, batch_norm=False,
            train=False, l2_reg=l2_reg, scope="LinearComponent")
        linear = tf.squeeze(linear) # shaped [None, ]
        interactive = fm_layer.apply(subgraph["X"], factor_dim, l2_weight=0.01, scope="InteractiveComponent") # shaped [None, factor_dim]
        interactive = tf.squeeze(tf.math.reduce_sum(interactive, axis=1))
        score = linear + interactive
        subgraph.register_global_output(score)
        pass

    @rec.traingraph.optimizergraph
    def train_optimizergraph(subgraph):
        losses = tf.math.add_n(subgraph.get_global_losses())
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
        subgraph.register_global_operation(optimizer.minimize(losses))
        pass

    @rec.traingraph.connector
    @rec.servegraph.connector
    def connector(graph):
        graph.usergraph["user_demography_vec"] = graph.inputgraph["user_demography_vec"]
        graph.usergraph["user_stat_vec"] = graph.inputgraph["user_stat_vec"]
        graph.usergraph["user_history_vec"] = graph.inputgraph["user_history_vec"]
        graph.usergraph["user_history_len"] = graph.inputgraph["user_history_len"]
        graph.contextgraph["context_hour"] = graph.inputgraph["context_hour"]
        graph.fusiongraph["user_vec"] = graph.usergraph["user_vec"]
        graph.fusiongraph["context_vec"] = graph.contextgraph["context_vec"]
        pass

    @rec.traingraph.connector.extend
    def train_connector(graph):
        graph.itemgraph["item_meta_vec_1"] = graph.inputgraph["item_meta_vec_1"]
        graph.itemgraph["item_stat_vec_1"] = graph.inputgraph["item_stat_vec_1"]
        graph.itemgraph["item_id_1"] = graph.inputgraph["item_id_1"]
        graph.itemgraph["item_meta_vec_2"] = graph.inputgraph["item_meta_vec_2"]
        graph.itemgraph["item_stat_vec_2"] = graph.inputgraph["item_stat_vec_2"]
        graph.itemgraph["item_id_2"] = graph.inputgraph["item_id_2"]
        graph.fusiongraph["item_vec_1"] = graph.itemgraph["item_vec_1"]
        graph.fusiongraph["item_vec_2"] = graph.itemgraph["item_vec_2"]
        graph.interactiongraph["X_1"] = graph.fusiongraph["X_1"]
        graph.interactiongraph["X_2"] = graph.fusiongraph["X_2"]
        graph.interactiongraph["dy"] = graph.inputgraph["dy"]
        pass

    @rec.servegraph.connector.extend
    def serve_connector(graph):
        graph.itemgraph["item_meta_vec"] = graph.inputgraph["item_meta_vec"]
        graph.itemgraph["item_stat_vec"] = graph.inputgraph["item_stat_vec"]
        graph.itemgraph["item_id"] = graph.inputgraph["item_id"]
        graph.fusiongraph["item_vec"] = graph.itemgraph["item_vec"]
        graph.interactiongraph["X"] = graph.fusiongraph["X"]
        pass

    return rec

if __name__ == "__main__":
    pass
