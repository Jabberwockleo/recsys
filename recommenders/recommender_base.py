#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : recommender_base.py
# Author            : Wan Li
# Date              : 24.01.2019
# Last Modified Date: 25.01.2019
# Last Modified By  : Wan Li

import tensorflow as tf
import numpy as np
import os  

class _RecommenderGraph(object):
    """
        Main graph, i.e. training/serving
    """
    class _SubGraph(object):
        """
            Component graph, i.e. itemgraph/usergraph/contextgraph etc.
        """
        class _Port(object):
            """
                Graph I/O abstract class
            """
            def __init__(self):
                """
                """
                self.s = None
        
        class _InPort(_Port):
            """
                Inputs of component graph
            """
            def assign(self, subgraph, key):
                """
                    Assign input
                    Params:
                        subgraph: upstream graph
                        key: key of the outputs of upstream graph
                """
                self.s = {'subgraph':subgraph, 'key':key}
            
            def retrieve(self):
                """
                    Retrieve input from upstream output
                """
                if self.s is None:
                    return None
                else:
                    return self.s['subgraph'][self.s['key']]

        class _OutPort(_Port):
            """
                Outputs of component graph
            """
            def assign(self, tensor):
                """
                    Assign output tensor to port
                """
                self.s = tensor

            def retrieve(self):
                """
                    Retrieve output tensor of this graph
                """
                return self.s

        def __init__(self, rec_graph):
            """
                Initializer of component graph
                Params:
                    rec_graph: main graph
            """
            self._super = rec_graph
            self._port_store = dict() # inports/outports
            self._build_funcs = []
            self._is_building_mode = False # False for connector connecting ports, True for building computation graph
            self._is_built = False
            
        def __getitem__(self, key):
            """
                Getter [], get InPorts or OutPorts, behaviour is different in modes
                e.g. graph_A["port_in"] = graph_B["port_out"] when build_mode is False
            """
            assert key in self._port_store, "%s port is not found." % key
            if self._is_building_mode:
                assert self._is_built, "[Build Error] Getting a value from an unconstructed graph."
                return self._port_store[key].retrieve()
            else:
                assert isinstance(self._port_store[key], self._OutPort), "[Connect Error] Getting a value from the %s in-port" % key
                return self, key # corresponds to InPort::assign(subgraph, key)
        
        def __setitem__(self, key, value):
            """
                Setter [], set InPorts or OutPorts, behaviour is different in modes
                e.g. graph_A["port_in"] = graph_B["port_out"]
            """
            assert key in self._port_store, "%s port is not found." % key
            if self._is_building_mode:
                assert isinstance(self._port_store[key], self._OutPort), "[Build Error] Assigning a value to the %s in-port" % key
                self._port_store[key].assign(value)
            else:
                assert isinstance(self._port_store[key], self._InPort), "[Connect Error] Assigning a value to the %s out-port" % key
                self._port_store[key].assign(value[0], value[1])

        def __call__(self, build_func=None, ins=[], outs=[]):
            """
                Caller (), decorator function, to use in a different way
            """
            assert isinstance(ins, list), "ins should be a list of strings."
            assert isinstance(outs, list), "outs should be a list of strings"
            
            self._port_store = {}
            self._build_funcs = []
            
            for in_ in ins:
                self._port_store[in_] = self._InPort()
            for out_ in outs:
                self._port_store[out_] = self._OutPort()
            
            if build_func is None:
                def add_build_func(build_func):
                    self._build_funcs.append(build_func)
                    return build_func
                return add_build_func
            else:
                self._build_funcs.append(build_func)
                return build_func

        def extend(self, build_func=None, ins=[], outs=[]):
            """
                Decorator function, to append instead
            """
            assert isinstance(ins, list), "ins should be a list of strings."
            assert isinstance(outs, list), "outs should be a list of strings"
            
            for in_ in ins:
                self._port_store[in_] = self._InPort()
            for out_ in outs:
                self._port_store[out_] = self._OutPort()
            
            if build_func is None:
                def add_build_func(build_func):
                    self._build_funcs.append(build_func)
                    return build_func
                return add_build_func
            else:
                self._build_funcs.append(build_func)
                return build_func

        def get_intrinsics(self):
            """
                Utilizing deep copy
            """
            return self._port_store, self._build_funcs

        def copy(self, subgraph):
            """
                Deep copy from another component graph
            """
            self._port_store, self._build_funcs = subgraph.get_intrinsics()

        def ready_to_build(self):
            """
                Mark as building mode
            """
            self._is_building_mode = True
            
        def build(self):
            """
                Do the build stuff of this component graph
            """
            if not self._is_built:
                self._is_built = True
                for build_func in self._build_funcs:
                    build_func(self)

        def register_global_input_mapping(self, input_mapping, identifier='default'):
            """
                Delegator
            """
            self._super.register_input_mapping(input_mapping, identifier)
        
        def update_global_input_mapping(self, update_input_mapping, identifier='default'):
            """
                Delegator
            """
            self._super.update_input_mapping(update_input_mapping, identifier)

        def register_global_operation(self, operation, identifier='default'):
            """
                Delegator
            """
            self._super.register_operation(operation, identifier)

        def register_global_loss(self, loss, identifier='default'):
            """
                Delegator
            """
            self._super.register_loss(loss, identifier)

        def register_global_output(self, output, identifier='default'):
            """
                Delegator
            """
            self._super.register_output(output, identifier)

        def register_global_summary(self, summary, identifier='default'):
            """
                Delegator
            """
            self._super.register_summary(summary, identifier)
        
        def get_global_input_mapping(self, identifier='default'):
            """
                Delegator
            """
            self._super.get_input_mapping(identifier)

        def get_global_operations(self, identifier='default'):
            """
                Delegator
            """
            return self._super.get_operations(identifier)

        def get_global_losses(self, identifier='default'):
            """
                Delegator
            """
            return self._super.get_losses(identifier)
            
        def get_global_outputs(self, identifier='default'):
            """
                Delegator
            """
            return self._super.get_outputs(identifier)

        def get_global_summarys(self, identifier='default'):
            """
                Delegator
            """
            self._super.get_summarys(identifier)

    class _Connector(object):
        """
            Connector of main graph, connectes component graphs
        """
        def __init__(self, global_graph):
            """
                Initializer
                Params:
                    global_graph: main graph
            """
            self._global_graph = global_graph
            self._connect_funcs = []
        
        def __call__(self, connect_func=None):
            """
                Caller (), decorator function
            """
            self._connect_funcs = []
            if connect_func is None:
                # creates another decorator function
                def add_connect_func(connect_func):
                    self._connect_funcs.append(connect_func)
                    return connect_func
                return add_connect_func
            else:
                # default usage
                self._connect_funcs.append(connect_func)
            return connect_func
        
        def extend(self, connect_func=None):
            """
                Decorator function, non-reset
            """
            if connect_func is None:
                def add_connect_func(connect_func):
                    self._connect_funcs.append(connect_func)
                    return connect_func
                return add_connect_func
            else:
                self._connect_funcs.append(connect_func)
            return connect_func
        
        def build(self):
            """
                Perform connection component graphs
            """
            assert len(self._connect_funcs) > 0, "Graph connection is not specified"
            for connect_func in self._connect_funcs:
                connect_func(self._global_graph)

    def __init__(self):
        """
            Initializer of main graph
        """
        self._tf_graph = tf.Graph()
        self.inputgraph = self._SubGraph(self)
        self.usergraph = self._SubGraph(self)
        self.itemgraph = self._SubGraph(self)
        self.contextgraph = self._SubGraph(self)
        self.fusiongraph = self._SubGraph(self)
        self.interactiongraph = self._SubGraph(self)
        self.optimizergraph = self._SubGraph(self)

        self.connector = self._Connector(self)
        
        self._operation_identifier_set = set() # operation identifiers for optimizer etc. (retrieved by tf.get_collection())
        self._loss_identifier_set = set() # tensor identifiers for loss (retrieved by tf.get_collection())
        self._output_identifier_set = set() # tensor identifiers for output (retrieved by tf.get_collection())
        self._input_mapping_dict = dict() # dict[identifier](dict[name]tensor) for feed dict
        self._summary_identifier_set = set() # tensor identifiers for summary (retrieved by tf.get_collection())

    def __setattr__(self, name, value):
        """
            Setter of main graph
        """        
        if name in set(['inputgraph', 'usergraph', 'itemgraph', 'contextgraph',
                      'fusiongraph', 'interactiongraph', 'optimizergraph']):
            if name in self.__dict__:
                self.__dict__[name].copy(value)
            else:
                self.__dict__[name] = value
        else:
            self.__dict__[name] = value

    @property
    def tf_graph(self):
        """
            Property getter            
        """
        return self._tf_graph

    def build(self):
         """
             Build the main graph
         """
         with self._tf_graph.as_default():
            
            self.connector.build()

            self.inputgraph.ready_to_build()
            self.usergraph.ready_to_build()
            self.itemgraph.ready_to_build()
            self.contextgraph.ready_to_build()
            self.fusiongraph.ready_to_build()
            self.interactiongraph.ready_to_build()
            self.optimizergraph.ready_to_build()
            
            with tf.variable_scope('inputgraph', reuse=tf.AUTO_REUSE):
                self.inputgraph.build()
            with tf.variable_scope('usergraph', reuse=tf.AUTO_REUSE):
                self.usergraph.build()
            with tf.variable_scope('itemgraph', reuse=tf.AUTO_REUSE):
                self.itemgraph.build()
            with tf.variable_scope('contextgraph', reuse=tf.AUTO_REUSE):
                self.contextgraph.build()
            with tf.variable_scope('fusiongraph', reuse=tf.AUTO_REUSE):
                self.fusiongraph.build()
            with tf.variable_scope('interactiongraph', reuse=tf.AUTO_REUSE):
                self.interactiongraph.build()
            with tf.variable_scope('optimizergraph', reuse=tf.AUTO_REUSE):  
                self.optimizergraph.build()

    def register_input_mapping(self, input_mapping, identifier='default'):
        """
            Register identifiers for computation graph
        """
        self._input_mapping_dict[identifier] = input_mapping
    
    def update_input_mapping(self, update_input_mapping, identifier='default'):
        """
            Update identifiers for computation graph
        """        
        self._input_mapping_dict[identifier].update(update_input_mapping)

    def register_operation(self, operation, identifier='default'):
        """
            Register identifiers for computation graph
        """
        self._operation_identifier_set.add(identifier)
        tf.add_to_collection('recsys.recommender.operations.'+identifier, operation)

    def register_loss(self, loss, identifier='default'):
        """
            Register identifiers for computation graph
        """
        self._loss_identifier_set.add(identifier)
        tf.add_to_collection('recsys.recommender.losses.'+identifier, loss)

    def register_output(self, output, identifier='default'):
        """
            Register identifiers for computation graph
        """
        self._output_identifier_set.add(identifier)
        tf.add_to_collection('recsys.recommender.outputs.'+identifier, output)

    def register_summary(self, summary, identifier='default'):
        """
            Register identifiers for computation graph
        """
        self._summary_identifier_set.add(identifier)
        tf.add_to_collection('recsys.recommender.summarys.' + identifier, summary)

    def get_input_mapping(self, identifier='default'):
        """
            Get dict[name]tensor input for computation graph
        """
        return self._input_mapping_dict[identifier]

    def get_operations(self, identifier='default'):
        """
            Get list of operations for computation graph
        """
        with self._tf_graph.as_default():
            return tf.get_collection('recsys.recommender.operations.'+identifier)

    def get_losses(self, identifier='default'):
        """
            Get list of losses for computation graph
        """
        with self._tf_graph.as_default():
            return tf.get_collection('recsys.recommender.losses.'+identifier)

    def get_outputs(self, identifier='default'):
        """
            Get list of output tensors for computation graph
        """
        with self._tf_graph.as_default():
            return tf.get_collection('recsys.recommender.outputs.'+identifier)

    def get_summarys(self, identifier='default'):
        """
            Get list of summary tensors for computation graph
        """
        with self._tf_graph.as_default():
            return tf.get_collection('recsys.recommender.summarys.' + identifier)

class Recommender(object):
    """
        Recommender Base Class
        Wrapper for train/serve main graphs and model load/save utilities
    """
    def __init__(self, init_model_dir=None, save_model_dir=None, train=True, serve=False):
        """
            Initializer
            Params:
                init_model_dir: default directory to restore model from
                save_model_dir: default directory to save built model to
                train: switch
                serve: switch
        """
        self._train = train
        self._serve = serve
        self._init_model_dir = init_model_dir
        self._save_model_dir = save_model_dir
        
        self._flag_updated = False # mark updated by training, needs updating serving graph
        self._flag_isbuilt = False # only after graph is built can it be used to train/serve
        
        self.traingraph = _RecommenderGraph() # main graph
        self.servegraph = _RecommenderGraph() # main graph

        self.T = self.traingraph # shortcut
        self.S = self.servegraph # shortcut

    def build(self):
        """
            Build all graphs
        """
        if self._train:
            self.traingraph.build()
            with self.traingraph.tf_graph.as_default():
                config = tf.ConfigProto()
                config.gpu_options.allow_growth=True
                self._tf_train_sess = tf.Session(config=config)
                self._tf_train_sess.run(tf.global_variables_initializer())
                self._tf_train_writer = tf.summary.FileWriter(self._save_model_dir + "/logs", self._tf_train_sess.graph)
                self._tf_train_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        if self._serve:
            self.servegraph.build()
            with self.servegraph.tf_graph.as_default():
                config = tf.ConfigProto()
                config.gpu_options.allow_growth=True
                self._tf_serve_sess = tf.Session(config=config)
                self._tf_serve_sess.run(tf.global_variables_initializer())
                self._tf_serve_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        
        if self._init_model_dir is not None:
            self.restore(save_model_dir=self._init_model_dir,
                        restore_train=self._train,
                        restore_serve=self._serve)
        
        self._flag_isbuilt = True
        
        return self
    
    def isbuilt(self):
        """
            Is built
        """
        return self._flag_isbuilt

    def train(self, batch_data, input_mapping_id='default', operations_id='default',
              losses_id='default', outputs_id='default', summary_id = 'default'):
        """
            Train
            Run build() first
            Params:
                input_mapping_id, operations_id, losses_id, outputs_id: TF graph collection identifiers
        """
        assert self._train, "Train is disabled"
        assert self._flag_isbuilt, "Train graph is not built"
        
        if input_mapping_id is None:
            feed_dict = {}
        else:
            feed_dict = self._generate_feed_dict(batch_data, 
                                            self.T.get_input_mapping(input_mapping_id))
        
        if operations_id is None:
            operations = []
        else:
            operations = self.T.get_operations(operations_id)
        
        if losses_id is None:
            losses = []
        else:
            losses = self.T.get_losses(losses_id)
            
        if outputs_id is None:
            outputs = []
        else:
            outputs = self.T.get_outputs(outputs_id)
        
        if summary_id is None:
            summarys = []
        else:
            summarys = self.T.get_summarys(summary_id)
        results = self._tf_train_sess.run([summarys, operations, losses, outputs],
                                 feed_dict=feed_dict)
        return_dict = {'losses': results[2],
                      'outputs': results[3],
                      'summarys': results[0]}
        
        self._flag_updated = True
        return return_dict

    def serve(self, batch_data, input_mapping_id='default', operations_id='default', losses_id='default', outputs_id='default'):
        """
            Serve
            Run build() and train()/restore() first
            Params:
                input_mapping_id, operations_id, losses_id, outputs_id: TF graph collection identifiers
        """
        assert self._serve, "serve is disabled"
        assert self._flag_isbuilt, "serve graph is not built"
        
        if self._flag_updated:
            self._save_and_load_for_serve()
            self._flag_updated = False
        
        if input_mapping_id is None:
            feed_dict = {}
        else:
            feed_dict = self._generate_feed_dict(batch_data, self.S.get_input_mapping(input_mapping_id))
        
        if operations_id is None:
            operations = []
        else:
            operations = self.S.get_operations(operations_id)
        
        if losses_id is None:
            losses = []
        else:
            losses = self.S.get_losses(losses_id)
            
        if outputs_id is None:
            outputs = []
        else:
            outputs = self.S.get_outputs(outputs_id)
        results = self._tf_serve_sess.run(operations+losses+outputs, 
                            feed_dict=feed_dict)

        return {'losses': results[len(operations):len(operations)+len(losses)], 
                'outputs': results[-len(outputs):]}
    
    def export(self, export_model_dir=None, input_mapping_id='default', outputs_id='default',
            top_k=None, as_text=False):
        """
            Export pb model
            Params:
                top_k: None or int, if not None, if output is one tensor, apply tf.nn.top_k
                       and outputs y_values and y_indices instead
        """
        if export_model_dir is None:
            export_model_dir = self._save_model_dir + '_exported_pb'
        assert self._serve is not None, 'serve is not enabled.'
        with self.servegraph.tf_graph.as_default():
            input_mapping_dict = self.S.get_input_mapping(input_mapping_id)
            input_signature_def_dict = dict(
                [(k, tf.saved_model.utils.build_tensor_info(v)) for k, v in input_mapping_dict.items()])
            output_tensor_list = self.S.get_outputs(outputs_id) # original outputs
            output_signature_def_dict = dict() # signature

            if top_k is not None and isinstance(top_k, int) and len(output_tensor_list) == 1:
                logits_top_values, logits_top_indices = tf.nn.top_k(tf.squeeze(output_tensor_list[0]), k=top_k)
                output_signature_def_dict["y_values"] = tf.saved_model.utils.build_tensor_info(logits_top_values)
                output_signature_def_dict["y_indices"] = tf.saved_model.utils.build_tensor_info(logits_top_indices)
            else:
                for index in range(len(output_tensor_list)):
                    output_signature_def_dict["y_" + str(index)] = tf.saved_model.utils.build_tensor_info(output_tensor_list[index])

            MODEL_DIR = export_model_dir
            SIGNATURE_NAME = "serving_default"
            builder = tf.saved_model.builder.SavedModelBuilder(MODEL_DIR)
            builder.add_meta_graph_and_variables(self._tf_serve_sess, [tf.saved_model.tag_constants.SERVING], signature_def_map= {
                    SIGNATURE_NAME: tf.saved_model.signature_def_utils.build_signature_def(
                        inputs= input_signature_def_dict,
                        outputs= output_signature_def_dict,
                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
                    })
            builder.save(as_text=as_text)

    def predict_pb(self, feed_name_dict, export_model_dir=None, input_mapping_id='default', outputs_id='default',
            top_k=None):
        """
            Predict using pb model
            Params:
                top_k: None or not, if not none, model has two outputs: y_values and y_indices
        """
        if export_model_dir is None:
            export_model_dir = self._save_model_dir + '_exported_pb'
        assert self._serve is not None, 'serve is not enabled.'
        with tf.Session(graph=tf.Graph()) as sess:
            input_mapping_dict = self.S.get_input_mapping(input_mapping_id)
            input_signature_def_dict = dict(
                [(k, None) for k, _ in input_mapping_dict.items()])
            output_tensor_list = self.S.get_outputs(outputs_id)
            output_signature_def_dict = dict()
            if top_k is None:
                for index in range(len(output_tensor_list)):
                    output_signature_def_dict["y_" + str(index)] = None
            else:
                output_signature_def_dict["y_values"] = None
                output_signature_def_dict["y_indices"] = None

            MODEL_DIR = export_model_dir
            SIGNATURE_NAME = "serving_default"
            meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], MODEL_DIR)
            signature = meta_graph_def.signature_def
            input_tensor_dict = dict()
            output_tensor_dict = dict()
            for k in input_signature_def_dict.keys():
                input_tensor_dict[k] = sess.graph.get_tensor_by_name(
                    signature[SIGNATURE_NAME].inputs[k].name)
            for k in output_signature_def_dict.keys():
                output_tensor_dict[k] = sess.graph.get_tensor_by_name(
                    signature[SIGNATURE_NAME].outputs[k].name)
            feed_dict = dict([(input_tensor_dict[name], val) for name, val in feed_name_dict.items()])
            return dict(list(zip(
                output_signature_def_dict.keys(),
                sess.run(list(output_tensor_dict.values()), feed_dict=feed_dict))))

    def save(self, save_model_dir=None, global_step=None):
        """
            Save model
        """
        if save_model_dir is None:
            save_model_dir = self._save_model_dir
        with self.traingraph.tf_graph.as_default():
            self._tf_train_saver.save(self._tf_train_sess, 
                os.path.join(save_model_dir, 'model.ckpt'), global_step=global_step)

    def restore(self, save_model_dir=None, restore_train=False, restore_serve=False):
        """
            Restore model
        """
        if save_model_dir is None:
            save_model_dir = self._save_model_dir
        if restore_train:
            assert self._train is not None, 'train is not enabled.'
            with self.traingraph.tf_graph.as_default():
                self._optimistic_restore(self._tf_train_sess, os.path.join(save_model_dir, 'model.ckpt'))
        if restore_serve:
            assert self._serve is not None, 'serve is not enabled.'
            with self.servegraph.tf_graph.as_default():
                self._optimistic_restore(self._tf_serve_sess, os.path.join(save_model_dir, 'model.ckpt'))

    def _generate_feed_dict(self, batch_data, input_map):
        """
            Generate feed dict dict[tensor]np.array
            Params:
                batch_data: np.ndarray() dtype corresponds to input_map.keys()
                input_map: dict[name]tensor
        """
        feed_dict = dict()
        
        if type(batch_data) is np.ndarray:
            keys = batch_data.dtype.names
        elif type(batch_data) is dict:
            keys = batch_data.keys()
        else:
            assert False, "Invalid batch data format, data: {}".format(batch_data)

        for key in keys:
            feed_dict[input_map[key]] = batch_data[key]
        return feed_dict

    def _optimistic_restore(self, session, save_file):
        """
            Restore necessary variables in computation graph
        """
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map() 
        
        restore_vars = []
        for var in tf.global_variables():
            var_name = var.name.split(':')[0]
            if var_name in saved_shapes and len(var.shape) > 0:
                if var.get_shape().as_list() == saved_shapes[var_name]:
                    restore_vars.append(var)
        
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)

    def _save_and_load_for_serve(self):
        """
            Save and update serving graph
        """
        assert self._save_model_dir is not None, 'save_model_dir is not specified'
        if self._train:
            self.save()
        if self._serve:
            self.restore(restore_serve=True)

    def train_inspect_ports(self, batch_data, ports=[], input_mapping_id='default'):
        """
            Debug ports
            Params:
                ports: tensors
        """
        assert self._train, "Train is disabled"
        assert self._flag_isbuilt, "Train graph is not built"
        
        feed_dict = self._generate_feed_dict(batch_data, 
                                            self.T.get_input_mapping(input_mapping_id))
        
        results = self._tf_train_sess.run(ports,
                                 feed_dict=feed_dict)
        return results

    def serve_inspect_ports(self, batch_data, ports=[], input_mapping_id='default'):
        """
            Debug ports
            Params:
                ports: tensors
        """
        assert self._serve, "serve graph is disabled"
        assert self._flag_isbuilt, "serve graph is not built"
        
        if self._flag_updated:
            self._save_and_load_for_serve()
            self._flag_updated = False
        
        if input_mapping_id is None:
            feed_dict = {}
        else:
            feed_dict = self._generate_feed_dict(batch_data, 
                                            self.S.get_input_mapping(input_mapping_id))
        
        results = self._tf_serve_sess.run(ports,
                                 feed_dict=feed_dict)
        return results

    def train_writer_handler(self):
        """
            Get log writer handler        
        """
        return self._tf_train_writer

if __name__ == "__main__":
    rec = Recommender()
