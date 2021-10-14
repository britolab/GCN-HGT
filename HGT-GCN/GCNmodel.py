# -*- coding: UTF-8 -*- 

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import numpy as np
import sys

#define flags
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
        
del_all_flags(tf.flags.FLAGS)

tf.app.flags.DEFINE_string('f', '', 'kernel')

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("i", None, "add configuration")
flags.DEFINE_string("e", None, "add configuration")
flags.DEFINE_string("p", None, "add configuration")
flags.DEFINE_string("o", None, "add configuration")
flags.DEFINE_string("n", None, "add configuration")

#learning rate; epochs; size of hidden layers.
flags.DEFINE_float('learning_rate', 0.00005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails 
    for very large sparse tensors (>1M elements)"""
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)"""
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim], minval=-init_range,
        maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict 

class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=lambda x: x):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        # computes max(M(adj) * M(hidden1) * M(weights), 0)
        with tf.name_scope(self.name):        
            x = inputs # input is "hidden1"(output from GraphConvolutionSparse()) as SparseTensor
            x = tf.nn.dropout(x, 1-self.dropout) #Dropout for sparse tensors "x" 
            x = tf.matmul(x, self.vars['weights']) # "x" * "Weights" (32 * 16)
            x = tf.sparse_tensor_dense_matmul(self.adj, x) # "adj" * "x" 
            outputs = self.act(x) # Computes rectified linear: max(x, 0)
        return outputs

class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        # computes max(M(adj) * M(features) * M(weights), 0)
        with tf.name_scope(self.name):
            x = inputs  #input is "placeholders['features']" as SparseTensor 
            x = dropout_sparse(x, 1-self.dropout, self.features_nonzero) #Dropout for sparse tensors. 
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights']) #SparseTensor "x" * denseM "weights"(6336*32).
            x = tf.sparse_tensor_dense_matmul(self.adj, x) #Multiply SparseTensor adj by dense matrix "x"
            outputs = self.act(x) #Computes rectified linear: max(x, 0)
        return outputs
    
class InnerProductDecoder():
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, name, dropout=0., act=lambda x: x):
        self.name = name
        self.issparse = False
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs):
        # computes sigmoid(M(hidden2).dropout * M(hidden2).dropout.transpose)
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1-self.dropout)  # input is "hidden2"(output from GraphConvolution()) as SparseTensor
            x = tf.transpose(inputs) # transpose 
            x = tf.matmul(inputs, x) # hidden2.dropout * hidden2.dropout.transpose
            x = tf.reshape(x, [-1]) 
            outputs = self.act(x) # outputs = self.act(x) # Computes sigmoid of x element-wise.
        return outputs

class GCNModel():
    def __init__(self, placeholders, num_features, features_nonzero, name):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        with tf.variable_scope(self.name):
            self.build()

    def build(self):
        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=FLAGS.hidden1,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.dropout)(self.inputs)

        self.embeddings = GraphConvolution(
            name='gcn_dense_layer',
            input_dim=FLAGS.hidden1,
            output_dim=FLAGS.hidden2,
            adj=self.adj,
            act=lambda x: x,
            dropout=self.dropout)(self.hidden1)

        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=FLAGS.hidden2, 
            act=lambda x: x)(self.embeddings)
#            act=lambda x: x)(self.embeddings)

class Optimizer():
    # def __init__(self, model, preds, labels, num_nodes, num_edges):
    def __init__(self, preds, labels, num_nodes, num_edges):
        
  # This optimizer uses two hyperparameters in its training: pos_weight and norm
        
        pos_weight = float(num_nodes**2 - num_edges) / num_edges
        
        norm = num_nodes**2 / float((num_nodes**2 - num_edges) * 2 )  
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(                # average
                        tf.nn.weighted_cross_entropy_with_logits(  # cross-entropy 
                            logits=preds_sub,               # between predictions
                            targets=labels_sub,           # and pos/neg labels
                            pos_weight=pos_weight          # punishing positives 
                                                          # with adjusted weights
                        )) 

        # use l1 regularizer on both layer
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.001)
        weights = tf.trainable_variables()
        self.regularizer = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # AdamOptimizer
        self.opt_op = self.optimizer.minimize(self.cost + self.regularizer)
