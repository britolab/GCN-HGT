# -*- coding: UTF-8 -*-  
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#ignore future warnings of packages
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# import packages
import sys
import tensorflow as tf # v.1.13.1
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import pandas as pd
import numpy as np
import random
import time
import scipy.sparse as sp
import argparse
import datetime
from tqdm import tqdm

# from scipy.special import softmax
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

#pasers
parser = argparse.ArgumentParser(description='train GCN using the HGT network and node features')
parser.add_argument('-e','--epochs_num', type=int, metavar='',required=True,
                   help='maximum epoch number of training')
parser.add_argument('-p','--p_observed', type=float, metavar='',required=True,
                   help='percent of observed network')
parser.add_argument('-o','--output_path', type=str, metavar='',required=True,
                   help='output path prefix')
arg=parser.parse_args()


# plot setstyle
sns.set(font_scale=1.3)
sns.set_style('white')

#make output directory
rannum=random.randint(10000000, 99999999)
timestr = time.strftime("%Y%m%d-%H%M%S")
outdir='.../experiment_'\
+str(arg.output_path)+'_epochs'+str(arg.epochs_num)+'_'+str(arg.p_observed)+'seen_'+timestr+str(rannum)
os.mkdir(outdir)
os.chdir(outdir)

# load HGT files 
D16S_HGT=pd.read_pickle('.../HGT.all.pkl')

# positive and negative edges
df_nozero=D16S_HGT[D16S_HGT.HGT==1]
df_zero=D16S_HGT[D16S_HGT.HGT==0]
df_zero_np=np.asarray(df_zero)

# all nodes
node_list=list(pd.concat([D16S_HGT.G1,  D16S_HGT.G2]).unique())
nodelist=sorted(node_list, key=lambda k: random.random())

os.chdir(outdir)

# load KO annotations
features_orig=pd.read_csv(".../final_KO.annotation_with_direct_alignment.txt",sep=',',index_col=0,low_memory=False)

# import taxa and 16S distance file
highQ_taxa=pd.read_csv(".../Genome.clean.highQ.noIMG.csv",index_col=0)
taxa_species=highQ_taxa[['species']]
close_16S = pd.read_csv('.../genome.pairs.16Smore97.csv', header = None, sep = ' ')

# randomly select test nodes
test_nodes = nodelist[0:500]

# find close relatives to all test nodes
close_16S_test = list(set(close_16S[close_16S[0].isin(test_nodes)][1].to_list() + close_16S[close_16S[1].isin(test_nodes)][0].to_list()))

# defind train and val nodes by excluding same species and close relatives with any test nodes
train_val_nodes = []
test_taxa_list = np.asarray(taxa_species.loc[test_nodes])
for i in range(len(nodelist)-len(test_nodes)):
    if np.asarray(taxa_species.loc[nodelist[len(test_nodes)+i]]) not in test_taxa_list and nodelist[len(test_nodes)+i] not in close_16S_test :  
        train_val_nodes.append(nodelist[len(test_nodes)+i])
val_nodes = random.sample(train_val_nodes,500) # randomly select 500 nodes for validation set
train_nodes = [x for x in train_val_nodes if x not in val_nodes]

# mkdir data and save raw nodes
os.mkdir('raw_data')
np.save('raw_data/test_nodes',test_nodes)
np.save('raw_data/train_nodes',train_nodes)
np.save('raw_data/val_nodes',val_nodes)

# import model
os.chdir('.../')
from GCN_model import *
os.chdir(outdir)

# sigmoid 
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# return actual node features
def nodes_to_features(nodes_names):
    subgraph_len = len(nodes_names)
    data_list = []

    for i in range(0,subgraph_len):
        if nodes_names[i] in list(features_orig.index): 
            data_list.append(list(features_orig.loc[nodes_names[i]]))
    data_list = pd.DataFrame(data_list)
    return data_list

# make dummy feature file 
def nodes_to_dummy_features(nodes_names):
    subgraph_len=len(nodes_names)
    data_list=[]

    for i in range(0,subgraph_len):
        if nodes_names[i] in list(features_dummy.index): 
                data_list.append(list(features_dummy.loc[nodes_names[i]]))
    data_list=pd.DataFrame(data_list)
    return data_list

# make adjcency matrix using subset of nodes
def nodes_to_adj(nodes):
    df_nodes=D16S_HGT[D16S_HGT['G1'].isin(nodes) & D16S_HGT['G2'].isin(nodes)]
    G_sub = nx.from_pandas_edgelist(df_nodes,'G1','G2', 'HGT')
    node_list_Gsub = list(G_sub.nodes())
    adj_sub = nx.to_numpy_matrix(G_sub, weight = 'HGT')
    adj_sub = sp.csr_matrix(adj_sub)

    return node_list_Gsub, adj_sub, df_nodes

# defind negative edges for training set
def make_false_edges(node_list,num):
    df_zero_np=np.asarray(df_zero)
    np.random.shuffle(df_zero_np)
    edges_false=[]
    for i in range(len(df_zero)):
        if len(edges_false) < num and df_zero_np[i][0] in node_list and df_zero_np[i][1] in node_list :
            edges_false.append([node_list.index(df_zero_np[i][0]),node_list.index(df_zero_np[i][1])])
    return edges_false

# create censored edges
def creat_censor_edges(censor_G_node_list,total_edges_list,nodes_names):
    censor_true_edges=[]
    total_edges_list_sub = total_edges_list[total_edges_list.G1.isin(nodes_names) 
                                            | total_edges_list.G2.isin(nodes_names)]
    total_edges_list_sub_pos = total_edges_list_sub[total_edges_list_sub.HGT == 1]
    for i in range(len(total_edges_list_sub_pos)):
        censor_true_edges.append([censor_G_node_list.index(np.asarray(total_edges_list_sub_pos)[i][0]),
                                  censor_G_node_list.index(np.asarray(total_edges_list_sub_pos)[i][1])])
    return censor_true_edges

# define negative edges for censored set 
def make_false_edges_censor(node_list_subset,num,node_list):
    edges_false = df_zero[df_zero['G1'].isin(node_list_subset) & df_zero['G1'].isin(node_list) &\
                         df_zero['G2'].isin(node_list_subset) & df_zero['G2'].isin(node_list)].\
    reset_index().drop('index',axis=1)
    
    edges_false_np = np.asarray(edges_false.sample(num))
    
    edges_false_final=[]
    for i in range(len(edges_false_np)):
        edges_false_final.append([node_list.index(edges_false_np[i][0]),node_list.index(edges_false_np[i][1])])
    
    return edges_false_final

# normalized adjacency matrix
def preprocess_graph_second(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

# train nodes and edges
node_list_train, train_adj, train_edges_list = nodes_to_adj(train_nodes)

# train adjacency matrix
adj_orig = train_adj - sp.dia_matrix((train_adj.diagonal()[np.newaxis, :], [0]), shape=train_adj.shape)
adj_orig.eliminate_zeros()

# feature tables for training data
feature_train=nodes_to_features(node_list_train)
feature_train_table_spx=sp.csr_matrix(feature_train)
features=sparse_to_tuple(feature_train_table_spx)
num_features = features[2][1]
features_nonzero = features[1].shape[0]
num_nodes = train_adj.shape[0]
num_edges = train_adj.sum()

# positive and negative edges of training sets 
train_edges=random.sample(sparse_to_tuple(train_adj)[0].tolist(),20000)
train_edges_false=make_false_edges(node_list_train,20000)

# nodes of censored data
censor_val_nodes = list(val_nodes + train_nodes)
censor_test_nodes= list(train_nodes + test_nodes)

# make censored edges
censor_test_G_nodelist, censor_test_adj, censor_test_edges_list=nodes_to_adj(censor_test_nodes)
censor_val_G_nodelist, censor_val_adj, censor_val_edges_list =nodes_to_adj(censor_val_nodes)

# create censor edges
censor_test_edges_true=creat_censor_edges(censor_test_G_nodelist, censor_test_edges_list, test_nodes)
seen_test=np.asarray(pd.DataFrame(censor_test_edges_true).sample(int(len(censor_test_edges_true)*arg.p_observed))).tolist()
unseen_test=[x for x in censor_test_edges_true if x not in seen_test]

censor_val_edges_true=creat_censor_edges(censor_val_G_nodelist, censor_val_edges_list, val_nodes)
seen_val=np.asarray(pd.DataFrame(censor_val_edges_true).sample(int(len(censor_val_edges_true)*arg.p_observed))).tolist()
unseen_val=[x for x in censor_val_edges_true if x not in seen_val]

# create censored adjacency matrix
censor_test_adj_seen=sp.csr_matrix(censor_test_adj.toarray())
for i in range(len(unseen_test)):
    censor_test_adj_seen[unseen_test[i][0],unseen_test[i][1]] = 0
    censor_test_adj_seen[unseen_test[i][1],unseen_test[i][0]] = 0

censor_val_adj_seen=sp.csr_matrix(censor_val_adj.toarray())
for i in range(len(unseen_val)):
    censor_val_adj_seen[unseen_val[i][0],unseen_val[i][1]] = 0
    censor_val_adj_seen[unseen_val[i][1],unseen_val[i][0]] = 0
    
# remove zeros from censored adjacency matrix
censor_test_adj_seen.eliminate_zeros()
censor_val_adj_seen.eliminate_zeros()

# create negative edges for censored sets
censor_test_edges_false = make_false_edges_censor(test_nodes,len(unseen_test),censor_test_G_nodelist)
censor_val_edges_false = make_false_edges_censor(val_nodes,len(unseen_val),censor_val_G_nodelist)

# feature tables for censored sets
feature_val_censor = sp.csr_matrix(nodes_to_features(censor_val_G_nodelist))
feature_test_censor = sp.csr_matrix(nodes_to_features(censor_test_G_nodelist))

# create the normalized adjacency matrix for train, val and test sets
adj_temp = train_adj
adj_norm = preprocess_graph(adj_temp)
adj_temp = censor_val_adj_seen
adj_norm_val = preprocess_graph_second(adj_temp)
adj_temp = censor_test_adj_seen
adj_norm_test = preprocess_graph_second(adj_temp)

# define the training 
def train_GCN_model_censor(adj_norm_train, adj_norm_val, adj_norm_test,dropout_rate, epochs, features, feature_val, feature_test):
    # evaluation scores of training 
    def get_roc_score_train(edges_pos, edges_neg):
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.embeddings, feed_dict=feed_dict)

        # Predict on test set of edges
        adj_rec = sigmoid(np.dot(emb, emb.T))
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(adj_rec[e[0], e[1]])
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(adj_rec[e[0], e[1]])
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
        # _, rec_score, _ = precision_recall_curve(labels_all, preds_all)

        return roc_score, ap_score

    # evaluation scores of validation and test  
    def get_preds(features_input,adj_norm_input,adj_orig_input,edges_pos, edges_neg):
        adj_label_temp = adj_orig_input + sp.eye(adj_orig_input.shape[0])
        adj_label = sparse_to_tuple(adj_label_temp)

        var1=[v for v in tf.trainable_variables() if v.name == current_model_name + '/gcn_sparse_layer_vars/weights:0'][0]
        layer1_weights = sess.run(var1)

        var2=[v for v in tf.trainable_variables() if v.name == current_model_name + '/gcn_dense_layer_vars/weights:0'][0]
        layer2_weights = sess.run(var2)
        layer1_output = np.maximum((np.matmul(adj_norm_input.toarray(),np.matmul(features_input.toarray(),layer1_weights))),0)
        layer2_output = np.matmul(adj_norm_input.toarray(),np.matmul(layer1_output,layer2_weights))
        adj_rec = sigmoid(np.dot(layer2_output,layer2_output.T))
            
        # Predict on test set of edges
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(adj_rec[e[0], e[1]])
            pos.append(adj_orig_input[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(adj_rec[e[0], e[1]])
            neg.append(adj_orig_input[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        precision, recall, _ = precision_recall_curve(labels_all, preds_all)
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return layer1_weights, layer2_weights, preds_all,labels_all, precision, recall, roc_score, ap_score
    
    def get_roc_score(features_input,adj_norm_input,adj_orig_input,edges_pos, edges_neg):
        
        layer1_weights, layer2_weights, preds_all,labels_all, precision, recall, roc_score, ap_score = get_preds(features_input,adj_norm_input,adj_orig_input,edges_pos, edges_neg)
        
        return roc_score, ap_score
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label_temp = train_adj + sp.eye(train_adj.shape[0])
    adj_label = sparse_to_tuple(adj_label_temp)

    saver = tf.train.Saver()
    train_loss= []
    roc_train = []
    roc_test = []
    ap_test= []
    roc_val = []
    ep = []

    # Train model
    for epoch in range(epochs):
        t = time.time()
        if epoch <= ... : ## define the stopping point of training

            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm_train, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout']: dropout_rate})

            # One update of parameter matrices
            _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

            # Performance on validation set
            from sklearn.metrics import recall_score
            roc_curr2, ap_curr2 = get_roc_score_train(train_edges[0:20000], train_edges_false[0:20000])
            roc_curr, ap_curr = get_roc_score(feature_val, adj_norm_val, censor_val_adj, unseen_val[0:20000], censor_val_edges_false[0:20000])
            roc_curr1, ap_curr1 = get_roc_score(feature_test, adj_norm_test, censor_test_adj, unseen_test[0:20000], censor_test_edges_false[0:20000])
            train_loss.append(avg_cost)
            roc_val.append(roc_curr)
            roc_test.append(roc_curr1)
            roc_train.append(roc_curr2)
            ap_test.append(ap_curr1)
            roc_val_max=np.max(roc_val)
            ep.append(epoch + 1)

            print("Epoch:", '%04d' % (epoch + 1), 
                  "train_loss=", "{:.5f}".format(avg_cost),
                  "train_roc=","{:.5f}".format(roc_curr2),
                  "val_roc=", "{:.5f}".format(roc_curr),
                  "test_roc=", "{:.5f}".format(roc_curr1),
                  "test_precision=", "{:.5f}".format(ap_curr1))
        else:
             break

    print('Optimization Finished!', epoch)
    layer1_weights, layer2_weights, preds_all, labels_all, precision, recall, _, _ = get_preds(feature_test, adj_norm_test, censor_test_adj, unseen_test[0:20000], censor_test_edges_false[0:20000])
    return ep, train_loss, roc_val, roc_test, roc_train, ap_test, precision, recall, labels_all, preds_all, layer1_weights, layer2_weights

#save processed data
import pickle
os.mkdir('GCN_processed_data')

with open('GCN_processed_data/node_list_train ', 'wb') as f:
    pickle.dump(node_list_train, f)  
with open('GCN_processed_data/censor_test_G_nodelist ', 'wb') as f:
    pickle.dump(censor_test_G_nodelist, f)  
with open('GCN_processed_data/censor_val_G_nodelist ', 'wb') as f:
    pickle.dump(censor_val_G_nodelist, f)  
with open('GCN_processed_data/adj_norm ', 'wb') as f:
    pickle.dump(adj_norm, f)    
with open('GCN_processed_data/features', 'wb') as f:
    pickle.dump(features, f)    
with open('GCN_processed_data/train_edges', 'wb') as f:
    pickle.dump(train_edges, f)    
with open('GCN_processed_data/train_edges_false', 'wb') as f:
    pickle.dump(train_edges_false, f)    
with open('GCN_processed_data/unseen_test', 'wb') as f:
    pickle.dump(unseen_test, f)
with open('GCN_processed_data/censor_test_edges_false', 'wb') as f:
    pickle.dump(censor_test_edges_false, f)
with open('GCN_processed_data/unseen_val', 'wb') as f:
    pickle.dump(unseen_val, f)
with open('GCN_processed_data/censor_val_edges_false', 'wb') as f:
    pickle.dump(censor_val_edges_false, f)
    
#save processed data into npz
sp.save_npz('GCN_processed_data/feature_train',feature_train_table_spx)
sp.save_npz('GCN_processed_data/adj_norm_val',adj_norm_val)
sp.save_npz('GCN_processed_data/adj_norm_test',adj_norm_test)
sp.save_npz('GCN_processed_data/feature_val_censor',feature_val_censor)
sp.save_npz('GCN_processed_data/feature_test_censor',feature_test_censor)
sp.save_npz('GCN_processed_data/censor_test_adj',censor_test_adj)
sp.save_npz('GCN_processed_data/censor_val_adj',censor_val_adj)
sp.save_npz('GCN_processed_data/train_adj',train_adj)
sp.save_npz('GCN_processed_data/adj_orig',adj_orig)

# train 
for i in range(0, 1):
    epoches = arg.epochs_num
    
    tf.reset_default_graph()
    
    placeholders = {
                    'features': tf.sparse_placeholder(tf.float32),
                    'adj': tf.sparse_placeholder(tf.float32),
                    'adj_orig': tf.sparse_placeholder(tf.float32),
                    'dropout': tf.placeholder_with_default(0., shape=())
                    }

    # Create model
    model = GCNModel(placeholders, num_features, features_nonzero, name='HGT_GCN_censor')
    # Create optimizer
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], 
            validate_indices=False), [-1]),
            num_nodes=num_nodes,
            num_edges=num_edges)

    current_model_name=str(tf.trainable_variables()[-1]).split("'")[1].split("/")[0]

    epoch_i, train_loss_i, roc_val_i, roc_test_i, roc_train_i, ap_test_i, precision_i, recall_i, labels_readadj_censor, preds_readadj_censor, layer1_weights_readadj_censor, layer2_weights_readadj_censor =   train_GCN_model_censor(adj_norm, adj_norm_val, adj_norm_test, 0.1,  epoches , features, feature_val_censor, feature_test_censor)
    
    if i == 0:
        ROC_all_readadj_censor=pd.DataFrame({"Epoch": epoch_i,"ROC_val": roc_val_i, "ROC_test": roc_test_i, "ROC_train": roc_train_i})
        Precision_readadj_censor=pd.DataFrame({"Precision": precision_i})
        Recall_readadj_censor=pd.DataFrame({"Recall": recall_i})
    else:
        ROC_all_readadj_censor=ROC_all_readadj_censor.append(pd.DataFrame({"Epoch": epoch_i,"ROC_val": roc_val_i, "ROC_test": roc_test_i, "ROC_train": roc_train_i}))
        Precision_readadj_censor=Precision_readadj_censor.append(pd.DataFrame({"Precision": precision_i}))
        Recall_readadj_censor=Recall_readadj_censor.append(pd.DataFrame({"Recall": recall_i}))

# make graphs -- Pred vs. label.pdf; learning curve.pdf;
os.mkdir('GCN_graph')

pred_label=pd.DataFrame([labels_readadj_censor,preds_readadj_censor]).T

fig = plt.figure()
sns.set(font_scale = 1.2)
sns.violinplot(0,1, data=pred_label)
plt.xlabel("Label")
plt.ylabel("Pred")
fig.savefig("GCN_graph/Pred_label.pdf")

fig = plt.figure()
sns.set(font_scale = 1.3)
sns.set_style("whitegrid") 
sns.lineplot(x='Epoch', y='ROC_val', data=ROC_all_readadj_censor)
sns.lineplot(x='Epoch', y='ROC_test', data=ROC_all_readadj_censor)
sns.lineplot(x='Epoch', y='ROC_train', data=ROC_all_readadj_censor)
plt.legend(labels=['ROC_val', 'ROC_test', 'ROC_train'], fontsize=15)
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("")
fig.savefig("GCN_graph/Learining_curve_all_KO_with_adj.pdf",transparent=True)

# evaluations score when testing using dummy features
def get_roc_score_test(features_input,adj_norm_input,adj_orig_input,edges_pos, edges_neg):
    adj_label_temp = adj_orig_input + sp.eye(adj_orig_input.shape[0])
    adj_label = sparse_to_tuple(adj_label_temp)

    layer1_weights = layer1_weights_readadj_censor
    layer2_weights = layer2_weights_readadj_censor

    layer1_output = np.maximum((np.matmul(adj_norm_input.toarray(),np.matmul(features_input.toarray(),layer1_weights))),0)
    layer2_output= np.matmul(adj_norm_input.toarray(),np.matmul(layer1_output,layer2_weights)) 
    adj_rec = sigmoid(np.dot(layer2_output,layer2_output.T))

    preds = []
    pos = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])
        pos.append(adj_orig_input[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])
        neg.append(adj_orig_input[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score, labels_all, preds_all

# make dummy feature tables
features_dummy = pd.DataFrame(np.random.randint(2, size=(features_orig.shape[0], features_orig.shape[1])))
features_dummy.index=features_orig.index
features_dummy.columns=features_orig.columns
feature_test_censor_dummy = sp.csr_matrix(nodes_to_dummy_features(censor_test_G_nodelist))

# return the dummy score
roc_score_dummy, ap_score_dummy, labels_all_dummy, preds_all_dummy=get_roc_score_test(feature_test_censor_dummy, adj_norm_test, censor_test_adj, unseen_test[0:20000], censor_test_edges_false[0:20000])

fpr_true, tpr_true, thr_true = roc_curve(labels_readadj_censor, preds_readadj_censor)
fpr_dummy, tpr_dummy, thr_dummy = roc_curve(labels_all_dummy, preds_all_dummy)
precision_dummy, recall_dummy, _ = precision_recall_curve(labels_all_dummy, preds_all_dummy)

#generate reconstructed adj
layer1_weights = layer1_weights_readadj_censor
layer2_weights = layer2_weights_readadj_censor

layer1_output = np.maximum((np.matmul(adj_norm_test.toarray(),np.matmul(feature_test_censor.toarray(),layer1_weights))),0)
layer2_output = np.matmul(adj_norm_test.toarray(),np.matmul(layer1_output,layer2_weights)) 
adj_rec=sigmoid(np.dot(layer2_output,layer2_output.T))

layer1_output = np.maximum((np.matmul(adj_norm_test.toarray(),np.matmul(feature_test_censor_dummy.toarray(),layer1_weights))),0)
layer2_output = np.matmul(adj_norm_test.toarray(),np.matmul(layer1_output,layer2_weights))
adj_rec_dummy = sigmoid(np.dot(layer2_output,layer2_output.T))

# plots ROC/PR-Recall curve
fig = plt.figure()
sns.set_style('white')
plt.plot(fpr_true,tpr_true,color='brown',lw=2)
plt.plot(fpr_dummy,tpr_dummy,color='darkblue',lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate',size=15)
plt.ylabel('True Positive Rate',size=15)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(('True','Random'),fontsize=10)
plt.tick_params(labelsize=15)
fig.savefig("GCN_graph/ROC_curve.all.pdf")

fig = plt.figure ()
plt.step(Recall_readadj_censor,Precision_readadj_censor, color='brown')
plt.step(recall_dummy,precision_dummy, color='darkblue')
plt.hlines(0.5,0,1,color='navy', lw=1, linestyle='--')
plt.xlabel('Recall',size=15)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Precision',size=15)
plt.legend(('True','Random'),fontsize=10)
plt.gca().set_aspect('equal', adjustable='box')
fig.savefig("GCN_graph/Precision_recall.all.pdf")

#save the evaluation and prediction data
fpr, tpr, thr = roc_curve(labels_readadj_censor, preds_readadj_censor)
os.mkdir('GCN_evaluation')
np.save('GCN_evaluation/All_KO_fpr',fpr)
np.save('GCN_evaluation/All_KO_tpr',tpr)
np.save('GCN_evaluation/All_KO_pre',Precision_readadj_censor.values)
np.save('GCN_evaluation/All_KO_rec',Recall_readadj_censor.values)
np.save('GCN_evaluation/fpr_dummy',fpr_dummy)
np.save('GCN_evaluation/tpr_dummy',tpr_dummy)
np.save('GCN_evaluation/pre_dummy',precision_dummy)
np.save('GCN_evaluation/rec_dummy',recall_dummy)
np.save('GCN_prediction/labels_readadj_censor',labels_readadj_censor)
np.save('GCN_prediction/preds_readadj_censor',preds_readadj_censor)
np.save('GCN_prediction/layer1_weights_readadj_censor',layer1_weights_readadj_censor)
np.save('GCN_prediction/layer2_weights_readadj_censor',layer2_weights_readadj_censor)
np.save('GCN_prediction/adj_rec',adj_rec)
np.save('GCN_prediction/adj_rec_dummy',adj_rec_dummy)
features_dummy.to_pickle('GCN_evaluation/features_dummy.pkl')

