# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:16:14 2019

@author: SHEN xiao

Please cite our paper as:
"Xiao Shen, Quanyu Dai, Fu-lai Chung, Wei Lu, and Kup-Sze Choi. Adversarial Deep Network Embedding for Cross-Network Node Classification. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), pages 2991-2999, 2020."

"""

import numpy as np
import pickle
import utils
from scipy.sparse import lil_matrix
import tensorflow.compat.v1 as tf

from ACDNE.evalModel import train_and_evaluate
#from src.Utils.combined_graph_union_node_attributes import marvel_graph
#from src.Enron_refactoring.Create_input_ACDNE_enron import csc_matrix_enron

from Preprocessing_Enron.Create_network import labels_array_enron
from Preprocessing_EU.Preprocess_labels_eu import labels_array_eu

#from Utils.combined_graph_union_node_attributes import lil_attr_matrix_second_graph, lil_attr_matrix_enron

with open('../pickle_temporary_data/adjacency_matrix_enroneu.pickle', 'rb') as f:
    csc_matrix_enron = pickle.load(f)

with open("../pickle_temporary_data/combined_attribute_matrix_enroneu.pickle", 'rb') as f:
   lil_attr_matrix_enron = pickle.load(f)

with open("../pickle_temporary_data/combined_attribute_matrix_euenron.pickle", 'rb') as f:
    lil_attr_matrix_second_graph = pickle.load(f)

#from src.Marvel_refactoring.Create_input_ACDNE_marvel import csc_matrix_marvel, nodes_marvel
with open("../pickle_temporary_data/adjacency_matrix_eu.pickle", 'rb') as f:
    csc_matrix_second_graph = pickle.load(f)



tf.disable_v2_behavior()
tf.set_random_seed(0)
np.random.seed(0)
#test
source = 'enron'
target = 'eu'
emb_filename = str(source) + '_' + str(target)
Kstep = 3

####################
# Load source data
####################
#A_s, X_s, Y_s = utils.load_network('/Users/freekstijfs/Documents/Uva/Thesis/DutchPoliceMasterThesis/Data/citationv1.mat')
'''compute PPMI'''
A_s = csc_matrix_enron
X_s = lil_attr_matrix_enron
Y_s = labels_array_enron

A_k_s = utils.AggTranProbMat(A_s, Kstep)
PPMI_s = utils.ComputePPMI(A_k_s)
n_PPMI_s = utils.MyScaleSimMat(PPMI_s)  # row normalized PPMI
X_n_s = np.matmul(n_PPMI_s, lil_matrix.toarray(X_s))  # neibors' attribute matrix

####################
# Load target data
####################
#A_t, X_t, Y_t = utils.load_network('../Data/dblpv7.mat')
A_t = csc_matrix_second_graph
X_t = lil_attr_matrix_second_graph
Y_t = labels_array_eu
#just being used for the f measure for as far as I can see
#Y_t = np.zeros((num_nodes_marvel,3))

'''compute PPMI'''
A_k_t = utils.AggTranProbMat(A_t, Kstep)
PPMI_t = utils.ComputePPMI(A_k_t)
n_PPMI_t = utils.MyScaleSimMat(PPMI_t)  # row normalized PPMI
X_n_t = np.matmul(n_PPMI_t, lil_matrix.toarray(X_t))  # neighbors' attribute matrix
#print("vstack",vstack([X_s, X_t]))

##input data
input_data = dict()
input_data['PPMI_S'] = PPMI_s
input_data['PPMI_T'] = PPMI_t
input_data['attrb_S'] = X_s
input_data['attrb_T'] = X_t
input_data['attrb_nei_S'] = X_n_s
input_data['attrb_nei_T'] = X_n_t
input_data['label_S'] = Y_s
input_data['label_T'] = Y_t

###model config
config = dict()
config['clf_type'] = 'multi-label'
config['dropout'] = 0.5
config['num_epoch'] = 10 # maximum training iteration
config['batch_size'] = 100
config['n_hidden'] = [512, 128]  # dimensionality for each k-th hidden layer of FE1 and FE2
config['n_emb'] = 20  # embedding dimension d
config['l2_w'] = 1e-3  # weight of L2-norm regularization
config['net_pro_w'] = 0.1  # weight of pairwise constraint
config['emb_filename'] = emb_filename  # output file name to save node representations
config['lr_ini'] = 0.02  # initial learning rate

numRandom = 5
microAllRandom = []
macroAllRandom = []

print('source and target networks:', str(source), str(target))
for random_state in range(numRandom):
    print("%d-th random initialization " % (random_state + 1))
    micro_t, macro_t = train_and_evaluate(input_data, config, random_state)

    microAllRandom.append(micro_t)
    macroAllRandom.append(macro_t)

'''avg F1 scores over 5 random splits'''
micro = np.mean(microAllRandom)
macro = np.mean(macroAllRandom)
micro_sd = np.std(microAllRandom)
macro_sd = np.std(macroAllRandom)

print("The avergae micro and macro F1 scores over %d random initializations are:  %f +/- %f and %f +/- %f: " % (
numRandom, micro, micro_sd, macro, macro_sd))

