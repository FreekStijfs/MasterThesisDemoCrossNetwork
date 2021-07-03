from src.ACDNE_model.ACDNE_model import ACDNE
from src.Utils.combined_graph_union_node_attributes import lil_attr_matrix_marvel, lil_attr_matrix_enron
from src.Enron_refactoring.Create_network import labels_array_enron

import tensorflow.compat.v1 as tf

config = dict()
config['clf_type'] = 'multi-label'
config['dropout'] = 0.5
config['num_epoch'] = 100 # maximum training iteration
config['batch_size'] = 100
config['n_hidden'] = [512, 128]  # dimensionality for each k-th hidden layer of FE1 and FE2
config['n_emb'] = 20  # embedding dimension d
config['l2_w'] = 1e-3  # weight of L2-norm regularization
config['net_pro_w'] = 0.1  # weight of pairwise constraint
config['emb_filename'] = 'TEST'  # output file name to save node representations
config['lr_ini'] = 0.02  # initial learning rate
###model config
clf_type = config['clf_type']
dropout = config['dropout']
num_epoch = config['num_epoch']
batch_size = config['batch_size']
n_hidden = config['n_hidden']
n_emb = config['n_emb']
l2_w = config['l2_w']
net_pro_w = config['net_pro_w']
emb_filename = config['emb_filename']
lr_ini = config['lr_ini']


X_s = lil_attr_matrix_enron
Y_s = labels_array_enron
n_input = X_s.shape[1]
num_class = Y_s.shape[1]

with tf.Graph().as_default():
    model = ACDNE(n_input, n_hidden, n_emb, num_class, clf_type, l2_w, net_pro_w, batch_size)

    with tf.Session() as sess:
        sess.run(model.train_op)
