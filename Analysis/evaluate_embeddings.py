from karateclub import GraRep, GraphWave, DeepWalk, Role2Vec, Node2Vec,ASNE
import networkx as nx
import scipy.io as sio

import pandas as pd

import pickle
import numpy as np
#from src.Enron_refactoring.Create_input_ACDNE_enron import coo_attr_matrix_enron
from Analysis.Evaluate_results import evaluate_embeddings,evaluate_embeddings_example_key_actors, evaluate_active_nodes, plot_embeddings
from Preprocessing_Enron.Preprocess_labels import label_dictionary
from scipy.sparse import csc_matrix, lil_matrix, coo_matrix
from Utils.Graph_Analyzer import Embedding_creator

"""Specify model being used, as well as its hyper parameters and the location of the embeddings/results to be stored"""
'''----------------------------------------------------------------------------------------------------------------'''
model = None
has_attributes = False
embeddings_known = True
ACDNE_used = True
if ACDNE_used:
    location_embeddings_acdne = "../Data/enron_euemb.mat"

if has_attributes:
    with open('../pickle_temporary_data/attr_matrix_enron.pickle',
            'rb') as f:
        attr_matrix = pickle.load(f)

location_results = "../Results/ACDNE.txt"

'''----------------------------------------------------------------------------------------------------------------'''

"""Get largest component of the networks"""
'''----------------------------------------------------------------------------------------------------------------'''
network1 = nx.read_gexf("../Data/Complete_graph.gexf")
network_largest_component = max(nx.connected_components(network1), key=len)
network1 = network1.subgraph(network_largest_component)
location_embedding_network1 = "../Data/embedding_grarep_network1.npy"

network2 = nx.read_gexf("../Data/Largest_subGraph_graph_Marvel.gexf")
network_largest_component = max(nx.connected_components(network2), key=len)
network2 = network2.subgraph(network_largest_component)
location_embedding_network2 = "../Data/embedding_grarep_network2.npy"

'''----------------------------------------------------------------------------------------------------------------'''

"""Create and save embeddings"""
'''----------------------------------------------------------------------------------------------------------------'''
embedding_creator = Embedding_creator(network1, model)
if embeddings_known == False:
    if has_attributes:
        embeddings,numeric_dictionary1 = embedding_creator.create_embeddings_ASNE(attr_matrix)
    else:
        embeddings, numeric_dictionary1 = embedding_creator.create_embeddings()
    embedding_creator.write_embedding(location_embedding_network1)
    embeddings = np.load(location_embedding_network1)
    embedding_dictionary_source = embedding_creator.create_embeddings_dictionary_nodes(list(network1.nodes()))
elif ACDNE_used:
    mat = sio.loadmat(location_embeddings_acdne)
    embedding_source = mat['rep_S']  # variable in mat file)
    embedding_creator.embedding = embedding_source
    embedding_dictionary_source = embedding_creator.create_embeddings_dictionary_nodes(list(network1.nodes()))
else:
    embeddings = np.load(location_embedding_network1)
    embedding_dictionary_source = embedding_creator.create_embeddings_dictionary_nodes(list(network1.nodes()))




embedding_creator = Embedding_creator(network2, model)
if embeddings_known == False:
    if has_attributes:
        embeddings,numeric_dictionary1 = embedding_creator.create_embeddings(attr_matrix)
    else:
        embeddings, numeric_dictionary1 = embedding_creator.create_embeddings()
    embedding_creator.write_embedding(location_embedding_network2)
    embeddings = np.load(location_embedding_network2)
    embedding_dictionary_target = embedding_creator.create_embeddings_dictionary()

elif ACDNE_used:
    mat = sio.loadmat(location_embeddings_acdne)
    embedding_target = mat['rep_T']  # variable in mat file)
    embedding_creator.embedding = embedding_target
    embedding_dictionary_target = embedding_creator.create_embeddings_dictionary_nodes(list(network2.nodes()))
else:
    embeddings = np.load(location_embedding_network2)
    embedding_dictionary_arget = embedding_creator.create_embeddings_dictionary_nodes(list(network2.nodes()))


'''----------------------------------------------------------------------------------------------------------------'''

"""Evaluate results"""
'''----------------------------------------------------------------------------------------------------------------'''
target_graph = network2
"""Specify the key actor(s) for who you want to find similar actors"""
key_actors = [k for k,v in label_dictionary.items() if v == 'Key Actor']
evaluate_embeddings(key_actors,["eigenvector","pagerank"],embedding_dictionary_source, embedding_dictionary_target, target_graph, top_n_embeddings=int((len(target_graph.nodes)/10)), top_n_percent=10,location_result_file=location_results)

