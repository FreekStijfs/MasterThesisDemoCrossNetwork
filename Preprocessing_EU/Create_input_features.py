import networkx as nx
import pandas as pd
import pickle
from karateclub import DeepWalk

import scipy as sp
import numpy as np
from scipy.io import savemat
from scipy.sparse import csc_matrix, lil_matrix
global csc_matrix_enron, lil_attr_matrix, nodes_eu, eu_graph, num_nodes

#from src.Enron_refactoring.Create_input_ACDNE_enron import Graph_analyzer, create_combined_attribute_dictionary
from Utils.utils import create_combined_attribute_dictionary, Graph_analyzer,create_encoded_dictionary
print("Creating input ACDNE model EU")
eu_graph = nx.read_gexf("../Data/Complete_graph_eu.gexf")
nodes_eu = eu_graph.nodes()
location_edgelist = "../Data/eu_graph.edgelist"

dictionary_pagerank_eu = nx.pagerank(eu_graph)
dictionary_eigenvector_eu = nx.eigenvector_centrality(eu_graph)
dictionary_degree_centrality_eu = nx.degree_centrality(eu_graph)
dictionary_in_degree_centrality_eu = nx.degree_centrality(eu_graph)
closeness_centrality_eu = nx.closeness_centrality(eu_graph)
#nx.set_node_attributes(graph, dictionary_pagerank, "pagerank")
dictionary_pagerank_eu = create_encoded_dictionary(dictionary_pagerank_eu, "pagerank", nodes_eu)
dictionary_eigenvector_eu = create_encoded_dictionary(dictionary_eigenvector_eu, "eigenvector", nodes_eu)
dictionary_degree_centrality_eu = create_encoded_dictionary(dictionary_degree_centrality_eu, "degree", nodes_eu)
dictionary_in_degree_centrality_eu = create_encoded_dictionary(dictionary_in_degree_centrality_eu, "in_degree", nodes_eu)
closeness_centrality_eu = create_encoded_dictionary(closeness_centrality_eu, "closeness", nodes_eu)

contacts_dictionary = {}
for node in eu_graph.nodes():
    contacts_dictionary[node] = eu_graph.edges(node)


def num_spaths(graph):
    n_spaths = dict.fromkeys(graph, 0.0)
    spaths = dict(nx.all_pairs_shortest_path(graph))
    for source in graph.nodes():
        for path in spaths[source].values():
            for node in path[1:]: # ignore firs element (source == node)
                n_spaths[node] += 1 # this path passes through `node`

    return n_spaths
number_of_shortest_paths = num_spaths(eu_graph)

mappings_to_be_added = [dictionary_eigenvector_eu, dictionary_pagerank_eu,contacts_dictionary, dictionary_degree_centrality_eu, dictionary_in_degree_centrality_eu,closeness_centrality_eu,number_of_shortest_paths]

combined_attribute_dictionary = create_combined_attribute_dictionary(eu_graph.nodes, mappings_to_be_added)

nx.set_node_attributes(eu_graph, combined_attribute_dictionary, "pagerank")


threshold = 0.00000000000001
degree_centrality_df = pd.DataFrame(list(dictionary_degree_centrality_eu.items()),columns = ['User','Degree_Centrality'])

Eu_graph_analyzer = Graph_analyzer(eu_graph, degree_centrality_df,threshold)
mat= Eu_graph_analyzer.create_adjacency_matrix(False)

csc_matrix_eu = csc_matrix(mat)

attr_matrix = nx.attr_matrix(eu_graph)
lil_attr_matrix_eu = lil_matrix(attr_matrix[0], shape=(len(attr_matrix),1))

with open('../pickle_temporary_data/adjacency_matrix_eu.pickle', 'wb') as f:
    pickle.dump(csc_matrix_eu, f)

with open('../pickle_temporary_data/attr_matrix_eu.pickle', 'wb') as f:
    pickle.dump(lil_attr_matrix_eu, f)

nx.write_edgelist(eu_graph, location_edgelist)