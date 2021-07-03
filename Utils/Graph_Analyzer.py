import networkx as nx
import pandas as pd
import scipy as sp
from scipy.sparse import csc_matrix, lil_matrix, coo_matrix


import numpy as np
from numpy import asarray
from numpy import save

from scipy.io import savemat
import networkx as nx

from scipy.sparse import csc_matrix, lil_matrix
global csc_matrix_enron, lil_attr_matrix, nodes_enron, graph, nodes_enron

class Graph_analyzer(object):

    def __init__(self, graph, centrality_scores_df, centrality_threshold):
        self.graph = graph
        self.threshold = centrality_threshold
        self.centrality_scores = centrality_scores_df

    def filter_graph(self):
        '''Filters the graph based on threshold values as predefined by user.'''
        #todo adapt to make it valueable for all measures.
        #get unique nodes using the threshold
        #self.centrality_scores = self.centrality_scores[self.centrality_scores > self.threshold]
        #print(self.centrality_scores.columns)
        self.centrality_scores_filtered = self.centrality_scores.loc[self.centrality_scores['Degree_Centrality'] > self.threshold]
        self.node_list = self.centrality_scores_filtered['User'].tolist()
        self.subgraph = self.graph.subgraph(self.node_list)
        return self.subgraph

    def get_largest_component(self):
        network_largest_component = max(nx.connected_components(self.graph), key=len)
        self.graph = self.graph.subgraph(network_largest_component)
        return self.graph

    def find_highest_centrality_degree(self, top_n):
        self.centrality_scores_highest = self.centrality_scores.nlargest(top_n, 'Degree_Centrality')
        return self.centrality_scores_highest['User'].tolist()

    def create_adjacency_matrix(self,filtered):
        if filtered == True:
            self.graph = self.filter_graph()
            #return nx.adjacency_matrix(self.graph)
            return nx.to_numpy_array(self.graph)
        elif filtered == False:
            #return nx.adjacency_matrix(self.graph)
            return nx.to_numpy_array(self.graph)

    def get_page_rank(self,top_n):
        ordered_dictionary = nx.pagerank(self.graph)
        sort_orders = sorted(ordered_dictionary.items(), key=lambda x: x[1], reverse=True)
        return list(sort_orders)[:top_n]

    def get_full_page_rank(self):
        return nx.pagerank(self.graph)

    def get_betweenness_rank(self, top_n):
        scored_dictionary = nx.current_flow_betweenness_centrality(self.graph)
        sort_orders = sorted(scored_dictionary.items(), key=lambda x: x[1], reverse=True)
        return list(sort_orders)[:top_n]

    def summarize_graph(self):
        self.general_info = nx.info(self.graph)
        print("General information: ", self.general_info)
        self.density = nx.density(self.graph)
        print("Network density:", self.density)
        self.connected = (nx.is_connected(self.graph))
        print("Graph is connected: ", self.connected)
        if self.connected == False:
            self.k_components = nx.k_components(self.graph)
            print("Number of components (approximated): ", self.k_components)




class Embedding_creator(object):

    def __init__(self, graph, embeddings_method):
        self.graph = graph
        self.embeddings_method = embeddings_method

    def create_embeddings(self):
        self.numeric_indices = [index for index in range(self.graph.number_of_nodes())]
        self.node_indices = sorted([node for node in self.graph.nodes()])
        # assert numeric_indices == node_indices
        self.label_dictionary_numeric = zip(self.node_indices, self.numeric_indices)
        self.label_dictionary_numeric = dict(self.label_dictionary_numeric)
        self.graph_numeric_labels = nx.relabel_nodes(self.graph, self.label_dictionary_numeric)
        self.model = self.embeddings_method
        self.model.fit(self.graph_numeric_labels)
        self.embedding = self.model.get_embedding()
        return self.embedding, self.label_dictionary_numeric

    def create_embeddings_ASNE(self, attr_matrix):
        self.numeric_indices = [index for index in range(self.graph.number_of_nodes())]
        self.node_indices = sorted([node for node in self.graph.nodes()])
        # assert numeric_indices == node_indices
        self.label_dictionary_numeric = zip(self.node_indices, self.numeric_indices)
        self.label_dictionary_numeric = dict(self.label_dictionary_numeric)
        self.graph_numeric_labels = nx.relabel_nodes(self.graph, self.label_dictionary_numeric)
        self.model = self.embeddings_method
        print(len(self.graph.nodes()))
        coo_attr_matrix_enron = coo_matrix(attr_matrix[0])
        print(coo_attr_matrix_enron.shape[0])
        self.model.fit(self.graph_numeric_labels, coo_attr_matrix_enron)
        self.embedding = self.model.get_embedding()
        return self.embedding, self.label_dictionary_numeric

    def write_embedding(self, location):
        self.embedding = asarray(self.embedding)
        save(location,self.embedding)


    def create_embeddings_dictionary(self):
        self.embeddings_dictionary = {}
        for emb in range(len(self.embedding)):
            for node_name, numeric_value in  self.label_dictionary_numeric.items():
                if emb == numeric_value:
                    self.embeddings_dictionary[node_name] = self.embedding[emb]
        return self.embeddings_dictionary

    def create_embeddings_dictionary_nodes(self, nodes):
        self.embeddings_dictionary = {}
        for i in range(1, self.embedding.shape[0]):
            self.embeddings_dictionary[nodes[i]] = self.embedding[i]
        return self.embeddings_dictionary

