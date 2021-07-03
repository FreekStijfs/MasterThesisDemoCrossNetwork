import networkx as nx
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, lil_matrix
global lil_attr_matrix_marvel, lil_attr_matrix_enron, marvel_graph
print(r"Creating shared node attributes")

def create_shared_node_attributes(name1, name2):
    if name1 =='enron':
        graph1 = nx.read_gexf("../Data/Trimmed_Enron.gexf")
    if name2 == 'eu':
        graph2 = nx.read_gexf("../Data/Complete_graph_eu.gexf")
    else:
        print("input not correct")
    nodes_graph1 = graph1.nodes()
    nodes_graph2 = graph2.nodes()
    num_nodes = len(graph2.nodes())

    combined_graph = nx.compose(graph1, graph2)
    #get node attributes
    attr_matrix = nx.attr_matrix(combined_graph)

    attributes = nx.get_node_attributes(combined_graph,"attributes")
    mat = sp.dok_matrix((len(combined_graph.nodes), 8), dtype=np.int8)
    for key, value in attributes.items():
        mat[key, value] = 1

    mat = mat.transpose().tocsr()

    lil_attr_matrix = lil_matrix(attr_matrix[0], shape=(len(attr_matrix),1))
    shape = lil_attr_matrix.get_shape()

    lil_attr_matrix_array = lil_attr_matrix.toarray()

    lil_attr_matrix_array_graph1 = lil_attr_matrix_array[0:len(nodes_graph1)]
    lil_attr_matrix_array_graph1 = lil_matrix(lil_attr_matrix_array_graph1)

    lil_attr_matrix_array_graph2 = lil_attr_matrix_array[len(nodes_graph1):]
    lil_attr_matrix_array_graph2 = lil_matrix(lil_attr_matrix_array_graph2)
    # perform a test to verify whether the shape is correct
    assert lil_attr_matrix_array_graph2.get_shape()[0] == num_nodes and lil_attr_matrix_array_graph2.get_shape()[1] == num_nodes + len(nodes_graph1)
    with open("../pickle_temporary_data/combined_attribute_matrix_"+name1+name2+'.pickle', 'wb') as f:
        pickle.dump(lil_attr_matrix_array_graph1, f)
    with open("../pickle_temporary_data/combined_attribute_matrix_"+name2+name1+'.pickle', 'wb') as f:
        pickle.dump(lil_attr_matrix_array_graph2, f)


create_shared_node_attributes('enron', 'eu')
