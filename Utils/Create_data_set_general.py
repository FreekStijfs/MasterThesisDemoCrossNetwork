import pandas as pd
import csv
import numpy as np
from scipy.sparse import csc_matrix
import networkx as nx
import pickle
from Utils.Input_creation import retrieve_unique_nodes, create_data_set

from Utils.utils import ACDNE_input_creator, create_combined_attribute_dictionary, Graph_analyzer, create_encoded_dictionary,create_encoded_dictionary_more_options, create_messages_sent_received_ratio
from Utils.combined_graph_union_node_attributes import create_shared_node_attributes

#load in the data
data = pd.read_csv("../../Data/enron.csv")
def clean_data(data):
    data2 = data[['From','To','user']]
    data = data[['From','To']]
    data['From'] = data['From'].str[12:]
    data['From'] = data['From'].str[:-3]
    data['To'] = data['To'].str[12:]
    data['To'] = data['To'].str[:-3]
    return data

data = clean_data(data)
#data['From'] = data['From'].map(lambda x: x.lstrip('"'))
sender_data = retrieve_unique_nodes(data)
#node lists are all the unique
data_filtered = data[data['From'].apply(lambda x: len(x.split(', ')) < 2)]
node_list = data_filtered.From.unique()

df2 = pd.DataFrame()
df2["nodes"] =node_list
df2.to_csv("../../Data/node_list1.csv", index=False)
#print(node_list)
myset = set(sender_data)

unique_node_list = list(myset)
# unique node list is the same as the node_list defined earlier

edge_data, unique_node_list_enron = create_data_set(unique_node_list,data, only_direct_messages=False)
edge_data.to_csv("../../Data/edge_list1.csv",index = False)
df2 = pd.DataFrame()
df2["nodes"] = unique_node_list_enron
df2 = df2.drop_duplicates()
df2.to_csv("../../Data/node_list_enron1.csv", index=False)


with open("../../Data/node_list_enron1.csv", 'r') as nodecsv: # Open the file
    nodereader = csv.reader(nodecsv) # Read the csv
    # Retrieve the data (using Python list comprhension and list slicing to remove the header row, see footnote 3)
    nodes = [n for n in nodereader][1:]
node_names = [n[0] for n in nodes] # Get a list of only the node names

with open("../../Data/edge_list1.csv", 'r') as edgecsv: # Open the file
    edgereader = csv.reader(edgecsv) # Read the csv
    edges = [tuple(e) for e in edgereader][1:] # Retrieve the data

G = nx.Graph()
G.add_nodes_from(node_names)
G.add_edges_from(edges)
unique_nodes = G.nodes
nx.write_gexf(G,"../../Data/Complete_graph.gexf")
nx.write_gpickle(G, "../../pickle_temporary_data/graph_complete_enron1.gpickle")

'''------------------------------------------------------------------------------------------------------------------------------'''
''r'Create labels'''

message_received_ratio = create_messages_sent_received_ratio(unique_nodes, data)

from Preprocessing_Enron.Preprocess_labels import label_dictionary

def create_label_array(nodes, label_dictionary):
    ''r'Create numpy array that will be used as labels. Please be aware that the order is important here.'''
    node_label_array = []
    count_key_actors_encoded = 0
    for node in nodes:
        if node in label_dictionary:
            if label_dictionary[node] == 'Key Actor':
                node_label_array.append([0, 1])
                count_key_actors_encoded += 1
            else:
                node_label_array.append([1, 0])
        else:
            node_label_array.append([1, 0])
    node_label_array = np.array(node_label_array)
    return node_label_array

node_label_array = create_label_array(unique_nodes, label_dictionary)

'''------------------------------------------------------------------------------------------------------------------------------'''
''r'Create attributes'''
dictionary_pagerank = nx.pagerank(G)
dictionary_eigenvector = nx.eigenvector_centrality(G)
dictionary_degree_centrality = nx.degree_centrality(G)
dictionary_in_degree_centrality = nx.degree_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
#nx.set_node_attributes(graph, dictionary_pagerank, "pagerank")
#create number of nodes dictionary
contacts_dictionary = {}
for node in G.nodes():
    contacts_dictionary[node] = len(G.edges(node))

dictionary_pagerank = create_encoded_dictionary_more_options(dictionary_pagerank,  unique_nodes)
dictionary_eigenvector = create_encoded_dictionary_more_options(dictionary_eigenvector, unique_nodes)
dictionary_degree_centrality = create_encoded_dictionary_more_options(dictionary_degree_centrality,unique_nodes)
dictionary_in_degree_centrality = create_encoded_dictionary_more_options(dictionary_in_degree_centrality, unique_nodes)
closeness_centrality = create_encoded_dictionary_more_options(closeness_centrality, unique_nodes)

mappings_to_be_added = [dictionary_eigenvector, dictionary_pagerank, contacts_dictionary,dictionary_degree_centrality, dictionary_in_degree_centrality,closeness_centrality]
combined_attribute_dictionary = create_combined_attribute_dictionary(unique_nodes,mappings_to_be_added)
nx.set_node_attributes(G, combined_attribute_dictionary, "eigenvector")
Graph = Graph_analyzer(G,dictionary_degree_centrality, 0.02)
mat= Graph.create_adjacency_matrix(False)

#Graph.summarize_graph()
csc_matrix_enron = csc_matrix(mat)

def create_shared_node_attribute_matrices(name_graph1, name_graph2):
    create_shared_node_attributes(name_graph1, name_graph2)
#create_shared_node_attribute_matrices('enron', 'marvel')

with open('../../pickle_temporary_data/adjacency_matrix_enron1.pickle', 'wb') as f:
    pickle.dump(csc_matrix_enron, f)

with open('../../pickle_temporary_data/nodes_array_labels1.pickle', 'wb') as f:
    pickle.dump(node_label_array, f)

nx.write_gpickle(G, '../../pickle_temporary_data/enron_graph.gpickle')