import csv
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from Preprocessing_Enron.Preprocess_labels import label_dictionary
global labels_array_enron

with open('../Data/node_list.csv', 'r') as nodecsv: # Open the file
    nodereader = csv.reader(nodecsv) # Read the csv
    # Retrieve the data (using Python list comprhension and list slicing to remove the header row, see footnote 3)
    nodes = [n for n in nodereader][1:]
node_names = [n[0] for n in nodes] # Get a list of only the node names

with open('../Data/edge_list.csv', 'r') as edgecsv: # Open the file
    edgereader = csv.reader(edgecsv) # Read the csv
    edges = [tuple(e) for e in edgereader][1:] # Retrieve the data

G = nx.Graph()
G.add_nodes_from(node_names)
G.add_edges_from(edges)

unique_nodes = G.nodes

#relabeling actualy changes the name of the node, it is not just adding the label. (good to note)
#G = nx.relabel_nodes(G, label_dictionary)

components = nx.connected_components(G)

largest_component = max(components, key=len)
#create a subgraph consisting of the largest component
subgraph = G.subgraph(largest_component)

# for plotting the subgraph
def plot_graph(subgraph):
    pos = nx.spring_layout(subgraph)
    nx.draw_networkx(subgraph, pos,with_labels=False, width=0.01)
    plt.show()


nx.write_gexf(G, "../Data/Complete_graph.gexf")
nx.write_gexf(subgraph, "../Data/Subgraph.gexf")

''r'Create labels'''
'''Ensuring that the labels are created in the same order as the nodes are in'''
new_label_dictionary = {}
for node in subgraph.nodes:
    if node in label_dictionary:
        print("yes")
        new_label_dictionary[node] = label_dictionary[node]
    else:
        new_label_dictionary[node] = 'N/A'


L = [new_label_dictionary]
u_value = set( val for new_label_dictionary in L for val in new_label_dictionary.values())
label_dictionary_sorted = sorted(new_label_dictionary.values())

label_mapping = {}
count = 1
for value in u_value:
    label_mapping[value] = count
    count += 1

one_hot_encoded_dictionary ={}
new_dictionary = {}
for key, value in new_label_dictionary.items():
    encoding = np.zeros((11,))
    encoding[label_mapping[value]] = 1
    #encoding = list(encoding)
    one_hot_encoded_dictionary[key] = encoding
    new_dictionary[key] = label_mapping[value]

#print(one_hot_encoded_dictionary)

one_hot_encoded_array = list(one_hot_encoded_dictionary.values())
one_hot_encoded_array = np.array(one_hot_encoded_array)

idx = sorted(new_dictionary.values())
# Initialise a matrix of zeros.
label_array = np.zeros((len(idx), max(idx) + 1))
# Assign 1 to appropriate indices.
label_array[np.arange(len(label_array)), idx] = 1
labels_array_enron = label_array
with open("../pickle_temporary_data/labels_array_enron.pickle", 'wb') as f:
    pickle.dump(labels_array_enron, f)
