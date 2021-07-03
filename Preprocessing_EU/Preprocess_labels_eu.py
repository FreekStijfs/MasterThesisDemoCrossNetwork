import networkx as nx
import numpy as np
#from src.Marvel_refactoring.Create_input_ACDNE_marvel import num_nodes
global labels_array_eu
eu_graph = nx.read_gexf("../Data/Complete_graph_eu.gexf")
num_nodes = len(eu_graph.nodes())
dictionary_pagerank_eu = nx.pagerank(eu_graph)
dict_top_ten_percent = dict(sorted(dictionary_pagerank_eu.items(), key=lambda item: item[1])[:(int(len(dictionary_pagerank_eu) / 10))])
dict_rest = dict(sorted(dictionary_pagerank_eu.items(), key=lambda item: item[1])[int((len(dictionary_pagerank_eu)) / 10):num_nodes])


def create_dictionary(dict_top_ten_percent, dic_rest):
    label_dictionary = {}
    for key, value in dict_top_ten_percent.items():
        label_dictionary[key] = 'Key Actor'
    for key, value in dict_rest.items():
        label_dictionary[key] = 'N/A'
    return label_dictionary


label_dictionary = create_dictionary(dict_top_ten_percent, dict_rest)
#make sure it is one-hot encoded
L = [label_dictionary]
# get the uniqe values in set form
u_value = set( val for label_dictionary in L for val in label_dictionary.values())
label_dictionary_sorted = sorted(label_dictionary.values())

label_mapping = {}
count = 1
for value in u_value:
    label_mapping[value] = count
    count += 1

one_hot_encoded_dictionary ={}
new_dictionary = {}
for key, value in label_dictionary.items():
    encoding = np.zeros((len(u_value)+1,))
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
labels_array_eu = label_array