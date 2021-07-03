import csv
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('../Data/node_list_eu.csv', 'r') as nodecsv: # Open the file
    nodereader = csv.reader(nodecsv) # Read the csv
    # Retrieve the data (using Python list comprhension and list slicing to remove the header row, see footnote 3)
    nodes = [n for n in nodereader][1:]
node_names = [n[0] for n in nodes] # Get a list of only the node names

with open('../Data/edge_list_eu.csv', 'r') as edgecsv: # Open the file
    edgereader = csv.reader(edgecsv) # Read the csv
    edges = [tuple(e) for e in edgereader][1:] # Retrieve the data

G = nx.Graph()
G.add_nodes_from(node_names)
G.add_edges_from(edges)
nodes_eu  = G.nodes()
nx.write_gexf(G, "../Data/Complete_graph_eu.gexf")

with open('../pickle_temporary_data/nodes_eu.pickle', 'wb') as f:
    pickle.dump(nodes_eu, f)

