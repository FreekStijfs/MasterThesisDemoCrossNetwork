import networkx as nx
import pandas as pd
import pickle
graph = nx.read_gexf(r"C:\Users\255955\PycharmProjects\ThesisProject\Data\Complete_graph.gexf")
data = pd.read_csv(r"C:\Users\255955\PycharmProjects\ThesisProject\Data\enron_05_17_2015_with_labels_v2_100K_chunk_1_of_6.csv")

data = data[['Date','From','To']]
data['From'] = data['From'].str[12:]
data['From'] = data['From'].str[:-3]
data['To'] = data['To'].str[12:]
data['To'] = data['To'].str[:-3]
data['Date']= pd.to_datetime(data['Date'])


def filter_graph(Graph, Data, message_threshold):
    nodes_tbr = []
    count = 0
    for node in Graph.nodes():
        count += 1
        data_filtered = Data.loc[Data['From'] == node]
        if len(data_filtered) < message_threshold:
            nodes_tbr.append(node)
    return nodes_tbr

from_dictionary = data.From.value_counts().to_dict()

to_dictionary = data.To.value_counts().to_dict()
nodes_toberemoved = []
for key, value in from_dictionary.items():
    if value < 5:
        nodes_toberemoved.append(key)
    else:
        if key in to_dictionary:
            if value > (3 * to_dictionary[key]):
                nodes_toberemoved.append(key)
        else:
            nodes_toberemoved.append(key)
for key, value in to_dictionary.items():
    if value < 5:
        nodes_toberemoved.append(key)
with open('length_activity_dictionary_enron.pickle', 'rb') as f:
    length_activity = pickle.load(f)
active_users = []
for key, value in length_activity.items():
    if value > 10:
        active_users.append(key)

nodes = list(graph.nodes())
nodes_to_stay = []
for node in nodes:
    if node in active_users and node not in nodes_toberemoved:
        nodes_to_stay.append(node)

with open('number_of_contacts_enron.pickle', 'rb') as f:
    number_of_contacts = pickle.load(f)

final_nodes_to_stay  =[]
for node in nodes_to_stay:
    if number_of_contacts[node] > 5:
        final_nodes_to_stay.append(node)

nodes_to_stay =final_nodes_to_stay

with open(r"C:\Users\255955\PycharmProjects\ThesisProject\pickle_temporary_data\nodes_to_stay_enron.pickle", 'wb') as f:
    pickle.dump(nodes_to_stay, f)

filtered_graph = graph.subgraph(nodes_to_stay)

mission_graph_largest_component = max(nx.connected_components(filtered_graph), key=len)
mission_graph = filtered_graph.subgraph(mission_graph_largest_component)
nx.write_gexf(filtered_graph, r"C:\Users\255955\PycharmProjects\ThesisProject\Data\Filtered_enron.gexf")

data2 = data[data['From'].isin(nodes_to_stay)]
data2.to_csv(r"C:\Users\255955\PycharmProjects\ThesisProject\Data\data_filtered_enron.csv")

