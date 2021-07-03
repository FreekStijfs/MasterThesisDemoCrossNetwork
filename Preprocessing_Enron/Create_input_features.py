import pandas as pd
import pickle
from Utils.Input_creation import retrieve_unique_nodes, create_data_set
from Utils.utils import create_encoded_dictionary_more_options, ACDNE_input_creator, create_average_response_time_dictionary, create_average_responding_time_dictionary, create_combined_attributes_dict,create_messages_sent_received_ratio,create_combined_attribute_dictionary, create_combined_attributes_dict, create_length_activity, create_average_messages_per_day_dictionary,create_daily_usage_distribution_dictionary
import networkx as nx
import csv

location_edgelist = "../Data/enron_edgelist.edgelist"
data = pd.read_csv("../Data/enron.csv")

data = data[['Date','From','To']]
data['From'] = data['From'].str[12:]
data['From'] = data['From'].str[:-3]
data['To'] = data['To'].str[12:]
data['To'] = data['To'].str[:-3]
data['Date']= pd.to_datetime(data['Date'])

graph = nx.read_gexf("../Data/Complete_graph.gexf")
network_largest_component = max(nx.connected_components(graph), key=len)
graph = graph.subgraph(network_largest_component)
node_list = graph.nodes()
nx.write_gexf(graph, "../Data/Trimmed_Enron.gexf")

'''Creating node attributes.'''
def create_attributes():
    length_activity = create_length_activity(data, node_list)
    with open('length_activity_dictionary_enron.pickle', 'wb') as handle:
        pickle.dump(length_activity, handle)
    with open('length_activity_dictionary_enron.pickle', 'rb') as f:
        length_activity = pickle.load(f)
    average_messages_per_day, average_messages_per_hour = create_average_messages_per_day_dictionary(length_activity, data, node_list)
    with open('average_messages_per_day_enron.pickle', 'wb') as handle:
         pickle.dump(average_messages_per_day,handle)
    message_received_ratio = create_messages_sent_received_ratio(node_list, data)
    with open('message_received_ratio_enron.pickle', 'wb') as handle:
        pickle.dump(message_received_ratio, handle)
    average_response_time = create_average_response_time_dictionary(data, node_list)
    with open('average_response_time_enron.pickle', 'wb') as handle:
        pickle.dump(average_response_time, handle)
    average_responding_time = create_average_responding_time_dictionary(data, node_list)
    with open('average_responding_time_enron.pickle', 'wb') as handle:
        pickle.dump(average_responding_time, handle)
    number_of_contacts = {}
    for node in graph.nodes():
        number_of_contacts[node] = len(graph.edges(node))
    with open('number_of_contacts_enron.pickle', 'wb') as handle:
        pickle.dump(number_of_contacts, handle)

    daily_hours_of_usage, most_frequent_usage_hours = create_daily_usage_distribution_dictionary(data, node_list)
    with open('daily_hours_of_usage_enron.pickle', 'wb') as handle:
        pickle.dump(daily_hours_of_usage, handle)
    with open('most_frequent_usage_hours_enron.pickle', 'wb') as handle:
        pickle.dump(most_frequent_usage_hours, handle)

#create_attributes()

with open('length_activity_dictionary_enron.pickle', 'rb') as f:
    length_activity = pickle.load(f)
print(length_activity)
with open('average_messages_per_day_enron.pickle', 'rb') as f:
    average_messages_per_day = pickle.load(f)
with open('message_received_ratio_enron.pickle', 'rb') as f:
    message_received_ratio = pickle.load(f)
with open('average_response_time_enron.pickle', 'rb') as f:
    average_response_time = pickle.load(f)
with open('average_responding_time_enron.pickle', 'rb') as f:
    average_responding_time = pickle.load(f)
with open('number_of_contacts_enron.pickle', 'rb') as f:
    number_of_contacts = pickle.load(f)
with open('daily_hours_of_usage_enron.pickle', 'rb') as f:
    daily_hours_of_usage = pickle.load(f)
with open('most_frequent_usage_hours_enron.pickle', 'rb') as f:
    most_frequent_usage_hours = pickle.load(f)

dictionary_eigenvector = nx.eigenvector_centrality(graph)
mappings_to_be_added_ = [length_activity,average_messages_per_day,message_received_ratio, average_responding_time, average_response_time, number_of_contacts, daily_hours_of_usage, most_frequent_usage_hours, dictionary_eigenvector]

mappings_to_be_added = []
'''Create the encoded dictionaries'''
for dict in mappings_to_be_added_:
    dict_to_add = create_encoded_dictionary_more_options(dict,node_list)
    mappings_to_be_added.append(dict_to_add)

combined_attribute_dictionary_list = create_combined_attributes_dict(node_list,mappings_to_be_added)
with open("../pickle_temporary_data/attribute_dictionary_enron.pickle", 'wb') as f:
    pickle.dump(combined_attribute_dictionary_list, f)
combined_attribute_dictionary = create_combined_attributes_dict(node_list,mappings_to_be_added)
nx.set_node_attributes(graph, combined_attribute_dictionary, "attributes")
input_creator = ACDNE_input_creator(graph)


attr_matrix = nx.attr_matrix(graph)
with open("../pickle_temporary_data/attr_matrix_enron.pickle", 'wb') as f:
    pickle.dump(attr_matrix, f)

lil_attr_matrix = input_creator.create_node_attributes(mappings_to_be_added)


csc_matrix = input_creator.create_adjacency_matrix()
input_creator.create_shared_node_attributes('enron','eu')

with open("../pickle_temporary_data/adjacency_matrix_enroneu.pickle", 'wb') as f:
    pickle.dump(csc_matrix, f)

nx.write_edgelist(graph, location_edgelist)
