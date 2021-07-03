import statistics
from Utils.combined_graph_union_node_attributes import create_shared_node_attributes
def create_encoded_dictionary(metric_dictionary, metricname, nodes):
    top_10_percent = len(nodes)/10
    ranked_dictionary = dict(sorted(metric_dictionary.items(), key=lambda item: item[1], reverse=True)[:int(top_10_percent)])
    encoding = {}
    for node in nodes:
        if node in ranked_dictionary:
            encoding[node] = 1
        else:
            encoding[node] = 0
    return encoding

def create_encoded_dictionary_more_options(metric_dictionary, nodes):
    list_values_all = list(metric_dictionary.values())
    list_values = []
    for i in list_values_all:
        if type(i) == int or type(i) == float:
            list_values.append(i)
    sd = statistics.stdev(list_values)
    encoding =  {}
    for node in nodes:
        try:
            if metric_dictionary[node] < sd:
                if metric_dictionary[node] < (2 * sd):
                    encoding[node] = -2
                else:
                    encoding[node] = -1
            elif metric_dictionary[node] > sd:
                if metric_dictionary[node] > (2 * sd):
                    encoding[node] = 2
                else:
                    encoding[node] = 1
            else:
                encoding[node] = 0
        except:
            encoding[node] = 0

    return encoding

def create_combined_attribute_dictionary(nodes, dictionaries):
    combined_attribute_dicionary = {}
    for node in nodes:
        node_attribute_dictionary = {}
        attr_index = 0
        for mapping in dictionaries:
            if node not in mapping:
                mapping[node] = 0
            node_attribute_dictionary[attr_index] = mapping[node]
            attr_index += 1

        combined_attribute_dicionary[node] = node_attribute_dictionary
    return combined_attribute_dicionary


def create_combined_attributes_dict(nodes, dictionaries):
    print("yes")
    combined_attribute_dictionary = {}
    for node in nodes:
        node_attribute_list = []
        for mapping in dictionaries:
            if node not in mapping:
                mapping[node] = 0
            node_attribute_list.append(mapping[node])
        combined_attribute_dictionary[node] = node_attribute_list
    return combined_attribute_dictionary

import networkx as nx
import pandas as pd
import scipy as sp
import numpy as np
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


import numpy as np
import networkx as nx
from scipy.sparse import csc_matrix, lil_matrix, coo_matrix
from Utils.Input_creation import create_label_array


class ACDNE_input_creator(object):

    def __init__(self, graph):
        self.graph = graph

    def create_adjacency_matrix(self):
        mat = nx.to_numpy_array(self.graph)
        csc_matrix_enron = csc_matrix(mat)
        return csc_matrix_enron

    def create_node_attributes(self, mappings):
        nodes = self.graph.nodes()
        combined_attribute_dictionary = create_combined_attribute_dictionary(nodes, mappings)
        nx.set_node_attributes(self.graph, combined_attribute_dictionary, "attributes")
        attr_matrix = nx.attr_matrix(self.graph)
        lil_attr_matrix_marvel = lil_matrix(attr_matrix[0], shape=(len(attr_matrix), 1))
        return lil_attr_matrix_marvel

    def create_node_labels(self, dictionary):
        nodes = self.graph.nodes()
        label_array = create_label_array(nodes, dictionary)
        return label_array

    def create_shared_node_attributes(self, name1, name2):
        create_shared_node_attributes(name1, name2)


def create_messages_sent_received_ratio(nodes, data):
    feature_dict = {}
    for node in nodes:
        data_filtered = data.loc[data['From'] == node]
        #print(data_filtered)
        number_of_messages_sent = len(data_filtered)
        data_filtered2 = data.loc[data['To'] == node]
        number_of_messages_received = len(data_filtered2)
        if number_of_messages_received is not 0 and number_of_messages_sent is not 0:
            ratio = number_of_messages_sent/number_of_messages_received
        feature_dict[node] = ratio

    return feature_dict

def calculate_number_of_shortest_paths(nodes):
    number_of_shortest_paths = {}
    for node in nodes:
        number_of_shortest_paths[node] = len(shortest_paths[node])
    return number_of_shortest_paths

def perform_shortest_paths(graph):
    #create topological proximity matrix
    numpy_matrix = nx.floyd_warshall_numpy(graph)
    proximity_matrix_enron2 = csc_matrix(numpy_matrix)
    return proximity_matrix_enron2

import seaborn as sb

import matplotlib.pyplot as plt
def plot_metrics_dictionary(metric_dictionary):
    '''Plot the dictionary distribution'''
    x = list(metric_dictionary.values())
    sb.displot(x=x)
    plt.show()

from karateclub import DeepWalk
def perform_deepwalk(graph):
    ''r'Creating deepwalk embeddings'''
    #labels are now being used as index. They should be numeric
    numeric_indices = [index for index in range(graph.number_of_nodes())]
    node_indices = sorted([node for node in graph.nodes()])
    label_dictionary_numeric = zip(node_indices,numeric_indices)
    label_dictionary_numeric = dict(label_dictionary_numeric)


    graph_numeric_labels = nx.relabel_nodes(graph, label_dictionary_numeric)#create deepwalk embedding
    numeric_indices = [index for index in range(graph_numeric_labels.number_of_nodes())]
    node_indices = sorted([node for node in graph_numeric_labels.nodes()])
    model = DeepWalk(dimensions=len(node_indices))
    model.fit(graph_numeric_labels)

    embedding = model.get_embedding()
    proximity_matrix_enron = csc_matrix(embedding)
    return proximity_matrix_enron

def create_average_response_time_dictionary(data, nodes):
    average_response_time_dict = {}


    for node in nodes:
        count = 0
        data_to = data[data['To'] == node]
        response_time = 0
        if len(data_to) > 100:
            data_from = data_to.head(100)
        for row_to in data_to.iterrows():
            date_sent = row_to[1][0]
            data_from = data[data['From'] == node]
            data_from = data_from[data_from['Date'] > date_sent]
            if len(data_from) > 100:
                data_from = data_from.head(100)
            for row_from in data_from.iterrows():
                date_response = row_from[1][0]
                delta = date_sent - date_response
                # I assume here that one can not respond in less than 10 second to a mail sent
                if delta.days < 10 and delta.seconds > 10:
                    response_time += delta.seconds // 3600
                    count += 1
            # threshold for messages received set to 10
            if count > 10:
                average_response_time = response_time / count
            else:
                average_response_time = "Nan"

        average_response_time_dict[node] = average_response_time
    return average_response_time_dict

# Create node features.
def create_average_response_time_dictionary(data, nodes):
    average_response_time_dict = {}
    for node in nodes:
        print(node)
        count = 0
        data_to = data[data['To'] == node]
        response_time = 0
        if len(data_to) > 100:
            data_to = data_to.head(100)
        for row_to in data_to.iterrows():
            date_sent = row_to[1]
            print("row",date_sent)
            date_sent = date_sent['Date']
            print("date_sent", date_sent)

            print("date_sent", date_sent)
            data_from = data[data['From'] == node]
            data_from = data_from[data_from['Date'] > date_sent]
            if len(data_from) > 100:
                data_from = data_from.head(100)
            for row_from in data_from.iterrows():
                date_response = row_from[1]['Date']

                print("date_response",date_response)
                delta = date_sent - date_response
                # I assume here that one can not respond in less than 10 second to a mail sent
                if delta.days < 10 and delta.seconds > 10:
                    response_time += delta.seconds // 3600
                    count += 1
            # threshold for messages received set to 10
            if count > 10:
                average_response_time = response_time / count
            else:
                average_response_time = "Nan"

        average_response_time_dict[node] = average_response_time
    return average_response_time_dict

def create_average_responding_time_dictionary(data, nodes):
    average_responding_time_dict = {}
    count_nodes = 0
    for node in nodes:
        count = 0
        data_from = data[data['From'] == node]
        response_time = 0
        if len(data_from) > 100:
            data_from = data_from.head(100)
        for row_from in data_from.iterrows():
            date_sent = row_from[1][0]
            if type(date_sent) == int:
                date_sent = row_from[1][1]
            data_to = data[data['To'] == node]
            data_to = data_to[data_to['Date'] > date_sent]
            if len(data_to) > 100:
                data_to = data_to.head(100)
            for row_to in data_to.iterrows():
                date_response = row_to[1][0]
                if type(date_response) == int:
                    date_response = row_to[1][1]
                delta = date_sent - date_response
                # I assume here that one can not respond in less than 10 second to a mail sent and that the response can only be a response if the mail is sent within 10 days.
                if delta.days < 10 and delta.seconds > 10:
                    response_time += delta.seconds//3600
                    count +=1
            # threshold for messages received set to 10
            if count > 10:
                average_response_time = response_time/count
            else:
                average_response_time = "Nan"
            average_responding_time_dict[node] = average_response_time
        count_nodes += 1
        print(len(nodes), " / ", count_nodes)
    return average_responding_time_dict

def create_length_activity(data, nodes):
    activity_length_dictionary = {}
    for node in nodes:
        data_node = data[data['From'] == node]
        first_message = data_node.Date.min()
        final_message = data_node.Date.max()
        delta = final_message - first_message
        activity_length_dictionary[node] = delta.days
    for node in nodes:
        if node not in activity_length_dictionary:
            activity_length_dictionary[node] = "Nan"
    return activity_length_dictionary


def create_length_activity(data, nodes):
    activity_length_dictionary = {}
    for node in nodes:
        data_node = data[data['From'] == node]
        first_message = data_node.Date.min()
        final_message = data_node.Date.max()
        delta = final_message - first_message
        activity_length_dictionary[node] = delta.days
    for node in nodes:
        if node not in activity_length_dictionary:
            activity_length_dictionary[node] = "Nan"
    return activity_length_dictionary


def create_average_messages_per_day_dictionary(length_activity, data, nodes):
    average_message_per_day = {}
    average_message_per_hour = {}
    for node in nodes:
        activity = length_activity[node]
        if activity == 0:
            activity  =1
        #convert to days if needed
        data_node = data[data['From']==node]
        average_message_per_day[node] = len(data_node)/activity
        average_message_per_hour[node] = len(data_node)/(activity*24)
    return average_message_per_day, average_message_per_hour

def create_daily_usage_distribution_dictionary(data, nodes):

    def most_frequent(list):
        return max(set(list), key = list.count)

    daily_hours_of_usage = {}
    most_frequent_usage = {}
    for node in nodes:
        hours_of_day = []
        data_node = data[data['From'] == node]
        data_grouped = data_node.set_index('Date').groupby(pd.Grouper(freq='D'))

        for day in data_grouped:
            day_df =  day[1]
            data_node = day_df[day_df['From']==node]
            data_node['time_hour'] = data_node.index.hour
            for hour in data_node['time_hour']:
                hours_of_day.append(hour)
                print(hours_of_day)
        if len(hours_of_day) != 0:
            most_frequent_hour = most_frequent(hours_of_day)
            most_frequent_usage[node] = most_frequent_hour
            set_hours = set(hours_of_day)
            usage = len(set_hours)/24
        else:
            print("some lists were empty")
            print(node)
            most_frequent_usage[node] = 12

        daily_hours_of_usage[node] = usage
    return daily_hours_of_usage, most_frequent_usage
