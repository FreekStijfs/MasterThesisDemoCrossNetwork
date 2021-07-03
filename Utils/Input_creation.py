import pandas as pd
import numpy as np
def create_data_set(node_list,data, only_direct_messages = True):
    '''This function creates the edgelist csv. Depending on the only_directed_messages, the massive forwarded mails are taken into account.'''
    #Todo : think about a way of given more wight to direct mail contacts compared to mailing lists.
    df =  pd.DataFrame()
    unique_node_list_enron = []
    sender_list = []
    receiver_list = []
    count = 0
    for node in node_list:
        contacted_list = []
        node_data = data.loc[data['From'] == node]
        for i in range(len(node_data)):
            entry = data.iloc[i]
            if type(entry['To']) is not str:
                if type(entry['To']) is not float:
                    print("there is a problem", entry['To'])
                    print(type(entry['To']))
                continue
            elif ',' in entry['To'] and only_direct_messages == False:
                splitted_nodes = entry['To'].split(", ")
                for unique_node in splitted_nodes:
                    unique_node = unique_node.replace("\'","")
                    if "@enron.com" in unique_node:
                        contacted_list.append(unique_node)
                        unique_node_list_enron.append(unique_node)
                    else:
                        continue
            elif ',' not in entry['To']:
                if ',' not in entry['From'] and "@enron.com" in entry['From']:
                    sender = entry['From']
                    unique_node_list_enron.append(sender)

                    receiver = entry['To']
                    if "@enron.com" in entry['To']:
                        contacted_list.append(receiver)
                    unique_contacted_set = set(contacted_list)
                    node_column_list = [sender] * len(unique_contacted_set)
                    sender_list.extend(node_column_list)
                    receiver_list.extend(unique_contacted_set)
                else:
                    continue
        count += 1
    df['From'] = sender_list
    df['To'] = receiver_list
    df.drop_duplicates()
    return df, unique_node_list_enron

def retrieve_unique_nodes(data):
    column_date = data.From
    column_date = column_date.values.tolist()
    for i in column_date:
        if ',' in i:
            list_of_splitted_words = i.split("', '")
            for user in list_of_splitted_words:
                column_date.append(user)
        else:
            continue
    return column_date


def create_label_array(nodes, label_dictionary):
    ''r'Create numpy array, that will be used as labels. Please be aware that the order is important here.'''
    node_label_array = []
    count_key_actors_encoded = 0
    for node in nodes:
        if node in label_dictionary:
            if label_dictionary[node] == 'Key Actor':
                node_label_array.append([0, 0, 1])
                count_key_actors_encoded += 1
            else:
                node_label_array.append([0, 1, 0])
        else:
            node_label_array.append([0, 1, 0])
    node_label_array = np.array(node_label_array)
    return node_label_array