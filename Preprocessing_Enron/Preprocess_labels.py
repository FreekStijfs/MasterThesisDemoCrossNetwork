import pandas as pd
import difflib
from difflib import get_close_matches
import numpy as np
import networkx as nx
global label_dictionary
global one_hot_encoded_dictionary
global one_hot_encoded_array
global node_label_array


print("Preprocessing labels")
labels_df = pd.read_csv('../Data/labels.txt', delimiter="\t", names=['full'])
labels_df = pd.DataFrame(labels_df.full.str.split('</td><td>').tolist(),
                                 columns = ['id','user','name',"role","department"])
print(len(labels_df))
spec_chars = ["<tr>","<td>","</td>","</tr>"]
for char in spec_chars:
    for column in labels_df.columns:
        labels_df[column] = labels_df[column].str.replace(char, '')

#merge with the enrondata set to create attribute csv
data = pd.read_csv("../Data/enron.csv", low_memory=False)
data = data[['From','user']]
data= data.drop_duplicates()
unique_users = pd.read_csv("../Data/node_list.csv", header=0)
#graph = nx.read_gexf("../Data/Complete_graph.gexf")
#unique_users = graph.nodes()

def create_email(name):
    name = name.replace(' ','.')
    name =  name + "@enron.com"
    name = name.replace(" ",'')
    return name


def create_label_dictionary(labels_df,detailed):
    name_df = labels_df[['name']]
    email_list = []
    label_dictionary = {}
    count_key_actors = 0
    for name in name_df['name']:
        role = labels_df.loc[labels_df['name'] == name, 'role'].iloc[0]
        if detailed == True:
            name = name.lower()
            email = create_email(name)
            email_list.append(email)
            label_dictionary[email] = role
        elif detailed == False:
            if role in ['Vice President', 'Director', 'Manager', r'CEO']:
                role = 'Key Actor'
                count_key_actors +=1
            else:
                role = 'N/A'
            name = name.lower()
            email = create_email(name)
            email_list.append(email)
            label_dictionary[email] = role
    return label_dictionary, count_key_actors

label_dictionary, count_key_actors = create_label_dictionary(labels_df, False)


#make sure it is one-hot encoded
node_label_array = []
count_key_actors_encoded = 0
unique_users = set(unique_users)
for node in unique_users:
    print(node)
    if node in label_dictionary:
        if label_dictionary[node] == 'Key Actor':
            node_label_array.append([0, 0, 1])
            count_key_actors_encoded += 1
        else:
            node_label_array.append([0,1,0])

    else:
        node_label_array.append([0, 1, 0])

node_label_array =  np.array(node_label_array)

