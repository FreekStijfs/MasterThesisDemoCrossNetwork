import pandas as pd
import csv

#load in the data
data = pd.read_csv("../Data/enron.csv")


data2 = data[['From','To','user']]
data = data[['From','To']]


data['From'] = data['From'].str[12:]
data['From'] = data['From'].str[:-3]

data['To'] = data['To'].str[12:]
data['To'] = data['To'].str[:-3]

#data['From'] = data['From'].map(lambda x: x.lstrip('"'))
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

sender_data = retrieve_unique_nodes(data)

#node lists are all the unique
data_filtered = data[data['From'].apply(lambda x: len(x.split(', ')) < 2)]

node_list = data_filtered.From.unique()

df2 = pd.DataFrame()
df2["nodes"] =node_list
df2.to_csv("../Data/node_list.csv", index=False)
#print(node_list)
myset = set(sender_data)

unique_node_list = list(myset)
# unique node list is the same as the node_list defined earlier
count = 0


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
            if type(entry[1]) is not str:
                print("there is a problem", entry[1])
                continue
            elif ',' in entry[1] and only_direct_messages == False:
                print(entry[1])
                splitted_nodes = entry[1].split(", ")
                for unique_node in splitted_nodes:
                    unique_node = unique_node.replace("\'","")

                    if "@enron.com" in unique_node:
                        contacted_list.append(unique_node)
                        unique_node_list_enron.append(unique_node)
                    else:
                        continue
            elif ',' not in entry[1]:
                if ',' not in entry[0] and "@enron.com" in entry[0]:
                    sender = entry[0]
                    unique_node_list_enron.append(sender)

                    receiver = entry[1]
                    if "@enron.com" in entry[1]:
                        contacted_list.append(receiver)
                else:
                    continue
            unique_contacted_set = set(contacted_list)
            node_column_list = [sender] * len(unique_contacted_set)
            sender_list.extend(node_column_list)
            receiver_list.extend(unique_contacted_set)
        count += 1
    df['From'] = sender_list
    df['To'] = receiver_list
    df.drop_duplicates()
    return df, unique_node_list_enron

edge_data, unique_node_list_enron = create_data_set(unique_node_list,data, only_direct_messages=False)
edge_data.to_csv('../Data/edge_list.csv',index = False)
df2 = pd.DataFrame()
df2["nodes"] = unique_node_list_enron
df2 = df2.drop_duplicates()
df2.to_csv("../Data/node_list_enron.csv", index=False)