import pandas as pd


input = pd.read_csv('../Data/email-Eu-core.txt', delimiter=" ", names= ['From','To'])

#input_from = input.From
#input_from_list = input_from.values.tolist()
node_list = input.From.unique()
df2 = pd.DataFrame()
df2["nodes"] =node_list
df2.to_csv("../Data/node_list_eu.csv", index=False)

def create_data_set(dataframe):
    '''This function creates the edgelist csv. Depending on the only_directed_messages, the massive forwarded mails are taken into account.'''
    #Todo : think about a way of given more wight to direct mail contacts compared to mailing lists.
    df =  pd.DataFrame()
    sender_list = []
    receiver_list = []
    for i in range(len(dataframe)):
        entry = dataframe.iloc[i]
        sender = entry[0]
        sender_list.append(sender)
        receiver = entry[1]
        receiver_list.append(receiver)
    df['From'] = sender_list
    df['To'] = receiver_list
    df.drop_duplicates()
    return df

edge_data = create_data_set(input)
edge_data.to_csv('../Data/edge_list_eu.csv',index = False)
