import networkx as nx
import pickle
from Preprocessing_Enron.Preprocess_labels import label_dictionary
from Utils.create_similarity_dictionary import create_similarity_dictionary

#key_actors = [k for k,v in label_dictionary.items() if v == 'Key Actor']

#target_graph = nx.read_gexf(r"C:\Users\255955\PycharmProjects\ThesisProject\Data/Complete_graph_Marvel.gexf")


def evaluate_embeddings(key_actors, metric_list, embeddings_dictionary_source, embeddings_dictionary_target, target_graph, top_n_embeddings, top_n_percent, location_result_file):
    overal_score_dictionary = {}
    top_n_percent_value = (top_n_percent*0.01)
    top_percent = len(target_graph.nodes()) * top_n_percent_value
    for actor in key_actors:
        if actor in embeddings_dictionary_source:
            embedding_to_match = embeddings_dictionary_source[actor]
            similarity_dictionary = create_similarity_dictionary(embeddings_dictionary_target, "euclidean", embedding_to_match)
            #get the top 20 most similar actors
            sorted_similarity_dictionary = dict(sorted(similarity_dictionary.items(), key=lambda item: item[1], reverse=True)[:top_n_embeddings])
            for key, value in sorted_similarity_dictionary.items():
                if key in overal_score_dictionary:
                    overal_score_dictionary[key] = overal_score_dictionary[key] + value
                else:
                    overal_score_dictionary[key] = value
    file = open(location_result_file, "w")
    for metric in metric_list:
        if metric == 'pagerank':
            dictionary_metric_network = nx.pagerank(target_graph)
        elif metric == 'betweenness':
            dictionary_metric_network = nx.betweenness_centrality(target_graph)
        elif metric == 'eigenvector':
            dictionary_metric_network = nx.eigenvector_centrality(target_graph)
        else:
            print("metric ill defined")
        metric_dictionary_network = dict(
            sorted(dictionary_metric_network.items(), key=lambda item: item[1], reverse=True)[:int(top_percent)])
        counted_in_top = 0
        for key, score in overal_score_dictionary.items():
            if key in metric_dictionary_network:
                counted_in_top += 1
        score = (counted_in_top / len(overal_score_dictionary)) * 100
        result = ("Using ", top_n_embeddings, " most similar nodes for each key actor, ", score,
              " percent of the nodes where in the top ", top_n_percent, " percent according to ", metric)
        result = str(result)
        result = result.replace(',','')
        result = result.replace("'","")
        file.write(str(result)+ '\n')
    return score, sorted_similarity_dictionary


def evaluate_embeddings_average_key_actors(key_actors,metric, embeddings_dictionary_source, embeddings_dictionary_target, target_graph, top_n_embeddings = 20, top_n_percent = 10):
    overal_score_dictionary = {}
    top_n_percent_value = (top_n_percent * 0.01)
    top_percent = len(target_graph.nodes()) * top_n_percent_value
    print(r"Creating Accumulative Score Dictionary")
    #first create average embedding
    #embedding_to_match = [0] * len(key_actors)
    count_used_actors = 0
    for actor in key_actors:
        if actor in embeddings_dictionary_source:
            embedding_to_add = embeddings_dictionary_source[actor]
            try:
                embedding_to_match += embedding_to_add
            except:
                embedding_to_match = embedding_to_add
            count_used_actors +=1
    print("number of embeddings used", count_used_actors)
    embedding_to_match  = embedding_to_match / count_used_actors
    similarity_dictionary = create_similarity_dictionary(embeddings_dictionary_target, "euclidean", embedding_to_match)
    #get the top 20 most similar actors

    sorted_similarity_dictionary = dict(sorted(similarity_dictionary.items(), key=lambda item: item[1], reverse=True)[:top_n_embeddings])
    sorted_similarity_dictionary_all = dict(sorted(similarity_dictionary.items(), key=lambda item: item[1], reverse=True))
    for key, value in sorted_similarity_dictionary.items():
        if key in overal_score_dictionary:
            overal_score_dictionary[key] = overal_score_dictionary[key] + value
        else:
            overal_score_dictionary[key] = value

    print(r"Calculating PageRank and Eigenvector centrality on target network")
    top_n_percent_value = 100/top_n_percent
    top_ten_percent = len(target_graph.nodes())/top_n_percent_value
    if metric == 'pagerank':
        dictionary_pagerank_marvel = nx.pagerank(target_graph)
    elif metric == 'betweenness':
        with open('betweenness_centrality_mission.pickle',
          'rb') as f:
            dictionary_pagerank_marvel = pickle.load(f)

    elif metric == 'eigenvector':
        dictionary_pagerank_marvel = nx.eigenvector_centrality(target_graph)
    else:
        print("metric ill defined")
    dictionary_eigenvector_marvel = dict(sorted(dictionary_pagerank_marvel.items(), key=lambda item: item[1], reverse=True)[:int(top_percent)])
    print(r"Calculating Final score")
    counted_in_top = 0
    for key, score in overal_score_dictionary.items():
        if key in dictionary_eigenvector_marvel:
            counted_in_top += 1

    score = (counted_in_top / len(overal_score_dictionary)) * 100
    print("Using ", top_n_embeddings, " most similar nodes for each key actor, ", score, " percent of the nodes where in the top ", top_percent, " number of nodes ", metric)

    return score, sorted_similarity_dictionary, sorted_similarity_dictionary_all

def evaluate_embeddings_example_key_actors(key_actors, metrics_list, embeddings_dictionary_source, target_graph,  top_n_embeddings, top_n_percent, location_result_file):
    overal_score_dictionary = {}
    top_n_percent_value = (top_n_percent*0.01)
    top_percent = len(target_graph.nodes()) * top_n_percent_value
    # first create average embedding
    # embedding_to_match = [0] * len(key_actors)
    count_used_actors = 0
    for actor in key_actors:
        if actor in embeddings_dictionary_source:
            embedding_to_add = embeddings_dictionary_source[actor]
            try:
                embedding_to_match += embedding_to_add
            except:
                embedding_to_match = embedding_to_add
            count_used_actors += 1
    embedding_to_match = embedding_to_match / count_used_actors
    similarity_dictionary = create_similarity_dictionary(embeddings_dictionary_source, "euclidean", embedding_to_match)
    print(similarity_dictionary)
    sorted_similarity_dictionary = dict(
        sorted(similarity_dictionary.items(), key=lambda item: item[1], reverse=True)[:int(top_n_embeddings)])
    sorted_similarity_dictionary_all = dict(
        sorted(similarity_dictionary.items(), key=lambda item: item[1], reverse=True))
    for key, value in sorted_similarity_dictionary.items():
        if key in overal_score_dictionary:
            overal_score_dictionary[key] = overal_score_dictionary[key] + value
        else:
            overal_score_dictionary[key] = value

    print(r"Calculating PageRank and Eigenvector centrality on target network")
    file = open(location_result_file, "w")
    for metric in metrics_list:
        if metric == 'pagerank':
            dictionary_metric_network = nx.pagerank(target_graph)
        elif metric == 'betweenness':
            dictionary_metric_network = nx.betweenness_centrality(target_graph)
        elif metric == 'eigenvector':
            dictionary_metric_network = nx.eigenvector_centrality(target_graph)
        else:
            print("metric ill defined")
        metric_dictionary_network = dict(
            sorted(dictionary_metric_network.items(), key=lambda item: item[1], reverse=True)[:int(top_percent)])
        print(r"Calculating Final score")
        counted_in_top = 0
        for key, score in overal_score_dictionary.items():
            if key in metric_dictionary_network:
                counted_in_top += 1
        score = (counted_in_top / len(overal_score_dictionary)) * 100
        result = ("Using ", top_n_embeddings, " most similar nodes for each key actor, ", score,
              " percent of the nodes where in the top ", top_n_percent, " percent according to ", metric)
        result = str(result)
        result = result.replace(',','')
        result = result.replace("'","")
        file.write(str(result)+ '\n')
    return score, sorted_similarity_dictionary, sorted_similarity_dictionary_all, dictionary_metric_network



def evaluate_embeddings_example_key_actors_separated(key_actors, metric, embeddings_dictionary_source, target_graph,  top_n_embeddings, top_n_percent):
    overal_score_dictionary = {}
    top_n_percent_value = 100 /top_n_percent
    top_ten_percent = len(target_graph.nodes()) / top_n_percent_value
    print(r"Creating Accumulative Score Dictionary")
    # first create average embedding
    # embedding_to_match = [0] * len(key_actors)
    count_used_actors = 0
    for actor in key_actors:
        if actor in embeddings_dictionary_source:
            embedding_to_match = embeddings_dictionary_source[actor]
            similarity_dictionary = create_similarity_dictionary(embeddings_dictionary_source, "Cosine",
                                                                 embedding_to_match)
            # get the top 20 most similar actors
            sorted_similarity_dictionary = dict(
                sorted(similarity_dictionary.items(), key=lambda item: item[1], reverse=True)[:top_n_embeddings])
            for key, value in sorted_similarity_dictionary.items():
                if key in overal_score_dictionary:
                    overal_score_dictionary[key] = overal_score_dictionary[key] + value
                else:
                    overal_score_dictionary[key] = value

    print(r"Calculating PageRank and Eigenvector centrality on target network")
    top_n_percent_value = 100 / top_n_percent
    top_ten_percent = len(target_graph.nodes()) / top_n_percent_value
    if metric == 'pagerank':
        dictionary_pagerank_marvel = nx.pagerank(target_graph)
    elif metric == 'betweenness':
        dictionary_pagerank_marvel = nx.betweenness_centrality(target_graph)
    elif metric == 'eigenvector':
        dictionary_pagerank_marvel = nx.eigenvector_centrality(target_graph)
    else:
        print("metric ill defined")
    dictionary_eigenvector_marvel = dict(
        sorted(dictionary_pagerank_marvel.items(), key=lambda item: item[1], reverse=True)[:int(top_ten_percent)])
    sorted_similarity_dictionary_all = dict(
        sorted(dictionary_pagerank_marvel.items(), key=lambda item: item[1], reverse=True))
    print(r"Calculating Final score")
    counted_in_top = 0
    for key, score in overal_score_dictionary.items():
        if key in dictionary_eigenvector_marvel:
            counted_in_top += 1

    score = (counted_in_top / len(overal_score_dictionary)) * 100
    print("Using ", top_n_embeddings, " most similar nodes for each key actor, ", score,
          " percent of the nodes where in the top ", top_n_percent, " percent according to ", metric)

    return score, sorted_similarity_dictionary, sorted_similarity_dictionary_all, dictionary_pagerank_marvel







def evaluate_embeddings_ground_truth(key_actors1, metric,key_actors2, embeddings_dictionary_source, embeddings_dictionary_target, target_graph, top_n_embeddings = 20, top_n_percent = 10):
    overal_score_dictionary = {}
    top_n_percent_value = 100 * top_n_percent
    top_ten_percent = len(target_graph.nodes())/top_n_percent_value
    print(r"Creating Accumulative Score Dictionary")
    #first create average embedding
    #embedding_to_match = [0] * len(key_actors)
    count_used_actors = 0
    for actor in key_actors:
        if actor in embeddings_dictionary_source:
            embedding_to_add = embeddings_dictionary_source[actor]
            try:
                embedding_to_match += embedding_to_add
            except:
                embedding_to_match = embedding_to_add
            count_used_actors +=1
        else:
            pass
    embedding_to_match  = embedding_to_match / count_used_actors
    similarity_dictionary = create_similarity_dictionary(embeddings_dictionary_target, "jensen", embedding_to_match)
    #get the top 20 most similar actors
    sorted_similarity_dictionary = dict(sorted(similarity_dictionary.items(), key=lambda item: item[1], reverse=True)[:top_n_embeddings])
    for key, value in sorted_similarity_dictionary.items():
        if key in overal_score_dictionary:
            overal_score_dictionary[key] = overal_score_dictionary[key] + value
        else:
            overal_score_dictionary[key] = value

    print(r"Calculating PageRank and Eigenvector centrality on target network")
    top_n_percent_value = 100/top_n_percent
    top_ten_percent = len(target_graph.nodes())/top_n_percent_value
    if metric == 'pagerank':
        dictionary_pagerank_marvel = nx.pagerank(target_graph)
    elif metric == 'betweenness':
        dictionary_pagerank_marvel = nx.betweenness_centrality(target_graph)
    elif metric == 'eigenvector':
        dictionary_pagerank_marvel = nx.eigenvector_centrality(target_graph)
    else:
        print("metric ill defined")

    dictionary_eigenvector_marvel = dict(sorted(dictionary_pagerank_marvel.items(), key=lambda item: item[1], reverse=True)[:int(top_ten_percent)])
    print(r"Calculating Final score")
    counted_in_top = 0
    for key, score in overal_score_dictionary.items():
        if key in key_actors2:
            counted_in_top += 1
    return counted_in_top/len(key_actors2)


def evaluate_active_nodes(active_nodes, ranking):
    count = 0
    for node in active_nodes:
        if node in ranking.keys():
            count += 1
    return count / len(active_nodes)


import seaborn as sns
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

def create_plot_input(embedding_target, embeddings_dictionary,sorted_metric_dictionary, key_actors, ranking):
    svd = TruncatedSVD(n_components=2)
    svd.fit(embedding_target)
    reduced_embedding = svd.transform(embedding_target)
    y = []
    x = []
    group = []
    for emb in reduced_embedding:
        y.append(emb[1])
        x.append(emb[0])
    for embedding in embeddings_dictionary.keys():

        if embedding in key_actors:
            group.append(3)
        elif embedding in sorted_metric_dictionary.keys():
            group.append(4)
        elif embedding in ranking.keys():
            group.append(1)
        else:
            group.append(2)
    cdict ={1:'red',2:'blue', 3 :'green', 4:'pink'}
    return y,x, cdict, group
def plot_embeddings(embedding_target, embedding_dictionary_target, sorted_metric_dictionary, key_actors, ranking):
    y1,x1,cdict, group = create_plot_input(embedding_target, embedding_dictionary_target, sorted_metric_dictionary, key_actors, ranking)
    graph = sns.scatterplot(x1, y1, hue=group, palette=cdict)
    #plt.scatter(x1,y1,lw=0, alpha=0.5, c=cdict)
    plt.show()

