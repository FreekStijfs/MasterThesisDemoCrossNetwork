from scipy import spatial
from scipy.spatial import distance
import numpy as np

def create_similarity_dictionary(embeddings_dictionary_target,scoring_measure, embedding):
    similarity_dictionary = {}
    for actor, embedding_marvel in embeddings_dictionary_target.items():
        #embedding_marvel = embedding_marvel[embedding_marvel != 0]
        if scoring_measure == r"Cosine":
            result = 1 - spatial.distance.cosine(embedding,embedding_marvel)
            similarity_dictionary[actor] = result
        elif scoring_measure == "jensen":
            result = 1 - distance.jensenshannon(embedding,embedding_marvel)
            similarity_dictionary[actor] = result
        elif scoring_measure == 'euclidean':
            embedding = np.array(embedding)
            embedding_marvel = np.array(embedding_marvel)
            result = np.linalg.norm(embedding - embedding_marvel)
            similarity_dictionary[actor] = result
    return similarity_dictionary