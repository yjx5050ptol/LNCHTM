from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
import numpy as np

class NpClustering(object):
    def __init__(self, option):
        self.option = option
        
    def train(self, affinity, preference = None):
        if self.option == "ap":
            clustering = AffinityPropagation(affinity = "precomputed", preference = preference)
        predicted_labels = clustering.fit_predict(affinity)
        formed_centers_indices = clustering.cluster_centers_indices_
        return predicted_labels, formed_centers_indices

