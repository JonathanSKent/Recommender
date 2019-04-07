"""
Handles all aspects of the clusterer, which is responsible for speeding
up computation regarding finding the closest papers to a user
"""

import joblib
import sklearn.cluster
import numpy as np

import settings

# Defines the clustering device
class clustering_device:
    # The behavior of the initilization function is as follows:
    # If it is given a matrix of abstract vectors, it will
    # train a new clusterer, and leave it in memory.
    # Otherwise, it will load the clusterer that has been stored
    def __init__(self, location = settings.clustering_device_location, vector_matrix = []):
        self.location = location
        if len(vector_matrix):
            self.clusterer = sklearn.cluster.KMeans(n_clusters = settings.cluster_count)
            self.clusterer.fit(vector_matrix)
            joblib.dump(self.clusterer, self.location)
        else:
            try:
                self.clusterer = joblib.load(self.location)
            except:
                print('ERROR: Lack of vectors / Lack of original file for clustering device')
                
    # When given a particular vector, returns an integer representing what cluster
    # it belongs to
    def vector_to_cluster(self, vector):
        if np.linalg.norm(vector) > 0:
            return(self.clusterer.predict([vector])[0])
        else:
            return(-1)