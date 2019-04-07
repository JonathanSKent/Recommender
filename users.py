"""
Handles users
"""

import numpy as np
import joblib

import papers
import settings
import clusterer

# Defines a user
class user:
    def __init__(self, ID = '', paper_ID_list = [], vector = [], cluster = 0, user_type = 'keyword', keywords = []):
        self.ID = ID
        self.user_type = user_type
        self.keywords = [word.lower() for word in keywords]
        self.paper_IDs = paper_ID_list
        self.vector = vector
        self.cluster = cluster
        
    # Takes a paper object, and returns the distance between it and the given paper
    def distance(self, paper):
        return(np.linalg.norm(self.vector - paper.vector))
        
    # Given a list of papers, updates the user based on those papers
    def update(self, paper_list):
        self.paper_IDs = self.paper_IDs + [paper.ID for paper in paper_list]
        new_vector = np.mean([paper.vector for paper in paper_list], axis = 0)
        if np.linalg.norm(self.vector) > 0 and np.linalg.norm(self.vector) != np.nan:
            alpha = settings.user_learn_rate
            self.vector = ((1 - alpha) * self.vector) + (alpha * new_vector)
        else:
            self.vector = new_vector
        if np.linalg.norm(self.vector):
            self.vector /= np.linalg.norm(self.vector)
            
    # Given a list of papers, returns the papers that it will recommend
    def recommend(self, paper_list):
        in_consideration = [paper for paper in paper_list if not paper.ID in self.paper_IDs]
        distances = np.array([self.distance(paper) for paper in in_consideration])
        return([(in_consideration[index], distances[index]) for index in distances.argsort()[:settings.recommendations]])
        
# Saves the list of users
def save_user_list(user_list):
    joblib.dump(user_list, settings.user_list_location)
    
# Loads the list of users
def load_user_list():
    return(joblib.load(settings.user_list_location))
    
# Updates the keyword users based on a new set of papers
def update_keyword_users(paper_list):
    clust_dev = clusterer.clustering_device()
    user_list = load_user_list()
    for user in user_list:
        current_papers = [paper for paper in paper_list if any([word in paper.clean_abstract for word in user.keywords])]
        user.update(current_papers)
        user.cluster = clust_dev.vector_to_cluster(user.vector)
    save_user_list(user_list)
    
# Creates a new user based on a list of keywords
def add_user(ID, keywords):
    user_list = load_user_list()
    user_list.append(user(ID = ID, keywords = keywords))
    save_user_list(user_list)