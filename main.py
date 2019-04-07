"""
Encodes central functions
"""

import os
import joblib
import unicodecsv

import csvhandling
import vectorizer
import clusterer
import settings
import users
import papers

# Given the location of a training corpus, creates and saves
# the new vectorizing device, the new clustering device, 
# and returns a list of papers that can be used to create
# initial users
def initialize(csv_location):
    print("Some initial setup...")
    if not os.path.isdir(settings.file_location):
        os.mkdir(settings.file_location)
    if not os.path.isfile(settings.user_list_location):
        joblib.dump([], settings.user_list_location)
    print("Initial setup complete. Reading paper abstracts into memory.")
    paper_abstracts = csvhandling.clean_abstract_list(csvhandling.read_csv_into_abstracts(csv_location))
    print("Abstracts read into memory. Developing vector device.")
    vect_dev = vectorizer.vectorizing_device(abstracts = paper_abstracts)
    print("Vector device developed. Preparing vector training set.")
    paper_vectors = vectorizer.abstract_list_to_vector_list(paper_abstracts)
    print("Vector training set prepared. Developing clustering device.")
    clust_dev = clusterer.clustering_device(vector_matrix = paper_vectors)
    print("Clustering device developed.")
    del(vect_dev)
    del(paper_vectors)
    del(clust_dev)

# Adds a user that adds all papers given to it based on a set of keywords
def add_user(ID, keywords):
    users.add_user(ID, keywords)
    
# Given a new csv representing the day's new papers, updated the users
# and gets their recommendations, before formatting it into and saving a csv at
# the target location
def day(csv_location, target_location = settings.default_target_location):
    paper_list = papers.lines_to_papers(csvhandling.read_csv_into_lines(csv_location))
    users.update_keyword_users(paper_list)
    clusters = papers.papers_to_cluster_list(paper_list)
    recommendations = [(user.ID, user.recommend(clusters[user.cluster])) for user in users.load_user_list()]
    del(paper_list)
    del(clusters)
    lines = []
    for rec in recommendations:
        new_line = [rec[0]]
        rec_papers = rec[1]
        for rec_pap in rec_papers:
            new_line.append(rec_pap[1])
            new_line.append(rec_pap[0].ID)
            new_line.append(rec_pap[0].URL)
        lines.append(new_line)
    with open(target_location, 'wb') as file:
        writer = unicodecsv.writer(file, encoding = 'utf-8', delimiter = ',')
        for line in lines:
            writer.writerow(line)