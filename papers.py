"""
Handles papers
"""

import numpy as np

import vectorizer
import clusterer
import csvhandling
import settings

# Defines a paper
class paper:
    def __init__(self, paper_ID, paper_vector, paper_URL, paper_cluster, paper_abstract, paper_clean_abstract):
        self.ID = paper_ID
        self.vector = paper_vector
        self.URL = paper_URL
        self.cluster = paper_cluster
        self.abstract = paper_abstract
        self.clean_abstract = paper_clean_abstract
        
# Given a list of lines from a CSV, returns a list of papers
def lines_to_papers(lines):
    clust_dev = clusterer.clustering_device()
    vect_dev = vectorizer.vectorizing_device()
    papers = []
    for line in lines:
        new_abstract = csvhandling.clean_abstract(line[2])
        vector = vect_dev.abstract_to_vector(new_abstract)
        cluster = clust_dev.vector_to_cluster(vector)
        papers.append(paper(line[9], vector, line[0], cluster, line[2], new_abstract))
    return(papers)
    
    
# Given a list of papers, returns a list of each paper indexed by cluster
# e.g. a paper in cluster 5 will appear in the 5-indexed portion of the list
# Null vectors, in cluster -1, will appear in the last portion
def papers_to_cluster_list(papers):
    cluster_list = [[] for k in range(settings.cluster_count + 1)]
    for paper in papers:
        cluster_list[paper.cluster].append(paper)
    return(cluster_list)