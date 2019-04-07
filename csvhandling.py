"""
This file handles CSVs for the purpose of reading into the rest of the program
"""

import unicodecsv
import multiprocessing
import numpy as np

import settings
import vectorizer
import papers

# Given the directory pointing towards a CSV, returns each line in the CSV
# as a list of strings
def read_csv_into_lines(location):
    with open(location, 'rb') as file:
        reader = unicodecsv.reader(file, encoding = 'utf-8', delimiter = ',')
        return([line for line in reader])
        
# Given the location of a CSV, returns a list containing the 2-indexed element of
# each line in the CSV. In Academic Sequitur formatting, this will be the abstract.
# The abstracts on their own can be used in order to train the clusterer and vectorizer
# models. Provides only the non-zero abstracts
def read_csv_into_abstracts(location):
    with open(location, 'rb') as file:
        reader = unicodecsv.reader(file, encoding = 'utf-8', delimiter = ',')
        corpus = [line[2] for line in reader if len(line[2])]
        return(np.random.choice(corpus, size = int(settings.training_proportion * len(corpus))))
        
# Given an abstract, cleans and renders it into a more cleanly vectorizible form
def clean_abstract(abstract, punctuation_to_remove = settings.punctuation_to_remove):
    return("".join([char for char in abstract.lower() if not char in punctuation_to_remove]))
    
# An alternate form of clean_abstract, which can be called on to clean a list of abstracts
# This will ignore the order of the abstracts given to it
def clean_abstract_list(abstracts):
    return([clean_abstract(abstract) for abstract in abstracts])
    
# Given a list of cleaned abstracts, returns an array of vectorized abstracts, without
# regard for order. This is parallelized
def full_abstract_list_to_array(abstracts, location = settings.vectorizing_device_location):
    chunks = np.array_split(abstracts, settings.parallel_processes)
    pool = multiprocessing.Pool(processes = settings.parallel_processes)
    output = pool.imap_unordered(vectorizer.abstract_list_to_vector_list, chunks)
    return(np.concatenate([vector_list for vector_list in output if len(vector_list)]))
    
