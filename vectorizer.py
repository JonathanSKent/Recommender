"""
Handles all aspects of the vectorizer, which is responsible for turning
paper abstracts into vectors
"""

import sklearn.feature_extraction
import joblib
import numpy as np

import settings

# Defines the vectorizing device
class vectorizing_device:
    # The behavior of the initializing function for the vectorizing device
    # is as such:
    # If it is given a list of abstracts, then it will train a new vectorizing
    # device, save it in the appropriate location, and retain it in memory.
    # If it is not given a list of abstracts, then it will not train a new
    # device, and it will load whatever version exists in the appropriate location
    #######################################################
    # WHEN GIVING THE VECTORIZER ABSTRACTS TO TRAIN FROM  #
    # GIVE IT PRE-PROCESSED ABSTRACTS                     #
    #######################################################
    def __init__(self, location = settings.vectorizing_device_location, abstracts = []):
        self.location = location
        if len(abstracts):
            self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
            self.vectorizer.fit(abstracts)
            joblib.dump(self.vectorizer, self.location)
        else:
            try:
                self.vectorizer = joblib.load(self.location)
            except:
                print('ERROR: Lack of abstracts / Lack of original file for vector device')
                
    # When given a particular abstract, returns a numpy vector
    # Once again, make sure to provide a pre-processed abstract
    # Note: all nonzero vectors are projected to the unit sphere
    # Zero vectors are just left alone
    def abstract_to_vector(self, abstract):
        vector = np.array(self.vectorizer.transform([abstract]).todense())[0]
        if np.linalg.norm(vector) > 0:
            vector /= np.linalg.norm(vector)
        return(vector)

# Given a list of pre-cleaned abstracts, returns a list of abstract vectors
# Can be run with itself in parallel
def abstract_list_to_vector_list(abstracts, location = settings.vectorizing_device_location):
    device = vectorizing_device(location = location)
    return([device.abstract_to_vector(abstract) for abstract in abstracts])