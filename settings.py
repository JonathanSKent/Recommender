"""
Stores the settings being used elsewhere in the program
"""

import os

######################
# DIRECTORY SETTINGS #
######################

# The location in which the program is operating
base_directory = os.getcwd()

# Gives the location of the required/generated files used by the program
file_location = base_directory + '/files'

# Gives the location of the vectorizing device
vectorizing_device_location = file_location + '/vectorizing_device.joblib'

# Gives the location of the clustering device
clustering_device_location = file_location + '/clustering_device.joblib'

# Gives the location of the list of users
user_list_location = file_location + '/user_list.joblib'

# Gives the default location where the CSV with recommendations is saved
default_target_location = file_location + '/recommendations.csv'

#######################

# A string of all the punctuation marks to be removed from abstracts, before
# they are vectorized
punctuation_to_remove = '"' + "`~1!2@3#4$5%6^7&8*9(0)-_=+[{]}\|;:',<.>/?"

# Gives the number of simultaneous processes to run when parallelizing tasks
parallel_processes = 4

# Gives the number of clusters into which abstract vectors or users should be grouped
cluster_count = 1

# The portion of a user's vector that is determined on any given update cycle
user_learn_rate = 0.1

# The number of recommendations each user receives per pass through
recommendations = 5

# The portion of the training dataset that is actually used
training_proportion = 0.02
