Recommender Readme

############################

WHAT IS ITS PURPOSE?

This codebase was created in order to solve - at least partially, via one methodology - the
problem of making recommendations of academic papers to graduate students and the like. The issue
was that I had no access to any feedback from anyone regarding what papers they liked to read
in the past, so whatever was going to be recommended was done so based on only on what papers
they were already being given by the website, i.e. they request a set of papers following a set of
clear-cut guidelines like author, journal, or keyword, and then they get sent links to all of those
papers in one place, rather than sifting through the literature themselves. So, this program
seeks to make recommendations of what papers someone might want to read, based on the papers
they've read so far.

############################

WHAT ALGORITHM DOES IT USE?

The algorithm that it uses is a homebrew one that I came up with, combining some elements
of existing NLP algorithms with things like naive bag-of-words and Bayesian inference.

Step 1) Using a corpus of abstracts from academic papers, train a TF-IDF vectorizing model
using SKLearn. This handles cleaning out stopwords and whatnot from the model, because if all
abstracts are zero-valued for the word "the," any differentiation between abstracts would be
completely unaffected by the existence of the word. As a note, all abstracts are pre-processed
to remove punctuation and make all letters lower-case. It helps to clean things up.

Step 1a) Project all vectors to the unit sphere by dividing them by their norms. This just
helps to clear out effects due to the lengths of papers or weird density issues. Not a full step,
just a trick to help with a few things. This is true for all the vectors that get talked about with
this program.

Step 2) Using the same corpus of abstracts and the TF-IDF vectorizer model, create a corpus
of vectorized abstracts. This will give me the data I need to train a K-Means cluster model using
SKLearn. This cluster model doesn't actually end up doing anything per se, but it helps speed
up computation once in production. It'll make a bit more sense later.

Step 3) Create users based on the abstracts of the papers that they've read. Pretty
straightforward, just consider a user to have a vector equivalent to the mean of the vectors
of the papers that they've already read. Re-project the user vectors just for the sake of
consistency.

Step 4) Assign all the users a cluster using the K-Means model.

Step 5) Get they day's new papers, vectorize them, and assign them each a cluster with the K-Means
model.

Step 6) For all users, update their vector by taking the existing vector, multiplying it by
(1-alpha) for some learning rate alpha, adding to it alpha * mean(vectors of papers the user got), 
and re-projecting it. This will let user preferences change over time, as well as adapt to new
preferences by the user without recalculating everything all over again. To make sure everything's
still good, recalculate the user's cluster with the K-Means model applied to the user's vector.

Step 7) Assign as recommendations for a givenecho "# Recommender" >> README.md
git init
git add README.md
git commit -m "first commit"
git remote add origin https://github.com/JonathanSKent/Recommender.git
git push -u origin master user the n papers of that day within the user's
cluster whose vector has the least Euclidean distance to the vector of the paper. This
is where the clusters come into play. Chances are, if the paper isn't given the same cluster
by the K-Means model as the user, it almost certainly won't be in the top few papers that the
user would be assigned. So, instead of checking every paper against every user to find the best
papers, you can just check the papers that have been assigned the same cluster and get nearly
the same effect.

Step 8) Goto 5.

#########################

HOW HAS IT BEEN IMPLEMENTED?

Each file in the program has been commented for what each function or class does, but this will
give a broader overview.

clusterer.py handles the K-Means clusterer model, including saving and loading it from storage.
Because neither the K-Means nor TF-IDF models update over time, they can be computed using a very
large corpus of data once, and then saved to the hard drive, instead of needing to redo the
rather expensive calculations every time.

csvhandling.py handles turning CSVs containing paper data into usable information.
read_csv_into_lines takes a directory location, and returns a list of lists of strings representing
the context of the CSV. Be careful with the formatting. read_csv_into_abstracts does much the same,
except returning only the 2-indexed element of each line instead of the full line. This helps
with speeding up the creation of an abstract corpus for training the TF-IDF and K-Means models.
clean_abstract just cleans up the abstracts into a more easily vectorizible string.
clean_abstract_list cleans up a list of abstracts.
full_abstract_list_to_array is my partially successful attempt to parallelize the process. I
would recommend against using it.

main.py handles some central functions of the program. initialize creates the K-Means and TF-IDF
models, and puts them into storage, as well as preparing where the users will be stored.
add_user adds a user to the database that reads every paper with where at least one of the keywords
is found in the abstract, and day handles steps 5 through 7.

papers.py handles papers. It creates paper objects to more easily process them by keeping
vectors, abstracts, cleaned abstracts, clusters, IDs, and whatnot all in the same place.
lines_to_papers takes the lines read by csvhandling.read_csv_into_lines, and turns it
into a list of papers. papers_to_cluster_list organizes the papers by what cluster they fall into,
to make help with making recommendations.

run.py just runs the program for testing purposes.

settings.py holds constant values that are used elsehwere in the program. It automatically
determines the directory that the program is working in, in order to put files like the
two models and the list of users in an appropriate location.

users.py covers how users work. The class holds onto a user ID, the list of papers it has already
read - so it doesn't recommend a paper the user has seen before - the user's cluster, and some
other information. It also covers getting the distance to a paper, updating the user, and
recommending papers. save_user_list and load_user_list help with databasing.

vectorizer.py handles the TF-IDF vectorizing model, as well as saving and loading it, and
abstract_list_to_vector_list helps prepare the training corpus of vectors for the K-Means model.

##########################

I'M SORRY, WHAT?

It's a bit of a spiderweb. If you need some help understaing what I've built here, you can
contact me at kent.jonathan.s@gmail.com.

-Jonathan S. Kent, March 4th, 2019

