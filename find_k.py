from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AffinityPropagation, MeanShift
from sklearn.model_selection import RandomizedSearchCV
import pickle
import numpy as np
from sklearn import metrics
import pandas as pd
import seaborn as sns
from find_centroids import find_centroids
import matplotlib.pyplot as plt
from feature_extraction import feature_extraction, process_data
from sklearn.base import BaseEstimator, TransformerMixin
import argparse
import matplotlib.cm as cm
from pylab import *
import csv

class KMeansTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, max_df=0, max_features=0, n_clusters=10, init='k-means++', n_init=1, init_size=1000, batch_size=1000):
       
        # Initialize parameters
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.init_size = init_size
        self.batch_size = batch_size
        
        # Initialize model
        self.model = MiniBatchKMeans(
            n_clusters = n_clusters,
            init = init,
            n_init = n_init,
            init_size = init_size,
            batch_size = batch_size)
        
        # Custom params
        self.max_df = max_df
        self.max_features = max_features
        
    def fit(self, X):
        feature_extraction(X, self.max_features, self.max_df)
        processed = pickle.load(open("processed-data.pkl","rb"))
        self.X = processed
        
        # Fit
        self.model.fit(self.X)

    def transform(self, X):
        pred = self.model.predict(X)
        return np.hstack([self.X, pred.reshape(-1, 1)])


    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def score(self, sample_size=1000):
        self.y = self.model.labels_
        score = metrics.silhouette_score(self.X, self.y, sample_size=1000)
        return score

raw_data = pickle.load(open("data.pkl","rb"))
param_dist = {'max_features': [500, 1000, 1500, 2000, 2500, 3000],
              'max_df': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
              'n_clusters': [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]}
clf = RandomizedSearchCV(KMeansTransformer(), param_dist, n_iter=10)
search = clf.fit(raw_data)
df = pd.DataFrame(clf.cv_results_)
df.to_csv("grid_search.csv", index=False)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--trials',
#         type=int,
#         required=True,
#         help='numbers of trials per k',
#         default=10,
#         )
#     parser.add_argument(
#         '--max_k',
#         type=int,
#         required=True,
#         help='maximum number of clusters to evaluate',
#         default=60,
#         )
#     parser.add_argument(
#         '--num_features',
#         type=int,
#         required=True,
#         help='maximum number of clusters to evaluate',
#         default=1000,
#         )
#     FLAGS, unparsed = parser.parse_known_args()
#     trials = FLAGS.trials
#     max_k = FLAGS.max_k
#     features = FLAGS.num_features
#     data = pickle.load(open("data.pkl","rb"))
#     feature_extraction(data, features, 0.5)
    # processed = pickle.load(open("processed-data.pkl","rb"))
    # output = find_k(processed, trials, max_k)