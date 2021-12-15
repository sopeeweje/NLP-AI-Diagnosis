from sklearn.cluster import MiniBatchKMeans
import pickle
import numpy as np
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from feature_extraction import feature_extraction
import argparse
from pylab import *

def find_k(data, trials, k):
    # Test different cluster sizes
    init_vals = np.random.choice(np.arange(100), size=10, replace=False)
    silhouette_vals = []
    sse_vals = []
    clusters = [5*i for i in list(range(1,int(k)//5))]
    
    for selected_k in np.array(clusters):
      print("Cluster: {}".format(str(selected_k)))
      for i in np.arange(trials):
        km = MiniBatchKMeans(n_clusters=selected_k, init='k-means++', verbose=0, max_no_improvement=None)
        km.fit(data)
        score = metrics.silhouette_score(data, km.labels_)
        silhouette_vals.append(score)
        print("Rep: {}, Score: {}".format(str(i), str(score)))
        sse_vals.append(km.inertia_)
    n_clusters = np.repeat(np.array(clusters), trials)
    
    
    # Plot silhouette scores
    to_plot_d = {'Init_Value': np.repeat(init_vals[0:trials], len(clusters)), 
                 'Silhouette': silhouette_vals,
                 'SSE': sse_vals,
                 'Num_Clusters': n_clusters}
    to_plot_df = pd.DataFrame(data=to_plot_d)
    to_plot_df.to_csv("finding_k.csv", index=False)
    
    plt.figure()
    ax = sns.lineplot(x="Num_Clusters", y="Silhouette", data=to_plot_df)
    ax.set(xlabel='Number of Clusters', ylabel='Silhouette score')
    plt.savefig('k_selection.eps', format='eps') 
    
    plt.figure()
    ax = sns.lineplot(x="Num_Clusters", y="SSE", data=to_plot_df)
    ax.set(xlabel='Number of Clusters', ylabel='Sum of Squared Errors')
    plt.savefig('figures/k_selection_sse.eps', format='eps')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trials',
        type=int,
        required=True,
        help='numbers of trials per k',
        default=10,
        )
    parser.add_argument(
        '--max_k',
        type=int,
        required=True,
        help='maximum number of clusters to evaluate',
        default=60,
        )
    parser.add_argument(
        '--num_features',
        type=int,
        required=True,
        help='maximum number of clusters to evaluate',
        default=1000,
        )
    FLAGS, unparsed = parser.parse_known_args()
    trials = FLAGS.trials
    max_k = FLAGS.max_k
    features = FLAGS.num_features
    data = pickle.load(open("data/data.pkl","rb"))
    feature_extraction(data, features, 0.1)
    processed = pickle.load(open("data/processed-data.pkl","rb"))
    output = find_k(processed, trials, max_k)