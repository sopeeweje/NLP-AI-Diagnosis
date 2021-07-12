from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AffinityPropagation, MeanShift
import pickle
import numpy as np
from sklearn import metrics
import pandas as pd
import seaborn as sns
from feature_extraction import text_process
import matplotlib.pyplot as plt
from feature_extraction import feature_extraction, process_data
import argparse
import matplotlib.cm as cm
from pylab import *

def find_k(data, trials, k):
    # Test different cluster sizes
    init_vals = np.random.choice(np.arange(100), size=10, replace=False)
    silhouette_vals = []
    sse_vals = []
    clusters = list(range(2,int(k/1)))
    clusters = [1*i for i in clusters]
    
    for selected_k in np.array(clusters):
      print("Cluster: {}".format(str(selected_k)))
      for i in np.arange(trials):
        print("Rep: {}".format(str(i)))
        km = MiniBatchKMeans(n_clusters=selected_k, init='k-means++', n_init=1, init_size=1000, 
                         batch_size=1000, random_state = init_vals[i])
        km.fit(data)
        silhouette_vals.append(metrics.silhouette_score(data, km.labels_, sample_size=1000))
        sse_vals.append(km.inertia_)
    n_clusters = np.repeat(np.array(clusters), trials)
    
    
    # Plot silhouette scores
    to_plot_d = {'Init_Value': np.repeat(init_vals[0:trials], len(clusters)), 
                 'Silhouette': silhouette_vals,
                 'SSE': sse_vals,
                 'Num_Clusters': n_clusters}
    to_plot_df = pd.DataFrame(data=to_plot_d)
    
    plt.figure()
    ax = sns.lineplot(x="Num_Clusters", y="Silhouette", data=to_plot_df)
    ax.set(xlabel='Number of Clusters', ylabel='Silhouette score')
    plt.savefig('k_selection.png')
    
    plt.figure()
    ax = sns.lineplot(x="Num_Clusters", y="SSE", data=to_plot_df)
    ax.set(xlabel='Number of Clusters', ylabel='Sum of Squared Errors')
    plt.savefig('k_selection_sse.png')

def find_k_silhouette():
    X = pickle.load(open("processed-data.pkl","rb"))
    range_n_clusters = [20, 30, 40, 50, 60]
    
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots()
        fig.set_size_inches(18, 7)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, np.shape(X)[0] + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=1, init_size=1000, 
                         batch_size=1000)
        cluster_labels = clusterer.fit_predict(X)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = metrics.silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = metrics.silhouette_samples(X, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        plt.show()

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
    data = pickle.load(open("data.pkl","rb"))
    feature_extraction(data, features, 0.5)
    processed = pickle.load(open("processed-data.pkl","rb"))
    output = find_k(processed, trials, max_k)