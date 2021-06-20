from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AffinityPropagation, MeanShift
import pickle
import numpy as np
from sklearn import metrics
import pandas as pd
import seaborn as sns
from feature_extraction import text_process
import matplotlib.pyplot as plt
from feature_extraction import feature_extraction, process_data

def test_k(processed, data, trials):
    # Test different cluster sizes
    init_vals = np.random.choice(np.arange(100), size=10, replace=False)
    silhouette_vals = []
    ch_vals = []
    db_vals = []
    sse_vals = []
    clusters = list(range(1,11))
    clusters = [5*i for i in clusters]
    
    for selected_k in np.array(clusters):
      print("Cluster: {}".format(str(selected_k)))
      for i in np.arange(trials):
        print("Rep: {}".format(str(i)))
        km = MiniBatchKMeans(n_clusters=selected_k, init='k-means++', n_init=1, init_size=1000, 
                         batch_size=1000, random_state = init_vals[i])
        km.fit(processed)
        silhouette_vals.append(metrics.silhouette_score(processed, km.labels_, sample_size=1000))
        ch_vals.append(metrics.calinski_harabasz_score(processed.toarray(), km.labels_))
        db_vals.append(metrics.davies_bouldin_score(processed.toarray(), km.labels_))
        sse_vals.append(km.inertia_)
        
    n_clusters = np.repeat(np.array(clusters), trials)
    
    
    # Plot silhouette scores
    to_plot_d = {'Init_Value': np.repeat(init_vals[0:trials], len(clusters)), 
                 'Silhouette': silhouette_vals, 'Squared_Errors_Sum': sse_vals, 
                 'CH': ch_vals, 'DB': db_vals,
                 'Num_Clusters': n_clusters}
    to_plot_df = pd.DataFrame(data=to_plot_d)
    
    plt.figure()
    ax = sns.lineplot(x="Num_Clusters", y="Silhouette", data=to_plot_df)
    ax.set(xlabel='Number of Clusters', ylabel='Silhouette score')
    
    plt.figure()
    ax = sns.lineplot(x="Num_Clusters", y="DB", data=to_plot_df)
    ax.set(xlabel='Number of Clusters', ylabel='DB')
    
    plt.figure()
    ax = sns.lineplot(x="Num_Clusters", y="CH", data=to_plot_df)
    ax.set(xlabel='Number of Clusters', ylabel='CH')
    
    # Plot Sum of Squared Errors values
    plt.figure()
    ax = sns.lineplot(x="Num_Clusters", y="Squared_Errors_Sum", data=to_plot_df)
    ax.set(xlabel='Number of Clusters', ylabel='Sum of Squared Errors')
    
def test_k_maxdf(data, trials):
    # Test different cluster sizes
    init_vals = np.random.choice(np.arange(100), size=10, replace=False)
    silhouette_vals_all = []
    ch_vals_all = []
    db_vals_all = []
    sse_vals_all = []
    clusters = list(range(1,11))
    clusters = [5*i for i in clusters]
    max_dfs = [0.25, 0.5, 0.75]
    
    for j in max_dfs:
        feature_extraction(data, 10000000000000000, j)
        silhouette_vals = []
        ch_vals = []
        db_vals = []
        sse_vals = []
        for selected_k in np.array(clusters):
            print("Cluster: {}".format(str(selected_k)))
            for i in np.arange(trials):
                print("Rep: {}".format(str(i)))
                km = MiniBatchKMeans(n_clusters=selected_k, init='k-means++', n_init=1, init_size=1000, 
                                 batch_size=1000, random_state = init_vals[i])
                km.fit(processed)
                silhouette_vals.append(metrics.silhouette_score(processed, km.labels_, sample_size=1000))
                ch_vals.append(metrics.calinski_harabasz_score(processed.toarray(), km.labels_))
                db_vals.append(metrics.davies_bouldin_score(processed.toarray(), km.labels_))
                sse_vals.append(km.inertia_)
        
        silhouette_vals_all.append(silhouette_vals)
        ch_vals_all.append(ch_vals)
        db_vals_all.append(db_vals)
        sse_vals_all.append(sse_vals)
        
    n_clusters = np.repeat(np.array(clusters), trials)
    
    plt.figure()
    for i in range(len(silhouette_vals_all)):
        ax = sns.lineplot(n_clusters, silhouette_vals_all[i])
        #plt.plot(n_clusters, silhouette_vals_all[i])
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.legend([str(i) for i in max_dfs], title="Maximum document frequency")
    
    return silhouette_vals_all, ch_vals_all, db_vals_all, sse_vals_all

def test_DBSCAN(processed):
    clustering = DBSCAN().fit(processed)
    print(clustering.core_sample_indices_)
    return clustering.core_sample_indices_

def test_affprop(processed):
    clustering = AffinityPropagation().fit(processed)
    return clustering.cluster_centers_

def test_meanshift(processed):
    clustering = MeanShift().fit(processed.toarray())
    return clustering.cluster_centers_

if __name__ == "__main__":
    processed = pickle.load(open("processed-data.pkl","rb"))
    data = pickle.load(open("data.pkl","rb"))
    output = test_k_maxdf(data, 10)