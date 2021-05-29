from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle
import numpy as np
from sklearn import metrics
import pandas as pd
import seaborn as sns
from feature_extraction import text_process
import matplotlib.pyplot as plt

def test_k(posts_tfidf_bow, data, trials):
    # Test different cluster sizes
    init_vals = np.random.choice(np.arange(100), size=10, replace=False)
    silhouette_vals = []
    sse_vals = []
    clusters = list(range(2,50))
    
    for selected_k in np.array(clusters):
      print("Cluster: {}".format(str(selected_k)))
      for i in np.arange(trials):
        print("Rep: {}".format(str(i)))
        km = MiniBatchKMeans(n_clusters=selected_k, init='k-means++', n_init=1, init_size=1000, 
                         batch_size=1000, random_state = init_vals[i])
        km.fit(posts_tfidf_bow)
        silhouette_vals.append(metrics.silhouette_score(posts_tfidf_bow, km.labels_, sample_size=1000))
        sse_vals.append(km.inertia_)
        
    n_clusters = np.repeat(np.array(clusters), trials)
    
    
    # Plot silhouette scores
    to_plot_d = {'Init_Value': np.repeat(init_vals[0:trials], len(clusters)), 'Silhouette': silhouette_vals, 'Squared_Errors_Sum': sse_vals, 
                 'Num_Clusters': n_clusters}
    to_plot_df = pd.DataFrame(data=to_plot_d)
    plt.figure()
    ax = sns.lineplot(x="Num_Clusters", y="Silhouette", data=to_plot_df)
    
    # Plot Sum of Squared Errors values
    plt.figure()
    ax = sns.lineplot(x="Num_Clusters", y="Squared_Errors_Sum", data=to_plot_df)
    


if __name__ == "__main__":
    posts_tfidf_bow = pickle.load(open("processed-data.pkl","rb"))
    data = pickle.load(open("data.pkl","rb"))
    test_k(posts_tfidf_bow, data, 10)