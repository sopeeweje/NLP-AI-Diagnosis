import csv
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, AffinityPropagation, SpectralClustering
import matplotlib.pyplot as plt
import pickle
import numpy as np
from feature_extraction import text_process
import sklearn.metrics as metrics
from yellowbrick.cluster import InterclusterDistance
import sklearn.neighbors
from scipy.optimize import curve_fit
import umap.umap_ as umap

# https://github.com/danielmlow/reddit/blob/master/Unsupervised_Clustering_Pipeline.ipynb

def get_clusters(selected_k, data_file, processed_file, centers, years):
    # Load data as dictionary
    data = pickle.load(open(data_file,"rb"))
    
    # Transformed data
    X_transformed = pickle.load(open(processed_file,"rb"))
    
    # Perform k means # KMeans(n_clusters=selected_k)#
    km = MiniBatchKMeans(n_clusters=selected_k, init=centers, n_init=10, init_size=3000, batch_size=1000, verbose=0, max_no_improvement=None)
    clusters = km.fit_predict(X_transformed)
    print(clusters)
    
    # Output data
    cluster_all = []
    costs = []
    yoy = []
    size = []
    
    for i in range(0,selected_k):
        print(i)
        
        # indices of cluster k
        cluster = [idx for idx, element in enumerate(clusters) if element == i]
        
        # get points
        cluster_data = [data[ind] for ind in cluster]
        cluster_all.append(cluster_data)
        
        # calculate average cost and std
        try:
            average_cost = sum([item["cost"] for item in cluster_data])/len(cluster_data)
            #std = statistics.pstdev([item["cost"] for item in cluster_data])
        except:
            average_cost = 0
            #std = 0
        costs.append(average_cost)
        
        cost_trend = []
        for year in years:
            year_data = [data[ind]["cost"] for ind in cluster if data[ind]["year"] == year]
            if len(year_data) == 0:
                cost_trend.append(0)
            else:
                year_cost = sum(year_data)/len(year_data)
                cost_trend.append(year_cost)
        
        yoy.append(cost_trend)
        
        size.append(len(cluster))
    
    # Get centroids 
    # Identify the top terms for each cluster, using the TF-IDF terms with the highest values in the centroid
    # Adapted From: https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    vectorizer = pickle.load(open("vectorizer.pkl","rb"))
    terms = vectorizer.get_feature_names()
    centroids = []
    centroid_file = open("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/centroid", "w")
    for i in range(selected_k):
        centroid_file.write("Cluster %d:" % i)
        centroid_list = []
        for ind in order_centroids[i, :20]:
            centroid_file.write(" %s" % terms[ind])
            centroid_list.append(terms[ind])
        centroids.append(centroid_list)
        centroid_file.write("\n")
    centroid_file.close()
    
    score = metrics.silhouette_score(X_transformed, km.labels_, sample_size=1000)
        
    return costs, yoy, size, cluster_all, centroids, score, km, order_centroids, clusters

def umap_visualization(X_transformed, cluster_labels):
    outlier_scores = sklearn.neighbors.LocalOutlierFactor(contamination=0.1).fit_predict(X_transformed)
    X_transformed = X_transformed[outlier_scores != -1]
    cluster_labels = cluster_labels[outlier_scores != -1]
    
    n_subset = len(cluster_labels)
    selected_cells = np.random.choice(np.arange(X_transformed.shape[0]), size = n_subset, replace = False)
    mapper = umap.UMAP(metric='hellinger', random_state=42).fit(X_transformed[selected_cells,:])
    
    embedding = mapper.transform(X_transformed[selected_cells,:])
    print(np.shape(embedding))
    print(np.shape(cluster_labels[selected_cells]))
    
    # Plot Clusters on UMAP
    plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=5, c=cluster_labels[selected_cells])
    plt.gca().set_aspect('equal', 'datalim')
    num_clust = len(np.unique(cluster_labels[selected_cells]))
    plt.colorbar(boundaries=np.arange(num_clust+1)-0.5).set_ticks(np.arange(num_clust))
    plt.title('UMAP Projection of Awards, TF-IDF', fontsize=14);

def viz_centroids(chosen):
    order_centroids = chosen[3].cluster_centers_.argsort()[:, ::-1]
    centroid_file = open("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/centroid", "w")
    vectorizer = pickle.load(open("vectorizer.pkl","rb"))
    terms = vectorizer.get_feature_names()
    for i in range(len(chosen[0])):
        centroid_file.write("Cluster %d:" % i)
        for ind in order_centroids[i, :20]:
            centroid_file.write(" %s" % terms[ind])
        centroid_file.write("\n")
    centroid_file.close()
        
    model = chosen[3]
    X_transformed = pickle.load(open("processed-data.pkl","rb"))
    plt.figure()
    visualizer = InterclusterDistance(model, random_state=0)
    visualizer.fit(X_transformed)     # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure

if __name__ == "__main__":
    
    years = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
    results = {}
    scores = []
    chosen = (0,0,0,0,0)
    selected_k = 15
    centers = 'k-means++'
    for i in range(30):
        # Generate clusters for a selected k
        costs, cost_trend, size, clusters, centroids, score, km, centers, cluster_labels = get_clusters(selected_k, "data.pkl", "processed-data.pkl", 'k-means++', years)
        j = 0
        for thing in clusters:
            for item in thing:
                try:
                    results[item["id"]].append(centroids[j])
                except:
                    results[item["id"]] = [item["id"],item["title"],item["cost"],centroids[j]]
            j+=1
        scores.append(score)
        if score >= max(scores):
            chosen = (clusters, centroids, score, km, cluster_labels, costs, cost_trend)
    
    X_transformed = pickle.load(open("processed-data.pkl","rb"))
    umap_visualization(X_transformed, cluster_labels)
    viz_centroids(chosen)
    with open("model.pkl", 'wb') as handle:
        pickle.dump(chosen[3], handle)
    
    print(results)  
    resultscsv = []
    for key in results:
        resultscsv.append(results[key])
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(resultscsv)
    
    
    plt.figure()
    plt.xlabel("Year")
    plt.ylabel("Funding")
    plt.title("Funding by cluster")
    for i in range(len(chosen[6])):
        plt.plot(years, chosen[6][i], label=str(i))
    plt.xlabel("Year")
    plt.ylabel("Amount ($)")
    plt.legend()
    plt.show()
    
    plt.figure()
    x = np.arange(selected_k)
    avg_cluster_cost = [sum([item["cost"] for item in group])/len(group) for group in chosen[0]]
    plt.bar(x, avg_cluster_cost, 0.4, label = 'Average award', color='r')
    plt.title('Average award')
    plt.ylabel("Amount ($)")
    plt.xlabel("Year")
    plt.show()
    
    plt.figure()
    x = np.arange(selected_k)
    avg_cluster_funding = [sum([item["funding"] for item in group])/len(group) for group in chosen[0]]
    plt.bar(x, avg_cluster_funding, 0.4, label = 'Average institutional total for 2020', color='b')
    plt.xlabel("Year")
    plt.title('Average institutional total for 2020')
    plt.show()
    
    plt.figure()
    plt.scatter(avg_cluster_funding, avg_cluster_cost)
    plt.ylabel('Average award')
    plt.xlabel('Average institutional total for 2020')
    plt.title('Average award vs. Average institutional total for 2020')
    
    # Create grid that highlights each projection with 95% CI
    # https://stackoverflow.com/questions/39434402/how-to-get-confidence-intervals-from-curve-fit
    plt.figure()
    years_int = list(range(0,21))
    for i in range(len(chosen[6])):
        # fit = np.polyfit(years_int, np.log([j+0.00000001 for j in chosen[6][i]]), 1) #, w=np.sqrt(chosen[6][i]))
        popt, pcov = curve_fit(lambda t,a,b: a*np.exp(b*t),  years_int,  chosen[6][i],  p0=(4, 0.1))
        x = np.linspace(0,25,400)
        ypred = [popt[0]*np.exp(popt[1]*point)-popt[0] for point in x]
        plt.plot(x, ypred, label=str(i))
        plt.legend()
   