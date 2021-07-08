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
from colorsys import hls_to_rgb
from pylab import *
from datetime import datetime
import os
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm
import argparse
import shutil

def rainbow_color_stops(n=10, end=1, shade=0.9):
    return [ hls_to_rgb(end * i/(n-1)*shade, 0.5*shade, 1*shade) for i in range(n) ]

def get_features():
    vector = pickle.load(open("vectorizer.pkl","rb"))
    centroid_file = open("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/features", "w")
    for i in vector.get_feature_names():
        centroid_file.write(i)
        centroid_file.write("\n")
    centroid_file.close()
    print(len(vector.get_feature_names()))

def get_clusters(selected_k, data_file, processed_file, centers, years, save_folder):
    # Load data as dictionary
    data = pickle.load(open(data_file,"rb"))
    
    # Transformed data
    X_transformed = pickle.load(open(processed_file,"rb"))
    
    # Perform k means # KMeans(n_clusters=selected_k)#
    km = MiniBatchKMeans(n_clusters=selected_k, init=centers, n_init=10, init_size=3000, batch_size=3000, verbose=0, max_no_improvement=None)
    # km = KMeans(n_clusters=selected_k, init=centers, n_init=10)
    clusters = km.fit_predict(X_transformed)
    scores = metrics.silhouette_samples(X_transformed, clusters)
    
    # Output data
    cluster_all = []
    costs = []
    yoy = []
    size = []
    
    for i in range(0,selected_k):
        
        # indices of cluster k
        cluster = [idx for idx, element in enumerate(clusters) if element == i]
        
        # get points
        cluster_data = [data[ind] for ind in cluster]
        cluster_scores = [scores[ind] for ind in cluster]
        for i in range(len(cluster_data)):
            cluster_data[i]["score"] = cluster_scores[i]
        cluster_all.append(cluster_data)
        
        # get scores
        
        # calculate average cost and std
        try:
            average_cost = sum([item["cost"] for item in cluster_data])/len(cluster_data)
        except:
            average_cost = 0
        costs.append(average_cost)
        
        cost_trend = []
        for year in years:
            year_data = [data[ind]["cost"] for ind in cluster if data[ind]["year"] == year]
            if len(year_data) == 0:
                cost_trend.append(0)
            else:
                year_cost = sum(year_data) # /len(year_data)
                cost_trend.append(year_cost)
        
        yoy.append(cost_trend)
        
        size.append(len(cluster))
    
    # Get centroids 
    # Identify the top terms for each cluster, using the TF-IDF terms with the highest values in the centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    vectorizer = pickle.load(open("vectorizer.pkl","rb"))
    terms = vectorizer.get_feature_names()
    centroids = []
    centroid_file = open("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/{}/centroid".format(save_folder), "w")
    for i in range(selected_k):
        centroid_file.write("Cluster %d:" % i)
        centroid_list = []
        for ind in order_centroids[i, :10]:
            centroid_file.write(" %s" % terms[ind])
            centroid_list.append(terms[ind])
        centroids.append(centroid_list)
        centroid_file.write("\n")
    centroid_file.close()
    
    score = metrics.silhouette_score(X_transformed, km.labels_)
    
    output = {
        "yr_avg_cost": costs,
        "yr_total_cost": yoy,
        "size": size,
        "data_by_cluster": cluster_all,
        "centroids": centroids,
        "score": score,
        "model": km,
        "complete_centroids": order_centroids,
        "labels": clusters
        }
    return output

def umap_visualization(X_transformed, cluster_labels, save_folder):
    outlier_scores = sklearn.neighbors.LocalOutlierFactor(contamination=0.1).fit_predict(X_transformed)
    X_transformed = X_transformed[outlier_scores != -1]
    cluster_labels = cluster_labels[outlier_scores != -1]
    
    n_subset = len(cluster_labels)
    selected_cells = np.random.choice(np.arange(X_transformed.shape[0]), size = n_subset, replace = False)
    mapper = umap.UMAP(metric='hellinger', random_state=42).fit(X_transformed[selected_cells,:])
    
    embedding = mapper.transform(X_transformed[selected_cells,:])
    
    # Plot Clusters on UMAP
    plt.figure()
    plt.grid(b=None)
    plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=5, c=cluster_labels[selected_cells])
    plt.gca().set_aspect('equal', 'datalim')
    num_clust = len(np.unique(cluster_labels[selected_cells]))
    plt.colorbar(boundaries=np.arange(num_clust+1)-0.5).set_ticks(np.arange(num_clust))
    plt.title('UMAP Projection of Awards, TF-IDF', fontsize=14)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig('{}/umap.png'.format(save_folder))

def graph_funding_projections(data, save_folder):
    # Create grid that highlights each projection with 95% CI
    # https://stackoverflow.com/questions/39434402/how-to-get-confidence-intervals-from-curve-fit
    
    # 1. Determine dimensions for plot
    k = len(data["size"])
    factors = []
    for i in range(1, k+1):
        if k / i == i:
            factors.extend([i,i])
        elif k % i == 0:
            factors.append(i)
    dim1, dim2 = factors[int(len(factors)/2)], factors[int(len(factors)/2-1)]
    
    # 2. Create plot
    fig, axs = plt.subplots(dim1, dim2, sharex='all', sharey='all')
    
    # 3. Create hidden frame for shared labels
    fig.add_subplot(111, frameon=False)
    plt.grid(b=None)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Years from 2000")
    plt.ylabel("Funding ($100 millions)")
    
    # 4. Plot each projection with scatter plot
    colors = cm.get_cmap('Spectral', k) #rainbow_color_stops(k)
    years_int = list(range(0,21))
    m = np.repeat(list(range(dim1)), dim2)
    n = np.tile(list(range(dim2)), dim1)
    maxy = 0
    projection = []
    growth = []
    bounds = []
    for i in range(len(data["yr_total_cost"])):
        popt, pcov = curve_fit(lambda t,a,b: a*np.exp(b*t),  years_int,  data["yr_total_cost"][i],  p0=(4000, 0.1))
        std = np.sqrt(np.diagonal(pcov))
        x = np.linspace(0,21,400)
        upper0 = popt[0]+1.96*std[0]
        lower0 = popt[0]-1.96*std[0]
        upper1 = popt[1]+1.96*std[1]
        lower1 = popt[1]-1.96*std[1]
        
        ypred = [popt[0]*np.exp(popt[1]*point) for point in x] #-popt[0]
        projection.append(ypred[-1])
        growth.append(popt[1])
        bounds.append([lower1, upper1])
        maxy = max([max(ypred), maxy])
        upper = [upper0*np.exp(upper1*point) for point in x]
        lower = [lower0*np.exp(lower1*point) for point in x]
        #color = matplotlib.colors.rgb2hex(rgba)
        axs[m[i],n[i]].set_title("Cluster {}".format(str(i)), size=10, weight='bold', position=(0.5, 0.7))
        axs[m[i],n[i]].plot(x, ypred, color=colors(i))
        axs[m[i],n[i]].fill_between(x, upper, lower, color=colors(i), alpha=0.1)
        axs[m[i],n[i]].scatter(years_int, data["yr_total_cost"][i], s=20, color=colors(i))
        axs[m[i],n[i]].set_ylim(-100000,maxy+100000)
        axs[m[i],n[i]].set_xlim(0,21.00001)
        axs[m[i],n[i]].grid(False)
    
    plt.savefig('{}/funding_by_year.png'.format(save_folder))
    
    # 5. Return 2021 projections and growth rate
    return projection, growth, bounds
    
def viz_centroids(data):
    order_centroids = data["model"].cluster_centers_.argsort()[:, ::-1]
    centroid_file = open("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/centroid", "w")
    vectorizer = pickle.load(open("vectorizer.pkl","rb"))
    terms = vectorizer.get_feature_names()
    for i in range(len(data["size"])):
        centroid_file.write("Cluster %d:" % i)
        for ind in order_centroids[i, :20]:
            centroid_file.write(" %s" % terms[ind])
        centroid_file.write("\n")
    centroid_file.close()
        
    model = data["model"]
    X_transformed = pickle.load(open("processed-data.pkl","rb"))
    plt.figure()
    visualizer = InterclusterDistance(model, random_state=0)
    visualizer.fit(X_transformed)     # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure

def predict_clusters(test_data, selected_k):
    test_data = pickle.load(open(test_data,"rb"))
    model = pickle.load(open("model.pkl","rb"))
    vectorizer = pickle.load(open("vectorizer.pkl","rb"))
    input_text = [item["text"] for item in test_data]
    test_transformed = vectorizer.transform(input_text)
    
    labels = model.predict(test_transformed)
    
    # Output data
    cluster_all = []
    costs = []
    yoy = []
    size = []
    
    for i in range(0,selected_k):
        
        # indices of cluster k
        cluster = [idx for idx, element in enumerate(labels) if element == i]
        
        # get points
        cluster_data = [test_data[ind] for ind in cluster]
        cluster_all.append(cluster_data)
        
        # calculate average cost and std
        try:
            average_cost = sum([item["cost"] for item in cluster_data])/len(cluster_data)
        except:
            average_cost = 0
        costs.append(average_cost)
        
        cost_trend = []
        for year in years:
            year_data = [test_data[ind]["cost"] for ind in cluster if test_data[ind]["year"] == year]
            if len(year_data) == 0:
                cost_trend.append(0)
            else:
                year_cost = sum(year_data) #/len(year_data)
                cost_trend.append(year_cost)
        
        yoy.append(cost_trend)
        
        size.append(len(cluster))
        
    return cluster_all, size  

def get_best_cluster(selected_k, num_trials, centers, years, save_folder):
    """

    Parameters
    ----------
    selected_k : TYPE
        DESCRIPTION.
    num_trials : TYPE
        DESCRIPTION.
    centers : TYPE
        DESCRIPTION.
    years : TYPE
        DESCRIPTION.

    Returns
    -------
    chosen : TYPE
        DESCRIPTION.

    """
    scores = []
    results = {}
    print("Optimizing model...")
    for i in range(num_trials):
        # Generate clusters for a selected k
        data = get_clusters(selected_k, "data.pkl", "processed-data.pkl", 'k-means++', years, save_folder)
        j = 0
        for thing in data["data_by_cluster"]:
            for item in thing:
                try:
                    results[item["id"]].append(centroids[j])
                except:
                    results[item["id"]] = [item["id"],item["title"],item["cost"],data["centroids"][j]]
            j+=1
        print("Trial {}: Score = {:.3f}".format(str(i+1), data["score"]))
        scores.append(data["score"])
        if data["score"] >= max(scores):
            chosen = data
    
    return chosen

def get_citations(clusters):
    """
    
    Parameters
    ----------
    clusters : nested lists of dictionaries representing each award in a cluster.

    Returns
    -------
    total_citations : list of total citations by cluster
    total_papers : list of total papers by cluster

    """
    
    # Get clusters by project number
    clusters_by_project = []
    for cluster in clusters:
        cluster = [item["project_number"] for item in cluster]
        cluster = list(set(cluster)) # Remove duplicates
        clusters_by_project.append(cluster)
    
    # Get number of citations by paper
    output = {}
    with open("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/citations.csv", newline='') as csvfile:
       raw_data = list(csv.reader(csvfile))
       for i in range(1,len(raw_data)):
           output[raw_data[i][0]] = {"citations": raw_data[i][6]}
    
    # Get project number by paper
    with open("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/papers.csv", newline='') as csvfile:
       raw_data = list(csv.reader(csvfile))
       for i in range(1,len(raw_data)):
           if raw_data[i][13] in output.keys():
               output[raw_data[i][13]]["project"] = raw_data[i][0]
    
    # Calculate total number of citations and papers for each cluster
    total_citations = []
    total_papers = []
    for cluster in clusters_by_project:
        cluster_citations = []
        num_papers = 0
        for idd in cluster:
            papers = [int(output[key]["citations"]) for key in output if output[key]["project"]==idd]
            num_papers += len(papers)
            cluster_citations.append(sum(papers))
        total_citations.append(sum(cluster_citations))        
        total_papers.append(num_papers)

    return total_citations, total_papers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--k',
        type=int,
        required=True,
        help='number of clusters',
        default=30,
        )
    parser.add_argument(
        '--trials',
        type=int,
        required=True,
        help='number of trials',
        default=50,
        )
    FLAGS, unparsed = parser.parse_known_args()
    
    
    years = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
    scores = []
    selected_k = FLAGS.k
    num_trials = FLAGS.trials
    centers = pickle.load(open("lda_centroids.pkl","rb"))
    
    # Create folder to save results
    now = datetime.now()
    if not os.path.exists("results"):
        os.makedirs("results")
    save_folder = "results/"+now.strftime("%m-%d-%Y--%H%M%S")
    os.mkdir(save_folder)
    
    # Move LDA centroids and topic chart to results folder
    shutil.move("lda_centroids.pkl", "{}/lda_centroids.pkl".format(save_folder)) 
    shutil.move("topic_chart.png", "{}/topic_chart.png".format(save_folder)) 
    
    # Get best clustering
    data = get_best_cluster(selected_k, num_trials, centers, years, save_folder)
    with open("{}/model_clustering.pkl".format(save_folder), 'wb') as handle:
        pickle.dump(data, handle)
    
    # Final cluster files
    num = 0
    os.mkdir(save_folder+"/clusters")
    for cluster in data["data_by_cluster"]:
        keys = cluster[0].keys()
        with open('{}/clusters/cluster-{}.csv'.format(save_folder,str(num)), 'w', newline='')  as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(cluster)
        num+=1
    
    # Silhouette score by cluster
    print("")
    print("------Silhouette scores------")
    X_transformed = pickle.load(open("processed-data.pkl","rb"))
    scores = metrics.silhouette_samples(X_transformed, data["labels"])
    tabulated = []
    pairs = [(scores[i],data["labels"][i]) for i in range(len(scores))]
    for i in range(selected_k):
        avg_score = np.mean([j[0] for j in pairs if j[1] == i])
        print("Cluster {}: {}".format(str(i), str(avg_score)))
        tabulated.append(avg_score)
    print("----------------------------")
    print("")
    
    # Final centroids
    order_centroids = data["model"].cluster_centers_.argsort()[:, ::-1]
    vectorizer = pickle.load(open("vectorizer.pkl","rb"))
    terms = vectorizer.get_feature_names()
    centroids = []
    centroid_file = open("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/{}/centroid".format(save_folder), "w")
    for i in range(selected_k):
        centroid_file.write("Cluster %d:" % i)
        centroid_list = []
        for ind in order_centroids[i, :10]:
            centroid_file.write(" %s," % terms[ind])
            centroid_list.append(terms[ind])
        centroids.append(centroid_list)
        centroid_file.write("\n")
    centroid_file.close()
        
    # UMAP Visualization
    X_transformed = pickle.load(open("processed-data.pkl","rb"))
    umap_visualization(X_transformed, data["labels"], save_folder)
    
    # Save model
    with open("model.pkl", 'wb') as handle:
        pickle.dump(data["model"], handle)
    
    # Projected funding by year
    projection, growth, bounds = graph_funding_projections(data, save_folder) # 2021 prediction
    
    # 2021 total awards by predicted cluster
    clusters_test, size_test = predict_clusters("test-data.pkl", selected_k)
    x = np.arange(selected_k)
    cluster_cost_2021 = [(sum([item["cost"] for item in group]) if len(group) > 0 else 0) for group in clusters_test]
    
    # Perform linear regression
    y = cluster_cost_2021
    x = projection
    X = sm.add_constant(x)
    
    re = sm.OLS(y, X).fit()
    print("2021 projected vs. actual R2: {:.3f}".format(re.rsquared))
    
    prstd, iv_l, iv_u = wls_prediction_std(re)

    st, reg_data, ss2 = summary_table(re, alpha=0.05)
    
    predicted = reg_data[:, 2]
    predict_mean_se  = reg_data[:, 3]
    predict_mean_ci_low, predict_mean_ci_upp = reg_data[:, 4:6].T
    predict_ci_low, predict_ci_upp = reg_data[:, 6:8].T
        
    # Actual vs. projected awards
    fig, ax = plt.subplots()
    ax.grid(b=None)
    colors = cm.get_cmap('Spectral', selected_k)
    colors = [colors(i) for i in range(selected_k)]
    ax.scatter(projection, cluster_cost_2021, c=colors)
    ax.plot([projection[np.argmin(projection)],projection[np.argmax(projection)]], [predicted[np.argmin(projection)],predicted[np.argmax(projection)]], color="#808080")
    ax.plot(sort(x), sort(predict_mean_ci_low), 'r--', lw=2)
    ax.plot(sort(x), sort(predict_mean_ci_upp), 'r--', lw=2)
    axin1 = ax.inset_axes([0.05, 0.5, 0.4, 0.4])
    pairs = [[projection[i], cluster_cost_2021[i], i] for i in range(len(projection))]
    pairs = [item for item in pairs if item[1] < 1.5e7]
    pairs = list(map(list, zip(*pairs))) 
    x1, x2, y1, y2 = 0, 0.2e8, 0, 1.5e7
    axin1.set_xlim(x1, x2)
    axin1.set_ylim(y1, y2)
    axin1.set_xticklabels('')
    axin1.set_yticklabels('')
    axin1.grid(b=None)
    axin1.scatter(pairs[0], pairs[1], c=[colors[i] for i in pairs[2]])
    ax.indicate_inset_zoom(axin1, edgecolor="black")
    superscript = str.maketrans("2", "²")
    ax.text(6e7, 0.1, "r2 = {:.3f}".format(re.rsquared).translate(superscript))
    ax.set_ylabel('Actual 2021 award to date ($10 millions)')
    ax.set_xlabel('Projected 2021 award ($100 millions)')
    plt.savefig('{}/actual_vs_projected.png'.format(save_folder))
    
    # Save 2021 clusters
    num = 0
    os.mkdir("{}/clusters_test".format(save_folder))
    for cluster in clusters_test:
        try:
            keys = cluster[0].keys()
        except:
            num+=1
            continue
        with open('{}/clusters_test/cluster-{}.csv'.format(save_folder,str(num)), 'w', newline='')  as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(cluster)
        num+=1
    
    # Citations and papers
    citations, papers = get_citations(data["data_by_cluster"])
    
    # Total funding
    total_cluster_funding = [sum([item["cost"] for item in group]) for group in data["data_by_cluster"]]
    
    # All data
    output = [["Cluster", "Size", "Total", "Citations", "Papers", "Citations per $1mil funding", "Projected 2021 Award", "Actual 2021 Award To Date", "Growth Rate", "Bounds", "Bounds", "Score", "Centroids"]]
    for i in range(selected_k):
        output.append([i, data["size"][i], total_cluster_funding[i], citations[i], papers[i], citations[i]/total_cluster_funding[i]*1e6, projection[i], cluster_cost_2021[i], growth[i], bounds[i][0], bounds[i][1], tabulated[i], centroids[i]])
    with open('{}/final_data.csv'.format(save_folder), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)
    
    print("Complete.")