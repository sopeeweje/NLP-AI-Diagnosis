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
import statistics
from scipy import stats
from sklearn.linear_model import LinearRegression
from progress.bar import ChargingBar
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import seaborn as sns
from pylab import *

# https://github.com/danielmlow/reddit/blob/master/Unsupervised_Clustering_Pipeline.ipynb

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

def get_clusters(selected_k, data_file, processed_file, centers, years):
    # Load data as dictionary
    data = pickle.load(open(data_file,"rb"))
    
    # Transformed data
    X_transformed = pickle.load(open(processed_file,"rb"))
    
    # Perform k means # KMeans(n_clusters=selected_k)#
    km = MiniBatchKMeans(n_clusters=selected_k, init=centers, n_init=10, init_size=3000, batch_size=3000, verbose=0, max_no_improvement=None)
    # km = KMeans(n_clusters=selected_k, init=centers, n_init=10)
    clusters = km.fit_predict(X_transformed)
    
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
        cluster_all.append(cluster_data)
        
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

def umap_visualization(X_transformed, cluster_labels):
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
    plt.savefig('umap.png')

def graph_funding_projections(data):
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
    
    plt.savefig('funding_by_year.png')
    
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

def get_best_cluster(selected_k, num_trials, centers, years):
    scores = []
    results = {}
    for i in range(num_trials):
        # Generate clusters for a selected k
        data = get_clusters(selected_k, "data.pkl", "processed-data.pkl", 'k-means++', years)
        j = 0
        print(i)
        for thing in data["data_by_cluster"]:
            for item in thing:
                try:
                    results[item["id"]].append(centroids[j])
                except:
                    results[item["id"]] = [item["id"],item["title"],item["cost"],data["centroids"][j]]
            j+=1
        scores.append(data["score"])
        if data["score"] >= max(scores):
            chosen = data
    
    return chosen

if __name__ == "__main__":
    
    years = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
    scores = []
    selected_k = 30
    num_trials = 50
    centers = pickle.load(open("lda_centroids.pkl","rb"))
    #centers = 'k-means++'
    
    # Get best clustering
    data = get_best_cluster(selected_k, num_trials, centers, years)
    
    # Results across trials 
    # resultscsv = []
    # for key in results:
    #     resultscsv.append(results[key])
    # with open('results.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(resultscsv)
    
    # Final cluster files
    num = 0
    for cluster in data["data_by_cluster"]:
        keys = cluster[0].keys()
        with open('clusters/cluster-{}.csv'.format(str(num)), 'w', newline='')  as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(cluster)
        num+=1
    
    # Silhouette score by sample
    X_transformed = pickle.load(open("processed-data.pkl","rb"))
    scores = metrics.silhouette_samples(X_transformed, data["labels"])
    tabulated = []
    pairs = [(scores[i],data["labels"][i]) for i in range(len(scores))]
    for i in range(selected_k):
        avg_score = np.mean([j[0] for j in pairs if j[1] == i])
        print("Cluster {}: {}".format(str(i), str(avg_score)))
        tabulated.append(avg_score)
    
    # Final centroids
    order_centroids = data["model"].cluster_centers_.argsort()[:, ::-1]
    vectorizer = pickle.load(open("vectorizer.pkl","rb"))
    terms = vectorizer.get_feature_names()
    centroids = []
    centroid_file = open("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/centroid", "w")
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
    umap_visualization(X_transformed, data["labels"])
    
    # Save model
    with open("model.pkl", 'wb') as handle:
        pickle.dump(data["model"], handle)
    
    # Projected funding by year
    projection, growth, bounds = graph_funding_projections(data) # 2021 prediction
    
    # 2021 total awards by predicted cluster
    clusters_test, size_test = predict_clusters("test-data.pkl", selected_k)
    x = np.arange(selected_k)
    cluster_cost_2021 = [(sum([item["cost"] for item in group]) if len(group) > 0 else 0) for group in clusters_test]
    
    # Linear regression for actual vs. projected
    reg = LinearRegression().fit(np.array(projection).reshape(-1,1), np.array(cluster_cost_2021).reshape(-1,1))
    print(reg.score(np.array(projection).reshape(-1,1), np.array(cluster_cost_2021).reshape(-1,1)))
    predicted = reg.predict(np.array(projection).reshape(-1,1))
    
    # Actual vs. projected awards
    # plt.figure()
    fig, ax = plt.subplots()
    ax.grid(b=None)
    colors = cm.get_cmap('Spectral', k)
    ax.scatter(projection, cluster_cost_2021)
    ax.plot(projection, predicted)
    axin1 = ax.inset_axes([0+0.5e8, 0.2e8+0.5e8, 0.5e8, 1e7])
    ax.ylabel('Actual 2021 award to date ($)')
    ax.xlabel('Projected 2021 award ($)')
    ax.title('Actual vs. Projected 2021 Award')
    ax.savefig('actual_vs_projected.png')
    plt.show()
    
    # Save 2021 clusters
    num = 0
    for cluster in clusters_test:
        try:
            keys = cluster[0].keys()
        except:
            num+=1
            continue
        with open('clusters_test/cluster-{}.csv'.format(str(num)), 'w', newline='')  as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(cluster)
        num+=1
    
    # Average award size by cluster
    plt.figure()
    plt.grid(b=None)
    x = np.arange(selected_k)
    cost_data = []
    for group in data["data_by_cluster"]:
        cost_data.append([item["cost"] for item in group])
    avg_cluster_cost = [sum([item["cost"] for item in group])/len(group) for group in data["data_by_cluster"]] # actual award
    F, p = stats.f_oneway(*cost_data)
    print("Average award size: {}".format(str(p)))
    
    #stds_cost = [statistics.pstdev([item["cost"] for item in group]) for group in chosen[0]]
    sem_cost = [stats.sem([item["cost"] for item in group]) for group in data["data_by_cluster"]]
    ci95_cost = [list(stats.norm.interval(0.95, loc=avg_cluster_cost[i], scale=sem_cost[i])) for i in range(len(sem_cost))]
    ci95_cost = list(map(list, zip(*ci95_cost)))
    ci95_cost = [[abs(ci95_cost[i][j]-avg_cluster_cost[j]) for j in range(len(avg_cluster_cost))] for i in range(2)]
    plt.bar(x, avg_cluster_cost, yerr=ci95_cost, width=0.4, label = 'Average award', color='r')
    plt.title('Average award')
    plt.ylabel("Amount ($)")
    plt.xlabel("Cluster")
    plt.xticks(np.arange(0, selected_k, step=1))
    plt.savefig('avg_award.png')
    plt.show()
    
    #create DataFrame to hold data
    df = pd.DataFrame({'score': [item["cost"] for item in pickle.load(open("data.pkl","rb"))],
                       'group': data["labels"]}) 
    
    # perform Tukey's test
    tukey = pairwise_tukeyhsd(endog=df['score'],
                              groups=df['group'],
                              alpha=0.05)
    df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    df.to_csv('anova_award.csv', index=False, header=True)
    piv = pd.pivot_table(df, values="p-adj",index=["group1"], columns=["group2"], fill_value=0)
    plt.figure()
    plt.title("Average award size (ANOVA)")
    ax = sns.heatmap(piv, vmax=0.05)
    plt.savefig('avg_award_ANOVA.png')
    
    # Average institutional funding
    plt.figure()
    plt.grid(b=None)
    x = np.arange(selected_k)
    funding_data = []
    for group in data["data_by_cluster"]:
        funding_data.append([item["funding"] for item in group])
    F, p = stats.f_oneway(*funding_data)
    print("Average institutional funding: {}".format(str(p)))
    avg_cluster_funding = [sum([item["funding"] for item in group])/len(group) for group in data["data_by_cluster"]]
    #stds_funding = [statistics.pstdev([item["funding"] for item in group]) for group in chosen[0]]
    sem_funding = [stats.sem([item["funding"] for item in group]) for group in data["data_by_cluster"]]
    ci95_funding = [list(stats.norm.interval(0.95, loc=avg_cluster_funding[i], scale=sem_funding[i])) for i in range(len(sem_funding))]
    ci95_funding = list(map(list, zip(*ci95_funding)))
    ci95_funding = [[abs(ci95_funding[i][j]-avg_cluster_funding[j]) for j in range(len(avg_cluster_funding))] for i in range(2)]
    plt.bar(x, avg_cluster_funding, yerr=ci95_funding, width=0.4, label = 'Average institutional total for 2020', color='b')
    plt.xlabel("Cluster")
    plt.xticks(np.arange(0, selected_k, step=1))
    plt.ylabel("Amount ($)")
    plt.title('Average institutional total for 2020')
    plt.savefig('avg_institutional.png')
    plt.show()
    
    #create DataFrame to hold data
    df = pd.DataFrame({'score': [item["funding"] for item in pickle.load(open("data.pkl","rb"))],
                       'group': data["labels"]}) 
    
    # perform Tukey's test
    tukey = pairwise_tukeyhsd(endog=df['score'],
                              groups=df['group'],
                              alpha=0.05)
    df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    df.to_csv('anova_institutional.csv', index=False, header=True)
    piv = pd.pivot_table(df, values="p-adj",index=["group1"], columns=["group2"], fill_value=0)
    plt.figure()
    plt.title("Average institutional total for 2020 (ANOVA)")
    ax = sns.heatmap(piv, vmax=0.05)
    plt.savefig('avg_institutional_ANOVA.png')
    
    # Total award
    plt.figure()
    plt.grid(b=None)
    x = np.arange(selected_k)
    total_cluster_funding = [sum([item["cost"] for item in group]) for group in data["data_by_cluster"]]
    plt.bar(x, total_cluster_funding, label = 'Total awards', color='b')
    plt.xlabel("Cluster")
    plt.xticks(np.arange(0, selected_k, step=1))
    plt.ylabel("Amount ($)")
    plt.title('Total award')
    plt.savefig('total_award.png')
    plt.show()
    
    # All data
    output = [["Cluster", "Size", "Total", "Average award (2000-2020)", "SD", "Average institutional award (2020)", "SD", "Projected 2021 Award", "Actual 2021 Award To Date", "Growth Rate", "Bounds", "Bounds", "Score", "Centroids"]]
    for i in range(len(avg_cluster_cost)):
        output.append([i, data["size"][i], data["size"][i]*avg_cluster_cost[i], avg_cluster_cost[i], ci95_cost[0][i], avg_cluster_funding[i], ci95_funding[0][i], projection[i], cluster_cost_2021[i], growth[i], bounds[i][0], bounds[i][1], tabulated[i], centroids[i]])
    with open('final_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)
            
# from pylab import *
# cmap = cm.get_cmap('Spectral', 12)    # PiYG
# for i in range(cmap.N):
#     rgba = cmap(i)
#     # rgb2hex accepts rgb or rgba
#     print(matplotlib.colors.rgb2hex(rgba))