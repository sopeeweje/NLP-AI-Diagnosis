import pickle
from analyze_clusters import get_best_cluster
from feature_extraction import feature_extraction
from find_centroids import find_centroids
import matplotlib.pyplot as plt
import numpy as np
import argparse
import csv
    
def find_num_features(data, test, trials, k):
    results = []
    lower = []
    upper = []
    output = [["Feature", "Document Frequency", "Trial 1", "Trial 2", "Trial 3", "Trial 4", "Trial 5"]]
    print("Finding optimal number of features...")
    print("K: {}".format(str(k)))
    max_features = [500, 1000, 1500, 2000, 2500, 3000]
    max_df = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for i in max_features:
        for j in max_df:
            feature_extraction(data, i, j)
            centroids = find_centroids(data, test, j, i, k)
            years = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
            chosen, scores = get_best_cluster(k, trials, centroids, years, save=False)
            
            #print("Number of features: {}".format(str(i)))
            # print("Silhouette score: {}".format(str(chosen["score"])))
            # print("")
            
            std = np.std(scores)
            mean = np.mean(scores)
            vectorizer = pickle.load(open("vectorizer.pkl","rb"))
            # terms = vectorizer.get_feature_names()
            
            results.append(mean)
            lower.append(mean-1.96*std)
            upper.append(mean+1.96*std)
            new = [str(i), str(j)]
            new.extend(scores)
            output.append(new)
            print(new)
        
        # plt.figure()
        # plt.grid(b=None)
        # plt.plot(max_features, results)
        # plt.fill_between(max_features, upper, lower, alpha=0.1)
        # plt.ylabel('Silhouette score')
        # plt.xlabel('Number of features')
        # plt.savefig('feature_selection.png')
    return output

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
        '--k',
        type=int,
        required=True,
        help='number of clusters',
        default=60,
        )
    FLAGS, unparsed = parser.parse_known_args()
    trials = FLAGS.trials
    k = FLAGS.k
    data = pickle.load(open("data.pkl", "rb"))
    test = pickle.load(open("test-data.pkl", "rb"))
    
    output = find_num_features(data, test, trials, k)
    with open('find_features.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)
