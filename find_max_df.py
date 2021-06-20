#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 19:06:13 2021

@author: Sope
"""
import pickle
from analyze_clusters import get_best_cluster
from feature_extraction import feature_extraction, process_data
from find_centroids import find_centroids
import matplotlib.pyplot as plt
import numpy as np

def find_k():
    results = []
    lower = []
    upper = []
    ks = list(range(10,61))
    # [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
    for i in ks:
        feature_extraction(data, 500, 0.5)
        data = pickle.load(open("data.pkl","rb"))
        test = pickle.load(open("test-data.pkl","rb"))
        centroids, score = find_centroids(data, test, 0.5, 500, i)
        scores = []
        for j in range(5):
            chosen = get_best_cluster(i, 1, centroids, years)
            scores.append(chosen["score"])
        std = np.std(scores)
        mean = np.mean(scores)
        vectorizer = pickle.load(open("vectorizer.pkl","rb"))
        terms = vectorizer.get_feature_names()
        
        results.append(mean)
        lower.append(mean-1.96*std)
        upper.append(mean+1.96*std)
        
        print("K: {}".format(str(i)))
        print("Max_df: {}".format(str(0.5)))
        print("Silhouette score: {}".format(str(chosen["score"])))
        print("Size: {}".format(str(max(chosen["size"]))))
        print("Features: {}".format(str(len(terms))))
        print("")
        
        order_centroids = chosen["complete_centroids"]
        vectorizer = pickle.load(open("vectorizer.pkl","rb"))
        terms = vectorizer.get_feature_names()
        centroids = []
        centroid_file = open("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/centroid", "w")
        for selected_k in range(0,i):
            centroid_file.write("Cluster %d:" % selected_k)
            centroid_list = []
            for ind in order_centroids[selected_k, :10]:
                centroid_file.write(" %s," % terms[ind])
                centroid_list.append(terms[ind])
            centroids.append(centroid_list)
            centroid_file.write("\n")
        centroid_file.close()
    
    plt.figure()
    plt.grid(b=None)
    plt.plot(ks, results)
    plt.fill_between(ks, upper, lower, alpha=0.1)
    plt.ylabel('Silhouette score')
    plt.xlabel('K')
    plt.savefig('k_selection.png')
    
def find_max_df(data, test):
    results = []
    lower = []
    upper = []
    k = 20
    max_dfs = [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
    for i in max_dfs:
        feature_extraction(data, i, 0.5)
        data = pickle.load(open("data.pkl","rb"))
        test = pickle.load(open("test-data.pkl","rb"))
        centroids, score = find_centroids(data, test, 0.5, i, k)
        scores = []
        for j in range(5):
            chosen = get_best_cluster(k, 1, centroids, years)
            scores.append(chosen["score"])
        std = np.std(scores)
        mean = np.mean(scores)
        vectorizer = pickle.load(open("vectorizer.pkl","rb"))
        terms = vectorizer.get_feature_names()
        
        results.append(mean)
        lower.append(mean-1.96*std)
        upper.append(mean+1.96*std)
        
        print("K: {}".format(str(k)))
        print("Max_df: {}".format(str(0.5)))
        print("Silhouette score: {}".format(str(chosen["score"])))
        print("Size: {}".format(str(max(chosen["size"]))))
        print("Features: {}".format(str(len(terms))))
        print("")
        
        order_centroids = chosen["complete_centroids"]
        vectorizer = pickle.load(open("vectorizer.pkl","rb"))
        terms = vectorizer.get_feature_names()
        centroids = []
        centroid_file = open("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/centroid", "w")
        for selected_k in range(0,k):
            centroid_file.write("Cluster %d:" % selected_k)
            centroid_list = []
            for ind in order_centroids[selected_k, :10]:
                centroid_file.write(" %s," % terms[ind])
                centroid_list.append(terms[ind])
            centroids.append(centroid_list)
            centroid_file.write("\n")
        centroid_file.close()
    
    plt.figure()
    plt.grid(b=None)
    plt.plot(max_dfs, results)
    plt.fill_between(max_dfs, upper, lower, alpha=0.1)
    plt.ylabel('Silhouette score')
    plt.xlabel('Number of features')
    plt.savefig('feature_selection.png')

file = '/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/raw data.csv'
funding_file = '/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/institution-funding.csv'
years = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
data, test = process_data(file, funding_file)
find_max_df(data, test)
