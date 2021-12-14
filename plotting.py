#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 18:52:14 2021

@author: Sope
"""
import pickle
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import summary_table
import warnings
from feature_extraction import LemmaStemmerTokenizer
import scipy.stats as scist
import math
import sys
import random
import os
import matplotlib.image as mpimg
import pandas as pd
import matplotlib.gridspec as gridspec
from operator import itemgetter
from analyze_clusters import get_citations
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.multicomp import pairwise_tukeyhsd
csv.field_size_limit(sys.maxsize)

def create_table_1(data):
    # Get total awards for 1985, 2020, overall
    awards_1985 = sum([int(data[i]["year"]) <= 2000 for i in range(len(data))])
    awards_2020 = sum([int(data[i]["year"]) > 2000 for i in range(len(data))])
    awards_overall = len(data)
    
    # Get total funding for 1985, 2020, overall
    funding_1985 = sum([data[i]["funding"] for i in range(len(data)) if int(data[i]["year"]) <= 2008])
    funding_2020 = sum([data[i]["funding"] for i in range(len(data)) if int(data[i]["year"]) > 2008])
    funding_overall = sum([data[i]["funding"] for i in range(len(data))])
    
    # Get text from 1985 and 2020
    text_1985 = ""
    text_2020 = ""
    text_all = ""
    for award in range(len(data)):
        if int(data[award]["year"]) <= 2008:
            text_1985 += data[award]["text"]
        else:
            text_2020 += data[award]["text"]
        text_all += data[award]["text"]
    
    tokenizer = LemmaStemmerTokenizer()
    text_1985 = tokenizer(text_1985)
    text_2020 = tokenizer(text_2020)
    text_all = tokenizer(text_all)
    print(len(text_all))
    
    # Calculate log likelihood ratios
    vector = pickle.load(open("vectorizer.pkl","rb"))
    features = vector.get_feature_names()
    word_map = {}
    for word in features:
        print(word)
        prob_1985 = sum([i == word for i in text_1985])
        prob_2020 = sum([i == word for i in text_2020])
        prob_all = sum([i == word for i in text_all])
        try:
            llr_1985 = 2*(math.log10(prob_1985) - math.log10(prob_all))
        except:
            llr_1985 = -100000
        try:
            llr_2020 = 2*(math.log10(prob_2020) - math.log10(prob_all))
        except:
            llr_2020 = -100000
        word_map[word] = {"2020": llr_2020, "1985": llr_1985}
    
    # Get the most enriched features
    enriched_1985 = sorted(word_map, key = lambda x: word_map[x]["1985"], reverse=True)[0:10]
    enriched_2020 = sorted(word_map, key = lambda x: word_map[x]["2020"], reverse=True)[0:10]    
    
    # Get number of citations, apt, and publication year by paper
    output = {}
    with open("citations.csv", newline='', encoding='utf8') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)): # "rcr": float(raw_data[i][6]),
            output[raw_data[i][0]] = {"citations": float(raw_data[i][14]) if raw_data[i][14]!='' else 0, "apt": float(raw_data[i][11])}

    # Get project number and year by paper
    with open("papers.csv", newline='', encoding='utf8') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)):
            if raw_data[i][13] in output.keys():
                output[raw_data[i][13]]["project"] = raw_data[i][0]
                output[raw_data[i][13]]["year"] = int(raw_data[i][2])
               
    # Get clusters by project number
    projects_1985 = []
    projects_2020 = []
    projects_all = []
    
    for award in data:
        if int(award["year"]) <= 2008:
            projects_1985.append(award["project_number"])
        else:
            projects_2020.append(award["project_number"])
        projects_all.append(award["project_number"])
    
    # Get citation data by year for 1985
    citations_1985 = 0
    apt_1985 = []
    papers_1985 = 0
    # availability_1985 = 0
    for idd in projects_1985:
        papers = [output[key]["citations"] for key in output if output[key]["project"]==idd] # list of all papers associated with cluster by citation count
        apt = [output[key]["apt"] for key in output if output[key]["project"]==idd]
        # availability_1985 += sum([max(0, 2021-output[key]["year"]) for key in output if output[key]["project"]==idd])
        
        apt_1985.extend(apt)
        papers_1985 += len(papers)
        citations_1985 += sum(papers)
            
    apt_interval_1985 = scist.norm.interval(alpha=0.95, loc=np.mean(apt_1985), scale=scist.sem(apt_1985))
    avg_apt_1985_str = "{} ({} - {})".format(str(np.mean(apt_1985)), str(apt_interval_1985[0]), str(apt_interval_1985[1]))
    cit_1985_str = citations_1985/funding_1985*1e6
    
    # Get citation data by year for 2020
    citations_2020 = 0
    apt_2020 = []
    papers_2020 = 0
    # availability_2020 = 0
    for idd in projects_2020:
        papers = [output[key]["citations"] for key in output if output[key]["project"]==idd] # list of all papers associated with cluster by citation count
        apt = [output[key]["apt"] for key in output if output[key]["project"]==idd]
        # availability_2020 += sum([max(0, 2021-output[key]["year"]) for key in output if output[key]["project"]==idd])
        
        apt_2020.extend(apt)
        papers_2020 += len(papers)
        citations_2020 += sum(papers)
            
    apt_interval_2020 = scist.norm.interval(alpha=0.95, loc=np.mean(apt_2020), scale=scist.sem(apt_2020))
    avg_apt_2020_str = "{} ({} - {})".format(str(np.mean(apt_2020)), str(apt_interval_2020[0]), str(apt_interval_2020[1]))
    cit_2020_str = citations_2020/funding_2020*1e6
    
    # Get citation data by overall
    citations_all = 0
    apt_all = []
    papers_all = 0
    # availability_all = 0
    for idd in projects_all:
        papers = [output[key]["citations"] for key in output if output[key]["project"]==idd] # list of all papers associated with cluster by citation count
        apt = [output[key]["apt"] for key in output if output[key]["project"]==idd]
        # availability_all += sum([max(0, 2021-output[key]["year"]) for key in output if output[key]["project"]==idd])
        
        apt_all.extend(apt)
        papers_all += len(papers)
        citations_all += sum(papers)
            
    apt_interval_all = scist.norm.interval(alpha=0.95, loc=np.mean(apt_all), scale=scist.sem(apt_all))
    avg_apt_all_str = "{} ({} - {})".format(str(np.mean(apt_all)), str(apt_interval_all[0]), str(apt_interval_all[1]))
    cit_all_str = citations_all/funding_overall*1e6
    
    # Final data
    data_to_csv = [['', '2008 and before', 'After 2008', 'Overall', 'p-value'],
                   ['Number of Awards', str(awards_1985), str(awards_2020), str(awards_overall), ''],
                   ['Total Funding', str(funding_1985), str(funding_2020), str(funding_overall), ''],
                   ['Number of Papers', str(papers_1985), str(papers_2020), str(papers_all), ''],
                   ['Citations Per Year of Availability', cit_1985_str, cit_2020_str, cit_all_str, ''],
                   ['Average APT score', avg_apt_1985_str, avg_apt_2020_str, avg_apt_all_str, str(scist.ttest_ind(apt_1985, apt_2020)[1])],
                   ['Enriched features', enriched_1985, enriched_2020, '', '']]
    
    # Write to CSV
    with open('table1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data_to_csv)
        
    return

def create_table_2_app(results_directory):
    years = list(range(1985,2021))
    
    # Load result
    result = pickle.load(open(f'{results_directory}model_clustering.pkl',"rb"))

    clusters = result["data_by_cluster"]
    annual_costs = result["yr_total_cost"]
    
    # Get labels by categories
    applications = {}
    with open(f'{results_directory}/final_data.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)):
            if (raw_data[i][18] != "N/A"):
                applications[raw_data[i][0]] = {
                    "description": raw_data[i][17], 
                    "category": raw_data[i][18], 
                    "data": clusters[int(raw_data[i][0])], 
                    "funding": int(raw_data[i][2]), 
                    "size": int(raw_data[i][1]),
                    "centroids": raw_data[i][20],
                    "yr_total_cost": annual_costs[int(raw_data[i][0])],
                    "score": float(raw_data[i][16])}
    
    # Get clusters by project number
    for cluster in applications:
        project_numbers = [item["project_number"] for item in applications[cluster]["data"]]
        # project_numbers = list(set(project_numbers)) # Remove duplicates
        applications[cluster]["projects"] = project_numbers

    # Get number of citations, apt, and publication year by paper
    output = {}
    with open("citations.csv", newline='', encoding='utf8') as csvfile:
       raw_data = list(csv.reader(csvfile))
       for i in range(1,len(raw_data)): # "rcr": float(raw_data[i][6]),
           output[raw_data[i][0]] = {"citations": float(raw_data[i][14]) if raw_data[i][14]!="" else 0, "apt": float(raw_data[i][11])}

    # Get project number and year by paper
    with open("papers.csv", newline='', encoding='utf8') as csvfile:
       raw_data = list(csv.reader(csvfile))
       for i in range(1,len(raw_data)):
           if raw_data[i][13] in output.keys():
               output[raw_data[i][13]]["project"] = raw_data[i][0]
               output[raw_data[i][13]]["year"] = int(raw_data[i][2])

    # Calculate total number of citations, total number of papers, average RCR, average APT for each cluster
    for cluster in applications:
        print(applications[cluster]["description"])
        cluster_citations = []
        cluster_apt = []
        num_papers = 0

        for idd in applications[cluster]["projects"]:
            papers = [output[key]["citations"] for key in output if output[key]["project"]==idd] # list of all papers associated with cluster by citation count
            apt = [output[key]["apt"] for key in output if output[key]["project"]==idd]

            cluster_apt.extend(apt)
            num_papers += len(papers)
            cluster_citations.append(sum(papers))
        
        # Get data by application
        applications[cluster]["size"] = applications[cluster]["size"]
        applications[cluster]["centroids"] = applications[cluster]["centroids"]
        applications[cluster]["total_citations"] = sum(cluster_citations)
        applications[cluster]["total_papers"] = num_papers
        applications[cluster]["caf"] = sum(cluster_citations)/applications[cluster]["funding"]*1e6
        applications[cluster]["funding"] = "${:,}".format(applications[cluster]["funding"])
        applications[cluster]["score"] = "{:.3f}".format(applications[cluster]["score"])
        
        popt, pcov = curve_fit(lambda t,a,b: a*(1+b)**(t-1985),  years,  applications[cluster]["yr_total_cost"])
        std = np.sqrt(np.diagonal(pcov))
        upper1 = popt[1]+1.96*std[1]
        lower1 = popt[1]-1.96*std[1]
        applications[cluster]["growth"] = "{:.3f} ({:.3f} - {:.3f})".format(popt[1], lower1, upper1)
        
        #create 95% confidence interval for population mean weight
        apts_interval = scist.norm.interval(alpha=0.95, loc=np.mean(cluster_apt), scale=scist.sem(cluster_apt))
        applications[cluster]["apt"] = "{:.3f} ({:.3f} - {:.3f})".format(np.mean(cluster_apt), apts_interval[0], apts_interval[1])
        applications[cluster].pop("data", None)
        applications[cluster].pop("yr_total_cost", None)
        applications[cluster].pop("projects", None)
    
    df = pd.DataFrame(applications)
    df = df.T
    df.to_csv(f'{results_directory}/table_2_app.csv', index=False)
    
    return

def create_table_2_cat(results_directory):
    
    # Load result
    result = pickle.load(open(f'{results_directory}model_clustering.pkl',"rb"))
    labels = result["labels"]
    clusters = result["data_by_cluster"]
    
    # Get labels by categories
    categories = {}
    with open(f'{results_directory}/final_data.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)):
            if (raw_data[i][18] != "N/A"):
                if raw_data[i][18] not in categories:
                    categories[raw_data[i][18]] = {"labels": [int(raw_data[i][0])], "funding": [int(raw_data[i][2])], "applications": [raw_data[i][17]]}
                else:
                    categories[raw_data[i][18]]["labels"].append(int(raw_data[i][0]))
                    categories[raw_data[i][18]]["funding"].append(int(raw_data[i][2]))
                    categories[raw_data[i][18]]["applications"].append(raw_data[i][17])
    
    # Get project numbers for each category
    for category in categories:
        labels = categories[category]["labels"]
        category_projects = []
        for label in labels:
            cluster = [item["project_number"] for item in clusters[label]]
            # cluster = list(set(cluster)) # Remove duplicates
            category_projects.extend(cluster)
        categories[category]["projects"] = category_projects

    # Get number of citations, apt, and publication year by paper
    output = {}
    with open("citations.csv", newline='', encoding='utf8') as csvfile:
       raw_data = list(csv.reader(csvfile))
       for i in range(1,len(raw_data)): # "rcr": float(raw_data[i][6]),
           output[raw_data[i][0]] = {"citations": float(raw_data[i][14]) if raw_data[i][14]!="" else 0, "apt": float(raw_data[i][11])}

    # Get project number and year by paper
    with open("papers.csv", newline='', encoding='utf8') as csvfile:
       raw_data = list(csv.reader(csvfile))
       for i in range(1,len(raw_data)):
           if raw_data[i][13] in output.keys():
               output[raw_data[i][13]]["project"] = raw_data[i][0]
               output[raw_data[i][13]]["year"] = int(raw_data[i][2])

    for category in categories:
        print(category)
        cluster_citations = []
        cluster_apt = []
        num_papers = 0
        # availability = []

        for idd in categories[category]["projects"]:
            papers = [output[key]["citations"] for key in output if output[key]["project"]==idd] # list of all papers associated with cluster by citation count
            apt = [output[key]["apt"] for key in output if output[key]["project"]==idd]

            cluster_apt.extend(apt)
            num_papers += len(papers)
            cluster_citations.append(sum(papers))
        
        categories[category]["total_citations"] = sum(cluster_citations)
        categories[category]["total_papers"] = num_papers
        categories[category]["funding"] = sum(categories[category]["funding"])
        categories[category]["caf"] = sum(cluster_citations)/categories[category]["funding"]*1e6
        categories[category]["apt"] = np.mean(cluster_apt)

        #create 95% confidence interval for population mean weight
        apts_interval = scist.norm.interval(alpha=0.95, loc=np.mean(cluster_apt), scale=scist.sem(cluster_apt))
        categories[category]["apt_lower"] = apts_interval[0]
        categories[category]["apt_upper"] = apts_interval[1]
    
    df = pd.DataFrame(categories)
    df = df.T
    df.to_csv(f'{results_directory}/table_2_cat.csv')
    return

def create_figure_1(data):
    """
    Load in data from data.pkl
    Generate table of award data listed by funding institute
    Save to by_funder_detailed.csv
    """
    # create dictionary with institute as key
    by_institute = dict()
    for item in data:
        institute = item["administration"]
        if institute in by_institute:
            by_institute[institute].add(item['project_number'])
        else:
            by_institute[institute] = {item['project_number']} #set

    # Abbreviations to full names
    institute_map = {}
    with open("nih_institutes.csv", newline='', encoding='utf8') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1, len(raw_data)):
            institute_map[raw_data[i][0]] = raw_data[i][1]
            

    # Get number of citations, apt, and publication year by paper
    output = {}
    with open("citations.csv", newline='', encoding='utf8') as csvfile:
       raw_data = list(csv.reader(csvfile))
       for i in range(1,len(raw_data)): # "rcr": float(raw_data[i][6]),
           output[raw_data[i][0]] = {"citations": float(raw_data[i][14]) if raw_data[i][14]!="" else 0, "apt": float(raw_data[i][11])}

    # Get project number and year by paper
    with open("papers.csv", newline='', encoding='utf8') as csvfile:
       raw_data = list(csv.reader(csvfile))
       for i in range(1,len(raw_data)):
           if raw_data[i][13] in output.keys():
               output[raw_data[i][13]]["project"] = raw_data[i][0]
               output[raw_data[i][13]]["year"] = int(raw_data[i][2])

    # iterate through institutes to get # awards, value, cpof, apt
    output_by_funder = [["Funder", "Number of awards", "Value of awards", "CPOF (adjusted by years since pub.)", "APT", "Avg. APT (95% CI)"]]

    for institute, project_set in by_institute.items():
        citations = 0
        apts = []

        for idd in project_set: #idd==project number
            citations += sum([output[key]["citations"] for key in output if output[key]["project"]==idd])
            apts.extend([output[key]["apt"] for key in output if output[key]["project"]==idd])
            # availability += sum([max(0, 2021-output[key]["year"]) for key in output if output[key]["project"]==idd])

        count = len([item for item in data if item["administration"] == institute]) #num of awards
        amount = sum([item["funding"] for item in data if item["administration"] == institute]) #value of awards

        # get apt 95% CI range
        if not apts: # is empty
            apt_range = "n/a"
        elif len(apts) == 1: # error thrown by interval calculation if <2 elements
            apt_range = "{:.4f}".format(apts[0])
        else:
            apt_avg = np.mean(apts)
            apts_interval = scist.norm.interval(alpha=0.95, loc=apt_avg, scale=scist.sem(apts))
            apt_range = "{:.4f} ({:.4f}-{:.4f})".format(apt_avg, apts_interval[0], apts_interval[1])
            
        output_by_funder.append([institute_map[institute], count, amount, citations/amount*1e6, apt_avg, apt_range])

    with open('by_funder_detailed.csv', 'w+', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output_by_funder)
    
    return

def create_etable_4(results_directory, N):
    # Pull up final data
    descriptions = [["Description", "Category", "Label"]]
    labels = []
    data = []
    with open(f'{results_directory}/final_data.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)):
            if raw_data[i][18] != "N/A":
                descriptions.append([raw_data[i][17], raw_data[i][18], raw_data[i][0]])
                labels.append(raw_data[i][0])
                
    for num in labels:
        num = str(num)
        with open(f'{results_directory}/clusters/cluster-{num}.csv', newline='') as csvfile:
            raw_data = list(csv.reader(csvfile))
            for i in range(1,len(raw_data)):
                data.append({
                    "title": raw_data[i][1],
                    "text": raw_data[i][0],
                    "label": num,
                    })
    
    rand_data = random.sample(data, N)
    
    # number label, description
    with open(f'{results_directory}/labels.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(descriptions)
    
    # title, text, label
    keys = rand_data[0].keys()
    with open(f'{results_directory}/score_sheet.csv', 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(rand_data)

def create_etable_5(result_directory):
    path, dirs, files = next(os.walk('{}/clusters'.format(result_directory)))
    file_count = len(files)
    output = [["Description", "Title", "Activity", "Organization", "Year"]]

    labels = {}
    with open(f'{result_directory}final_data.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1, len(raw_data)):
            if raw_data[i][17] == "N/A":
                continue
            labels[str(raw_data[i][0])] = raw_data[i][17]

    for i in range(file_count):
        if str(i) not in labels:
            continue
        unique_awards = {}
        titles = {}

        # open file
        with open('{}/clusters/cluster-{}.csv'.format(result_directory, str(i)), newline='', encoding='utf8') as csvfile:
            raw_data = list(csv.reader(csvfile))
            for j in range(1,len(raw_data)):
                title = raw_data[j][1]
                organization = raw_data[j][6]
                mechanism = raw_data[j][7]
                year = int(raw_data[j][8])
                score = float(raw_data[j][10])

                # If this is a new title
                if title.lower() not in titles:
                    unique_awards[title] = {
                        "organization": organization,
                        "activity": mechanism,
                        "year": year,
                        "score": score,
                        }
                    titles[title.lower()] = year

                # If the title is already there
                else:
                    current_year = titles[title.lower()]
                    # Use the most recent one
                    if year > current_year:
                        unique_awards[title] = {
                        "organization": organization,
                        "activity": mechanism,
                        "year": year,
                        "score": score,
                        }

        unique_awards_sorted = dict(sorted(unique_awards.items(), key = lambda item: -item[1]["score"]))
        unique_awards_list = list(unique_awards_sorted.items())[0:5]
        
        for j in range(len(unique_awards_list)):
            output.append([labels[str(i)], 
                           unique_awards_list[j][0], 
                           unique_awards_list[j][1]['activity'], 
                           unique_awards_list[j][1]['organization'],
                           str(unique_awards_list[j][1]['year'])])
    
    with open(f'{result_directory}/eTable 5.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)

def create_etable_7(results_directory):
    output = [["Application Category",
              "Application",
              "No. of R01 (% in application)",
              "No. of U01 (% in application)",
              "No. of R44 (% in application)",
              "No. of R21 (% in application)",
              "Total number of awards"]]
    
    with open(f'{results_directory}/final_data.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)):
            if (raw_data[i][18] != "N/A"):
                output.append([
                    raw_data[i][18],
                    raw_data[i][17],
                    raw_data[i][31],
                    raw_data[i][32],
                    raw_data[i][33],
                    raw_data[i][34],
                    int(raw_data[i][1]),
                    ])
    with open(f'{results_directory}/eTable 7.csv', 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)

def create_etable_8(results_directory, label_map):
    output = [["Category", "Year of implementation", "Total awards", "Matched awards", "Pct"]]
    
    with open('/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/raw_data.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        all_ids = []
        data = {}
        labels = label_map.keys()
        print("Raw data N: {}".format(str(len(raw_data))))
        for i in range(1,len(raw_data)):
            if (raw_data[i][6] in all_ids) or (raw_data[i][11][0] in ['Z','T']):
                #ids.append(raw_data[i][6])
                continue
            elif "No abstract available" in raw_data[i][1]:
                continue
            elif len(raw_data[i][49]) <= 1:
                continue
            else:
                all_ids.append(raw_data[i][6])
            data[raw_data[i][6]] = [raw_data[i][0].split("; "), raw_data[i][42]]
                
    for num in labels:
        ids = []
        total = 0
        present = 0
        num = str(num)
        with open(f'{results_directory}/clusters/cluster-{num}.csv', newline='') as csvfile:
            raw_data = list(csv.reader(csvfile))
            for i in range(1,len(raw_data)):
                ids.append(raw_data[i][2])
        for id_num in ids:
            total += 1
            #print(label_map[num])
            #print(data[id_num])
            if (label_map[num][0] in data[id_num][0]) and (int(data[id_num][1]) > int(label_map[num][1])):
                present += 1
        print("{}: {}/{} ({:.2f}%)".format(label_map[num][0], str(present), str(total), present/total*100))
        output.append([label_map[num][0], label_map[num][1], total, present, present/total])
        
    with open(f'{results_directory}/eTable 8.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)    

class EffectMeasurePlot:
    """Used to generate effect measure (AKA forest) plots. Estimates and confidence intervals are plotted in a diagram
    on the left and a table of the corresponding estimates is provided in the same plot. See the Graphics page on
    ReadTheDocs examples of the plots
    Parameters
    --------------
    label : list
        List of labels to use for y-axis
    effect_measure : list
        List of numbers for point estimates to plot. If point estimate has trailing zeroes,
        input as a character object rather than a float
    lcl : list
        List of numbers for upper confidence limits to plot. If point estimate has trailing
        zeroes, input as a character object rather than a float
    ucl : list
        List of numbers for upper confidence limits to plot. If point estimate has
        trailing zeroes, input as a character object rather than a float
    Examples
    -------------
    Setting up the data to plot
    >>> from matplotlib.pyplot as plt
    >>> from zepid.graphics import EffectMeasurePlot
    >>> lab = ['One','Two']
    >>> emm = [1.01,1.31]
    >>> lcl = ['0.90',1.01]  # String allows for trailing zeroes in the table
    >>> ucl = [1.11,1.53]
    Setting up the plot, measure labels, and point colors
    >>> x = EffectMeasurePlot(lab, emm, lcl, ucl)
    >>> x.labels(effectmeasure='RR')  # Changing label of measure
    >>> x.colors(pointcolor='r')  # Changing color of the points
    Generating matplotlib axes object of forest plot
    >>> x.plot(t_adjuster=0.13)
    >>> plt.show()
    """
    def __init__(self, label, effect_measure, lcl, ucl, center):
        self.df = pd.DataFrame()
        self.df['study'] = label
        self.df['OR'] = effect_measure
        self.df['LCL'] = lcl
        self.df['UCL'] = ucl
        self.df['OR2'] = self.df['OR'].astype(str).astype(float)
        if (all(isinstance(item, float) for item in lcl)) & (all(isinstance(item, float) for item in effect_measure)):
            self.df['LCL_dif'] = self.df['OR'] - self.df['LCL']
        else:
            self.df['LCL_dif'] = (pd.to_numeric(self.df['OR'])) - (pd.to_numeric(self.df['LCL']))
        if (all(isinstance(item, float) for item in ucl)) & (all(isinstance(item, float) for item in effect_measure)):
            self.df['UCL_dif'] = self.df['UCL'] - self.df['OR']
        else:
            self.df['UCL_dif'] = (pd.to_numeric(self.df['UCL'])) - (pd.to_numeric(self.df['OR']))
        self.em = 'OR'
        self.ci = '95% CI'
        self.scale = 'linear'
        self.center = center
        self.errc = 'dimgrey'
        self.shape = 'd'
        self.pc = 'k'
        self.linec = 'gray'

    def labels(self, **kwargs):
        """Function to change the labels of the outputted table. Additionally, the scale and reference
        value can be changed.
        Parameters
        -------------
        effectmeasure : string, optional
            Changes the effect measure label
        conf_int : string, optional
            Changes the confidence interval label
        scale : string, optional
            Changes the scale to either log or linear
        center : float, integer, optional
            Changes the reference line for the center
        """
        if 'effectmeasure' in kwargs:
            self.em = kwargs['effectmeasure']
        if 'conf_int' in kwargs:
            self.ci = kwargs['conf_int']
        if 'scale' in kwargs:
            self.scale = kwargs['scale']
        if 'center' in kwargs:
            self.center = kwargs['center']

    def colors(self, **kwargs):
        """Function to change colors and shapes.
        Parameters
        ---------------
        errorbarcolor : string, optional
            Changes the error bar colors
        linecolor : string, optional
            Changes the color of the reference line
        pointcolor : string, optional
            Changes the color of the points
        pointshape : string, optional
            Changes the shape of points
        """
        if 'errorbarcolor' in kwargs:
            self.errc = kwargs['errorbarcolor']
        if 'pointshape' in kwargs:
            self.shape = kwargs['pointshape']
        if 'linecolor' in kwargs:
            self.linec = kwargs['linecolor']
        if 'pointcolor' in kwargs:
            self.pc = kwargs['pointcolor']
    
    def plot(self, figsize=(3, 3), t_adjuster=0.01, decimal=3, size=3, max_value=None, min_value=None):
        """Generates the matplotlib effect measure plot with the default or specified attributes.
        The following variables can be used to further fine-tune the effect measure plot
        Parameters
        -----------------
        figsize : tuple, optional
            Adjust the size of the figure. Syntax is same as matplotlib `figsize`
        t_adjuster : float, optional
            Used to refine alignment of the table with the line graphs. When generate plots, trial and error for this
            value are usually necessary. I haven't come up with an algorithm to determine this yet...
        decimal : integer, optional
            Number of decimal places to display in the table
        size : integer,
            Option to adjust the size of the lines and points in the plot
        max_value : float, optional
            Maximum value of x-axis scale. Default is None, which automatically determines max value
        min_value : float, optional
            Minimum value of x-axis scale. Default is None, which automatically determines min value
        Returns
        ---------
        matplotlib axes
        """
        tval = []
        ytick = []
        for i in range(len(self.df)):
            if not np.isnan(self.df['OR2'][i]):
                if ((isinstance(self.df['OR'][i], float)) & (isinstance(self.df['LCL'][i], float)) &
                        (isinstance(self.df['UCL'][i], float))):
                    tval.append([round(self.df['OR2'][i], decimal), (
                                '(' + str(round(self.df['LCL'][i], decimal)) + ', ' +
                                str(round(self.df['UCL'][i], decimal)) + ')')])
                else:
                    tval.append(
                        [self.df['OR'][i], ('(' + str(self.df['LCL'][i]) + ', ' + str(self.df['UCL'][i]) + ')')])
                ytick.append(i)
            else:
                tval.append([' ', ' '])
                ytick.append(i)
        if max_value is None:
            if pd.to_numeric(self.df['UCL']).max() < 1:
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 0.05),
                             2)  # setting x-axis maximum for UCL less than 1
            if (pd.to_numeric(self.df['UCL']).max() < 9) and (pd.to_numeric(self.df['UCL']).max() >= 1):
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 1),
                             0)  # setting x-axis maximum for UCL less than 10
            if pd.to_numeric(self.df['UCL']).max() > 9:
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 10),
                             0)  # setting x-axis maximum for UCL less than 100
        else:
            maxi = max_value
        if min_value is None:
            if pd.to_numeric(self.df['LCL']).min() > 0:
                mini = round(((pd.to_numeric(self.df['LCL'])).min() - 0.1), 1)  # setting x-axis minimum
            if pd.to_numeric(self.df['LCL']).min() < 0:
                mini = round(((pd.to_numeric(self.df['LCL'])).min() - 0.05), 2)  # setting x-axis minimum
        else:
            mini = min_value
        plt.figure(figsize=figsize)  # blank figure
        gspec = gridspec.GridSpec(1, 6)  # sets up grid
        plot = plt.subplot(gspec[0, 0:4])  # plot of data
        tabl = plt.subplot(gspec[0, 4:])  # table of OR & CI
        plot.set_ylim(-1, (len(self.df)))  # spacing out y-axis properly
        if self.scale == 'log':
            try:
                plot.set_xscale('log')
            except:
                raise ValueError('For the log scale, all values must be positive')
        plot.axvline(self.center, color=self.linec, zorder=1)
        plot.errorbar(self.df.OR2, self.df.index, xerr=[self.df.LCL_dif, self.df.UCL_dif], marker='None', zorder=2,
                      ecolor=self.errc, elinewidth=(size / size), linewidth=0)
        plot.scatter(self.df.OR2, self.df.index, c=self.pc, s=(size * 6), marker=self.shape, zorder=3,
                     edgecolors='None')
        plot.xaxis.set_ticks_position('bottom')
        plot.yaxis.set_ticks_position('left')
        plot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plot.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        plot.set_yticks(ytick)
        plot.set_xlim([mini, maxi])
        plot.set_xticks([mini, self.center, maxi])
        plot.set_xticklabels([mini, "{:.3f}".format(self.center), maxi])
        plot.set_yticklabels(self.df.study, fontdict={'fontsize': 8})
        plot.yaxis.set_ticks_position('none')
        plot.invert_yaxis()  # invert y-axis to align values properly with table
        tb = tabl.table(cellText=tval, cellLoc='center', loc='right', colLabels=[self.em, self.ci],
                        bbox=[0, t_adjuster, 1, 1])
        tabl.axis('off')
        tb.auto_set_font_size(False)
        tb.set_fontsize(8)
        for key, cell in tb.get_celld().items():
            cell.set_linewidth(0)
        return plot

def create_etable_4_old(results_directory):
    
    search_terms = ["artificial intelligence", "machine learning", "deep learning", "supervised learning", "naive bayes", "decision tree", "random forest", "support vector machine", "K-nearest neighbor", "K-means", "singular value decomposition", "apriori", "hidden markov model", "principal component analysis", "hierarchical clustering", "gaussian mixture", "q-learning", "markov decision process", "artificial neural network", "convolutional neural network", "recurrent neural network", "long short-term memory", "knowledge representation", "logical representation", "propositional logic", "predicate logic", "semantic network", "production rules", "rule-based system", "frame representation", "frame language", "frame network", "slot-filter", "semantic frame", "expert system", "natural language processing", "named entity recognition", "sentiment analysis", "aspect mining", "topic modeling", "text mining"]
    
    # Pull up final data
    tocsv = [["Description", "Key words", "Number of awards", "Most frequent search terms (% of awards)", "", ""]]
    data = {}
    with open(f'{results_directory}/final_data.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)):
            if raw_data[i][17] != "N/A":
                data[raw_data[i][0]] = {
                    "description": raw_data[i][17],
                    "centroids": raw_data[i][20],
                    "awards": raw_data[i][1]
                    }
                
    for num in data:
        print(num)
        result = [data[num]["description"], data[num]["centroids"], data[num]["awards"]]
        with open(f'{results_directory}/clusters/cluster-{num}.csv', newline='') as csvfile:
            raw_data = list(csv.reader(csvfile))
            search_map = {}
            for term in search_terms:
                search_map[term] = 0
            for term in search_map: # for each term
                for i in range(1,len(raw_data)): # for each
                    if (term in raw_data[i][0].lower()): # count if award has that term
                        search_map[term] += 1
                search_map[term] = search_map[term]/(len(raw_data)-1)
            my_keys = sorted(search_map, key=search_map.get, reverse=True)[:3]
            for key in my_keys:
                value = search_map[key]*100
                result_string = f"'{key}' ({value:.1f}%)"
                result.append(result_string)
            tocsv.append(result)
    
    print(tocsv)
    with open(f'{results_directory}/eTable 4.csv', 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(tocsv)
                
    return

def create_figure_2_cummulative(results_directory):
    # get data from run
    data = pickle.load(open(f'{results_directory}/model_clustering.pkl',"rb"))
    
    
    # get labels -> topics/categories
    label_map = {}
    with open(f'{results_directory}/final_data.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)):
            if raw_data[i][18] == "N/A":
                continue
            if raw_data[i][18] not in label_map:
                label_map[raw_data[i][18]] = [int(raw_data[i][0])]
            else:
                label_map[raw_data[i][18]].append(int(raw_data[i][0]))
    
    funding = []
    years = []
    grouped = []
    measure = []
    lower = []
    upper = []
    labs = []
    with open('/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/by_year.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)):
            funding.append(int(raw_data[i][2]))
            years.append(int(raw_data[i][0]))
    for category in label_map:
        labels = label_map[category]
        total_cost = 0
        for i in labels:
            cost = data["yr_total_cost"][i]
            if total_cost == 0:
                total_cost = cost
            else:
                total_cost = zip(total_cost, cost)
                combined = []
                for (item1, item2) in total_cost:
                    combined.append(item1+item2)
                total_cost = combined
        
        popt, pcov = curve_fit(lambda t,a,b: a*(1+b)**(t-1985),  years,  total_cost)
        std = np.sqrt(np.diagonal(pcov))
        upper1 = popt[1]+1.96*std[1]
        lower1 = popt[1]-1.96*std[1]
        grouped.append([popt[1], lower1, upper1, category, "{:.3f} ({:.3f}-{:.3f})".format(popt[1], lower1, upper1)])
        print("{}: {} ({}-{})".format(category, popt[1], lower1, upper1))
       
    grouped.sort(key=itemgetter(0))
    for cluster in grouped:
        measure.append(cluster[0])
        lower.append(cluster[1])
        upper.append(cluster[2])
        labs.append(cluster[3])

    # Get cummulative
    popt, pcov = curve_fit(lambda t,a,b: a*(1+b)**(t-1985),  years,  funding)
    std = np.sqrt(np.diagonal(pcov))
    upper1 = popt[1]+1.96*std[1]
    lower1 = popt[1]-1.96*std[1]
    measure.append(popt[1])
    labs.append("NIH")
    lower.append(lower1)
    upper.append(upper1)
    print("NIH: {} ({}-{})".format(popt[1], lower1, upper1))
    
    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper, center=popt[1])
    p.labels(effectmeasure='Estimated AGR', fontsize=8)
    p.colors(pointshape="D")
    ax=p.plot(figsize=(10,5), t_adjuster=0.01, max_value=1.1, min_value=-0.5)
    ax.grid(False)
    ax.set_xlabel("Slower growth than overall       Faster growth than overall", fontsize=10)
    ax.xaxis.set_label_coords(0.50, -0.06)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    
    plt.savefig(f'{results_directory}/figure 2.eps', format="eps", bbox_inches='tight')
    with open(f'{results_directory}/table_2_cumgrowth.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(grouped) 
    return 

def create_figure_2(results_directory, ignore):
    # get data from run
    data = pickle.load(open(f'{results_directory}/model_clustering.pkl',"rb"))
    
    # get labels -> topics/categories
    label_map = {}
    with open(f'{results_directory}/final_data.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)):
            if raw_data[i][18] == "N/A":
                continue
            else:
                label_map[int(raw_data[i][0])] = [raw_data[i][17], raw_data[i][18]]
    
    funding = []
    years = []
    grouped = []
    measure = []
    lower = []
    upper = []
    labs = []
    with open('/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/by_year.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)):
            funding.append(int(raw_data[i][2]))
            years.append(int(raw_data[i][0]))
    
    for i in range(len(data["yr_total_cost"])):
        if i in ignore:
            continue
        popt, pcov = curve_fit(lambda t,a,b: a*(1+b)**(t-1985),  years,  data["yr_total_cost"][i])
        std = np.sqrt(np.diagonal(pcov))
        upper1 = popt[1]+1.96*std[1]
        lower1 = popt[1]-1.96*std[1]
        grouped.append([popt[1], lower1, upper1, label_map[i][0], label_map[i][1], data["size"][i]])
        print("{}: {} ({}-{})".format(i, popt[1], lower1, upper1))
        
    grouped.sort(key=itemgetter(0))
    for cluster in grouped:
        measure.append(cluster[0])
        lower.append(cluster[1])
        upper.append(cluster[2])
        labs.append(cluster[3])

    grouped.sort(key=itemgetter(4,5))
    pos = []
    for i in range(len(grouped)-1):
        if grouped[i][4] != grouped[i+1][4]:
            pos.append(i+1)
    
    acc = 0
    for i in range(len(pos)):
        grouped.insert(pos[i]+acc, ["", "", "", "", ""])
        acc += 1
        
    
    # Get cummulative
    popt, pcov = curve_fit(lambda t,a,b: a*(1+b)**(t-1985),  years,  funding)
    std = np.sqrt(np.diagonal(pcov))
    upper1 = popt[1]+1.96*std[1]
    lower1 = popt[1]-1.96*std[1]
    measure.append(popt[1])
    lower.append(lower1)
    upper.append(upper1)
    print("NIH: {} ({}-{})".format(popt[1], lower1, upper1))
    labs.append("NIH")
    
    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper, center=popt[1])
    p.labels(effectmeasure='Estimated AGR', fontsize=8)
    p.colors(pointshape="D")
    ax=p.plot(figsize=(5,10), t_adjuster=0.01, max_value=2, min_value=-0.5)
    ax.grid(False)
    ax.set_xlabel("Slower growth than overall       Faster growth than overall", fontsize=10)
    ax.xaxis.set_label_coords(0.35, -0.025)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    
    plt.savefig(f'{results_directory}/figure 2 old.eps', format="eps", bbox_inches='tight')
    with open(f'{results_directory}/table_2_growth.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(grouped) 
    return

def cluster_anova(results_directory, ignore):
    # Result labels
    result = pickle.load(open(f'{results_directory}model_clustering.pkl',"rb"))
    labels = result["labels"]
    clusters = result["data_by_cluster"]
    # funding = [sum(item["funding"] for item in clusters[i]) for i in range(len(clusters))]
    total_citations, total_papers, apts_95, apts, lower, upper, listed_apts = get_citations(clusters)
    
    all_apts = []
    labels = []
    for i in range(len(listed_apts)):
        if i in ignore:
            continue
        all_apts.extend(listed_apts[i])
        labels.extend([i for j in range(len(listed_apts[i]))])
        
    # Load data frame
    df = pd.DataFrame({'score': all_apts,
                   'group': labels}) 
    
    # perform Tukey's test
    tukey = pairwise_tukeyhsd(endog=df['score'],
                              groups=df['group'],
                              alpha=0.05)
    df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    for index in range(len(df)):
        group1 = df["group1"][index]
        group2 = df["group2"][index]
        p = df[df["group1"]==group1][df["group2"]==group2]["p-adj"][index]
        if p > 0.05:
            p = 5
        elif 0.01 < p <= 0.05:
            p = 4
        elif 0.005 < p <= 0.01:
            p = 3
        elif 0.001 < p <= 0.005:
            p = 2
        elif p <= 0.001:
            p = 1
        if np.mean(listed_apts[group2]) > np.mean(listed_apts[group1]):
            p = -p
        df.at[index,'p-adj']=p

    df.to_csv('anova_award.csv', index=False, header=True)
    piv = pd.pivot_table(df, values="p-adj",index=["group2"], columns=["group1"])
    plt.figure()
    myColors = ((1.0, 1.0, 1.0),
                #(255/256, 215/256, 205/256),
                (255/256, 175/256, 158/256),
                (254/256, 134/256, 112/256),
                (243/256, 88/256, 68/256),
                (228/256, 13/256, 25/256),
                (43/256, 12/256, 240/256),
                (117/256, 72/256, 245/256),
                (160/256, 118/256, 249/256),
                (196/256, 163/256, 252/256),
                #(227/256, 208/256, 254/256),
                (1.0, 1.0, 1.0))
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
    xlabels = []
    ylabels = []
    # with open('figures and tables/tick_labels.csv', newline='') as csvfile:
    #     raw_data = list(csv.reader(csvfile))
    #     for i in range(0,len(raw_data)-1):
    #         xlabels.append(raw_data[i][0])
    #         ylabels.append(raw_data[i+1][0])
    sns.set(font_scale=0.6)
    ax = sns.heatmap(piv, vmin=-5, vmax=5, xticklabels=True, yticklabels=True, cmap=cmap, cbar=True, square=True,) #'format': '%.0f%%', 
    ax.set_xlabel("Cluster 1")
    ax.set_ylabel("Cluster 2")
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels) 
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
    colorbar.set_ticklabels(['Cluster 2 > 1, p < 0.05',
                             'Cluster 2 > 1, p < 0.01',
                             'Cluster 2 > 1, p < 0.005',
                             'Cluster 2 > 1, p < 0.001', 
                             'Cluster 1 > 2, p < 0.001',
                             'Cluster 1 > 2, p < 0.005',
                             'Cluster 1 > 2, p < 0.01',
                             'Cluster 1 > 2, p < 0.05',])
    # ax.collections[0].colorbar.set_ticklabels(['10', '30'])
    for text in ax.texts:
        if abs(float(text.get_text())) > 0.05:
            text.set_text("")
    ax.figure.tight_layout()
    plt.savefig(f'{results_directory}avg_award_ANOVA.png')

def create_efigure_3(data):
    """

    Parameters
    ----------
    data : list
        List of dictionaries representing each award (loaded from data.pkl)

    Returns
    -------
    None.

    """
    years = np.unique(np.array([item["year"] for item in data]))
    awards = []
    values = []
    for year in years:
        awards.append(len([item for item in data if item["year"] == year]))
        values.append(sum([item["funding"] for item in data if item["year"] == year]))
        
    fig, ax1 = plt.subplots()
    ax1.set_xticks(np.arange(0, len(years)+1, 5))
    ax1.set_xticklabels([years[i] for i in np.arange(0, len(years)+1, 5)])
    

    # award values
    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Award Values (billions USD)', color=color)
    bar = ax1.bar(years, [v/1e9 for v in values], color=color) #yerr=errorbar
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.bar_label(bar, fmt='$%.1f', padding=1, size='x-small')
    ax2 = ax1.twinx()
    ax2.set_xticks(np.arange(0, len(years)+1, 5))
    ax2.set_xticklabels([years[i] for i in np.arange(0, len(years)+1, 5)])

    # award size (avg) = total value/num awards
    color = 'tab:green'
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Average Award Size (thousands USD)', color=color)
    ax2.plot(years, [values[i]/1e3/awards[i] for i in range(len(years))], color=color) #yerr=errorbar
    ax2.tick_params(axis='y', labelcolor=color)

    # plt.title('Total Awards by Year')
    plt.savefig('efigure2.png')#, format='eps')

if __name__ == "__main__":
    create_table_2_app("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/results/10-25-2021--134436/")
    create_table_2_cat("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/results/10-25-2021--134436/")
    # create_figure_2_cummulative("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/results/10-25-2021--134436/")
    # create_figure_2_cummulative("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/results/10-25-2021--134436/", [28, 73, 75, 39, 37])
    # create_figure_2("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/results/10-25-2021--134436/", [28, 73, 75, 39, 37])
    # create_etable_4("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/results/10-25-2021--134436/")
    # manual_verification("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/results/10-25-2021--134436/", 200)
    # data = pickle.load(open("data.pkl", "rb"))
    # create_efigure_2(data)
    # create_table_1(data)
    # create_table_2(data)
    # create_figure_4("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/results/10-18-2021--224935/", 10)
    # label_map = {
    #         "76": ["Asthma", "2008"],
    #         "16": ["Alzheimer's Disease", "2008"],
    #         "11": ["Autism", "2008"],
    #         "15": ["Breast Cancer", "2008"],
    #         "52": ["Dementia", "2008"],
    #         "9": ["HIV/AIDS", "2008"],
    #         "23": ["Kidney Disease", "2008"],
    #         "43": ["Liver Disease", "2008"],
    #         "38": ["Depression", "2015"],
    #         "45": ["Pain Research", "2012"],
    #         "18": ["Schizophrenia", "2008"],
    #         "13": ["Stroke", "2008"],
    #         "55": ["Suicide",  "2008"],
    #     }
    # create_etable_8("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/results/10-25-2021--134436/", label_map)
    # create_etable_6("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/results/10-25-2021--134436/")