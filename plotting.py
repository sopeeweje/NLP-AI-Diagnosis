#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 18:52:14 2021

@author: Sope
"""
import pickle
import csv
import numpy as np
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
csv.field_size_limit(sys.maxsize)

def create_figure_1(data):
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
    ax1.bar_label(bar, fmt='$%.1f', padding=1, size='x-small')
    ax2 = ax1.twinx()
    ax2.set_xticks(np.arange(0, len(years)+1, 5))
    ax2.set_xticklabels([years[i] for i in np.arange(0, len(years)+1, 5)])

    # award size (avg) = total value/num awards
    color = 'tab:green'
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Average Award Size (thousands USD)', color=color)
    ax2.plot(years, [values[i]/1e3/awards[i] for i in range(len(years))], color=color) #yerr=errorbar
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Total Awards by Year')
    plt.savefig('figure1.eps', format='eps')

def create_table_1(data):
    # Get total awards for 1985, 2020, overall
    awards_1985 = sum([int(data[i]["year"]) <= 2000 for i in range(len(data))])
    awards_2020 = sum([int(data[i]["year"]) > 2000 for i in range(len(data))])
    awards_overall = len(data)
    
    # Get total funding for 1985, 2020, overall
    funding_1985 = sum([data[i]["funding"] for i in range(len(data)) if int(data[i]["year"]) <= 2000])
    funding_2020 = sum([data[i]["funding"] for i in range(len(data)) if int(data[i]["year"]) > 2000])
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
        if award["year"] == "1985":
            projects_1985.append(award["project_number"])
        elif award["year"] == "2020":
            projects_2020.append(award["project_number"])
        else:
            projects_all.append(award["project_number"])
    
    # Get citation data by year for 1985
    citations_1985 = []
    apt_1985 = []
    papers_1985 = 0
    for idd in projects_1985:
        papers = [output[key]["citations"] for key in output if output[key]["project"]==idd] # list of all papers associated with cluster by citation count
        apt = [output[key]["apt"] for key in output if output[key]["project"]==idd]
        
        apt_1985.extend(apt)
        papers_1985 += len(papers)
        citations_1985.extend(papers)
            
    apt_interval_1985 = scist.norm.interval(alpha=0.95, loc=np.mean(apt_1985), scale=scist.sem(apt_1985))
    avg_apt_1985_str = "{} ({} - {})".format(str(np.mean(apt_1985)), str(apt_interval_1985[0]), str(apt_interval_1985[1]))
    cit_interval_1985 = scist.norm.interval(alpha=0.95, loc=np.mean(citations_1985), scale=scist.sem(citations_1985))
    cit_1985_str = "{} ({} - {})".format(str(np.mean(citations_1985)), str(cit_interval_1985[0]), str(cit_interval_1985[1]))
    
    # Get citation data by year for 2020
    citations_2020 = []
    apt_2020 = []
    papers_2020 = 0
    for idd in projects_2020:
        papers = [output[key]["citations"] for key in output if output[key]["project"]==idd] # list of all papers associated with cluster by citation count
        apt = [output[key]["apt"] for key in output if output[key]["project"]==idd]
        
        apt_2020.extend(apt)
        papers_2020 += len(papers)
        citations_2020.extend(papers)
        
    apt_interval_2020 = scist.norm.interval(alpha=0.95, loc=np.mean(apt_2020), scale=scist.sem(apt_2020))
    avg_apt_2020_str = "{} ({} - {})".format(str(np.mean(apt_2020)), str(apt_interval_2020[0]), str(apt_interval_2020[1]))
    cit_interval_2020 = scist.norm.interval(alpha=0.95, loc=np.mean(citations_2020), scale=scist.sem(citations_2020))
    cit_2020_str = "{} ({} - {})".format(str(np.mean(citations_2020)), str(cit_interval_2020[0]), str(cit_interval_2020[1]))
    
    # Get citation data by overall
    citations_all = []
    apt_all = []
    papers_all = 0
    for idd in projects_all:
        papers = [output[key]["citations"] for key in output if output[key]["project"]==idd] # list of all papers associated with cluster by citation count
        apt = [output[key]["apt"] for key in output if output[key]["project"]==idd]
        
        apt_all.extend(apt)
        papers_all += len(papers)
        citations_all.extend(papers)

    avg_apt_all = np.mean(apt_all)
    apt_interval_all = scist.norm.interval(alpha=0.95, loc=np.mean(apt_all), scale=scist.sem(apt_all))
    avg_apt_all_str = "{} ({} - {})".format(str(avg_apt_all), str(apt_interval_all[0]), str(apt_interval_all[1]))
    cit_interval_all = scist.norm.interval(alpha=0.95, loc=np.mean(citations_all), scale=scist.sem(citations_all))
    cit_all_str = "{} ({} - {})".format(str(np.mean(citations_all)), str(cit_interval_all[0]), str(cit_interval_all[1]))
    
    # Final data
    data_to_csv = [['', '1985', '2020', 'Overall'],
                   ['Number of Awards', str(awards_1985), str(awards_2020), str(awards_overall)],
                   ['Total Funding', str(funding_1985), str(funding_2020), str(funding_overall)],
                   ['Number of Papers', str(papers_1985), str(papers_2020), str(papers_all)],
                   ['Average Number of Citations Per Year of Availability', cit_1985_str, cit_2020_str, cit_all_str],
                   ['Average APT score', avg_apt_1985_str, avg_apt_2020_str, avg_apt_all_str],
                   ['Enriched features', enriched_1985, enriched_2020, '']]
    
    # Write to CSV
    with open('table1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data_to_csv)

def create_table_2(data):
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
            output[raw_data[i][0]] = {"citations": int(raw_data[i][13]), "apt": float(raw_data[i][11])}

    # Get project number and year by paper
    with open("papers.csv", newline='', encoding='utf8') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)):
            if raw_data[i][13] in output.keys():
                output[raw_data[i][13]]["project"] = raw_data[i][0]
                output[raw_data[i][13]]["year"] = int(raw_data[i][2])

    # iterate through institutes to get # awards, value, cpof, apt
    output_by_funder = [["Funder", "Number of awards", "Value of awards", "CPOF (adjusted by years since pub.)", "Avg. APT (95% CI)"]]

    for institute, project_set in by_institute.items():
        citations = 0
        apts = []
        availability = 0

        for idd in project_set: #idd==project number
            citations += sum([output[key]["citations"] for key in output if output[key]["project"]==idd])
            apts.extend([output[key]["apt"] for key in output if output[key]["project"]==idd])
            availability += sum([max(0, 2021-output[key]["year"]) for key in output if output[key]["project"]==idd])

        count = len([item for item in data if item["administration"] == institute]) #num of awards
        amount = sum([item["funding"] for item in data if item["administration"] == institute]) #value of awards

        # get apt 95% CI range
        if not apts: # is empty
            apt_range = "n/a"
        elif len(apts) == 1: # error thrown by interval calculation if <2 elements
            apt_range = "{:.2f}".format(apts[0])
        else:
            apt_avg = np.mean(apts)
            apts_interval = scist.norm.interval(alpha=0.95, loc=apt_avg, scale=scist.sem(apts))
            apt_range = "{:.2f} ({:.2f}-{:.2f})".format(apt_avg, apts_interval[0], apts_interval[1])

        if availability == 0:
            cpof_per_yr = "n/a"
        else:
            cpof_per_yr = citations/availability
            
        output_by_funder.append([institute_map[institute], count, amount, cpof_per_yr, apt_range])

    with open('by_funder_detailed.csv', 'w+', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output_by_funder)
    return

if __name__ == "__main__":
    data = pickle.load(open("data.pkl", "rb"))
    create_table_1(data)