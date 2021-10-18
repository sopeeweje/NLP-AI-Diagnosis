import requests
import argparse
import re
from collections import defaultdict
import json
import pandas as pd
import csv
import time

# # Getting the awards
# dfs = []
# years = list(range(1985,2022))
# for year in years:
#     for o in range(0,1000):
#         params = {
#           "criteria":
#             {
#               "fiscal_years": [year],
#               "advanced_text_search":
#               {
#                     "operator": "or", 
#                     "search_field": "projecttitle,terms,abstracttext", 
#                     "search_text": "artificial intelligence" or "machine learning" or "deep learning" or "supervised learning" or "logistic regression" or "naive bayes" or "decision tree" or "random forest" or "support vector machine" or "linear regression" or "K-nearest neighbor" or "K-means" or "singular value decomposition" or "apriori" or "hidden markov model" or "principal component analysis" or "hierarchical clustering" or "gaussian mixture" or "reinforcement learning" or "q-learning" or "markov decision process" or "monte carlo" or "knowledge representation" or "logical representation" or "propositional logic" or "predicate logic" or "ontology" or "semantic network" or "production rules" or "rule-based system" or "production system" or "frame representation" or "frame language" or "frame network" or "slot-filter" or "semantic frame" or "expert system" or "artificial neural network" or "convolutional neural network" or "recurrent neural network" or "long short-term memory" or "natural language processing" or "machine vision" or "computer vision" 
#               },
#               "exclude_subprojects": True,
#               "use_relevance": False,
#               "include_active_projects": False,
#             },
#           "offset":o*500,
#           "limit":500,
#           "sort_field":"fiscal_year",
#           "sort_order":"desc",
#         }
    
#         response = requests.post("https://api.reporter.nih.gov/v2/projects/search", json=params)
#         while response.status_code != 200:
#             print("Didn't work trying again, status code: {}".format(response.status_code))
#             time.sleep(30)
#             response = requests.post("https://api.reporter.nih.gov/v2/projects/search", json=params)
            
#         response_dict = response.json()
#         results = response_dict["results"]
#         print("Year: {}, {}: {}, {}".format(year, o, response, len(results)))
#         if len(results) < 500:
#             break
#         results = pd.json_normalize(results, sep='_')
#         df = pd.DataFrame.from_dict(results)
#         dfs.append(df)

# result = pd.concat(dfs)
# result.to_csv("new_raw.csv", index=False)

# ############################################

# # Getting the papers
# data_file = "new_raw.csv"
# with open(data_file, newline='', encoding='utf8') as csvfile:
#     raw_data = list(csv.reader(csvfile))
# application_ids = []
# for i in range(1,len(raw_data)):
#     application_ids.append(str(raw_data[i][6]))

# dfs = []
# for o in range(len(application_ids)//20):
#     params = {
#         "criteria": {
#             "appl_ids": application_ids[o*20:min(o*20+20, len(application_ids))],
#         },
#         # "offset": o*500,
#         # "limit": 500,
#         "sort_field":"appl_ids",
#         "sort_order":"desc"
#     }

#     response = requests.post("https://api.reporter.nih.gov/v2/publications/search", json=params)
#     while response.status_code != 200:
#         response = requests.post("https://api.reporter.nih.gov/v2/publications/search", json=params)
        
#     print("{}: {}".format(o,response))
#     response_dict = response.json()
#     results = response_dict["results"]
#     results = pd.json_normalize(results, sep='_')
#     df = pd.DataFrame.from_dict(results)
#     dfs.append(df)

# result = pd.concat(dfs)
# result.to_csv("new_publication.csv", index=False)

############################################

# Getting the citation data
# data_file = "new_publication.csv"
data_file = "papers.csv"
dfs = []

with open(data_file, newline='', encoding='utf8') as csvfile:
    raw_data = list(csv.reader(csvfile))
pmids = []
for i in range(1,len(raw_data)):
    pmids.append(raw_data[i][13])

for i in range(len(pmids)//1000):
    target_pmids = pmids[i*1000:min(i*1000+1000, len(pmids))]
    pmid_string = ",".join(target_pmids)
    query = "pmids=" + pmid_string + "&limit=1000"
    response = requests.get("https://icite.od.nih.gov/api/pubs?" + query)
    print("{}: {}".format(i,response))
    pub = response.json()
    for i in range(len(pub["data"])):
        pub["data"][i]['pmid'] = target_pmids[i]
    results = pd.json_normalize(pub["data"], sep='_')
    df = pd.DataFrame.from_dict(results)
    dfs.append(df)

result = pd.concat(dfs)
result.to_csv("citations.csv", index=False)