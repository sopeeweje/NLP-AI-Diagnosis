import requests
import argparse
import pandas as pd
import csv
import time
from progress.bar import Bar
from tqdm import tqdm
import codecs

def get_data(termsfile, start, end, operator):
    
    # Get query
    lines = []
    search_text = ""
    with open(termsfile) as f:
        lines = f.readlines()
        if operator != "advanced": # if "and" or "or" query (search_text.txt must be list of terms)
            for line in lines:
                line.strip()
                term = "\"" + line + "\", "
                search_text += term
            search_text = search_text[0:-2]
        else: # if "advanced query" (search_text.txt must be directly formated query)
            line = lines[0].strip()
            search_text = line
    print("Your query: {}".format(search_text))
            
    
    # Get awards from NIH RePORTER
    print("Getting awards...")
    dfs = []
    years = list(range(start,end))
    for year in years:
        for o in range(0,1000):
            params = {
              "criteria":
                {
                  "fiscal_years": [year],
                  "advanced_text_search":
                  {
                        "operator": operator, 
                        "search_field": "projecttitle,terms,abstracttext", 
                        "search_text": search_text
                  },
                  "exclude_subprojects": True,
                  "use_relevance": False,
                  "include_active_projects": False,
                },
              "offset":o*500,
              "limit":500,
              "sort_field":"fiscal_year",
              "sort_order":"desc",
            }
        
            response = requests.post("https://api.reporter.nih.gov/v2/projects/search", json=params)
            while response.status_code != 200:
                print("Didn't work trying again, status code: {}".format(response.status_code))
                time.sleep(30)
                response = requests.post("https://api.reporter.nih.gov/v2/projects/search", json=params)
                
            response_dict = response.json()
            results = response_dict["results"]
            print("Year: {}, {}: {}, {} awards".format(year, o, response, len(results)))
            results = pd.json_normalize(results, sep='_')
            df = pd.DataFrame.from_dict(results)
            dfs.append(df)
            if len(results) < 500:
                break
    
    result = pd.concat(dfs)
    result.to_csv("data/raw_data.csv", index=False)
    print("Got awards.")
    
    ############################################
    
    # Getting the papers
    print("Getting papers (5 awards at a time)...")
    data_file = "data/raw_data.csv"
    
    with open(data_file, newline='', encoding='utf8') as csvfile:
        reader = csv.reader(x.replace('\0', '') for x in csvfile)
        raw_data = list(reader)
        
    application_ids = []
    for i in range(1,len(raw_data)):
        application_ids.append(str(raw_data[i][0]))
    
    dfs = []
    for o in tqdm(range(len(application_ids)//5), position=0, leave=True):
        params = {
            "criteria": {
                "appl_ids": application_ids[o*5:min(o*5+5, len(application_ids))],
            },
            "sort_field":"appl_ids",
            "sort_order":"desc"
        }
    
        response = requests.post("https://api.reporter.nih.gov/v2/publications/search", json=params)
        while response.status_code != 200:
            response = requests.post("https://api.reporter.nih.gov/v2/publications/search", json=params)
            
        # print("{}: {}".format(o,response))
        response_dict = response.json()
        results = response_dict["results"]
        results = pd.json_normalize(results, sep='_')
        df = pd.DataFrame.from_dict(results)
        dfs.append(df)
    
    result = pd.concat(dfs)
    result.to_csv("data/publications.csv", index=False)
    print("Got papers.")
    
    ###########################################
    
    # Getting the citation data
    print("Getting citation data...")
    data_file = "data/publications.csv"
    dfs = []
    
    with open(data_file, newline='', encoding='utf8') as csvfile:
        raw_data = list(csv.reader(csvfile))
    pmids = []
    for i in range(1,len(raw_data)):
        pmids.append(raw_data[i][1])
    
    
    for i in tqdm(range(0, max(len(pmids)//1000,1)), position=0, leave=True):
        target_pmids = pmids[i*1000:min(i*1000+1000, len(pmids))]
        pmid_string = ",".join(target_pmids)
        query = "pmids=" + pmid_string + "&limit=1000"
        response = requests.get("https://icite.od.nih.gov/api/pubs?" + query)
        # print("{}: {}".format(i,response))
        pub = response.json()
        for i in range(len(pub["data"])):
            pub["data"][i]['pmid'] = target_pmids[i]
        results = pd.json_normalize(pub["data"], sep='_')
        df = pd.DataFrame.from_dict(results)
        dfs.append(df)
    
    result = pd.concat(dfs)
    result.to_csv("data/citations.csv", index=False)
    print("Got citation data.")

if __name__ == "__main__":
    
    # Arguments: path to search terms text file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--search_terms',
        type=str,
        required=True,
        help='Terms for NIH RePORTER query',
        default="search_terms.txt",
        )
    parser.add_argument(
        '--operator',
        type=str,
        required=True,
        help='Operator for NIH RePORTER query (and, or, advanced)',
        default="or",
        )
    parser.add_argument(
        '--start_year',
        type=int,
        required=True,
        help='Start year for search (inclusive)',
        default=1985,
        )
    parser.add_argument(
        '--end_year',
        type=int,
        required=True,
        help='End year for search (inclusive)',
        default=2021,
        )
    FLAGS, unparsed = parser.parse_known_args()
    
    # Run
    get_data(FLAGS.search_terms, FLAGS.start_year, FLAGS.end_year+1, FLAGS.operator)