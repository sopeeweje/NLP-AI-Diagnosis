import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
import statistics
import pickle
import numpy as np


def get_terms():
    ai_terms = [
        "Artificial Intelligence",
        "Machine Learning",
        "Deep Learning",
        "Natural Language Processing",
        "Random Forest",
        "Logistic Regression",
        "LSTM",
        "RNN",
        "CNN",
        "Federated Learning",
        "Decision Tree",
        "Support Vector Machine",
        "Bayesian Learning",
        "Gradient Boosting",
        "Computational Intelligence",
        "Naive Bayes",
        "Computer Vision",]
    
    purpose_terms = [
        "Diagnosis",
        "Early Detection",
        "Decision Support",
        "Screening"
        ]
    
    search_string = ""
    term_lists = [ai_terms, purpose_terms]
    for term_list in term_lists:
        for term in term_list:
            if term == term_list[0]:
                search_string += '("' + term + '" or '
            elif term == term_list[-1]:
                search_string += ('"' + term + '") and ')
            else:
                search_string += ('"' + term + '" or ')
    print(search_string)

def mk_int(s):
    s = s.strip()
    return int(s) if s else 0

def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    4. Remove words
    '''
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc =  [word.lower() for word in nopunc.split()]
    nopunc = [word for word in nopunc if word not in stopwords.words('english')]
    return [stemmer.lemmatize(word) for word in nopunc]

def process_data(data_file, funding_file):
    try:
        # Load data as dictionary
        data = pickle.load(open("data.pkl","rb"))
    
        # Transformed data
        test_data = pickle.load(open("test-data.pkl","rb"))
        print("loaded from pickle")
        
        return data, test_data
    except:
        pass
    
    funding_data = {}
    with open(funding_file, newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)):
            org = raw_data[i][0]
            funding = int(raw_data[i][5])
            funding_data[org] = funding
    
    data = []
    with open(data_file, newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        ids = []
        print(len(raw_data))
        for i in range(1,len(raw_data)):
            if (raw_data[i][6] in ids) or (raw_data[i][11][0] in ['Z','T']):
                continue
            else:
                ids.append(raw_data[i][6])
            abstract = raw_data[i][1].replace('\n',' ')
            title = raw_data[i][3]
            relevance = raw_data[i][4].replace('\n',' ')
            funding = funding_data.get(raw_data[i][31], 0)
            data.append({
                "text": title + " " + abstract + " " + relevance,
                "title": title,
                "id": raw_data[i][6],
                "terms": raw_data[i][2].split(";"),
                "administration": raw_data[i][5],
                "organization": raw_data[i][31],
                "mechanism": raw_data[i][11],
                "year": raw_data[i][42],
                "cost": mk_int(raw_data[i][49]),
                "funding": funding,
                })
    
    test_data = []
    for item in data:
        if item["cost"] == 0:
            data.remove(item)
        elif item["year"] == "2021":
            data.remove(item)
            test_data.append(item)

    
    with open("data.pkl", 'wb') as handle:
        pickle.dump(data, handle)
        
    with open("test-data.pkl", 'wb') as handle:
        pickle.dump(test_data, handle)
    
    print(len(data))
    return data, test_data

def feature_extraction(data, num_features, max_df):
    """

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    
    Returns 
    -------
    features:
        funding - institution's funding in 2020
        year - one-hot encoded year
        text - tfidf for all text data
    """
    input_text = [item["text"] for item in data]
    #vectorizer = TfidfVectorizer(analyzer=text_process, max_df=0.95, min_df=0.001, max_features=num_features).fit(input_text)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=max_df, max_features=num_features).fit(input_text) #, token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b'
    processed_text = vectorizer.transform(input_text)
    
    with open("processed-data.pkl", 'wb') as handle:
        pickle.dump(processed_text, handle)
        
    with open("vectorizer.pkl", 'wb') as handle:
        pickle.dump(vectorizer, handle)
    
if __name__ == "__main__":
    file = '/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/raw_data.csv'
    funding_file = '/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/institution-funding.csv'
    years = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
    data, test_data = process_data(file, funding_file)
    feature_extraction(data, 1000, 0.5)
    
    # By Funder
    funders = np.unique(np.array([item["administration"] for item in data]))
    output = [["Funder", "Number of awards", "Value of awards"]]
    for funder in funders:
        count = len([item for item in data if item["administration"] == funder])
        amount = sum([item["cost"] for item in data if item["administration"] == funder])
        output.append([funder, count, amount])
    with open('by_funder.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)
        
    # By Year
    years = np.unique(np.array([item["year"] for item in data]))
    output = [["Year", "Number of awards", "Value of awards"]]
    for year in years:
        count = len([item for item in data if item["year"] == year])
        amount = sum([item["cost"] for item in data if item["year"] == year])
        output.append([year, count, amount])
    with open('by_year.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)
        
    # By Mechanism
    mechanisms = np.unique(np.array([item["mechanism"] for item in data]))
    output = [["Mechanism", "Number of awards", "Value of awards"]]
    for mech in mechanisms:
        count = len([item for item in data if item["mechanism"] == mech])
        amount = sum([item["cost"] for item in data if item["mechanism"] == mech])
        output.append([mech, count, amount])
    with open('by_mechanism.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)