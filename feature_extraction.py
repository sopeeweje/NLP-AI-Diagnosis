import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import Counter
import statistics
import pickle
import numpy as np
import argparse

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
        print("Raw data N: {}".format(str(len(raw_data))))
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
                "project_number": raw_data[i][9][1:].split("-")[0],
                "terms": raw_data[i][2].split(";"),
                "administration": raw_data[i][5],
                "organization": raw_data[i][31],
                "mechanism": raw_data[i][11],
                "year": raw_data[i][42],
                "funding": mk_int(raw_data[i][49]),
                #"funding": funding, # by institution
                })
    
    new_data = []
    for item in data:
        if item["funding"] != 0:
            new_data.append(item)
    
    data = []
    test_data = []
    for item in new_data:
        if item["year"] != "2021":
            data.append(item)
        else:
            test_data.append(item)

    with open("data.pkl", 'wb') as handle:
        pickle.dump(data, handle)
        
    with open("test-data.pkl", 'wb') as handle:
        pickle.dump(test_data, handle)
    
    print("Processed data N: {}".format(str(len(data))))
    return data, test_data

def feature_extraction(data, num_features, max_df):
    """

    Parameters
    ----------
    data : parallels "data" from process_data
    
    Returns 
    -------
    features:
        funding - institution's funding in 2020
        year - one-hot encoded year
        text - tfidf for all text data
    """
    input_text = [item["text"] for item in data]
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=max_df, max_features=num_features).fit(input_text) #, token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b'
    processed_text = vectorizer.transform(input_text)
    
    with open("processed-data.pkl", 'wb') as handle:
        pickle.dump(processed_text, handle)
        
    with open("vectorizer.pkl", 'wb') as handle:
        pickle.dump(vectorizer, handle)
    print("Data vectorized.")

def get_features():
    """
    Parameters
    ----------
    None

    Returns
    -------
    A text file with a list of TF-IDF feature names
    """
    # Vectorizer to convert raw documents to TF-IDF features
    vector = pickle.load(open("vectorizer.pkl","rb"))
    
    # Get feature names and save to text file
    centroid_file = open("features", "w")
    for i in vector.get_feature_names():
        centroid_file.write(i)
        centroid_file.write("\n")
    centroid_file.close()
    print(len(vector.get_feature_names()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_features',
        type=int,
        required=True,
        help='number of features',
        default=2000,
        )
    parser.add_argument(
        '--max_df',
        type=float,
        required=True,
        help='maximum document frequency',
        default=0.5,
        )
    FLAGS, unparsed = parser.parse_known_args()
    
    file = '/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/raw_data.csv'
    funding_file = '/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/institution-funding.csv'
    data, test_data = process_data(file, funding_file)
    feature_extraction(data, FLAGS.max_features, FLAGS.max_df)
    data = data + test_data
    
    # By Funder
    funders = np.unique(np.array([item["administration"] for item in data]))
    output = [["Funder", "Number of awards", "Value of awards"]]
    for funder in funders:
        count = len([item for item in data if item["administration"] == funder])
        amount = sum([item["funding"] for item in data if item["administration"] == funder])
        output.append([funder, count, amount])
    with open('by_funder.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)
        
    # By Year
    years = np.unique(np.array([item["year"] for item in data]))
    output = [["Year", "Number of awards", "Value of awards"]]
    for year in years:
        count = len([item for item in data if item["year"] == year])
        amount = sum([item["funding"] for item in data if item["year"] == year])
        output.append([year, count, amount])
    with open('by_year.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)
        
    # By Mechanism
    mechanisms = np.unique(np.array([item["mechanism"] for item in data]))
    output = [["Mechanism", "Number of awards", "Value of awards"]]
    for mech in mechanisms:
        count = len([item for item in data if item["mechanism"] == mech])
        amount = sum([item["funding"] for item in data if item["mechanism"] == mech])
        output.append([mech, count, amount])
    with open('by_mechanism.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)
        