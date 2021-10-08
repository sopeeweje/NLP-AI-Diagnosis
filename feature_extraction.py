import csv
import nltk
from nltk.corpus import stopwords
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
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('punkt')
nltk.download('wordnet')
import argparse

def mk_int(s):
    s = s.strip()
    return int(s) if s else 0

def process_data(data_file):
    data = []
    with open(data_file, newline='', encoding='utf8') as csvfile:
        raw_data = list(csv.reader(csvfile))
        ids = []
        print("Raw data N: {}".format(str(len(raw_data))))
        for i in range(1,len(raw_data)):
            if (raw_data[i][6] in ids) or (raw_data[i][11][0] in ['Z','T']):
                #ids.append(raw_data[i][6])
                continue
            else:
                ids.append(raw_data[i][6])
            abstract = raw_data[i][1].replace('\n',' ')
            title = raw_data[i][3]
            relevance = raw_data[i][4].replace('\n',' ')
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

class LemmaStemmerTokenizer:
    """
    Tokenizer that lemmatizes and stems words
    """
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.ps = PorterStemmer()
    def __call__(self, doc):
        # leaving out stemming for now
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t.isalpha()]

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
    vectorizer = TfidfVectorizer(tokenizer=LemmaStemmerTokenizer(), stop_words='english', ngram_range=(1,2), max_df=max_df, max_features=num_features).fit(input_text) #, token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b'
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
    centroid_file = open("features", "w", encoding='utf8')
    for i in vector.get_feature_names():
        centroid_file.write(i)
        centroid_file.write("\n")
    centroid_file.close()
    # print(len(vector.get_feature_names()))
    

# get data from csv file
def get_by_year_data():
    years = []
    awards = []
    values = []

    with open('by_year.csv', newline='', encoding='utf8') as csvfile:
        raw_data = csv.reader(csvfile)
        next(raw_data) #skip the header

        for entry in raw_data:
            years.append(int(entry[0]))
            awards.append(int(entry[1]))
            values.append(float(entry[2]))

    return years, awards, values

# plot award number & values on same plot, 2 y-axes
def plot_by_year():
    years, awards, values = get_by_year_data()

    fig, ax1 = plt.subplots()

    # award values
    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Award Values (billions USD)', color=color)
    bar = ax1.bar(years, [v/1e9 for v in values], color=color) #yerr=errorbar
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.bar_label(bar, fmt='$%.1f', padding=1, size='x-small')
    ax2 = ax1.twinx()

    # award size (avg) = total value/num awards
    color = 'tab:green'
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Average Award Size (thousands USD)', color=color)
    ax2.plot(years, [values[i]/1e3/awards[i] for i in range(len(years))], color=color) #yerr=errorbar
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Total Awards by Year')
    plt.savefig('table2.png') 

    # plt.show()


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

    file = 'raw_data.csv'
    data, test_data = process_data(file)
    feature_extraction(data, FLAGS.max_features, FLAGS.max_df)
    get_features()
    # data = data + test_data

    # By Funder
    funders = np.unique(np.array([item["administration"] for item in data]))
    output = [["Funder", "Number of awards", "Value of awards"]]
    for funder in funders:
        count = len([item for item in data if item["administration"] == funder])
        amount = sum([item["funding"] for item in data if item["administration"] == funder])
        output.append([funder, count, amount])
    with open('by_funder.csv', 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)

    # By Year
    years = np.unique(np.array([item["year"] for item in data]))
    output = [["Year", "Number of awards", "Value of awards"]]
    for year in years:
        count = len([item for item in data if item["year"] == year])
        amount = sum([item["funding"] for item in data if item["year"] == year])
        output.append([year, count, amount])
    with open('by_year.csv', 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)
    plot_by_year()
    
    # By Mechanism
    mechanisms = np.unique(np.array([item["mechanism"] for item in data]))
    output = [["Mechanism", "Number of awards", "Value of awards"]]
    for mech in mechanisms:
        count = len([item for item in data if item["mechanism"] == mech])
        amount = sum([item["funding"] for item in data if item["mechanism"] == mech])
        output.append([mech, count, amount])
    with open('by_mechanism.csv', 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)
