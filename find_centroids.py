import pickle
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import seaborn as sns
import argparse
import csv
import numpy as np

def find_centroids(data, test, max_df, max_features, k, plot=False, score=False):
    # Load documents
    input_text = [item["text"] for item in data]
    titles = [item["title"] for item in data]

    # Create vectorizers
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=max_df, max_features=max_features).fit(input_text) # frequency - inverse frequency

    # Processed data
    tfidf_processed = tfidf_vectorizer.transform(input_text)

    # number of topics and number of defining words per topic
    num_topics = k
    n_top_words = 10

    # Fit data with LDA TFIDF
    lda = LatentDirichletAllocation(num_topics)
    lda.fit(tfidf_processed)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    if plot:
        plot_top_words(lda, tfidf_feature_names, n_top_words, 'Topics in LDA model', k)

    with open("lda_centroids.pkl", 'wb') as handle:
        pickle.dump(lda.components_, handle)

    all_features = []
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [tfidf_feature_names[i] for i in top_features_ind]
        all_features.append(top_features)

    by_doc = []
    for i in range(len(input_text)):
        weights = lda.transform(tfidf_processed[i])
        max_index_col = np.argmax(weights, axis=1)[0]
        new = [titles[i], max_index_col, np.amax(weights, axis=1), all_features[max_index_col]]
        by_doc.append(new)

    with open('lda_weights.csv', 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(by_doc)
    # perplexity = 0
    # if score:
    #     test_input = [item["text"] for item in test]
    #     test = tfidf_vectorizer.transform(test_input)
    #     perplexity = lda.perplexity(test)

    return lda.components_ #, perplexity

def plot_top_words(model, feature_names, n_top_words, title, k):
    factors = []
    for i in range(1, k+1):
        if k / i == i:
            factors.extend([i,i])
        elif k % i == 0:
            factors.append(i)
    dim1, dim2 = factors[int(len(factors)/2)], factors[int(len(factors)/2-1)]

    fig, axes = plt.subplots(dim1, dim2, sharex=True)
    axes = axes.flatten()
    features_csv = []
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        features_csv.append(top_features)
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}')
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major')
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title)

    with open('lda_features.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(features_csv)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.savefig('topic_chart.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_df',
        type=float,
        required=True,
        help='maximum document frequency',
        default=0.5,
        )
    parser.add_argument(
        '--max_features',
        type=int,
        required=True,
        help='maximum number of features',
        default=1000,
        )
    parser.add_argument(
        '--k',
        type=int,
        required=True,
        help='number of clusters',
        default=30,
        )
    FLAGS, unparsed = parser.parse_known_args()

    data = pickle.load(open("data.pkl","rb"))
    test = pickle.load(open("test-data.pkl","rb"))
    max_df = FLAGS.max_df
    max_features = FLAGS.max_features
    k = FLAGS.k
    print("Finding {} initial centroids...".format(str(k)))
    centroids = find_centroids(data, test, max_df, max_features, k, plot=True)
    print("Centroids located, see plot for topic descriptions.")
