import pickle
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import seaborn as sns

def find_centroids(data, test, max_df, max_features, k, plot=False, score=False):
    # Load documents
    input_text = [item["text"] for item in data]
    
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
        plot_top_words(lda, tfidf_feature_names, n_top_words, 'Topics in LDA model')
    
    with open("lda_centroids.pkl", 'wb') as handle:
        pickle.dump(lda.components_, handle)
    
    perplexity = 0
    if score:
        test_input = [item["text"] for item in test]
        test = tfidf_vectorizer.transform(test_input)
        perplexity = lda.perplexity(test)
    
    return lda.components_, perplexity

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(5, 6, sharex=True) #, figsize=(30, 15)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}')#,fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major')#, labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title)#, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    plt.savefig('topic chart.png')
    
if __name__ == "__main__":
    data = pickle.load(open("data.pkl","rb"))
    test = pickle.load(open("test-data.pkl","rb"))
    max_df = 0.5
    max_features = 1000
    scores = []
    ks = range(30,31)
    for k in ks:
        print(k)
        centroids, score = find_centroids(data, test, max_df, max_features, k, plot=True)
        scores.append(score)
    plt.figure()
    ax = sns.lineplot(list(ks), scores)
    ax.set(xlabel='Number of Clusters', ylabel='Perplexity score')
    
    #tf_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2), max_df=max_df, max_features=max_features).fit(input_text) # frequency only #, min_df=0.001
    #tf_processed = tfidf_vectorizer.transform(input_text)
    
    # Fit data with LDA TF
    # lda = LatentDirichletAllocation(num_topics)
    # lda.fit(tf_processed)
    # tf_feature_names = tf_vectorizer.get_feature_names()
    # plot_top_words(lda, tf_feature_names, n_top_words, 'Topics in LDA model')