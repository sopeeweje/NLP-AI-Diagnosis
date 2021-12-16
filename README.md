# NLP-AI-Medicine
<p>A natural language processing approach to determining frontiers in medical AI using NIH award data</p>

<h2>Quick Start</h2>

<h3>Environment</h3>
<p>Install pipenv if not already installed: <code>pip install pipenv</code>. Pipenv is used to create a python virtual environment that includes the libraries necessary to conduct the analysis.</p>

<h3>Analysis</h3>
<p>Perform the analysis by running the run.sh shell script from the project directory: <code>sh run.sh</code>.</p>
<p>This performs the following:</p>
<ul>
  <li><code>pipenv lock --clear</code> - initializes python virtual environment</li>
  <li><code>pipenv install</code> - initializes python virtual environment</li>
  <li><code>pipenv run python setup.py</code> - Set up directory structure and install necessary NLTK libraries</li>
  <li><code>pipenv run python setup.py</code> - pipenv run python nih_reporter_query.py --search_terms "search_terms.txt" --start_year 1985 --end_year 2021</li>
  <li><code>pipenv run python feature_extraction.py --max_df 0.1 --max_features 500</code> - Performs feature extraction with document corpus</li>
  <li><code>pipenv run python find_k.py --trials 5 --max_k 120 --num_features 500</code> - empiric search for K</li>
  <li><code>pipenv run python analyze_clusters.py --k ### --trials ### </code> - creates the clusters with K-Means Clustering and analyzes funding and citation data. k = number of clusters, trials = number of clustering trials to run</li>
</ul>

<h2>Details</h2>
<h3>Data collection</h3>
<p>Awards and publications are collected from the <a target="_blank" href="https://reporter.nih.gov/advanced-search">NIH RePORTER database</a> while citation data are collected from the <a target="_blank" href="https://icite.od.nih.gov/analysis">NIH iCite search tool</a>. Enter the search terms related to your topic of interest in "search_terms.txt" with a new line for each term. The query is executed with "OR" logic and excludes subprojects (see "nih_reporter_query.py", line 31 to modify criteria per the <a target="_blank" href="https://api.reporter.nih.gov/">NIH RePORTER API)</a>. Example advanced search:</p>

```
params = {

            "criteria":
            {
               "fiscal_years": [2021],
               "advanced_text_search":
               {
                  "operator": "advanced", 
                  "search_field": "projecttitle,terms,abstracttext", 
                  "search_text": " (\"dna\" or \"rna\" or \"gene\") and (\"machine learning\" or \"artificial intelligence\" or \"nlp\") "
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
```

<h3>Feature extraction</h3>  

<h3>Results</h3>
<p>Results from each run are returned in the "results" directory:</p>
<ul>
  <li>actual_vs_projected.png - linear regression comparing projected 2021 funding by cluster to actual 2021 funding. Quality of this analysis assumes that funding for your dataset has followed an exponential trend year-to-year</li>
  <li>centroid.txt - text file with centroid words listed for each cluster</li>
  <li>clusters - csv files containing awards assigned to each cluster (2000-2020)</li>
  <li>clusters_test - csv files containing awards assigned to each cluster (2021)</li>
  <li>final_data.csv - summary table</li>
  <li>funding_by_year.png - funding for each cluster plotted by year with exponential fit and 95% CI bounds</li>
  <li>umap.png -  <a target="_blank" href="https://arxiv.org/abs/1802.03426">UMAP</a> visualization of clusters</li>
  <li>lda_centroids.pkl - initial centroids from LDA</li>
  <li>topic chart.png - topic chart from LDA</li>
  <li>supp_info.docx - Microsoft word document with tables contain 5 representative awards from each cluster, selected by maximum <a target="_blank" href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html">silhouette score</a>.</li>
  <li>model_clustering.pkl - pickle containing dictionary with the following keys:
    <ul>
      <li>"yr_avg_cost" - average award funding by cluster</li>
      <li>"yr_total_cost" - total award funding by cluster</li>
      <li>"size" - cluster size</li>
      <li>"data_by_cluster" - nested lists of dictionaries representing individual awards assigned to each cluster</li>
      <li>"centroids" - list of lists of centroids by cluster (first 10 elements)</li>
      <li>"model" - MiniBatchKMeans model</li>
      <li>"complete_centroids" - list of lists of centroids by cluster (all elements)</li>
      <li>"labels" - ordered list of cluster labels by award (same order as data loaded by from data.pkl)</li>
    </ul>
  </li>
</ul>

<h2>Directory strucure</h2>

```
├── LICENSE
├── Pipfile
├── Pipfile.lock
├── README.md
├── analyze_clusters.py
├── data
│   ├── by_funder.csv
│   ├── by_mechanism.csv
│   ├── by_year.csv
│   ├── citations.csv
│   ├── data.pkl
│   ├── features
│   ├── nih_institutes.csv
│   ├── processed-data.pkl
│   ├── publications.csv
│   ├── raw_data.csv
│   ├── test-data.pkl
│   └── vectorizer.pkl
├── feature_extraction.py
├── figures
│   ├── ...
├── find_k.py
├── nih_reporter_query.py
├── results
│   ├── 12-15-2021--214341
│   │   ├── centroids
│   │   ├── clusters
│   │   │   ├── cluster-0.csv
│   │   │   ├── cluster-1.csv
│   │   │   ├── ...
│   │   │   ├── cluster-30.csv
│   │   ├── clusters_test
│   │   │   ├── cluster-0.csv
│   │   │   ├── cluster-1.csv
│   │   │   ├── ...
│   │   │   ├── cluster-30.csv
│   │   ├── final_data.csv
│   │   ├── model_clustering.pkl
│   │   ├── supp_info.docx
│   │   └── umap.png
├── run.sh
├── search_terms.txt
└── setup.py
```
