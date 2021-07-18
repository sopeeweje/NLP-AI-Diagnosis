# NLP-AI-Medicine
<p>A natural language processing approach to determining frontiers in medical AI using NIH award data</p>
<h3>Data collection</h3>
<p>Begin by collecting data to analyze as follows:</p>
<ol>
  <li>Go to the <a target="_blank" href="https://reporter.nih.gov/advanced-search">NIH RePORTER search tool</a> and complete your query of interest. Download the returned awards as a csv (including all columns, max 15,000 per download) to the project directory, and remove the initial lines of the csv that describe the query so you are left with the data table. Title the file "raw_data.csv".</li>
  <li>Export the papers associated with your RePORTER query as a csv. Download the file to the project directory and title the file "papers.csv".</li>
  <li>Go to the <a target="_blank" href="https://icite.od.nih.gov/analysis">NIH iCite search tool</a>. Copy the PMIDs from papers.csv (max 10,000 at a time) to query iCite. Go to the "Citations" tab of the return and select "Export (all modules)". Save the file in the project directory as "citations.csv".</li>
  <li>Export the 2020 NIH funding by institution data from <a target="_blank" href="https://report.nih.gov/award/index.cfm">NIH RePORT</a>. This data is not incorporated in the analysis but can be incorporated if desired.</li>
</ol>

<h3>Environment</h3>
<p>Install pipenv if not already installed: <code>pip install pipenv</code>. Pipenv is used to create a python virtual environment that includes the libraries necessary to conduct the analysis.</p>

<h3>Determine parameters</h3>
<p>Determine an appropriate number of <a target="_blank" href="https://monkeylearn.com/blog/what-is-tf-idf/">TF-IDF</a> features by running the feature identification script: <code>sh find_num_features.sh</code>.</p>
<p>Once the size of the feature set is determined, find an appropriate number of clusters to divide the dataset: <code>sh find_k.sh</code>.</p>
<p>Optimal parameter values can be selected by applying the "elbow method" to the resultant graphs.</p>

<h3>Analysis</h3>
<p>Perform the analysis by running the run.sh shell script from the project directory: <code>sh run.sh</code>.</p>
<p>This performs the following:</p>
<ul>
  <li><code>pipenv install</code> - initializes python virtual environment</li>
  <li><code>pipenv run python feature_extraction.py</code> - Performs feature extraction with document corpus</li>
  <li><code>pipenv run python find_centroids.py --max_df ### --max_features ### --k ###</code> - initializes centroids with LDA. max_df = <a target="_blank" href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html">maximum document frequency</a>, max_features = <a target="_blank" href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html">maximum n-gram features</a>, k = number of clusters</li>
  <li><code>pipenv run python analyze_clusters.py --k ### --trials ### </code> - creates the clusters with K-Means Clustering and analyzes funding and citation data. k = number of clusters, trials = number of clustering trials to run</li>
</ul>
<p>Parameters can be edited in run.sh to meet what was empirically determined to be optimal or left as naive defaults.</p>

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
