# NLP-AI-Diagnosis
<p>A natural language processing approach to determining frontiers in AI-based disease characterization using NIH award data</p>
<h3>Data collection</h3>
<p>Begin by collecting data to analyze as follows:</p>
<ol>
  <li>Go to the <a target="_blank" href="https://reporter.nih.gov/advanced-search">NIH RePORTER search tool</a> and complete your query of interest. Download the returned awards as a csv (including all columns, max 15,000 per download) to the project directory, and remove the initial lines of the csv that describe the query so you are left with the data table. Title the file "raw_data.csv".</li>
  <li>Export the papers associated with your RePORTER query as a csv. Download the file to the project directory and title the file "papers.csv".</li>
  <li>Go to the <a target="_blank" href="https://icite.od.nih.gov/analysis">NIH iCite search tool</a>. Copy the PMIDs from papers.csv (max 10,000 at a time) to query iCite. Go to the "Citations" tab of the return and select "Export (this module)". Save the file in the project directory as "citations.csv".</li>
</ol>

<h3>Determine TF-IDF parameters</h3>

<h3>Analysis</h3>
<p>Perform the analysis by running the run.sh shell script in command line/terminal: <code>sh run.sh</code>.</p>
<p>This performs the following:</p>
<ul>
  <li><code>pipenv install</code> - initializes python virtual environment</li>
  <li><code>pipenv run python feature_extraction.py</code> - Performs feature extraction with document corpus</li>
  <li><code>pipenv run python find_centroids.py --max_df ### --max_features ### --k ###</code> - initializes centroids with LDA. max_df = <a target="_blank" href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html">maximum document frequency</a>, max_features = <a target="_blank" href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html">maximum n-gram features</a>, k = number of clusters</li>
  <li><code>pipenv run python analyze_clusters.py --k ### --trials ### </code> - creates the clusters with K-Means Clustering and analyzes funding and citation data. k = number of clusters, trials = number of clustering trials to run</li>
</ul>
<p>Parameters can be edited in run.sh to meet what was empirically determined to be optimal.</p>

<h3>Results</h3>
