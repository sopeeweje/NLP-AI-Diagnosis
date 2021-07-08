# NLP-AI-Diagnosis
<p>A natural language processing approach to determining frontiers in AI-based disease characterization using NIH award data</p>
<h3>Data collection</h3>
<p>Begin by collecting data to analyze as follows:</p>
<ol>
  <li>Go to the <a target="_blank" href="https://www.adafruit.com/product/4026">NIH RePORTER search tool</a> and complete your query of interest. Download the returned awards as a csv (including all columns, max 15,000 per download) to the project directory, and remove the initial lines of the csv that describe the query so you are left with the data table. Title the file "raw_data.csv".</li>
  <li>Export the papers associated with your RePORTER query as a csv. Download the file to the project directory and title the file "papers.csv".</li>
  <li>Go to the <a target="_blank" href="https://icite.od.nih.gov/analysis">NIH iCite search tool</a>. Copy the PMIDs from papers.csv (max 10,000 at a time) to query iCite. Go to the "Citations" tab of the return and select "Export (this module)". Save the file in the project directory as "citations.csv".</li>
</ol>

<h3>Determine TF-IDF parameters</h3>

<h3>Analysis</h3>

<h3>Results</h3>
