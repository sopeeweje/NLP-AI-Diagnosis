# Set up virtual environment
pipenv lock --clear
pipenv install

# Set up project directory
pipenv run python setup.py

# Get projects, publications, and citation data
# pipenv run python nih_reporter_query.py --search_terms "search_terms.txt" --operator "advanced" --start_year 1985 --end_year 2021

# Extract features
pipenv run python feature_extraction.py --max_df 0.1 --max_features 1000

# Find k (uncomment if needed)
# pipenv run python find_k.py --trials 5 --max_k 120 --num_features 500

# Run analysis
pipenv run python analyze_clusters.py --k 50 --trials 1