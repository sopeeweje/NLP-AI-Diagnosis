pipenv install
pipenv run python feature_extraction.py 
pipenv run python find_centroids.py --max_df 0.5 --max_features 1000 --k 30
pipenv run python analyze_clusters.py --k 30 --trials 1