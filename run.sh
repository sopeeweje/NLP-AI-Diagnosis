pipenv install
pipenv run python feature_extraction.py --max_df 0.5 --max_features 1000
pipenv run python find_centroids.py --max_df 0.5 --max_features 1000 --k 32
pipenv run python analyze_clusters.py --k 32 --trials 50