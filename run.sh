pipenv lock --clear
pipenv install
pipenv run python feature_extraction.py --max_df 0.5 --max_features 1500
pipenv run python find_centroids.py --max_df 0.5 --max_features 1500 --k 60
#pipenv run python analyze_clusters.py --k 60 --trials 1