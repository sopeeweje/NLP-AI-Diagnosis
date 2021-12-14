import pandas as pd
import nltk
import os
import numpy as np
import random
from nltk.metrics import agreement
from sklearn.metrics import confusion_matrix

ratings = pd.read_csv("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/results/10-25-2021--134436/score_sheet_combined.csv") #TOCHANGE needs path to real data csv

#need to add "abstract id" to convert to tidy format
num = pd.DataFrame({"id": range(len(ratings))})
ratings_id = pd.concat([num, ratings], axis=1)

#create three pairwise comparisons: rater_1 (Rater 1 and NLP), rater_2 (Rater 2 and NLP), human (Rater 1 and Rater 2)
rater_1 = ratings_id.drop(columns = 'Rater 2')
rater_2 = ratings_id.drop(columns = 'Rater 1')
human = ratings_id.drop(columns = 'Actual')

def convert_to_tidy(df):
    df_str = df.applymap(str) #cohen's kappa needs ratings as strings
    df_long = pd.melt(df_str, id_vars = "id") #creates tidy format
    df_reorder = df_long[['variable', 'id', 'value']] #required order for cohens kappa
    rating_list = [df_str.iloc[:, 1].values.tolist(), df_str.iloc[:, 2].values.tolist()] #needed for agreement
    return(rating_list, df_reorder)

def print_values(rater_list, rater_tidy, desc): #prints out Kappa and agreement
    agree_obj = agreement.AnnotationTask(data = rater_tidy.values.tolist())
    # cm = confusion_matrix(rater_list[0], rater_list[1])
    
    print(desc + ":")
    print("Cohen's Kappa: " + str(agree_obj.kappa()))
    print("Agreement: " + str(agree_obj.avg_Ao()))
    
    return(agree_obj.kappa())


def generate_ratings(n_ratings, k): 
    variable = ["rater_1"] * n_ratings + ["rater_2"] * n_ratings
    rating_num = list(range(n_ratings)) * 2
    value = np.random.randint(1, k, n_ratings * 2).tolist() #random ratings from k categories
    
    rating_num_str = list(map(str, rating_num)) #convert to string
    value_str = list(map(str, value))
    
    ratings_list = [None] * n_ratings * 2
    for i in range(n_ratings * 2):
        ratings_list[i] = [variable[i], rating_num_str[i], value_str[i]]
    
    agree_obj = agreement.AnnotationTask(data = ratings_list)
    return(agree_obj.kappa())

rater_1_list, rater_1_tidy = convert_to_tidy(rater_1)
rater_2_list, rater_2_tidy = convert_to_tidy(rater_2)
human_list, human_tidy = convert_to_tidy(human)
    
rater_1_kappa = print_values(rater_1_list, rater_1_tidy, "NLP vs Rater 1")
rater_2_kappa = print_values(rater_2_list, rater_2_tidy, "NLP vs Rater 2")
human_kappa = print_values(human_list, human_tidy, "Rater 1 vs Rater 2")

# np.random.seed(30)
# n_iterations = 100000
# k_categories = 75 #TOCHANGE How many categories did Raters have access to?

# kappa = [None] * n_iterations
# for i in range(n_iterations):
#     kappa[i] = generate_ratings(n_ratings = 60, k = k_categories)

# rater_1_pval = len([val for val in kappa if val >= rater_1_kappa])/n_iterations #proportion of values as extreme or more
# rater_2_pval = len([val for val in kappa if val >= rater_2_kappa])/n_iterations
# human_pval = len([val for val in kappa if val >= human_kappa])/n_iterations

# print(rater_1_pval)
# print(rater_2_pval)
# print(human_pval)