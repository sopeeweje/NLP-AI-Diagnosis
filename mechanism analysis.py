import pandas as pd
import numpy as np
import scipy as sp
import itertools
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()

stats = importr('stats')

cat_data = pd.read_csv("/Users/Sope/Documents/GitHub/NLP-AI-Diagnosis/results/10-25-2021--134436/categories by mechanism.csv") #TOCHANGE

categories = cat_data.iloc[:, 0].tolist()
grant_types = cat_data.columns[1:]
cont_table = np.asarray(cat_data.iloc[:, 1:cat_data.shape[1]])

cat_pairwise = list(itertools.combinations(range(0, len(categories)), 2))

print(cont_table)

n_simulations = 100000 #TOCHANGE higher n_simulations leads to more accurate results - 100,000 is probably sufficient, but can do 1,000,000 if it is "final" data
alpha = 0.05 #TOCHANGE 0.05 is a good default for upper range of p-value

grant_cont_tables = []
for i in range(0, len(grant_types)):
    is_grant_i = cont_table[:, i]
    not_grant_i = np.delete(cont_table, i, axis = 1).sum(axis = 1)
    
    bin_cont_table = np.column_stack((is_grant_i, not_grant_i))
    
    grant_type_p = stats.fisher_test(bin_cont_table, 
                        simulate_p_value = True,
                        B = n_simulations)[0]
    
    grant_cont_tables.append(bin_cont_table)
    
    print("P-value for " + grant_types[i] + " vs Not is: " + str(grant_type_p))
    
grant_col = []
cat1_col = []
cat2_col = []
pval_col = []
sig_posthoc = []
cat1_prop = []
cat2_prop = []

for grant_index in range(0, len(grant_cont_tables)):
    bin_cont_table = grant_cont_tables[grant_index]
    
    for pairwise_index in range(0, len(cat_pairwise)):
        comparison = cat_pairwise[pairwise_index]
        
        pairwise_cont_table = bin_cont_table[comparison, :]
        cat_props = pairwise_cont_table[:, 0] / pairwise_cont_table.sum(axis = 1)
        category_1 = categories[comparison[0]]
        category_2 = categories[comparison[1]]

        pairwise_p = stats.fisher_test(pairwise_cont_table, 
                        simulate_p_value = True,
                        B = n_simulations)[0]
        bonf_sig = pairwise_p < alpha / len(cat_pairwise)

        grant_col.append(grant_types[grant_index])
        cat1_col.append(category_1)
        cat2_col.append(category_2)
        pval_col.append(pairwise_p)
        sig_posthoc.append(bonf_sig)
        cat1_prop.append(cat_props[0])
        cat2_prop.append(cat_props[1])
        
pairwise_fishers = pd.DataFrame({
    "grant_type" : grant_col,
    "category_1" : cat1_col,
    "category_2" : cat2_col,
    "uncorrected_pvalues" : pval_col,
    "sig_post_correction" : sig_posthoc,
    "category_1_prop" : cat1_prop,
    "category_2_prop" : cat2_prop
},
    index = list(range(len(grant_col))))

print(pairwise_fishers)
pairwise_fishers.to_csv("pairwise_fishers.csv", encoding='utf-8', index=False)