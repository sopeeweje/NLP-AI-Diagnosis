#https://gist.github.com/bosborne/8efbd21ffbe3dc8d057d80539114ab07
#https://api.reporter.nih.gov/
#https://github.com/muschellij2/fedreporter

def get_terms(technique_terms, purpose_terms):
    search_string = ""
    term_lists = [technique_terms, purpose_terms]
    for term_list in term_lists:
        for term in term_list:
            if term == term_list[0]:
                search_string += '("' + term + '" or '
            elif term == term_list[-1] and term_list != term_lists[-1]:
                search_string += ('"' + term + '") and ')
            elif term == term_list[-1] and term_list == term_lists[-1]:
                search_string += ('"' + term + '")')
            else:
                search_string += ('"' + term + '" or ')
    print(search_string)

if __name__ == "__main__":
    
    # technique_terms = [
    #     "Artificial Intelligence",
    #     "Machine Learning",
    #     "Deep Learning",
    #     "Natural Language Processing",
    #     "Random Forest",
    #     "Logistic Regression",
    #     "LSTM",
    #     "RNN",
    #     "CNN",
    #     "Federated Learning",
    #     "Decision Tree",
    #     "Support Vector Machine",
    #     "Bayesian Learning",
    #     "Gradient Boosting",
    #     "Computational Intelligence",
    #     "Naive Bayes",
    #     "Computer Vision",]
    
    # purpose_terms = [
    #     "Diagnosis",
    #     "Early Detection",
    #     "Decision Support",
    #     "Screening"
    #     ]
    
    technique_terms = [
        "Artificial Intelligence",
        "Machine Learning",
        "Deep Learning",
        "Natural Language Processing",
        "Random Forest",
        "Logistic Regression",
        "LSTM",
        "RNN",
        "CNN",
        "Federated Learning",
        "Decision Tree",
        "Support Vector Machine",
        "Bayesian Learning",
        "Gradient Boosting",
        "Computational Intelligence",
        "Naive Bayes",
        "Computer Vision",]
    
    purpose_terms = [
        "disparities",
        "disparity",
        "race",
        "gender",
        "demographic",
        "inequity",
        "inequities"]
    
    get_terms(technique_terms, purpose_terms)