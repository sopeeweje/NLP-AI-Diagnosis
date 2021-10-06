#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 18:52:14 2021

@author: Sope
"""
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import summary_table
import warnings


def cluster_anova():
    # Result labels
    result = pickle.load(open("results/07-18-2021--132946/model_clustering.pkl","rb"))
    labels = result["labels"]
    clusters = result["data_by_cluster"]
    funding = [sum(item["funding"] for item in clusters[i]) for i in range(len(clusters))]
    total_citations, total_papers, apts_95, apts, lower, upper, listed_apts = get_citations(clusters)
    
    all_apts = []
    labels = []
    for i in range(len(listed_apts)):
        all_apts.extend(listed_apts[i])
        labels.extend([i for j in range(len(listed_apts[i]))])
        
    # Load data frame
    df = pd.DataFrame({'score': all_apts,
                   'group': labels}) 
    
    # perform Tukey's test
    tukey = pairwise_tukeyhsd(endog=df['score'],
                              groups=df['group'],
                              alpha=0.05)
    df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    for index in range(len(df)):
        group1 = df["group1"][index]
        group2 = df["group2"][index]
        p = df[df["group1"]==group1][df["group2"]==group2]["p-adj"][index]
        if p > 0.05:
            p = 5
        elif 0.01 < p <= 0.05:
            p = 4
        elif 0.005 < p <= 0.01:
            p = 3
        elif 0.001 < p <= 0.005:
            p = 2
        elif p <= 0.001:
            p = 1
        if np.mean(listed_apts[group2]) > np.mean(listed_apts[group1]):
            p = -p
        df.at[index,'p-adj']=p

    df.to_csv('anova_award.csv', index=False, header=True)
    piv = pd.pivot_table(df, values="p-adj",index=["group2"], columns=["group1"])
    plt.figure()
    myColors = ((1.0, 1.0, 1.0),
                #(255/256, 215/256, 205/256),
                (255/256, 175/256, 158/256),
                (254/256, 134/256, 112/256),
                (243/256, 88/256, 68/256),
                (228/256, 13/256, 25/256),
                (43/256, 12/256, 240/256),
                (117/256, 72/256, 245/256),
                (160/256, 118/256, 249/256),
                (196/256, 163/256, 252/256),
                #(227/256, 208/256, 254/256),
                (1.0, 1.0, 1.0))
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
    xlabels = []
    ylabels = []
    with open('figures and tables/tick_labels.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(0,len(raw_data)-1):
            xlabels.append(raw_data[i][0])
            ylabels.append(raw_data[i+1][0])
    sns.set(font_scale=0.6)
    ax = sns.heatmap(piv, vmin=-5, vmax=5, xticklabels=True, yticklabels=True, cmap=cmap, cbar=True, square=True,) #'format': '%.0f%%', 
    ax.set_xlabel("Cluster 1")
    ax.set_ylabel("Cluster 2")
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels) 
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
    colorbar.set_ticklabels(['Cluster 2 > 1, p < 0.05',
                             'Cluster 2 > 1, p < 0.01',
                             'Cluster 2 > 1, p < 0.005',
                             'Cluster 2 > 1, p < 0.001', 
                             'Cluster 1 > 2, p < 0.001',
                             'Cluster 1 > 2, p < 0.005',
                             'Cluster 1 > 2, p < 0.01',
                             'Cluster 1 > 2, p < 0.05',])
    # ax.collections[0].colorbar.set_ticklabels(['10', '30'])
    for text in ax.texts:
        if abs(float(text.get_text())) > 0.05:
            text.set_text("")
    ax.figure.tight_layout()
    plt.savefig('avg_award_ANOVA.png')   


def generate_color_selection(max_color, min_color, n):
    """
    Parameters
    ----------
    max_color: tuple of rgb for max color in gradient, where each value is an int divided by 256
    min_color: tuple of rgb for min color in gradient, where each value is an int divided by 256
    n: number of rgb tuples for colors to generate

    Returns
    -------
    list of n rgb tuples

    """

    num_color_indices = len(max_color)  # should be 3 for rgb

    # initialize color_selection with min_color
    color_selection = [max_color]

    # find range of rgb values between min and max colors, divide based on n
    color_intervals = [(min_color[color_index] - max_color[color_index]) / (n - 1) for
                       color_index in range(num_color_indices)]

    # add to color_selection, incrementing by the color_interval
    for color_num in range(1, n):
        next_color = tuple([(max_color[color_index] + color_num * color_intervals[color_index])
                            for color_index in range(num_color_indices)])
        color_selection.append(next_color)

    return color_selection


def plot_translation_metrics(results_directory):
    apt = []
    cpof = []

    # technical and clinical clusters
    clinical_categories = []
    technical_categories = []

    with open(f'{results_directory}/final_data.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)):
            description = raw_data[i][18]
            category = raw_data[i][19]
            clinical_technical = raw_data[i][20]
            if category == 'N/A':
                continue
            apt.append([description, category, float(raw_data[i][5]), float(raw_data[i][6]), float(raw_data[i][7])])
            cpof.append([description, category, float(raw_data[i][9])])
            if clinical_technical == 'Clinical' and category not in clinical_categories:
                clinical_categories.append(category)
            elif clinical_technical == 'Technical' and category not in technical_categories:
                technical_categories.append(category)
            elif clinical_technical not in ['Clinical', 'Technical']:
                warnings.warn(f"Cluster {raw_data[i][0]} (description: {description}, category: {category}) "
                              f"not classified as 'Clinical' or 'Technical')")

    apt.sort(key=lambda x: x[2])
    ci95_apt = [[abs(i[2]-i[3]), abs(i[2]-i[4])] for i in apt]
    ci95_apt = list(map(list, zip(*ci95_apt)))
    cpof.sort(key=lambda x: x[2])
    categories = np.array(sorted(clinical_categories) + sorted(technical_categories))

    # color_selection = rainbow_color_stops(len(categories))
    min_clinical_color = (244/256, 191/256, 191/256)
    max_clinical_color = (255/256, 100/256, 55/256)
    num_clinical_categories = len(clinical_categories)
    clinical_color_selection = generate_color_selection(max_clinical_color, min_clinical_color, num_clinical_categories)

    min_technical_color = (191/256, 198/256, 244/256)
    max_technical_color = (55/256, 100/256, 255/256)
    num_technical_categories = len(technical_categories)
    technical_color_selection = generate_color_selection(max_technical_color, min_technical_color,
                                                         num_technical_categories)

    color_selection = clinical_color_selection + technical_color_selection
    clinical = list(range(0, num_clinical_categories))
    technical = list(range(num_clinical_categories, num_clinical_categories + num_technical_categories))
    plt.rcdefaults()
    
    # APT
    fig, ax = plt.subplots()
    ax.barh(np.arange(len(apt)), [i[2] for i in apt], xerr=ci95_apt, color=[color_selection[np.where(categories==j)[0][0]] for j in [i[1] for i in apt]])
    ax.set_yticks(np.arange(len(apt)))
    ax.set_yticklabels([i[0] for i in apt])
    handles = [plt.Rectangle((0,0),1,1, color=color_selection[i]) for i in clinical]
    legend1 = ax.legend(handles, [categories[i] for i in clinical], loc='best', bbox_to_anchor=(-0.45, 1))
    legend1.set_title(title="Clinical focus", prop={"weight": "bold"})
    handles = [plt.Rectangle((0,0),1,1, color=color_selection[i]) for i in technical]
    legend2 = ax.legend(handles, [categories[i] for i in technical], loc='best', bbox_to_anchor=(-0.45, 0.7), title="Technical focus")
    legend2.set_title(title="Technical focus", prop={"weight": "bold"})
    ax.set_xlabel("Average Approximate Potential to Translate (APT) score", weight="bold")
    ax.set_ylabel("Application", weight="bold")
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.tight_layout()
    plt.savefig(f'{results_directory}/apt.png')

    # CPOF
    fig, ax = plt.subplots()
    ax.barh(np.arange(len(cpof)), [i[2] for i in cpof], color=[color_selection[np.where(categories==j)[0][0]] for j in [i[1] for i in cpof]])
    ax.set_yticks(np.arange(len(cpof)))
    ax.set_yticklabels([i[0] for i in cpof])
    ax.set_xlabel("Citations per $1 million funding (CPOF)", weight="bold")
    ax.set_ylabel("Application", weight="bold")
    plt.tight_layout()
    plt.savefig(f'{results_directory}/cpof.png')

    # Projected funding by year
    # # Actual vs. projected awards
    with open(f'{results_directory}/final_data.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        descriptions = []
        clusters = []
        projections = []
        actuals = []
        for i in range(1,len(raw_data)):
            cluster = int(raw_data[i][0])
            description = raw_data[i][18]
            projection = float(raw_data[i][10])
            actual = float(raw_data[i][11])
            if len(description.split()) == 4:
                split = description.split(" ")
                description = split[0]+" "+split[1]+"\n"+split[2]+" "+split[3]
            elif len(description.split()) == 3:
                split = description.split(" ")
                description = split[0]+" "+split[1]+"\n"+split[2]
            elif len(description.split()) == 2:
                split = description.split(" ")
                description = split[0]+"\n"+split[1]
            descriptions.append(description)
            clusters.append(cluster)
            projections.append(projection)
            actuals.append(actual)
    

def graph_projections(results_directory):
    """
    creates projections plot in results_directory

    Parameters
    ----------
    results_directory: example is "results/07-18-2021--132946"

    Returns
    -------
    None

    """
    # 1. Determine dimensions for plot
    data = pickle.load(open(f"{results_directory}/model_clustering.pkl", "rb"))
    descriptions = []
    clusters = []
    projections = []
    actuals = []

    with open(f'{results_directory}/final_data.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1, len(raw_data)):
            cluster = int(raw_data[i][0])
            description = raw_data[i][18]
            category = raw_data[i][19]
            projection = float(raw_data[i][10])
            actual = float(raw_data[i][11])
            descriptions.append(description)
            clusters.append([cluster, description, category])
            projections.append(projection)
            actuals.append(actual)

    k = 12
    factors = []
    categories = np.unique([x[2] for x in clusters])
    categories = np.delete(categories, 6)
    for i in range(1, k+1):
        if k / i == i:
            factors.extend([i,i])
        elif k % i == 0:
            factors.append(i)
    dim1, dim2 = factors[int(len(factors)/2)], factors[int(len(factors)/2-1)]
    
    # 2. Create plot
    fig, axs = plt.subplots(dim1, dim2, sharex='all', sharey='all')
    
    # 3. Create hidden frame for shared labels
    fig.add_subplot(111, frameon=False)
    plt.grid(b=None)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Years from 2000")
    plt.ylabel("Funding ($100 millions)")
    
    # 4. Plot each projection with scatter plot
    years_int = list(range(0, 21))
    m = np.repeat(list(range(dim1)), dim2)
    n = np.tile(list(range(dim2)), dim1)
    maxy = 0
    projection = []
    growth = []
    bounds = []
    for j in range(k):
        filtered = list(filter(lambda x: x[2] == categories[j], clusters))
        if j in [0,1,2,6,7,8,9,10]:
            color_spectrum = 'Reds'
        else:
            color_spectrum = 'Blues'
        colors = cm.get_cmap(color_spectrum, len(filtered)+5)
        cluster_labels = [x[0] for x in filtered]
        for i in range(len(cluster_labels)):
            popt, pcov = curve_fit(lambda t,a,b: a*np.exp(b*t),  years_int,  data["yr_total_cost"][cluster_labels[i]],  p0=(4000, 0.1))
            std = np.sqrt(np.diagonal(pcov))
            x = np.linspace(0,21,400)
            upper0 = popt[0]+1.96*std[0]
            lower0 = popt[0]-1.96*std[0]
            upper1 = popt[1]+1.96*std[1]
            lower1 = popt[1]-1.96*std[1]
            
            ypred = [popt[0]*np.exp(popt[1]*point) for point in x] #-popt[0]
            projection.append(ypred[-1])
            growth.append(popt[1])
            bounds.append([lower1, upper1])
            maxy = max([max(ypred), maxy])
            upper = [upper0*np.exp(upper1*point) for point in x]
            lower = [lower0*np.exp(lower1*point) for point in x]
            #color = matplotlib.colors.rgb2hex(rgba)
            axs[m[j],n[j]].set_title(categories[j], size=10, weight='bold', position=(0.5, 0.8))
            axs[m[j],n[j]].plot(x, ypred, color=colors(i+5))
            axs[m[j],n[j]].fill_between(x, upper, lower, color=colors(i+5), alpha=0.1)
            axs[m[j],n[j]].scatter(years_int, data["yr_total_cost"][cluster_labels[i]], s=20, color=colors(i+5))
            axs[m[j],n[j]].set_ylim(-100000,maxy+100000)
            axs[m[j],n[j]].set_xlim(0,21.00001)
            axs[m[j],n[j]].grid(False)
        axs[m[j],n[j]].legend([x[1] for x in filtered], loc="center left", prop={'size': 5})
    
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.savefig(f'{results_directory}/projected.png')

    ############################################

    x = np.arange(len(clusters))
    # Perform linear regression
    y = actuals
    x = projections
    X = sm.add_constant(x)
    
    re = sm.OLS(y, X).fit()
    print("2021 projected vs. actual R2: {:.3f}".format(re.rsquared))
    
    prstd, iv_l, iv_u = wls_prediction_std(re)

    st, reg_data, ss2 = summary_table(re, alpha=0.05)
    
    predicted = reg_data[:, 2]
    predict_mean_se = reg_data[:, 3]
    predict_mean_ci_low, predict_mean_ci_upp = reg_data[:, 4:6].T
    predict_ci_low, predict_ci_upp = reg_data[:, 6:8].T
    
    colors = cm.get_cmap('Spectral', 12)
    mapping = [
        10,
        50,
        19,
        20,
        27,
        7,
        16,
        28,
        29,
        21,
        17,
        52,
        22,
        42,
        11,
        8,
        30,
        46,
        23,
        57,
        24,
        4,
        31,
        53,
        12,
        5,
        32,
        43,
        33,
        34,
        35,
        58,
        18,
        54,
        25,
        51,
        44,
        36,
        55,
        1,
        37,
        59,
        26,
        60,
        38,
        39,
        9,
        40,
        47,
        13,
        48,
        2,
        14,
        6,
        45,
        15,
        56,
        41,
        49,
        3,
        ]
    labels=[
        "Diet/Exercise",
        "Electronic health record",
        "Experimental techniques",
        "Genetics",
        "Imaging methods",
        "Nondescript",
        "Neurology",
        "Psychiatry",
        "Oncology",
        "Other",
        "Patient populations",
        "Training and Education"
        ]
    points = [
        [(-2.7e6, -0.5e7), colors(0), "Diet/Exercise"], #Obesity
        [(-0.5e7, 5e7), colors(0), "Diet/Exercise"], #Physical activity
        [(-5e6,1e7), colors(0), "Diet/Exercise"], #Microbiome
        [(-1e7,1e7), colors(1), "Electronic health record"], #Clinical decision support
        [(-1e7,-8e6), colors(1), "Electronic health record"], #NLP
        [(-1e7,6e6), colors(1), "Electronic health record"], #EMR phenotyping
        [(1e7,6e7), colors(2), "Experimental techniques"], #Single cell analysis
        [(0,-2e7), colors(2), "Experimental techniques"], #Protein structure analysis
        [(0,-5e7), colors(2), "Experimental techniques"], #Mass spec
        [(-1e7,5e6), colors(3), "Genetics"], #Dev genomics
        [(-0.5e7,6.5e7), colors(3), "Genetics"], #Epigenetics
        [(-5.4e6,-3e7),  colors(3), "Genetics"], #Regulatory gen
        [(0,-1e7),  colors(3), "Genetics"], #GWAS
        [(0,-1e7),  colors(3), "Genetics"], #Clinical genomics
        [(-2e7,4e7),  colors(3), "Genetics"], #RNA sequencing
        [(0.9e7,-8e7),  colors(4), "Imaging methods"], #PET
        [(-1.1e7,-6e7),  colors(4), "Imaging methods"], #CAD
        [(0,-2e7),  colors(4), "Imaging methods"], #MRI
        [(-0.5e7,0.5e7),  colors(5), "Nondescript"], #Nondescript
        [(-1.5e7,9e7),  colors(5), "Nondescript"], #Nondescript
        [(-2.5e7,2e7),  colors(5), "Nondescript"], #Nondescript
        [(-0.2e7,-8e7),  colors(5), "Nondescript"], #Nondescript
        [(-0.5e7,0.5e7),  colors(5), "Nondescript"], #Nondescript
        [(-0.5e7,6e7),  colors(5), "Nondescript"], #Abstract unavailable
        [(0,0.5e7),  colors(5), "Nondescript"], #Nondescript
        [(-1e7,-7e7),  colors(5), "Nondescript"], #Nondescript
        [(-0.5e7,8e7),  colors(10), "Psychiatry"], #Pain management
        [(0.5e7,-7e7),  colors(6), "Neurology"], #Dementia
        [(0.2e7,-0.5e7),  colors(6), "Neurology"], #Single-neuron analysis
        [(-0.45e7,1e7),  colors(6), "Neurology"], #Speech disorders
        [(-0.6e7,0.5e7),  colors(4), "Imaging methods"], #Functional MRI
        [(-0.2e7,-10e7),  colors(6), "Neurology"], #Neurocog
        [(0,0.5e7),  colors(6), "Neurology"], #Stroke
        [(-2e7,-4e7),  colors(6), "Neurology"], #Sleep
        [(0.4e7,-3.8e7),  colors(6), "Neurology"], #Neural circuits
        [(0.2e7,-1.8e7),  colors(6), "Neurology"], #Language development
        [(-1.2e7,7e7),  colors(6), "Neurology"], #Vision
        [(-2.5e7,10e7),  colors(6), "Neurology"], #Motor disorders
        [(0.1e7,-1e7),  colors(6), "Neurology"], #Autism
        [(-2e7,-1.2e7),  colors(6), "Neurology"], #Neuroscience
        [(-1.2e7,5e7),  colors(6), "Neurology"], #PD
        [(-2e7,-2.8e7),  colors(7), "Oncology"], #Cancer genomics
        [(0,-1e7),  colors(7), "Oncology"], #Lung cancer
        [(2e7,0.2e7),  colors(7), "Oncology"], #Breast cancer
        [(-0.1e7,-4e7),  colors(7), "Oncology"], #Cancer imaging
        [(-0.1e7,2e7),  colors(8), "Other"], #HIV
        [(-1.6e7,0.8e7),  colors(8), "Other"], #Environmental health
        [(-1.3e7,-2e7),  colors(8), "Other"], #Asthma
        [(1e7,2e7),  colors(8), "Other"], #Drug efficacy
        [(-3e7,-2.5e7),  colors(9), "Patient populations"], #Pediatrics
        [(2e7,-1.6e7),  colors(9), "Patient populations"], #Older adults
        [(1e7,7e7),  colors(10), "Psychiatry"], #Suicidality
        [(3e7,-2.1e7),  colors(10), "Psychiatry"], #AUD
        [(-2.4e7,-7e7),  colors(10), "Psychiatry"], #Cigarette smoking
        [(0.5e7,4e7),  colors(10), "Psychiatry"], #Mental health
        [(-2e7,-0.75e7),  colors(10), "Psychiatry"], #Emotion
        [(-2.4e7,0.9e7),  colors(11), "Training and Education"], #Research career programs
        [(-2e7,5e7),  colors(11), "Training and Education"], #Bibliometric analysis
        [(3e7, -1e7),  colors(11), "Training and Education"], #Big data education
        [(-0.5e7,-1e7),  colors(11), "Training and Education"], #Science education
    ]
    fig, ax = plt.subplots()
    ax.grid(b=None)
    for i in range(12):
        js = [k for k in range(len(clusters)) if points[k][2]==labels[i]]
        ax.scatter([projections[j] for j in js], [actuals[j] for j in js], color=colors(i), label=labels[i]) #points[i][1]
    ax.set_xlim(-0.2e8, 1.6e8)
    ax.set_ylim(-10e7, 14e7)
    ax.legend(loc="lower right")
    ann = []
    locations = [(projections[i], actuals[i]) for i in range(len(clusters))]
    for i in range(len(clusters)):
        ann.append(ax.annotate(descriptions[mapping[i]-1], xy=locations[mapping[i]-1], xytext=tuple(map(sum,zip(locations[mapping[i]-1],points[i][0]))), fontsize=8, arrowprops=dict(arrowstyle="-", color='k', lw=0.5)))
    # https://adjusttext.readthedocs.io/en/latest/_modules/adjustText.html#adjust_text
    # adjust_text(ann, projection, cluster_cost_2021, ax=ax, expand_text=(1.05,3), force_text=(0.25, 0.5), only_move={'points':'y', 'text':'y', 'objects':'y'}, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    ax.plot([projections[np.argmin(projections)],projections[np.argmax(projections)]], [predicted[np.argmin(projections)],predicted[np.argmax(projections)]], color="#808080")
    ax.plot(sorted(x), sorted(predict_mean_ci_low), color='#808080', linestyle="--", lw=2)
    ax.plot(sorted(x), sorted(predict_mean_ci_upp), color='#808080', linestyle="--", lw=2)
    ax.set_ylabel('Actual 2021 award to date ($10 millions)')
    ax.set_xlabel('Projected 2021 award ($100 millions)')
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.savefig(f'{results_directory}/actual_vs_projected.png')
