#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 14:56:26 2021

@author: Sope
"""

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
            c_t = raw_data[i][20]
            projection = float(raw_data[i][10])
            actual = float(raw_data[i][11])
            descriptions.append(description)
            clusters.append([cluster, description, category, c_t])
            projections.append(projection)
            actuals.append(actual)

    factors = []
    categories = np.unique([x[2] for x in clusters])
    k = len(categories)
    # categories = np.delete(categories, 6)
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
    # plt.grid(b=None)
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
    for j in range(len(categories)):
        filtered = list(filter(lambda x: x[2] == categories[j], clusters))
        if filtered[0][3] == "Clinical":
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
