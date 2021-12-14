#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:57:58 2021

@author: Sope
"""

# Get applications by categories
    categories = {}
    with open(f'{results_directory}/final_data.csv', newline='') as csvfile:
        raw_data = list(csv.reader(csvfile))
        for i in range(1,len(raw_data)):
            if (raw_data[i][18] != "N/A"):
                if raw_data[i][18] not in categories:
                    categories[raw_data[i][18]] = {"labels": [int(raw_data[i][0])], "funding": [int(raw_data[i][2])], "applications": [raw_data[i][17]]}
                else:
                    categories[raw_data[i][18]]["labels"].append(int(raw_data[i][0]))
                    categories[raw_data[i][18]]["funding"].append(int(raw_data[i][2]))
                    categories[raw_data[i][18]]["applications"].append(raw_data[i][17])
    
    # Get applications and categories by project number
    for category in categories:
        category_project_numbers = []
        for cluster in categories[category]["labels"]:
            project_numbers = [item["project_number"] for item in applications[cluster]["data"]]
            # project_numbers = list(set(project_numbers)) # Remove duplicates
            applications[cluster]["projects"] = project_numbers
            category_project_numbers.extend(project_numbers)
        applications[category]["projects"] = category_project_numbers

def create_figure_5(results_directory, top):
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
            cpof.append([description, category, float(raw_data[i][9]), float(raw_data[i][10]), float(raw_data[i][11])])
            if clinical_technical == 'Clinical' and category not in clinical_categories:
                clinical_categories.append(category)
            elif clinical_technical == 'Technical' and category not in technical_categories:
                technical_categories.append(category)
            elif clinical_technical not in ['Clinical', 'Technical']:
                warnings.warn(f"Cluster {raw_data[i][0]} (description: {description}, category: {category}) "
                              f"not classified as 'Clinical' or 'Technical')")
    
    apt.sort(key=lambda x: x[2])
    apt = apt[0:top] + apt[-top:]
    ci95_apt = [[abs(i[2]-i[3]), abs(i[2]-i[4])] for i in apt]
    ci95_apt = list(map(list, zip(*ci95_apt)))
    cpof.sort(key=lambda x: x[2])
    cpof = cpof[0:top] + cpof[-top:]
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
    plt.axhline(y=top-0.5, color='k', linestyle='--')
    ax.set_xlabel("Average Approximate Potential to Translate (APT) score", weight="bold")
    ax.set_ylabel("Application", weight="bold")
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.tight_layout()
    plt.savefig(f'{results_directory}/apt.eps', format='eps')

    # CPOF
    fig, ax = plt.subplots()
    ax.barh(np.arange(len(cpof)), [i[2] for i in cpof], color=[color_selection[np.where(categories==j)[0][0]] for j in [i[1] for i in cpof]])
    # handles = [plt.Rectangle((0,0),1,1, color=color_selection[i]) for i in clinical]
    # legend1 = ax.legend(handles, [categories[i] for i in clinical], loc='best', bbox_to_anchor=(0.3, 0.3))
    # legend1.set_title(title="Clinical focus", prop={"weight": "bold"})
    # handles = [plt.Rectangle((0,0),1,1, color=color_selection[i]) for i in technical]
    # legend2 = ax.legend(handles, [categories[i] for i in technical], loc='best', bbox_to_anchor=(0.7, 0.3), title="Technical focus")
    # legend2.set_title(title="Technical focus", prop={"weight": "bold"})
    # ax.add_artist(legend1)
    # ax.add_artist(legend2)
    ax.set_yticks(np.arange(len(cpof)))
    ax.set_yticklabels([i[0] for i in cpof])
    ax.set_xlabel("Citations per $1 million funding (CPOF)", weight="bold")
    ax.set_ylabel("Application", weight="bold")
    plt.axhline(y=top-0.5, color='k', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{results_directory}/cpof.eps', format='eps')
    
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

    # initialize color_selection with max_color
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