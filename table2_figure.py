import matplotlib.pyplot as plt
import numpy as np
import csv

"""
Created on Fri Sep 19 2021

Plot Table 2 as a figure
Award Number & Values by Year
"""

# get data from csv file
def get_by_year_data():
    years = []
    awards = []
    values = []

    with open('by_year.csv', newline='', encoding='utf8') as csvfile:
        raw_data = csv.reader(csvfile)
        next(raw_data) #skip the header

        for entry in raw_data:
            years.append(int(entry[0]))
            awards.append(int(entry[1]))
            values.append(float(entry[2])/1e9)

    return years, awards, values

# plot award number & values on same plot, 2 y-axes
def plot_by_year():
    years, awards, values = get_by_year_data()

    fig, ax1 = plt.subplots()

    # award values
    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Award Values (billions USD)', color=color)
    bar = ax1.bar(years, values, color=color) #yerr=errorbar
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.bar_label(bar, fmt='$%.1f', padding=1, size='x-small')
    ax2 = ax1.twinx()

    # award number
    color = 'tab:green'
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Number of Awards', color=color)
    ax2.plot(years, awards, color=color) #yerr=errorbar
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Total Awards by Year')

    plt.show()


if __name__ == "__main__":
    plot_by_year()
    print('Done.')
