import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot as plt
from matplotlib import rcParams
from lev_distance import lev_distance
from sklearn.model_selection import train_test_split
from cluster import *


# Add column with mean # of tries
def mean_tries(tries: list):
    s = 0
    w = 0
    for try_n, count in enumerate(tries, 1):
        s += try_n * count
        w += count
    return s/w


def main():
    wordle_path = "Data/Problem_C_Data_Wordle no typos.csv"
    wordle_df = pd.read_csv(wordle_path).dropna(axis=1)
    words = wordle_df['Word'].to_numpy()

    keys = ['Date', 'Contest number', 'Number of  reported results', 'Number in hard mode']

    attempt_data = wordle_df.drop(keys, axis=1).set_index('Word')

    attempt_data['Mean # of Tries'] = attempt_data.apply(lambda x: mean_tries([
        x['1 try'], x['2 tries'], x['3 tries'], x['4 tries'], x['5 tries'], x['6 tries'], x['7 or more tries (X)']]),
                                                            axis=1)
    train, test = train_test_split(attempt_data, test_size=0.1, random_state=1917)
    aff_prop = AffProp(lev_distance)
    df, clusters = aff_prop.aff_prop_clusters(train)
    df.sort_values('mean # of tries', inplace=True)
    for c in clusters:
        print(', '.join(c.words))

    plots(df)


def plots(df):
    rcParams['figure.figsize'] = 16, 8
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['lines.linewidth'] = 2.5
    # rcParams['axes.prop_cycle'] = cycler(color=['#424242'])
    rcParams['xtick.labelsize'] = 'xx-large'
    rcParams['ytick.labelsize'] = 'xx-large'

    plt.xticks(df['mean # of tries'], df.index.values)  # location, labels
    plt.plot(df['mean # of tries'])

    fig, ax = plt.subplots()

    ax.bar(df.index, df['mean # of tries'], color='orange')
    ax.set_xticklabels(df.index, rotation='vertical', fontdict={'fontsize': 10})
    ax.set_title('Tries per Word Class')
    ax.set_xlabel('Exemplar')
    ax.set_ylabel('Mean Tries')
    plt.show()


if __name__ == "__main__":
    main()
