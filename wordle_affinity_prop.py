import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot as plt
from matplotlib import rcParams
from wordle_distance import WordleDistance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from cluster import *
from pathlib import Path  


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
    w_d = WordleDistance(words)
    aff_prop = AffProp(lambda x, y: w_d.wordle_distance(x, y), cache=True, path='wordle affinity.numpy')
    df, clusters = aff_prop.aff_prop_clusters(train)
    df.sort_values('mean # of tries', inplace=True)
    for c in clusters:
        print(', '.join(c.words))
        
    test['Mean # of Tries Prediction'] = test.apply(lambda x: Cluster.best_cluster_mean_tries(clusters, x.name).mean_tries, axis=1)
    mse = relative_root_mean_squared_error(test['Mean # of Tries'], test['Mean # of Tries Prediction'])
    test['mse'] = test.apply(lambda x: 
        relative_root_mean_squared_error(x['Mean # of Tries'], x['Mean # of Tries Prediction']), axis=1)
    print(mse)
    filepath = Path('folder/subfolder/wor_out.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    test.to_csv(filepath)  
    plots(df)
    
def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

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
