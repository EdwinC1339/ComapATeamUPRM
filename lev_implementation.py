import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot as plt
from matplotlib import rcParams
from lev_distance import lev_distance
from sklearn.model_selection import train_test_split


class Cluster:
    def __init__(self, exemplar, words: np.array, cluster_mean_tries):
        self.words = words
        self.mean_tries = cluster_mean_tries

    def affinity(self, word):
        distances = np.array([lev_distance(w, word) for w in self.words])
        return distances.mean()

    @staticmethod
    def best_cluster_mean_tries(clusters, word):
        affinities = np.array([c.affinity(word) for c in clusters])  # This is actually -1 * affinity
        match = clusters[np.argmin(affinities)]  # So we use the minimum instead of max
        return match


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

    keys = ['Date', 'Contest number', 'Number of  reported results', 'Number in hard mode']

    attempt_data = wordle_df.drop(keys, axis=1).set_index('Word')

    attempt_data['Mean # of Tries'] = attempt_data.apply(lambda x: mean_tries([
        x['1 try'], x['2 tries'], x['3 tries'], x['4 tries'], x['5 tries'], x['6 tries'], x['7 or more tries (X)']]),
                                                            axis=1)
    train, test = train_test_split(attempt_data, test_size=0.25, random_state=1917)
    df, clusters = aff_prop_clusters(train)
    df.sort_values('mean # of tries', inplace=True)

    plots(df)


def aff_prop_clusters(train):
    # We can find the Affinity Propagation in Levenshtein distance
    # by defining the pairwise similarity of levenshtein distance.
    # We do this by simply multiplying the lev distance by -1.
    words = train.index.to_numpy()
    lev_similarity = -1*np.array([[lev_distance(w1, w2) for w1 in words] for w2 in words])

    # Affinity propagation using lev_distance
    # Affinity Propagation is 𝑂(𝑡×𝑛2)
    # where 𝑛
    # is the number of names, and 𝑡
    # is the number of iteration until convergence.
    affprop = AffinityPropagation(affinity="precomputed", damping=0.5, random_state=1917)
    l_s_float = lev_similarity.astype(float)
    affprop.fit(l_s_float)

    cluster_centers_indices = affprop.cluster_centers_indices_
    labels = affprop.labels_

    n_clusters_ = len(cluster_centers_indices)

    # print("Estimated number of clusters: %d" % n_clusters_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    # print(
    #     "Adjusted Mutual Information: %0.3f"
    #     % metrics.adjusted_mutual_info_score(labels_true, labels)
    # )
    # print(
    #     "Silhouette Coefficient: %0.3f"
    #     % metrics.silhouette_score(X, labels, metric="sqeuclidean")
    # )

    df = pd.DataFrame({'class': [],
                       'mean # of tries': [],
                       'cluster centers indices': [],
                       'labels': [],
                       }, index=[])

    clusters = []

    for cluster_id in np.unique(affprop.labels_):
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
        cluster_str = ", ".join(cluster)
        # print(" - *%s:* %s" % (exemplar, cluster_str))
        mean_number_of_tries = 0
        n = 0
        for word in cluster:
            mean_number_of_tries += train['Mean # of Tries'][word]
            n += 1
        mean_number_of_tries = mean_number_of_tries/n

        c_o = Cluster(exemplar, cluster, mean_number_of_tries)
        clusters.append(c_o)

        df.loc[exemplar] = [cluster_str,
                            mean_number_of_tries,
                            affprop.cluster_centers_indices_[cluster_id],
                            affprop.labels_[cluster_id]
                            ]

    return df, clusters


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
