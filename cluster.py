import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import AffinityPropagation


class Cluster:
    def __init__(self, exemplar, words: np.array, cluster_mean_tries, dist: callable):
        self.exemplar = exemplar
        self.words = words
        self.mean_tries = cluster_mean_tries
        self.dist = dist

    def affinity(self, word):
        # Distance is -1 * affinity, so we will return -1 * mean(distances)
        distances = np.array([self.dist(w, word) for w in self.words])
        return -1 * distances.mean()

    @staticmethod
    def best_cluster_mean_tries(clusters, word):
        # Given a list of clusters and a word, find the cluster with the maximum affinity for the word.
        affinities = np.array([c.affinity(word) for c in clusters])
        match = clusters[np.argmax(affinities)]
        return match


class AffProp:
    def __init__(self, dist: callable, cache=False, path=''):
        self.dist = dist
        self.cache = cache
        self.path = path

    def aff_prop_clusters(self, train):
        words = train.index.to_numpy()
        try:
            similarity = np.fromfile(self.path, dtype=float).reshape([len(words), len(words)])
        except FileNotFoundError:
            with ThreadPoolExecutor(400) as executor:
                results = [[executor.submit(self.dist, w1, w2) for w2 in words] for w1 in words]
            s_l = [[j.result() for j in i] for i in results]
            similarity = -1 * np.array(s_l, dtype=float)
            if self.cache:
                similarity.tofile(self.path)

        # Affinity propagation using dist
        # Affinity Propagation is 𝑂(𝑡×𝑛2)
        # where 𝑛
        # is the number of names, and 𝑡
        # is the number of iteration until convergence.
        affprop = AffinityPropagation(affinity="precomputed", damping=0.5, random_state=1917)
        affprop.fit(similarity)

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
            mean_number_of_tries = 0
            n = 0
            for word in cluster:
                mean_number_of_tries += train['Mean # of Tries'][word]
                n += 1
            mean_number_of_tries = mean_number_of_tries / n

            c_o = Cluster(exemplar, cluster, mean_number_of_tries, self.dist)
            clusters.append(c_o)

            df.loc[exemplar] = [cluster_str,
                                mean_number_of_tries,
                                affprop.cluster_centers_indices_[cluster_id],
                                affprop.labels_[cluster_id]
                                ]

        return df, clusters
