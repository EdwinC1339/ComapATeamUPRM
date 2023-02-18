import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from lev_distance import lev_distance

wordle_path = "Data/Problem_C_Data_Wordle no typos.csv"
wordle_df = pd.read_csv(wordle_path).dropna(axis=1)
words = wordle_df["Word"].values.tolist()
print(wordle_df)

words = np.asarray(words) 
lev_similarity = -1*np.array([[lev_distance(w1,w2) for w1 in words] for w2 in words])

affprop = AffinityPropagation(affinity="precomputed", damping=0.5)
affprop.fit(lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))
    