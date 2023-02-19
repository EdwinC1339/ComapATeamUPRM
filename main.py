import string

import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import log
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor

from lib.CharVectorizer import CharVectorizer
from lev_implementation import Cluster, aff_prop_clusters

print("Running on Pandas Version", pandas.__version__)


def main():
    wordle_path = "Data/Problem_C_Data_Wordle no typos.csv"
    word_path = "Data/unigram_freq.csv"

    wordle_df = pandas.read_csv(wordle_path).dropna(axis=1)
    word_df = pandas.read_csv(word_path).set_index('word')

    keys = ['Date', 'Contest number', 'Number of  reported results', 'Number in hard mode']
    attempt_data = wordle_df.drop(keys, axis=1).set_index('Word')

    # Add column with mean # of tries
    def mean_tries(tries: list):
        s = 0
        w = 0
        for try_n, count in enumerate(tries, 1):
            s += try_n * count
            w += count
        return s/w

    attempt_data['Mean # of Tries'] = attempt_data.apply(lambda x: mean_tries([
        x['1 try'], x['2 tries'], x['3 tries'], x['4 tries'], x['5 tries'], x['6 tries'], x['7 or more tries (X)']]),
                                                         axis=1)

    # Add column with frequency information
    def freq(word: str):
        try:
            count = word_df['count'][word.strip()]
        except KeyError:
            count = np.NaN
        return count

    attempt_data['Word Frequency'] = attempt_data.index.map(freq)
    attempt_data['Log Word Frequency'] = attempt_data['Word Frequency'].apply(log)

    # Split our data into two samples, the training sample used to gather information for our models, then
    # a prediction sample used to measure the error of the models
    # x_train, x_test, y_train, y_test = train_test_split(
    #     attempt_data['Log Word Frequency'].to_numpy(), attempt_data['Mean # of Tries'].to_numpy(),
    #     test_size=0.25, random_state=1917
    # )
    train, test, = train_test_split(attempt_data, test_size=0.25, random_state=1917)

    x_train = train['Log Word Frequency']
    y_train = train['Mean # of Tries']

    x_test = test['Log Word Frequency']
    y_test = test['Mean # of Tries']

    models = pandas.DataFrame(y_test).rename({'Mean # of Tries': 'Ground Truth'}, axis=1)

    tries_mean = y_train.mean()
    models['mean'] = tries_mean

    # Model the mean # of tries as a linear function of the natural log of a word's frequency
    lin_reg = LinearRegression().fit(x_train.to_numpy().reshape(-1, 1), y_train.to_numpy().reshape(-1, 1))
    models['linear'] = lin_reg.predict(x_test.to_numpy().reshape(-1, 1))

    # Make a polynomial regression
    poly = PolynomialFeatures(degree=3, include_bias=False)
    poly_features = poly.fit_transform(x_train.to_numpy().reshape(-1, 1))
    poly_reg = LinearRegression().fit(poly_features, y_train.to_numpy().reshape(-1, 1))
    models['polynomial'] = poly_reg.predict(poly.transform(x_test.to_numpy().reshape(-1, 1)))

    # Gradient Boosted Recessor model
    gbr = GradientBoostingRegressor(n_estimators=600,
                                    max_depth=5,
                                    learning_rate=0.01,
                                    min_samples_split=3)
    gbr.fit(x_train.to_numpy().reshape(-1, 1), y_train.to_numpy().reshape(-1, 1))
    models['gbr'] = gbr.predict(x_test.to_numpy().reshape(-1, 1))

    # Word Vector Model
    # Declare a vectorizer object
    vectorizer = CharVectorizer(string.ascii_lowercase)

    # Get a vector for each of our words
    train_windows = train.index.map(lambda s: s.strip())
    target_length = 5
    train_matrix = vectorizer.transform(train_windows, target_length)

    # Model the mean # of tries as a linear function of the word vector
    x_axis = np.array(train_matrix)
    y_axis = np.array(y_train)
    word_lin_reg = LinearRegression().fit(x_axis, y_axis)

    # Turn the words in the prediction sample into vectors and run the prediction
    predict_windows = test.index.map(lambda s: s.strip())
    predict_matrix = vectorizer.transform(predict_windows, target_length)
    models['word vec linear'] = word_lin_reg.predict(predict_matrix)
    # If we just leave it here, the model will occasionally predict values that are outside of the range [1,7],
    # which are not possible within the restrictions of Wordle.
    models['word vec linear'] = models['word vec linear'].apply(lambda x: np.clip(x, 1, 7))

    # Same idea but with poly regression
    poly_features = poly.fit_transform(x_axis)
    poly_predict_matrix = poly.fit_transform(predict_matrix)
    word_poly_reg = LinearRegression().fit(poly_features, y_axis)
    models['word vec poly'] = word_poly_reg.predict(poly_predict_matrix)

    # Making Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(n_estimators=600,
                                    max_depth=5,
                                    learning_rate=0.01,
                                    min_samples_split=3)
    # with default parameters
    # gbr = GradientBoostingRegressor()

    gbr.fit(x_axis, y_axis)
    models['word vec gbr'] = gbr.predict(predict_matrix)

    # Clustering
    # Get clusters
    _, clusters = aff_prop_clusters(train)
    models['lev distance clustering'] = models.index.map(lambda w:
                                                         Cluster.best_cluster_mean_tries(clusters, w).mean_tries)

    # Calculate MSE for each model

    mse_s = models.apply(lambda x: mean_squared_error(x, models['Ground Truth']), axis=0)
    
    # Mat plot lib graph style
    rcParams['figure.figsize'] = 16, 8
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['lines.linewidth'] = 2.5
    # rcParams['axes.prop_cycle'] = cycler(color=['#424242'])
    rcParams['xtick.labelsize'] = 'xx-large'
    rcParams['ytick.labelsize'] = 'xx-large'

    models.sort_values('Ground Truth', inplace=True)
    fig, ax = plt.subplots()
    ax.scatter(x=models.index, y=models['Ground Truth'], marker='*', label="Ground Truth", s=9, c="black")
    ax.scatter(x=models.index, y=models['linear'], c="red", marker='.', s=9, label="Linear Model")
    ax.scatter(x=models.index, y=models['polynomial'], c="orange", marker='.', s=9, label="Polynomial Model")
    ax.scatter(x=models.index, y=models['gbr'], c="cyan", marker='.', s=9, label="GBR Model")
    ax.scatter(x=models.index, y=models['word vec linear'], c="yellow", marker='.', s=9,
               label="Linear Word Vector Model")
    ax.scatter(x=models.index, y=models['word vec poly'], c="purple", marker='.', s=9,
               label="Polynomial Word Vector Model")
    ax.scatter(x=models.index, y=models['word vec gbr'], c="green", marker='.', s=9,
               label="GBR Word Vector Model")
    ax.scatter(x=models.index, y=models['lev distance clustering'], c="pink", marker='.', s=9,
               label="Levenshtein Distance Clustering Model")

    ax.set_title("Mean Tries Per Word")
    ax.set_xlabel("Word")
    ax.set_ylabel("Mean Tries")
    ax.set_xticklabels(models.index, rotation='vertical', fontdict={'fontsize': 6})
    fig.legend()

    fig, ax = plt.subplots()
    ax.bar(mse_s.index, mse_s, color='k')
    ax.set_ylabel('Mean Squared Error')
    ax.set_xlabel('Model')
    ax.set_xticklabels(mse_s.index, fontdict={'fontsize': 10})
    ax.set_title("Model Error")

    plt.show()


if __name__ == "__main__":
    main()
