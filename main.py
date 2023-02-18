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

    # # Word Vector Model
    # # Declare a vectorizer object
    # vectorizer = CharVectorizer(string.ascii_lowercase)
    #
    # # Get a vector for each of our words
    # train_windows = x_train.index.map(lambda s: s.strip())
    # target_length = 5
    # train_matrix = vectorizer.transform(train_windows, target_length)
    #
    # # Model the mean # of tries as a linear function of the word vector
    # x_axis = np.array(train_matrix)
    # y_axis = np.array(x_train['Mean # of Tries'])
    # word_lin_reg = LinearRegression().fit(x_axis, y_axis)
    #
    # # Turn the words in the prediction sample into vectors and run the prediction
    # predict_windows = x_test.index.map(lambda s: s.strip())
    # predict_matrix = vectorizer.transform(predict_windows, target_length)
    # x_test['word vectors linear fit'] = word_lin_reg.predict(predict_matrix)
    # # If we just leave it here, the model will occasionally predict values that are outside of the range [1,7],
    # # which are not possible within the restrictions of Wordle.
    # x_test['word vectors linear fit'] = x_test['word vectors linear fit'].apply(lambda x: np.clip(x, 1, 7))
    #
    # # Making Gradient Boosting Regressor
    # gbr = GradientBoostingRegressor(n_estimators=600,
    #     max_depth=5,
    #     learning_rate=0.01,
    #     min_samples_split=3)
    # # with default parameters
    # gbr = GradientBoostingRegressor()
    #
    # gbr.fit(x_axis, y_axis)
    # x_test['gbr fit'] = gbr.predict(predict_matrix)
    #
    # # Same idea but with poly regression
    # poly_features = poly.fit_transform(x_axis)
    # poly_predict_matrix = poly.fit_transform(predict_matrix)
    # word_poly_reg = LinearRegression().fit(poly_features, y_axis)
    # x_test['word vectors poly fit'] = word_poly_reg.predict(poly_predict_matrix)

    # Calculate MSE for each model
    # mse_mean = mean_squared_error(x_test['Mean # of Tries'], x_test['mean'])
    # mse_linear = mean_squared_error(x_test['Mean # of Tries'], x_test['linear regression'])
    # mse_poly = mean_squared_error(x_test['Mean # of Tries'], x_test['polynomial regression'])
    # mse_lin_vec = mean_squared_error(x_test['Mean # of Tries'], x_test['word vectors linear fit'])
    # mse_poly_vec = mean_squared_error(x_test['Mean # of Tries'], x_test['word vectors poly fit'])
    # mse_gbr = mean_squared_error(x_test['Mean # of Tries'], x_test['gbr fit'])

    mse_s = models.apply(lambda x: mean_squared_error(x, models['Ground Truth']), axis=0)
    
    # Mat plot lib graph style
    rcParams['figure.figsize'] = 16, 8
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['lines.linewidth'] = 2.5
    # rcParams['axes.prop_cycle'] = cycler(color=['#424242'])
    rcParams['xtick.labelsize'] = 'xx-large'
    rcParams['ytick.labelsize'] = 'xx-large'

    # print(x_train)
    # print(x_test)
    # print("Mean squared error for mean model:", mse_mean)
    # print("Mean squared error for linear model:", mse_linear)
    # print("Mean squared error for polynomial model:", mse_poly)
    # print("Mean squared error for linear vector model:", mse_lin_vec)
    # print("Mean squared error for polynomial vector model:", mse_poly_vec)
    # print("Mean squared error for gradient boosted regressor:", mse_gbr)

    models.sort_values('Ground Truth', inplace=True)
    plt.figure(1)
    plt.scatter(x=models.index, y=models['Ground Truth'], marker='*', label="Ground Truth", s=3, c="black")
    plt.scatter(x=models.index, y=models['linear'], c="red", marker='.', s=3, label="Linear Model")
    plt.scatter(x=models.index, y=models['polynomial'], c="orange", marker='.', s=3, label="Polynomial Model")
    plt.scatter(x=models.index, y=models['gbr'], c="cyan", marker='.', s=3, label="GBR Model")
    plt.xlabel("Word")
    plt.ylabel("Mean Tries")
    plt.xticks()
    plt.legend()

    plt.figure(2)
    plt.scatter(x=x_test.index.array,
                y=x_test['Mean # of Tries'], s=3, marker='.', label="Real Value", c="black")
    plt.plot(x_test.index.array,
             x_test['word vectors linear fit'], label="Linear Word Vector Model")
    plt.plot(x_test.index.array,
             x_test['word vectors poly fit'], label="Polynomial Word Vector Model")
    plt.plot(x_test.index.array,
             x_test['gbr fit'], label="Gradient Boosted Regressor Model")
    plt.xticks(color='w')  # Hide word tick labels
    plt.xlabel('Word')
    plt.ylabel('Model Prediction')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
