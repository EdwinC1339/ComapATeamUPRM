import string

import pandas
import numpy as np
import matplotlib.pyplot as plt
from math import log
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
    training_sample = attempt_data.iloc[:150, :].copy()
    prediction_sample = attempt_data.iloc[151:, :].copy()

    tries_mean = training_sample['Mean # of Tries'].mean()
    prediction_sample['mean'] = tries_mean

    # Model the mean # of tries as a linear function of the natural log of a word's frequency
    x_axis = np.array(training_sample['Log Word Frequency'])[:, None]  # We add an empty axis since sklearn expects 2D
    y_axis = np.array(training_sample['Mean # of Tries'])
    lin_reg = LinearRegression().fit(x_axis, y_axis)
    prediction_sample['linear regression'] = lin_reg.predict(
        np.array(prediction_sample['Log Word Frequency'])[:, None])

    # Make a polynomial regression
    poly = PolynomialFeatures(degree=3, include_bias=False)
    x_axis = np.array(training_sample['Log Word Frequency'])[:, None]  # We add an empty axis since sklearn expects 2D
    y_axis = np.array(training_sample['Mean # of Tries'])
    poly_features = poly.fit_transform(x_axis)
    poly_reg = LinearRegression().fit(poly_features, y_axis)
    prediction_series = poly.transform(prediction_sample['Log Word Frequency'].array.reshape(-1, 1))
    prediction_sample['polynomial regression'] = poly_reg.predict(prediction_series)

    # Word Vector Model
    # Declare a vectorizer object
    vectorizer = CharVectorizer(string.ascii_lowercase)

    # Get a vector for each of our words
    train_windows = training_sample.index.map(lambda s: s.strip())
    target_length = 5
    train_matrix = vectorizer.transform(train_windows, target_length)

    # Model the mean # of tries as a linear function of the word vector
    x_axis = np.array(train_matrix)
    y_axis = np.array(training_sample['Mean # of Tries'])
    word_lin_reg = LinearRegression().fit(x_axis, y_axis)

    # Turn the words in the prediction sample into vectors and run the prediction
    predict_windows = prediction_sample.index.map(lambda s: s.strip())
    predict_matrix = vectorizer.transform(predict_windows, target_length)
    prediction_sample['word vectors linear fit'] = word_lin_reg.predict(predict_matrix)
    # If we just leave it here, the model will occasionally predict values that are outside of the range [1,7],
    # which are not possible within the restrictions of Wordle.
    prediction_sample['word vectors linear fit'] = prediction_sample['word vectors linear fit'].apply(lambda x:
                                                                                                      np.clip(x, 1, 7))

    # Same idea but with poly regression
    poly_features = poly.fit_transform(x_axis)
    poly_predict_matrix = poly.fit_transform(predict_matrix)
    word_poly_reg = LinearRegression().fit(poly_features, y_axis)
    prediction_sample['word vectors poly fit'] = word_poly_reg.predict(poly_predict_matrix)

    # Calculate MSE for each model
    mse_mean = mean_squared_error(prediction_sample['Mean # of Tries'], prediction_sample['mean'])
    mse_linear = mean_squared_error(prediction_sample['Mean # of Tries'], prediction_sample['linear regression'])
    mse_poly = mean_squared_error(prediction_sample['Mean # of Tries'], prediction_sample['polynomial regression'])
    mse_lin_vec = mean_squared_error(prediction_sample['Mean # of Tries'], prediction_sample['word vectors linear fit'])
    mse_poly_vec = mean_squared_error(prediction_sample['Mean # of Tries'], prediction_sample['word vectors poly fit'])

    print(training_sample)
    print(prediction_sample)
    print("Mean squared error for mean model:", mse_mean)
    print("Mean squared error for linear model:", mse_linear)
    print("Mean squared error for polynomial model:", mse_poly)
    print("Mean squared error for linear vector model:", mse_lin_vec)
    print("Mean squared error for polynomial vector model:", mse_poly_vec)

    prediction_sample = prediction_sample.sort_values('Log Word Frequency')
    plt.scatter(x=prediction_sample["Log Word Frequency"], y=prediction_sample["Mean # of Tries"],
                marker='.', label="Real Value", s=3, c="black")
    plt.plot(prediction_sample["Log Word Frequency"], prediction_sample["linear regression"],
             c="red", label="Linear Model")
    plt.plot(prediction_sample["Log Word Frequency"],
             prediction_sample["polynomial regression"], c="cyan", label="Polynomial Model")
    plt.xlabel("ln of word frequency")
    plt.ylabel("model prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
