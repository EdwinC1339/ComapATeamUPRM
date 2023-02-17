import pandas
import numpy as np
import matplotlib.pyplot as plt
from math import log
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

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
    
    mse_mean = mean_squared_error(prediction_sample['Mean # of Tries'], prediction_sample['mean'])
    mse_linear = mean_squared_error(prediction_sample['Mean # of Tries'], prediction_sample['linear regression'])

    print(training_sample)
    print(prediction_sample)
    print("Mean squared error for mean model:", mse_mean)
    print("Mean squared error for linear model:", mse_linear)

    plt.scatter(x=training_sample["Mean # of Tries"], y=training_sample["Log Word Frequency"], marker='^')
    plt.xlabel("mean # of tries to guess word")
    plt.ylabel("ln of word frequency")
    plt.show()


if __name__ == "__main__":
    main()
