import pandas
import numpy as np
from sklearn.metrics import mean_squared_error

print("Running on Pandas Version", pandas.__version__)


def main():
    wordle_path = "Data/Problem_C_Data_Wordle.xlsx - Sheet1.csv"
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

    # Split our data into two samples, the training sample used to gather information for our models, then
    # a prediction sample used to measure the error of the models
    training_sample = attempt_data.iloc[:150, :].copy()
    prediction_sample = attempt_data.iloc[151:, :].copy()

    # Add column with frequency information
    def freq(word: str):
        try:
            count = word_df['count'][word]
        except KeyError:
            count = np.NaN
        return count

    training_sample['Word Frequency'] = training_sample.index.map(freq)

    print(attempt_data.head())

    tries_mean = training_sample['Mean # of Tries'].mean()
    prediction_sample['mean'] = tries_mean
    print(prediction_sample['mean'])
    
    mse = mean_squared_error(prediction_sample['Mean # of Tries'], prediction_sample['mean'])
    print(mse)
    



if __name__ == "__main__":
    main()
