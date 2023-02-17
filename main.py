import pandas
import numpy as np

print("Running on Pandas Version", pandas.__version__)


def main():
    wordle_path = "Data/Problem_C_Data_Wordle.xlsx - Sheet1.csv"
    word_path = "Data/unigram_freq.csv"

    wordle_df = pandas.read_csv(wordle_path).dropna(axis=1)
    word_df = pandas.read_csv(word_path).set_index('word')

    keys = ['Date', 'Contest number', 'Number of  reported results', 'Number in hard mode']
    attempt_data = wordle_df.drop(keys, axis=1).set_index('Word')
    attempt_data2 = attempt_data.iloc[:150,:]
    attempt_data3 = attempt_data.iloc[150:,:]

    # Add column with frequency information
    def freq(word: str):
        try:
            count = word_df['count'][word]
        except KeyError:
            count = np.NaN
        return count

    attempt_data['Word Frequency'] = attempt_data.index.map(freq)

    # Add column with mean # of tries
    def mean_tries(tries: list):
        s = 0
        w = 0
        for try_n, count in enumerate(tries, 1):
            s += try_n * count
            w += count
        return s/w

    attempt_data2['Mean # of Tries'] = attempt_data2.apply(lambda x: mean_tries([
        x['1 try'], x['2 tries'], x['3 tries'], x['4 tries'], x['5 tries'], x['6 tries'], x['7 or more tries (X)']]),
                                                         axis=1)
    attempt_data3['Mean # of Tries'] = attempt_data3.apply(lambda x: mean_tries([
        x['1 try'], x['2 tries'], x['3 tries'], x['4 tries'], x['5 tries'], x['6 tries'], x['7 or more tries (X)']]),
                                                         axis=1)

    print(attempt_data.head())
    
    from sklearn.metrics import mean_squared_error
    
    tries_mean = attempt_data2['Mean # of Tries'].mean()
    attempt_data3['mean'] = tries_mean
    print(attempt_data3['mean'])
    
    mse = mean_squared_error(attempt_data3['Mean # of Tries'], attempt_data3['mean'])
    print(mse)
    



if __name__ == "__main__":
    main()
