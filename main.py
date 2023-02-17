import pandas

print("Running on Pandas Version", pandas.__version__)


def main():
    wordle_path = "Data/Problem_C_Data_Wordle.xlsx - Sheet1.csv"
    word_path = "Data/unigram_freq.csv"

    wordle_df = pandas.read_csv(wordle_path).dropna(axis=1)
    word_df = pandas.read_csv(word_path).set_index('word')

    keys = ['Date', 'Contest number', 'Number of  reported results', 'Number in hard mode']
    attempt_data = wordle_df.drop(keys, axis=1).set_index('Word')

    print(attempt_data.head())
    print(word_df.head())



if __name__ == "__main__":
    main()
