import pandas

print("Running on Pandas Version", pandas.__version__)


def main():
    path = "Data/Problem_C_Data_Wordle.xlsx - Sheet1.csv"
    df = pandas.read_csv(path).dropna(axis=1)
    attempt_data = df.drop(['Date', 'Contest number', 'Number of  reported results', 'Number in hard mode'], axis=1)

    print(attempt_data)


if __name__ == "__main__":
    main()
