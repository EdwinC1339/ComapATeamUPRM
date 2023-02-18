import pandas
import numpy as np
import matplotlib.pyplot as plt
import datetime


def main():
    wordle_path = "Data/Problem_C_Data_Wordle no typos.csv"
    wordle_df = pandas.read_csv(wordle_path).dropna(axis=1)

    time_series = wordle_df.loc[:, ['Date', 'Number of  reported results', 'Number in hard mode']]

    time_series['Date'] = pandas.to_datetime(time_series['Date'])
    time_series['Day of Week'] = time_series['Date'].apply(lambda d: d.day_name())

    fft_n_results = np.abs(np.fft.fft(time_series['Number of  reported results']))
    fft_n_results /= np.size(fft_n_results)
    fft_hard_mode = np.abs(np.fft.fft(time_series['Number in hard mode']))
    fft_hard_mode /= np.size(fft_hard_mode)

    weekly = time_series.groupby('Day of Week')

    weekly_average = weekly.mean()
    sorter = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    sorter_index = dict(zip(sorter, range(len(sorter))))
    weekly_average['day_id'] = weekly_average.index.map(sorter_index)
    weekly_average.sort_values('day_id', inplace=True)
    weekly_average = weekly_average.drop('day_id', axis=1)

    cvs = weekly_average.std() / weekly_average.mean()
    print("Coefficient of variation for total reports:", cvs['Number of  reported results'])
    print("Coefficient of variation for hard mode reports:", cvs['Number in hard mode'])

    plt.figure(1)
    plt.scatter(x=time_series['Date'], y=time_series['Number of  reported results'])

    plt.figure(2)
    plt.scatter(x=np.arange(0, len(fft_n_results)), y=fft_n_results)

    ind = np.arange(7)
    width = 0.35
    offset = 0.05

    fig, ax = plt.subplots()
    bars_n = ax.bar(ind, weekly_average['Number of  reported results'],
                    width, color='black', label='Total')
    bars_hard_mode = ax.bar(ind + width + offset, weekly_average['Number in hard mode'],
                            width, color='orange', label='Hard Mode')
    ax.set_ylabel('Average number of reported results')
    ax.set_xlabel('Day of Week')
    ax.set_xticks(ind + (width + offset) / 2)
    ax.set_xticklabels(weekly_average.index, fontdict={'fontsize': 8})
    fig.legend()

    plt.show()


if __name__ == "__main__":
    main()
