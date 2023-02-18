import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

import datetime


def main():
    wordle_path = "Data/Problem_C_Data_Wordle no typos.csv"
    wordle_df = pandas.read_csv(wordle_path).dropna(axis=1)

    time_series = wordle_df.loc[:, ['Date', 'Number of  reported results', 'Number in hard mode']]

    time_series['Date'] = pandas.to_datetime(time_series['Date'])
    time_series['Day of Week'] = time_series['Date'].apply(lambda d: d.day_name())
    time_series['time'] = time_series['Date'].map(lambda t: t.timestamp())
    # time_series['time'] = (time_series['time'] - time_series['time'].min()) / (time_series['time'].max() - time_series['time'].min())

    fft_n_results = np.fft.fft(time_series['Number of  reported results'])
    fft_n_results_magnitude = np.absolute(fft_n_results)
    # fft_n_results_magnitude /= np.sum(fft_n_results_magnitude)  # normalize
    fft_n_results_angle = np.angle(fft_n_results)

    fft_hard_mode = np.fft.fft(time_series['Number in hard mode'])
    fft_hard_mode_magnitude = np.absolute(fft_hard_mode)
    # fft_hard_mode_magnitude /= np.sum(fft_hard_mode_magnitude)  # normalize
    fft_hard_mode_angle = np.angle(fft_hard_mode)

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

    # Regressions

    # Split our sample into a train and test sample

    # TODO: use word vector as part of the regression models
    x_train, x_test, y_train, y_test = train_test_split(
        time_series['time'].to_numpy(), time_series[['Number of  reported results', 'Number in hard mode']].to_numpy(),
        test_size=0.25, random_state=1917
    )

    models = pandas.DataFrame(y_test, index=x_test, columns=['Total Real Value', 'Hard Mode Real Value'])

    # Linear model
    lin_reg = LinearRegression().fit(x_train.reshape(-1, 1), y_train)
    lin_model = lin_reg.predict(x_test.reshape(-1, 1))
    models['Total Linear Model'] = lin_model[:, 0]
    models['Hard Mode Linear Model'] = lin_model[:, 1]

    # Poly model
    poly = PolynomialFeatures(degree=5, include_bias=False)
    poly_features = poly.fit_transform(x_train.reshape(-1, 1))
    poly_reg = LinearRegression().fit(poly_features, y_train)
    poly_model = poly_reg.predict(poly.fit_transform(x_test.reshape(-1, 1)))
    models['Total Polynomial Model'] = poly_model[:, 0]
    models['Hard Mode Polynomial Model'] = poly_model[:, 1]

    # Gradient Boosted Recessor model
    gbr = GradientBoostingRegressor(n_estimators=600,
                                    max_depth=5,
                                    learning_rate=0.01,
                                    min_samples_split=3)
    # GBR Only supports one dependent variable at a time so we will just do it twice, once for the totals and once
    # for the hard mode data.
    gbr.fit(x_train.reshape(-1, 1), y_train[:, 0])
    gbr_model_totals = gbr.predict(x_test.reshape(-1, 1))
    models['Total GBR Model'] = gbr_model_totals

    gbr.fit(x_train.reshape(-1, 1), y_train[:, 1])
    gbr_model_hard_mode = gbr.predict(x_test.reshape(-1, 1))
    models['Hard Mode GBR Model'] = gbr_model_hard_mode
    
    # Adaboost model
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(x_train.reshape(-1, 1), y_train[:, 0])
    ada_model_totals = clf.predict(x_test.reshape(-1,1))
    models['Total Ada Model'] = ada_model_totals
    
    clf.fit(x_train.reshape(-1, 1), y_train[:, 1])
    ada_model_hard_mode = clf.predict(x_test.reshape(-1, 1))
    models['Hard Mode Ada Model'] = ada_model_hard_mode
    
    # Linear SVR model
    svr = LinearSVR()
    # wrapper = MultiOutputRegressor(svr)
    svr.fit(x_train.reshape(-1, 1), y_train[:, 0])
    svr_model_totals = svr.predict(x_test.reshape(-1,1))
    models['Total SVR Model'] = svr_model_totals
    
    svr.fit(x_train.reshape(-1, 1), y_train[:, 1])
    svr_model_hard_mode = svr.predict(x_test.reshape(-1, 1))
    models['Hard Mode SVR Model'] = svr_model_hard_mode
    
    models.sort_index(inplace=True)

    # MSE Calculation
    mse_mean_t = models['Total Real Value'].var()
    mse_lin_t = mean_squared_error(models['Total Real Value'], models['Total Linear Model'])
    mse_poly_t = mean_squared_error(models['Total Real Value'], models['Total Polynomial Model'])
    mse_gbr_t = mean_squared_error(models['Total Real Value'], models['Total GBR Model'])
    mse_ada_t = mean_squared_error(models['Total Real Value'], models['Total Ada Model'])
    mse_svr_t = mean_squared_error(models['Total Real Value'], models['Total SVR Model'])

    mse_mean_h = models['Hard Mode Real Value'].var()
    mse_lin_h = mean_squared_error(models['Hard Mode Real Value'], models['Hard Mode Linear Model'])
    mse_poly_h = mean_squared_error(models['Hard Mode Real Value'], models['Hard Mode Polynomial Model'])
    mse_gbr_h = mean_squared_error(models['Hard Mode Real Value'], models['Hard Mode GBR Model'])
    mse_ada_h = mean_squared_error(models['Hard Mode Real Value'], models['Hard Mode Ada Model'])
    mse_svr_h = mean_squared_error(models['Hard Mode Real Value'], models['Hard Mode SVR Model'])

    print("Variance of totals:", mse_mean_t)
    print("Variance of hard mode:", mse_mean_h)

    print("Mean squared error of linear model over totals:", mse_lin_t)
    print("MSE/variance:", mse_lin_t/mse_mean_t)
    print("Mean squared error of linear model over hard mode data:", mse_lin_h)
    print("MSE/variance:", mse_lin_h/mse_mean_h)

    print("Mean squared error of polynomial model over totals:", mse_poly_t)
    print("MSE/variance:", mse_poly_t/mse_mean_t)
    print("Mean squared error of polynomial model over hard mode data:", mse_poly_h)
    print("MSE/variance:", mse_poly_h/mse_mean_h)

    print("Mean squared error of gradient boosted regressor model over totals:", mse_gbr_t)
    print("MSE/variance:", mse_gbr_t/mse_mean_t)
    print("Mean squared error of gradient boosted regressor model over hard mode data:", mse_gbr_h)
    print("MSE/variance:", mse_gbr_h/mse_mean_h)
    
    print("Mean squared error of ada regressor model over totals:", mse_ada_t)
    print("MSE/variance:", mse_ada_t/mse_mean_t)
    print("Mean squared error of ada regressor model over hard mode data:", mse_ada_h)
    print("MSE/variance:", mse_ada_h/mse_mean_h)
    
    print("Mean squared error of svr regressor model over totals:", mse_svr_t)
    print("MSE/variance:", mse_svr_t/mse_mean_t)
    print("Mean squared error of svr regressor model over hard mode data:", mse_svr_h)
    print("MSE/variance:", mse_svr_h/mse_mean_h)

    # Plots
    
    # Mat plot lib graph style
    rcParams['figure.figsize'] = 16, 8
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['lines.linewidth'] = 2.5
    # rcParams['axes.prop_cycle'] = cycler(color=['#424242'])
    rcParams['xtick.labelsize'] = 'xx-large'
    rcParams['ytick.labelsize'] = 'xx-large'
    

    plt.figure(1)
    plt.scatter(x=time_series['Date'], y=time_series['Number of  reported results'])

    plt.figure(2)
    plt.scatter(x=models.index, y=models['Total Real Value'], c="k", s=3, marker='.', label="Total Real Value")
    plt.plot(models.index, models['Total Linear Model'], label="Total Linear Model")
    plt.plot(models.index, models['Total Polynomial Model'], label="Total Polynomial Model")
    plt.plot(models.index, models['Total GBR Model'], label="Total GBR Model")
    plt.plot(models.index, models['Total Ada Model'], label="Total Ada Model")
    plt.plot(models.index, models['Total SVR Model'], label="Total SVR Model")

    plt.legend()

    fig = plt.figure(3)
    ax = fig.add_subplot(projection='3d')

    ax.scatter(np.arange(0, len(fft_n_results_magnitude)), fft_n_results_magnitude, fft_n_results_angle, c="y")
    ax.set_xlabel('Frequency Spectrum')
    ax.set_ylabel('Magnitude')
    ax.set_zlabel('Phase')
    ax.set_zticks(np.arange(- np.pi, np.pi, np.pi / 4))

    ind = np.arange(7)
    width = 0.4
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
