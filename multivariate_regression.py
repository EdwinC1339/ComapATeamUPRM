# Libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
##Metrics
from sklearn.metrics import mean_squared_error

# Regressors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor

if __name__ == "__main__":
    #read in the data
    wordle_df = pd.read_csv('Data/Problem_C_Data_Wordle no typos.csv')

    # time series
    time_series = wordle_df.loc[:, ['Date', '1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']]

    time_series['Date'] = pd.to_datetime(time_series['Date'])
    time_series['Day of Week'] = time_series['Date'].apply(lambda d: d.day_name())
    time_series['time'] = time_series['Date'].map(lambda t: t.timestamp())
    time_series = time_series.iloc[::-1]
    time_series = time_series.set_index('Date')

    x_train, x_test, y_train, y_test = train_test_split(
        time_series['time'], time_series[['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']].to_numpy(),
        test_size=0.20, random_state=1917, shuffle=False
    )


    estimators = {
        "K-nn": KNeighborsRegressor(),
        "Linear regression": LinearRegression(),
        "Ridge": RidgeCV(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(random_state=0),
        "RandomForestRegressor": RandomForestRegressor(max_depth=4, random_state=2),
        "Decision Tree Regressor":DecisionTreeRegressor(max_depth=5),
        "MultiO/P GBR" :MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5)),
        "MultiO/P AdaB" :MultiOutputRegressor(AdaBoostRegressor(n_estimators=5))
    }

    models = pd.DataFrame(y_test, index=x_test, columns=['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)'])
    long_term_prediction = pd.DataFrame(index=pd.date_range('1/1/2022', '1/1/2024', 360))
    long_term_prediction['time'] = long_term_prediction.index.map(lambda t: t.timestamp())

    model = dict()

    for name, estimator in estimators.items():
        estimator.fit(x_train.to_numpy().reshape(-1, 1), y_train)                    # fit() with instantiated object
        estimator_totals = estimator.predict(x_test.to_numpy().reshape(-1, 1))
        print(name)
        model[str(name)] = estimator_totals
        print('training score',estimator.score(x_train.to_numpy().reshape(-1,1),y_train))
        print('testing score',estimator.score(x_test.to_numpy().reshape(-1,1),y_test))

    models = pd.concat({k: pd.DataFrame(v).T for k, v in model.items()}, axis=0)
    for i, l in enumerate(['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']):
        s = pd.Series(y_test[:, i], name='Ground Truth/' + str(i))
        models = models.append(s)
    models = models.transpose()
    models.index = x_test.index

    print(models)

    # Plots

    # Mat plot lib graph style
    rcParams['figure.figsize'] = 16, 8
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['lines.linewidth'] = 2.5
    # rcParams['axes.prop_cycle'] = cycler(color=['#424242'])
    rcParams['xtick.labelsize'] = 'xx-large'
    rcParams['ytick.labelsize'] = 'xx-large'

    titles = ['1st Try', '2nd Try', '3rd Try', '4th Try', '5th Try', '6th Try', 'Fail']
    for i in range(7):
        fig = plt.figure(i + 1)
        ax = fig.add_subplot()
        title = 'Model Prediction: % of Games that ended in ' + titles[i] + ' Over Time'
        ax.scatter(
            x_test, models['Ground Truth/' + str(i)],
            label='Ground Truth')
        for j, e in enumerate(estimators.keys(), 1):
            ax.plot(x_test, models[e, i], label=e)

        ax.set_xticklabels(x_test.index, fontdict={'fontsize': 12})
        ax.margins(x=0)
        fig.legend()
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('% of Reports')
        fig.savefig("multivariate_figs\\" + titles[i] + ".svg", dpi=150)

    plt.show()

