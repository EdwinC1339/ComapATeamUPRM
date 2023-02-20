# Libraries
import string
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


def day_of_week_vec(day: int):
    index = pd.Index(np.array(('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')))
    v = np.zeros(7)
    v[day] = 1
    return pd.Series(v, index=index)


def simple_word_vec(word: str, alphabet):
    return pd.Series([word.count(c) for c in alphabet], index=list(alphabet), name=word)


def vectorize_words(words: pd.Series):
    return words.apply(lambda w: simple_word_vec(w, string.ascii_lowercase))


if __name__ == "__main__":
    #read in the data
    wordle_df = pd.read_csv('Data/Problem_C_Data_Wordle no typos.csv')

    try_names = ['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']
    # time series
    time_series = wordle_df.loc[:, ['Date', 'Word'] + try_names]

    word_vec_cols = [('Word Vector', c) for c in string.ascii_lowercase]
    try_cols = [('Try Percentage', t) for t in try_names]
    days_cols = [('Day of Week', d) for d in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')]

    columns = pd.MultiIndex.from_tuples(days_cols + word_vec_cols + try_cols, names=['Category', 'Feature'])

    time_series['Date'] = pd.to_datetime(time_series['Date'])
    time_series['Day of Week'] = time_series['Date'].apply(lambda d: d.dayofweek)
    time_series['time'] = time_series['Date'].map(lambda t: t.timestamp())

    word_matrix = vectorize_words(time_series['Word'])
    word_matrix.index = time_series['Word']
    day_of_week_matrix = pd.DataFrame(time_series['Day of Week'].apply(day_of_week_vec))
    day_of_week_matrix.index = time_series['Word']
    try_matrix = time_series.loc[:, try_names]
    try_matrix.index = time_series['Word']

    data_set = pd.DataFrame(index=time_series['Word'], columns=columns)

    data_set['Day of Week'] = day_of_week_matrix
    data_set['Word Vector'] = word_matrix
    data_set['Try Percentage'] = try_matrix

    eerie_raw = pd.Series(
        [pd.to_datetime('2023/3/1'), 'eerie'],
        index=['Date', 'Word']
    )
    eerie_vec = simple_word_vec(eerie_raw['Word'], string.ascii_lowercase)
    eerie_weekday = day_of_week_vec(eerie_raw['Date'].dayofweek)
    eerie_vec_matrix = eerie_vec.to_frame().transpose()
    eerie_weekday_matrix = eerie_weekday.to_frame().transpose()
    eerie_weekday_matrix.index = ['eerie']
    eerie = pd.DataFrame(columns=columns, index=['eerie'])
    eerie['Word Vector'] = eerie_vec_matrix
    eerie['Day of Week'] = eerie_weekday_matrix

    train, test = train_test_split(
        data_set,
        test_size=0.20, random_state=1917
    )

    estimators = {
        "K-nn": KNeighborsRegressor(),
        "Linear Regression": LinearRegression(),
        "Ridge": RidgeCV(),
        "Lasso": Lasso(random_state=0),
        "Elastic Net": ElasticNet(random_state=0),
        "Random Forest": RandomForestRegressor(max_depth=4, random_state=2),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=0),
        "MultiO/P GBR": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5)),
        "MultiO/P AdaB": MultiOutputRegressor(AdaBoostRegressor(n_estimators=5))
    }

    x_train = train.drop('Try Percentage', axis=1)
    x_test = test.drop('Try Percentage', axis=1)
    y_train = train['Try Percentage']
    y_test = test['Try Percentage']

    models_cols = pd.MultiIndex.from_product(
        [['Ground Truth'] + list(estimators.keys()),
         try_names])
    models = pd.DataFrame(index=x_test.index, columns=models_cols)
    models['Ground Truth'] = y_test

    for name, estimator in estimators.items():
        estimator.fit(x_train, y_train)
        estimator_totals = pd.DataFrame(estimator.predict(x_test))
        estimator_totals.index = x_test.index
        estimator_totals.columns = try_names
        models[name] = estimator_totals
        print(name)
        print('training score', estimator.score(x_train, y_train))
        print('testing score', estimator.score(x_test, y_test))

    eerie_predictions = pd.DataFrame(index=list(estimators.keys()), columns=try_names)

    for i, estimator in enumerate(estimators.values()):
        prediction = pd.Series(estimator.predict(eerie.drop('Try Percentage', axis=1)).flat)
        prediction.index = try_names
        prediction_frame = prediction.to_frame().transpose()

        eerie_predictions.iloc[i] = prediction_frame

    eerie_predictions = eerie_predictions.astype(float).round(3)
    eerie_predictions.to_csv('eerie try predictions.csv')

    mse_all_models = pd.DataFrame(
        columns=list(estimators.keys()),
        index=try_names)
    variances = models['Ground Truth'].var()
    for model_name in estimators.keys():
        errors = models[model_name] - models['Ground Truth']
        squared_errors = errors * errors
        mses = squared_errors.mean()
        mse_all_models[model_name] = mses / variances

    models.sort_values(('Ground Truth', '4 tries'), inplace=True)
    # Plots

    # Mat plot lib graph style
    rcParams['figure.figsize'] = 19, 10
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['lines.linewidth'] = 2.5
    # rcParams['axes.prop_cycle'] = cycler(color=['#424242'])
    rcParams['xtick.labelsize'] = 'medium'
    rcParams['ytick.labelsize'] = 'medium'

    titles = ['1st Try', '2nd Try', '3rd Try', '4th Try', '5th Try', '6th Try', 'Fail']
    for i in range(7):
        fig = plt.figure(i + 1)
        ax = fig.add_subplot()
        title = 'Model Prediction: % of Games that ended in ' + titles[i] + ' Over Time'
        ax.scatter(
            models.index, models['Ground Truth', try_names[i]],
            label='Ground Truth')
        for j, e in enumerate(estimators.keys(), 1):
            ax.scatter(models.index, models[e, try_names[i]], label=e)

        ax.set_xticklabels(x_test.index, fontdict={'fontsize': 12})
        fig.legend()
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('% of Reports')
        fig.savefig("multivariate_gigamodel_figs\\" + titles[i] + ".svg", dpi=150)

    # Relative MSEs
    # set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1)

    mses_arr = mse_all_models.to_numpy()
    mses_summary = mse_all_models.mean()

    c = ax1.pcolor(mses_arr)
    ax1.set_title('Relative Mean Squared Error per Variable per Model')

    ax1.set_xticks(np.arange(len(mse_all_models.columns)) + 0.5)
    ax1.set_yticks(np.arange(len(mse_all_models.index)) + 0.5)

    ax1.set_xticklabels(mse_all_models.columns)
    ax1.set_yticklabels(mse_all_models.index)
    ax1.set_ylabel('Variable')
    ax1.set_xlabel('Model')

    fig.colorbar(c, ax=ax1)

    ax2.bar(mses_summary.index, mses_summary, color='purple')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_xlabel('Model')
    ax2.set_title('Mean Squared Error across all Variables per Model')

    fig.savefig("multivariate_gigamodel_figs\\Relative MSE.svg", dpi=150)

    plt.show()

