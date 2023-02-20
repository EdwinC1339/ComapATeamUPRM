import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
from matplotlib import rcParams

lev = pd.read_csv ('folder/subfolder/lev_out.csv')
wordle = pd.read_csv ('folder/subfolder/wor_out.csv')

# Mat plot lib graph style
rcParams['figure.figsize'] = 19, 10
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['lines.linewidth'] = 2.5
# rcParams['axes.prop_cycle'] = cycler(color=['#424242'])
rcParams['xtick.labelsize'] = 'medium'
rcParams['ytick.labelsize'] = 'medium'

fig, ax2 = plt.subplots(1)

mse_all_models = pd.DataFrame()
mse_all_models['APC-Levenshtein'] = lev['mse'].copy()
mse_all_models['APC-Wordle'] = wordle['mse'].copy()

print(mse_all_models)

mses_arr = mse_all_models.to_numpy()
mses_summary = mse_all_models.mean()


ax2.bar(mses_summary.index, mses_summary, color='purple')
ax2.set_ylabel('Relative Root Mean Squared Error')
ax2.set_xlabel('Model')
ax2.set_title('Relative Root Mean Squared Error across all Variables per Model')


plt.show()