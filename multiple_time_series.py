import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams, cycler
import matplotlib.ticker as ticker 
import matplotlib.pyplot as plt

# read data
wordle_path = "Data/Problem_C_Data_Wordle no typos.csv"
wordle_df = pd.read_csv(wordle_path).dropna(axis=1)

time_series1 = wordle_df.loc[:, ['Date', '1 try']]
time_series2 = wordle_df.loc[:, ['Date', '2 tries']]
time_series3 = wordle_df.loc[:, ['Date', '3 tries']]
time_series4 = wordle_df.loc[:, ['Date', '4 tries']]
time_series5 = wordle_df.loc[:, ['Date', '5 tries']]
time_series6 = wordle_df.loc[:, ['Date', '6 tries']]
time_series7 = wordle_df.loc[:, ['Date', '7 or more tries (X)']]

time_series1['Date'] = pd.to_datetime(time_series1['Date'])
time_series1['Day of Week'] = time_series1['Date'].apply(lambda d: d.day_name())
time_series1['time'] = time_series1['Date'].map(lambda t: t.timestamp())

print(time_series1)

time_series2['Date'] = pd.to_datetime(time_series2['Date'])
time_series2['Day of Week'] = time_series2['Date'].apply(lambda d: d.day_name())
time_series2['time'] = time_series2['Date'].map(lambda t: t.timestamp())

print(time_series2)

time_series3['Date'] = pd.to_datetime(time_series3['Date'])
time_series3['Day of Week'] = time_series3['Date'].apply(lambda d: d.day_name())
time_series3['time'] = time_series3['Date'].map(lambda t: t.timestamp())

print(time_series3)

time_series4['Date'] = pd.to_datetime(time_series4['Date'])
time_series4['Day of Week'] = time_series4['Date'].apply(lambda d: d.day_name())
time_series4['time'] = time_series4['Date'].map(lambda t: t.timestamp())

print(time_series4)

time_series5['Date'] = pd.to_datetime(time_series5['Date'])
time_series5['Day of Week'] = time_series5['Date'].apply(lambda d: d.day_name())
time_series5['time'] = time_series5['Date'].map(lambda t: t.timestamp())

print(time_series5)

time_series6['Date'] = pd.to_datetime(time_series6['Date'])
time_series6['Day of Week'] = time_series6['Date'].apply(lambda d: d.day_name())
time_series6['time'] = time_series6['Date'].map(lambda t: t.timestamp())

print(time_series6)

time_series7['Date'] = pd.to_datetime(time_series7['Date'])
time_series7['Day of Week'] = time_series7['Date'].apply(lambda d: d.day_name())
time_series7['time'] = time_series7['Date'].map(lambda t: t.timestamp())

print(time_series7)
                              

# Mat plot lib graph style
rcParams['figure.figsize'] = 16, 8
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['lines.linewidth'] = 2.5
# rcParams['axes.prop_cycle'] = cycler(color=['#424242'])
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'

fig, ax = plt.subplots(figsize=(12,6))
sns.lineplot(x='Date',y='1 try',data=time_series1,label='attempt 1',ax=ax,color='black',legend=False)
plt.ylabel('# of attempts')
ax2 = ax.twinx()
sns.lineplot(x='Date',y='2 tries',data=time_series2,label='attempt 2',ax=ax,color='blue',legend=False)
ax3 = ax.twinx()
sns.lineplot(x='Date',y='3 tries',data=time_series3,label='attempt 3',ax=ax,color='red',legend=False)
ax4 = ax.twinx()
sns.lineplot(x='Date',y='4 tries',data=time_series4,label='attempt 4',ax=ax,color='green',legend=False)
ax5 = ax.twinx()
sns.lineplot(x='Date',y='5 tries',data=time_series5,label='attempt 5',ax=ax,color='yellow',legend=False)
ax6 = ax.twinx()
sns.lineplot(x='Date',y='6 tries',data=time_series6,label='attempt 6',ax=ax,color='purple',legend=False)
ax7 = ax.twinx()
sns.lineplot(x='Date',y='7 or more tries (X)',data=time_series7,label='attempt 7',ax=ax,color='pink',legend=False)
ax.figure.legend()
ax2.figure.legend()
ax3.figure.legend()
ax4.figure.legend()
ax5.figure.legend()
ax6.figure.legend()
ax7.figure.legend()
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax3.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax4.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax5.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax6.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax7.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.suptitle('Time series plot',size=20)
plt.show()

