# load required libraries

import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

from keras.models import Sequential
from keras.layers import Dense,Dropout, Flatten,LSTM,RepeatVector,TimeDistributed,Conv1D,MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from livelossplot.keras import PlotLossesCallback

from statsmodels.tsa.seasonal import seasonal_decompose


# Import the data set
df = pd.read_csv('data/total-rain-fall-in-mm.csv', sep = ';')



# Cleaning

df.insert(2, 'Day', '01')
df = df.replace({'Month':{'January':1, 'February':2 , 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December': 12}})                                                                                         

date_format = pd.to_datetime(df[['Year', 'Month', 'Day']])
df.insert(3, 'Date', date_format)

df

# Romving dublicate values 'Riyadh Old'


df['Region'].value_counts()

df.drop(df.index[df['Region'] == 'Riyadh Old'], inplace = True)


# Renaming 

df['Region'] = df['Region'].replace(['Riyadh New'],'Riyadh')
df['Region'] = df['Region'].replace(['Jeddah KAIA'],'Jeddah')

df.rename(columns={'Total Rainfall (mm)': 'Rainfall'}, inplace=True)

df




# Explore the data

df.info()

df.shape

# Glimplse on the first 10 rows

df.head(10)

df['Rainfall'].describe()


#set column as index
df = df.set_index('Date')
#df.reset_index()
#df.reset_index(drop=True)



# summary statistics
df.groupby(['Region'])['Year'].describe()

# What is the average  rainfall for each of the region?
df.groupby([df["Date"].dt.year, "Region"])["Rainfall"].mean()










# Plotting


# Plot rainfall data over the years

plt.figure(figsize=(30, 18))
plt.plot(df.Rainfall)
plt.title('Total Rainfall in KSA over 10 years', fontsize=50)
plt.ylabel('Total Rainfall', fontsize=45)
plt.xlabel('Trading Year', fontsize=34)
plt.grid(False)
plt.xticks(rotation=45)
plt.xticks(fontsize=34)
plt.yticks(fontsize=35)
plt.show()

df


# Seasonal Rainfall from Year 2009 to 2019 for all regions

df[['Year','Rainfall']].groupby("Year").mean().plot(figsize=(12,4));
plt.xlabel('Year',fontsize=20)
plt.ylabel(' Rainfall',fontsize=20)
plt.title(' Rainfall from Year 2009 to 2019 for all regions',fontsize=25)
plt.grid()
plt.ioff()


# Another plot
fig, axs = plt.subplots(figsize=(12, 4))
df.groupby(df["Date"].dt.year)["Rainfall"].mean().plot(kind='bar', rot=0, ax=axs)
plt.xlabel('Year',fontsize=20)
plt.ylabel('Rainfall',fontsize=20)


# Draw plot

def plot_df(df, x, y, title="c", xlabel='Date', ylabel='Rainfall', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(df, x=df.Year, y=df.Rainfall, title='Monthly rainfall from 2009 to 2019.')    



#