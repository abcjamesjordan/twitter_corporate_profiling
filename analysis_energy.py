import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

import matplotlib.pyplot as plt
import seaborn as sns

import os

from wordcloud import WordCloud

# Constants
ENERGY = ['coal', 'solar', 'wind', 'gas', 'petro']
TWITTER_USER_REGEX = r'@([a-zA-Z0-9_]+)'

# Paths
path = os.getcwd()

# Import data
df_dict = {}
for energy in ENERGY:
    path_current = os.path.join(path, 'data', energy+'.pkl')
    df_dict[energy] = pd.read_pickle(path_current)
    
print(df_dict['coal'].head(10))

# Drop empty columns
COLS_TO_DROP = ['timezone', 'place', 'language', 'cashtags', 'user_id', 'user_id_str', 'name', 'photos', 'video', 'retweet', 'search', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'retweet_date', 'translate', 'trans_src', 'trans_dest']
for key, df in df_dict.items():
    # drop empty columns
    df.drop(columns=COLS_TO_DROP, inplace=True)
    # date formating
    df['date'] = pd.to_datetime(df['date'])
    # add mentions list column
    df['mentions'] = df['tweet'].str.findall(TWITTER_USER_REGEX).apply(','.join).str.split(',')

print(df_dict['coal'])