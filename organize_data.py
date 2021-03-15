import pandas as pd
import numpy as numpy
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from wordcloud import WordCloud

from sklearn.preprocessing import MultiLabelBinarizer

# Constants
DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS_OF_YEAR = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Regex variables
twitter_handle_regex = r'@([a-zA-Z0-9_]+)'

def create_sums_variable(df_sums, mentions=False, hashtags=False):
    '''
    Returns either mentions or hashtags into a list of occurrences within tweets dataframe
    '''
    if mentions:
        s = df_sums['mentions']
        mlb = MultiLabelBinarizer()
        df_mentions = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_)
        sum_variable = df_mentions.sum(axis=0, skipna=True).sort_values(ascending=False)
        return sum_variable
    elif hashtags:
        s = df_sums['hashtags']
        mlb = MultiLabelBinarizer()
        df_hashtags = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_)
        sum_variable = df_hashtags.sum(axis=0, skipna=True).sort_values(ascending=False)
        return sum_variable
    else:
        print('Do you want hashtags or mentions??')
        
def get_tweets_by(df, unit_of_time):
    '''
    Returns number of tweets by day, hour, dayofweek, week, month, or year.
    '''
    if unit_of_time == 'day':
        per_day = df.groupby([df['date'].dt.date]).sum().loc[:, ['one', 'engagements']].reset_index().rename(columns={'one': 'total_tweets'})
        per_day['type'] = df['type'].iloc[0]
        per_day['month'] = pd.to_datetime(per_day['date']).dt.month_name()
        per_day['year'] = pd.to_datetime(per_day['date']).dt.year
        return per_day
    elif unit_of_time == 'hour':
        per_hour = df.groupby([df['date'].dt.hour]).sum().loc[:, ['one', 'engagements']].reset_index().rename(columns={'one': 'total_tweets'})
        per_hour['type'] = df['type'].iloc[0]
        return per_hour
    elif unit_of_time == 'dayofweek':
        per_week_day = df.groupby([df['date'].dt.day_name()]).sum().loc[:, ['one', 'engagements']].reset_index().rename(columns={'one': 'total_tweets'})
        per_week_day['type'] = df['type'].iloc[0]
        return per_week_day
    elif unit_of_time == 'week':
        per_week = df.groupby([df['date'].dt.isocalendar().week]).sum().loc[:, ['one', 'engagements']].reset_index().rename(columns={'one': 'total_tweets'})
        per_week['type'] = df['type'].iloc[0]
        return per_week
    elif unit_of_time == 'month':
        per_month = df.groupby([df['date'].dt.month_name()]).sum().loc[:, ['one', 'engagements']].reset_index().rename(columns={'one': 'total_tweets'})
        per_month['type'] = df['type'].iloc[0]
        return per_month
    elif unit_of_time == 'year':
        per_year = df.groupby([df['date'].dt.year]).sum().loc[:, ['one', 'engagements']].reset_index().rename(columns={'one': 'total_tweets'})
        per_year['type'] = df['type'].iloc[0]
        return per_year
    else:
        print('please choose either day, hour, dayofweek, month, or year')

def average_tweets_by(df, unit_of_time):
    '''
    Returns the average number of tweets by day, hour, week, month, or year.
    '''
    if unit_of_time == 'day':
        df['day/year'] = df['date'].apply(lambda x: '%d/%d' % (x.day, x.year))
        return df.groupby('day/year').size().mean()
    elif unit_of_time == 'hour':
        df['hour/year'] = df['date'].apply(lambda x: '%d/%d' % (x.hour, x.year))
        return df.groupby('hour/year').size().mean()
    elif unit_of_time == 'week':
        df['week/year'] = df['date'].apply(lambda x: '%d/%d' % (x.week, x.year))
        return df.groupby('week/year').size().mean()
    elif unit_of_time == 'month':
        df['month/year'] = df['date'].apply(lambda x: '%d/%d' % (x.month, x.year))
        return df.groupby('month/year').size().mean()
    elif unit_of_time == 'year':
        df['year'] = df['date'].dt.year
        return df.groupby('year').size().mean()
    else:
        print('please use either day, hour, week, month, or year')
    
def organize_df(df):
    ''' 
    Organize each dataframe. Returns dataframe, mentions sum df, and hashtag sum df.
    '''
    # Fix the dates
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['week'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Add mentions column similar to hashtags column
    twitter_handle_regex = r'@([a-zA-Z0-9_]+)'
    df['mentions'] = df['tweet'].str.findall(twitter_handle_regex).apply(','.join)
    df['mentions'] = df['mentions'].str.split(',')
    
    # Create the summing variables for mentions and hashtags
    sum_mentions = create_sums_variable(df, mentions=True)
    sum_hashtags = create_sums_variable(df, hashtags=True)
    
    # Reorder dataframe
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # Add counter column
    df['count'] = df.index + 1
    df['one'] = 1
    
    # Add engagements column
    df['engagements'] = df.loc[:, ['nlikes', 'nreplies', 'nretweets']].sum(axis=1)
    
    return df, sum_mentions, sum_hashtags

def import_dataframes():
    ''' 
    Imports all dataframes from pickle files
    '''
    path = os.getcwd()
    file_path_tweets = os.path.join(path, 'tweets_df.pkl')
    file_path_user = os.path.join(path, 'user_df.pkl')
    file_path_mention1 = os.path.join(path, 'tweets_mention1_df.pkl')
    file_path_mention2 = os.path.join(path, 'tweets_mention2_df.pkl')
    file_path_mention3 = os.path.join(path, 'tweets_mention3_df.pkl')
    df = pd.read_pickle(file_path_tweets)
    df_user = pd.read_pickle(file_path_user)
    df_mention1 = pd.read_pickle(file_path_mention1)
    df_mention2 = pd.read_pickle(file_path_mention2)
    df_mention3 = pd.read_pickle(file_path_mention3)
    
    return df, df_user, df_mention1, df_mention2, df_mention3

def create_per_df(df):
    '''
    Returns all dataframes of per_blank statistics (day, hour, dayofweek, week, month, year)
    '''
    per_day = get_tweets_by(df, 'day')
    per_hour = get_tweets_by(df, 'hour')
    per_dayofweek = get_tweets_by(df, 'dayofweek')
    per_week = get_tweets_by(df, 'week')
    per_month = get_tweets_by(df, 'month')
    per_year = get_tweets_by(df, 'year')
    
    return per_day, per_hour, per_dayofweek, per_week, per_month, per_year

# Import dataframes
df, df_user, df_mention1, df_mention2, df_mention3 = import_dataframes()

# Import top 3 mentions
top_3_mentions = []
with open('top_3_mentions.txt', 'r') as f:
    for line in f:
        top_3_mentions.append(str(line).replace('\n', ''))

# Organize all datasets
df, sum_mentions, sum_hashtags = organize_df(df)
df['type'] = str('@'+df_user['name'].iloc[0])
df_mention1, sum_mentions1, sum_hashtags1 = organize_df(df_mention1)
df_mention1['type'] = '@'+top_3_mentions[0]
df_mention2, sum_mentions2, sum_hashtags2 = organize_df(df_mention2)
df_mention2['type'] = '@'+top_3_mentions[1]
df_mention3, sum_mentions3, sum_hashtags2 = organize_df(df_mention3)
df_mention3['type'] = '@'+top_3_mentions[2]

# # Create various dataframes for plotting analysis
tweets_per_day, tweets_per_hour, tweets_per_dayofweek, tweets_per_week, tweets_per_month, tweets_per_year = create_per_df(df)
tweets_per_day1, tweets_per_hour1, tweets_per_dayofweek1, tweets_per_week1, tweets_per_month1, tweets_per_year1 = create_per_df(df_mention1)
tweets_per_day2, tweets_per_hour2, tweets_per_dayofweek2, tweets_per_week2, tweets_per_month2, tweets_per_year2 = create_per_df(df_mention2)
tweets_per_day3, tweets_per_hour3, tweets_per_dayofweek3, tweets_per_week3, tweets_per_month3, tweets_per_year3 = create_per_df(df_mention3)

# Master dataframes for plotting
df_master = pd.concat([df, df_mention1, df_mention2, df_mention3]).reset_index(drop=True)
df_master_per_day = pd.concat([tweets_per_day, tweets_per_day1, tweets_per_day2, tweets_per_day3]).reset_index(drop=True)
df_master_per_hour = pd.concat([tweets_per_hour, tweets_per_hour1, tweets_per_hour2, tweets_per_hour3]).reset_index(drop=True)
df_master_per_dayofweek = pd.concat([tweets_per_dayofweek, tweets_per_dayofweek1, tweets_per_dayofweek2, tweets_per_dayofweek3]).reset_index(drop=True)
df_master_per_month = pd.concat([tweets_per_month, tweets_per_month1, tweets_per_month2, tweets_per_month3]).reset_index(drop=True)
df_master_per_year = pd.concat([tweets_per_year, tweets_per_year1, tweets_per_year2, tweets_per_year3]).reset_index(drop=True)


### Plotting section
###
# Tweets over time
# sns.set_context('poster')
# sns.relplot(data=df_master, x='date', y='count', kind='line', hue='type', height=10, aspect=1.5, legend=True)
# plt.title('Total Tweets vs Top 3 Mentions', fontsize=18)
# plt.xlabel('Time')
# plt.ylabel('# of Tweets')
# plt.show()

###
# # Engagments vs Number of Tweets
# sns.relplot(data=tweets_per_day, x='date', y='engagements', size='total_tweets', sizes=(10, 1000), kind='scatter', height=10, aspect=1.5)
# plt.title('Engagments vs Number of Tweets per day', fontsize=18)
# plt.xlabel('Tweets per day')
# plt.ylabel('# of engagements')
# plt.show()

###
# Tweets by weekday
# sns.catplot(data=df_master_per_dayofweek, x='date', y='total_tweets', kind='point', hue = 'type', legend=True, height=10, aspect=1.5, order=DAYS_OF_WEEK)
# plt.title('Tweets By Weekday', fontsize=18)
# plt.xlabel('Day of week')
# plt.ylabel('# of tweets')
# plt.show()

# Tweets by month
# fig = plt.figure(figsize=(16, 8))
# sns.barplot(data=tweets_per_month, x='date', y='total_tweets', order=MONTHS_OF_YEAR)
# plt.title('Tweets By Month')
# plt.xlabel('Month')
# plt.ylabel('# of tweets')
# plt.show()

####
# Tweets by month per day averages
# fig = plt.figure(figsize=(16, 8))
# sns.barplot(data=tweets_per_day, x='month', y='total_tweets', order=MONTHS_OF_YEAR)
# plt.title('Tweets Per Day Each Month', fontsize=18)
# plt.xlabel('Month')
# plt.ylabel('# of tweets per day')
# plt.show()

#####
# Tweets by year per day averages
# fig = plt.figure(figsize=(16, 8))
# sns.barplot(data=tweets_per_day, x='year', y='total_tweets')
# plt.title('Tweets Per Day Each Year', fontsize=18)
# plt.xlabel('Year')
# plt.ylabel('# of tweets per day')
# plt.show()

# Engagements over time
sns.relplot(data=df, x='date', y='engagements', kind='scatter', height=10, aspect=1.5, legend=True)
plt.title('Engagements vs Top 3 Mentions')
plt.xlabel('Time')
plt.ylabel('Engagements')
plt.show()

# # Wordclouds
# # Mentions
# mentions_dict = sum_mentions.to_dict()
# wordcloud = WordCloud(width=1600, height=800).generate_from_frequencies(mentions_dict)

# plt.figure(figsize = (20,10), dpi=600, facecolor='k')
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.tight_layout(pad=0)
# plt.show()

# # Hashtags
# hashtags_dict = sum_hashtags.to_dict()
# wordcloud = WordCloud(width=1600, height=800).generate_from_frequencies(hashtags_dict)

# plt.figure(figsize = (20,10), dpi=600, facecolor='k')
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.tight_layout(pad=0)
# plt.show()








# # Database Stats
# NUM_OF_TWEETS = len(df)
# TOTAL_LIKES = df['nlikes'].sum()
# TOTAL_REPLIES = df['nreplies'].sum()
# TOTAL_RETWEETS = df['nretweets'].sum()







# # test_df = df.copy()
# # test_df['count'] = 1
# # print(test_df.groupby(['month']).mean())
# # print(test_df.groupby(['month']).sum().iloc[:, -1].mean())
# # print(test_df.columns)
# # print(sum_mentions)

# # //FIXME fix the data statstics
# # Database statistics
# AVG_TWEETS_PER_DAY = tweets_per_day['count'].mean()
# # AVG_TWEETS_PER_HOUR = tweets_per_hour['count'].mean()
# # AVG_TWEETS_PER_WEEK = tweets_per_week['count'].mean()
# # AVG_TWEETS_PER_MONTH = tweets_per_month['count'].mean()
# # AVG_TWEETS_PER_YEAR = tweets_per_year['count'].mean()
# # print('Number of tweets', NUM_OF_TWEETS)
# # print(AVG_TWEETS_PER_DAY, AVG_TWEETS_PER_HOUR, AVG_TWEETS_PER_WEEK, AVG_TWEETS_PER_MONTH, AVG_TWEETS_PER_YEAR)

# # print('Tweets per day', AVG_TWEETS_PER_DAY)


# # User statistics:
# print(df_user.columns)
# NUM_OF_FOLLOWERS = df_user['followers'].iloc[0]
# USER_NAME = df_user['name'].iloc[0]
# JOIN_DATE = df_user['join_date'].iloc[0]
# NUM_OF_TWEETS_ACTUAL = df_user['tweets'].iloc[0]

# print(USER_NAME)
# print(f'Number of Followers: {NUM_OF_FOLLOWERS}')
# print(f'Join Date: {JOIN_DATE}')
# print(f'Number of Tweets: {NUM_OF_TWEETS_ACTUAL}')