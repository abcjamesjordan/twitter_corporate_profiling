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
COMPANY = 'veevasystems'

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
        per_day['company'] = df['company'].iloc[0]
        per_day['month'] = pd.to_datetime(per_day['date']).dt.month_name()
        per_day['year'] = pd.to_datetime(per_day['date']).dt.year
        return per_day
    elif unit_of_time == 'hour':
        per_hour = df.groupby([df['date'].dt.hour]).sum().loc[:, ['one', 'engagements']].reset_index().rename(columns={'one': 'total_tweets'})
        per_hour['company'] = df['company'].iloc[0]
        return per_hour
    elif unit_of_time == 'dayofweek':
        per_week_day = df.groupby([df['date'].dt.day_name()]).sum().loc[:, ['one', 'engagements']].reset_index().rename(columns={'one': 'total_tweets'})
        per_week_day['company'] = df['company'].iloc[0]
        return per_week_day
    elif unit_of_time == 'week':
        per_week = df.groupby([df['date'].dt.isocalendar().week]).sum().loc[:, ['one', 'engagements']].reset_index().rename(columns={'one': 'total_tweets'})
        per_week['company'] = df['company'].iloc[0]
        return per_week
    elif unit_of_time == 'month':
        per_month = df.groupby([df['date'].dt.month_name()]).sum().loc[:, ['one', 'engagements']].reset_index().rename(columns={'one': 'total_tweets'})
        per_month['company'] = df['company'].iloc[0]
        return per_month
    elif unit_of_time == 'year':
        per_year = df.groupby([df['date'].dt.year]).sum().loc[:, ['one', 'engagements']].reset_index().rename(columns={'one': 'total_tweets'})
        per_year['company'] = df['company'].iloc[0]
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
    sum_mentions_10 = sum_mentions[0:10].drop('', axis=0)
    sum_mentions_10 = sum_mentions_10.to_frame()
    sum_mentions_10 = sum_mentions_10.rename(columns={0: 'mentions'})
    sum_hashtags = create_sums_variable(df, hashtags=True)
    
    # Reorder dataframe
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # Add counter column
    df['count'] = df.index + 1
    df['one'] = 1
    
    # Add engagements column
    df['engagements'] = df.loc[:, ['nlikes', 'nreplies', 'nretweets']].sum(axis=1)
    
    # df = add_mentions_counter(df)
    
    return df, sum_mentions, sum_hashtags, sum_mentions_10

def import_dataframes():
    ''' 
    Imports all dataframes from pickle files
    '''
    path = os.getcwd()
    file_path_tweets = os.path.join(path, 'data/tweets_df.pkl')
    file_path_user = os.path.join(path, 'data/user_df.pkl')
    file_path_mention1 = os.path.join(path, 'data/tweets_mention1_df.pkl')
    file_path_mention2 = os.path.join(path, 'data/tweets_mention2_df.pkl')
    file_path_mention3 = os.path.join(path, 'data/tweets_mention3_df.pkl')
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

def make_plots():
    '''
    Make all plots for PDF
    '''   
    sns.set_context('poster')
    # Top 10 Mentions
    fig = plt.figure(figsize=(16, 8))
    sns.barplot(x=sum_mentions_10.index, y=sum_mentions_10['mentions'])
    plt.title('Top 10 Mentions')
    plt.xlabel('Twitter handle')
    plt.ylabel('# of Mentions')
    plt.xticks(rotation=40)
    plt.savefig('images/top10_mentions.png', bbox_inches='tight')
    
    # Engagements vs mentions
    sns.catplot(data=engage_mention_concat, x='company', y='engagement_per_tweet', hue='mentioned', kind='bar', height=10, aspect=1.8)
    plt.title('Engagements When Twitter Handle Mentioned In Tweet')
    plt.xlabel('Company')
    plt.ylabel('Average # of engagements')
    plt.savefig('images/engagements_vs_mentions.png', bbox_inches='tight')
    
    # Engagements vs hashtags
    
    # Tweets over time
    sns.relplot(data=df_master, x='date', y='count', kind='line', hue='company', height=10, aspect=1.8, legend=True)
    plt.title('Total Tweets vs Top 3 Mentions Total Tweets')
    plt.xlabel('Time')
    plt.ylabel('# of Tweets')
    plt.savefig('images/tweets_over_time.png', bbox_inches='tight')
    
    sns.relplot(data=df, x='date', y='count', kind='line', hue='company', height=10, aspect=2, legend=False)
    plt.title('Total Tweets Over Time')
    plt.xlabel('Time')
    plt.ylabel('# of Tweets')
    plt.savefig('images/total_tweets.png', bbox_inches='tight')

    # Mentions Plots
    sns.relplot(data=df_long, x='date', y='index', kind='line', hue='company', height=10, aspect=1.8, legend=True)
    plt.title('Mentions Over Time')
    plt.xlabel('Time')
    plt.ylabel('# of Tweets')
    plt.savefig('images/mentions_over_time.png', bbox_inches='tight')

    # Engagements vs Number of Tweets
    sns.relplot(data=tweets_per_day, x='date', y='engagements', size='total_tweets', sizes=(10, 1000), kind='scatter', height=10, aspect=1.8)
    plt.title('Engagments vs Number of Tweets Per Day')
    plt.xlabel('Date')
    plt.ylabel('# of Engagements')
    plt.savefig('images/engagements_vs_tweets.png', bbox_inches='tight')

    # Tweets by weekday
    sns.catplot(data=df_master_per_dayofweek, x='date', y='total_tweets', kind='point', hue = 'company', legend=True, height=10, aspect=1.5, order=DAYS_OF_WEEK)
    plt.title('Tweets By Weekday')
    plt.xlabel('Day of week')
    plt.ylabel('# of Tweets')
    plt.savefig('images/tweets_weekday_all.png', bbox_inches='tight')

    # Tweets by month
    fig = plt.figure(figsize=(16, 8))
    sns.barplot(data=tweets_per_month, x='date', y='total_tweets', order=MONTHS_OF_YEAR)
    plt.title('Tweets By Month')
    plt.xlabel('Month')
    plt.ylabel('# of Tweets')
    plt.savefig('images/tweets_month.png', bbox_inches='tight')

    # Tweets by month per day averages
    fig = plt.figure(figsize=(16, 8))
    sns.barplot(data=tweets_per_day, x='month', y='total_tweets', order=MONTHS_OF_YEAR)
    plt.title('Tweets Per Day Each Month')
    plt.xlabel('Month')
    plt.ylabel('# of Tweets per day')
    plt.xticks(rotation=40)
    plt.savefig('images/tweets_month_per_day_avg.png', bbox_inches='tight')

    # Tweets by year per day averages
    fig = plt.figure(figsize=(16, 8))
    sns.barplot(data=tweets_per_day, x='year', y='total_tweets')
    plt.title('Tweets Per Day Each Year')
    plt.xlabel('Year')
    plt.ylabel('# of Tweets per day')
    plt.savefig('images/tweets_year_per_day_avg.png', bbox_inches='tight')
    
    # Wordclouds
    # Mentions
    mentions_dict = sum_mentions.to_dict()
    wordcloud = WordCloud(width=1600, height=800).generate_from_frequencies(mentions_dict)

    plt.figure(figsize = (20,10), dpi=100, facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('images/mentions_wordcloud.png', bbox_inches='tight')

    # Hashtags
    hashtags_dict = sum_hashtags.to_dict()
    wordcloud = WordCloud(width=1600, height=800).generate_from_frequencies(hashtags_dict)

    plt.figure(figsize = (20,10), dpi=100, facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('images/hashtags_wordcloud.png', bbox_inches='tight')
    
    return

def add_mentions_counter(df):
    df[top_3_mentions[0]] = [1 if bool(set(x) & set(top_3_mentions[0].split())) else 0 for x in df['mentions']]
    df[top_3_mentions[1]] = [1 if bool(set(x) & set(top_3_mentions[1].split())) else 0 for x in df['mentions']]
    df[top_3_mentions[2]] = [1 if bool(set(x) & set(top_3_mentions[2].split())) else 0 for x in df['mentions']]

    mention1 = df[df[top_3_mentions[0]] == 1]
    mention1 = mention1[['date']].reset_index(drop=True)
    mention1 = mention1.reset_index()
    mention1['company'] = top_3_mentions[0]

    mention2 = df[df[top_3_mentions[1]] == 1]
    mention2 = mention2[['date']].reset_index(drop=True)
    mention2 = mention2.reset_index()
    mention2['company'] = top_3_mentions[1]

    mention3 = df[df[top_3_mentions[2]] == 1]
    mention3 = mention3[['date']].reset_index(drop=True)
    mention3 = mention3.reset_index()
    mention3['company'] = top_3_mentions[2]

    mentions_df = pd.concat([mention1, mention2, mention3], ignore_index=True).reset_index(drop=True)
    
    return mentions_df

def engagement_by_mention(df, mention_name):
    df_temp = df.groupby(df[mention_name]).sum()
    df_temp = df_temp[['one', 'engagements']]
    df_temp['engagement_per_tweet'] = df_temp['engagements'] / df_temp['one']
    df_temp['company'] = mention_name
    df_temp['mentioned'] = 'No'
    df_temp['mentioned'].iloc[1] = 'Yes'
    return df_temp

def engagement_concat(df):
    engage_mention1 = engagement_by_mention(df, top_3_mentions[0])
    engage_mention2 = engagement_by_mention(df, top_3_mentions[1])
    engage_mention3 = engagement_by_mention(df, top_3_mentions[2])
    engage_mention_concat = pd.concat([engage_mention1, engage_mention2, engage_mention3], ignore_index=True).reset_index(drop=True)
    
    return engage_mention_concat

# Import dataframes
df, df_user, df_mention1, df_mention2, df_mention3 = import_dataframes()

# Import top 3 mentions
top_3_mentions = []
with open('data/top_3_mentions.txt', 'r') as f:
    for line in f:
        top_3_mentions.append(str(line).replace('\n', ''))

# Organize all datasets
df, sum_mentions, sum_hashtags, sum_mentions_10 = organize_df(df)
df_long = add_mentions_counter(df)
df['company'] = str('@'+df_user['name'].iloc[0])
df_mention1, sum_mentions1, sum_hashtags1, sum_mentions1_10 = organize_df(df_mention1)
df_mention1['company'] = '@'+top_3_mentions[0]
df_mention2, sum_mentions2, sum_hashtags2, sum_mentions2_10 = organize_df(df_mention2)
df_mention2['company'] = '@'+top_3_mentions[1]
df_mention3, sum_mentions3, sum_hashtags2, sum_mentions3_10 = organize_df(df_mention3)
df_mention3['company'] = '@'+top_3_mentions[2]

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

# Engagements by mention
engage_mention_concat = engagement_concat(df)

### Plotting section
make_plots()