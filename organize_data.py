import pandas as pd
import numpy as numpy

from sklearn.preprocessing import MultiLabelBinarizer

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
        per_day = pd.to_datetime(df['date']).dt.date.value_counts().sort_index().reset_index()
        per_day.columns = ['DATE','count']
        return per_day
    elif unit_of_time == 'hour':
        per_hour = pd.to_datetime(df['date']).dt.hour.value_counts().sort_index().reset_index()
        per_hour.columns = ['HOUR', 'count']
        return per_hour
    elif unit_of_time == 'dayofweek':
        per_week_day = pd.to_datetime(df['date']).dt.day_name().value_counts().sort_index().reindex(DAYS_OF_WEEK).reset_index()
        per_week_day.columns = ['WEEKDAY', 'count']
        per_week_day.fillna({'count':0}, inplace=True)
        per_week_day['count'] = per_week_day['count'].astype('int')
        return per_week_day
    elif unit_of_time == 'week':
        per_week = pd.to_datetime(df['date']).dt.week.value_counts().sort_index().reset_index()
        per_week.columns = ['WEEK', 'count']
        return per_week
    elif unit_of_time == 'month':
        per_month = pd.to_datetime(df['date']).dt.month_name().value_counts().sort_index().reindex(MONTHS_OF_YEAR).reset_index()
        per_month.columns = ['MONTH', 'count']
        per_month.fillna({'count':0}, inplace=True)
        per_month['count'] = per_month['count'].astype('int')
        return per_month
    elif unit_of_time == 'year':
        per_year = pd.to_datetime(df['date']).dt.year.value_counts().sort_index().reset_index()
        per_year.columns = ['YEAR', 'count']
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
    

# Import that dataframe from the get_day.py output
df = pd.read_pickle('tweets_df.pkl')

# Constants
DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS_OF_YEAR = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

NUM_OF_TWEETS = len(df)
TOTAL_LIKES = df['nlikes'].sum()
TOTAL_REPLIES = df['nreplies'].sum()
TOTAL_RETWEETS = df['nretweets'].sum()

# Regex variables
twitter_handle_regex = r'@([a-zA-Z0-9_]+)'

# Fix the date column to US/Eastern Time Zone
df['date'] = pd.to_datetime(df['date'], utc=True)
df['date'] = pd.to_datetime(df['date']).dt.tz_convert('US/Eastern')

# Create a mentions column similar to the hashtags column
df['mentions'] = df['tweet'].str.findall(twitter_handle_regex).apply(','.join)
df['mentions'] = df['mentions'].str.split(',')

# Create the summing variables for mentions and hashtags
sum_mentions = create_sums_variable(df, mentions=True)
sum_hashtags = create_sums_variable(df, hashtags=True)

# Create various dataframes for plotting analysis
tweets_per_day = get_tweets_by(df, 'day')
tweets_per_hour = get_tweets_by(df, 'hour')
tweets_per_week_day = get_tweets_by(df, 'dayofweek')
tweets_per_week = get_tweets_by(df, 'week')
tweets_per_month = get_tweets_by(df, 'month')
tweets_per_year = get_tweets_by(df, 'year')

# Database statistics
AVG_TWEETS_PER_DAY = tweets_per_day['count'].mean()
AVG_TWEETS_PER_HOUR = tweets_per_hour['count'].mean()
AVG_TWEETS_PER_WEEK = tweets_per_week['count'].mean()
AVG_TWEETS_PER_MONTH = tweets_per_month['count'].mean()
AVG_TWEETS_PER_YEAR = tweets_per_year['count'].mean()

# print('Number of tweets', NUM_OF_TWEETS)
# print(AVG_TWEETS_PER_DAY, AVG_TWEETS_PER_HOUR, AVG_TWEETS_PER_WEEK, AVG_TWEETS_PER_MONTH, AVG_TWEETS_PER_YEAR)

# print(df.head(10))
# print(df.columns)

# test_df = df.copy()
# test_df['week/year'] = test_df['date'].apply(lambda x: "%d/%d" % (x.week, x.year))
# print(test_df.groupby('week/year').size().mean())

# avg_day = tweets_per_day

print(tweets_per_day['count'].mean(axis=0))

week_avg = 0
print(df['date'].dt.week)

test_df = df.copy()
test_df['count'] = 1
test_df['month'] = df['date'].dt.month
print(test_df.groupby(['month']).mean())
print(test_df.groupby(['month']).sum().iloc[:, 5])


