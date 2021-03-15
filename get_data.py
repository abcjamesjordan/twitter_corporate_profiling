import twint
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

def tweets_scrape(username, pandas_bool=True, hide_output_bool=True, retweets_bool=True, filter_retweets_bool=False):
    c = twint.Config()
    c.Username = username
    c.Pandas = pandas_bool
    c.Hide_output = hide_output_bool
    c.Retweets = retweets_bool
    c.Filter_retweets = filter_retweets_bool
    
    # Run the twint twitter scraper function
    twint.run.Search(c)

    # Store the results into a pandas dataframe
    tweets_df = twint.storage.panda.Tweets_df

    # Remove unncessary data from the dataframe
    tweets_df.drop(columns=['id', 'conversation_id', 'created_at', 'timezone', 'place', 'language', 'cashtags', 'user_id', 'user_id_str', 'username', 'name', 'link', 'thumbnail', 'retweet', 'quote_url', 'search', 'near',
                            'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'retweet_date', 'translate', 'trans_src', 'trans_dest'], inplace=True)
    
    return tweets_df

def profile_lookup(username, pandas_bool=True, hide_output_bool=True):
    c = twint.Config()
    c.Username = username
    c.Pandas = pandas_bool
    c.Hide_output = hide_output_bool

    twint.run.Lookup(c)
    
    user_df = twint.storage.panda.User_df
    user_df.to_pickle('./user_df.pkl')
    
    return user_df

# Get the original company twitter data
df_tweets = tweets_scrape('veevasystems')
df_tweets.to_pickle('./tweets_df.pkl')
df_user = profile_lookup('veevasystems')


# 1. Find top 3 mentions within those tweets
twitter_handle_regex = r'@([a-zA-Z0-9_]+)'
df_tweets['mentions'] = df_tweets['tweet'].str.findall(twitter_handle_regex).apply(','.join)
df_tweets['mentions'] = df_tweets['mentions'].str.split(',')

s = df_tweets['mentions']
mlb = MultiLabelBinarizer()
df_mentions = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_)
sum_mentions = df_mentions.sum(axis=0, skipna=True).sort_values(ascending=False)
sum_mentions.pop(str(df_user['username'].iloc[0]))
sum_mentions.pop('')

# 1a. Export the top mentions
top_3_mentions = sum_mentions[0:3].index.tolist()
print(top_3_mentions)

# 2. Scrape same data from those three sources
tweets_mention_1 = tweets_scrape(top_3_mentions[0])
tweets_mention_2 = tweets_scrape(top_3_mentions[1])
tweets_mention_3 = tweets_scrape(top_3_mentions[2])

# 3. Return that data into new variables
tweets_mention_1.to_pickle('./data/tweets_mention1_df.pkl')
tweets_mention_2.to_pickle('./data/tweets_mention2_df.pkl')
tweets_mention_3.to_pickle('./data/tweets_mention3_df.pkl')

with open('data/top_3_mentions.txt', 'w') as f:
    for item in top_3_mentions:
        f.write("%s\n" % item)
        
print('All done!')