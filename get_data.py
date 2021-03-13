import twint
import pandas as pd

# Configure the twint scrapper
c = twint.Config()
c.Username = 'SPGlobalPlatts'
c.Limit = 2000
c.Pandas = True
c.Hide_output = True
c.Retweets = True
c.Filter_retweets = False

# Run the twint twitter scraper function
twint.run.Search(c)

# Store the results into a pandas dataframe
tweets_df = twint.storage.panda.Tweets_df

# Remove unncessary data from the dataframe
tweets_df.drop(columns=['id', 'conversation_id', 'created_at', 'timezone', 'place', 'language', 'cashtags', 'user_id', 'user_id_str', 'username', 'name', 'link', 'thumbnail', 'retweet', 'quote_url', 'search', 'near',
                        'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'retweet_date', 'translate', 'trans_src', 'trans_dest'], inplace=True)

# Export the pandas dataframe to a pickle file
tweets_df.to_pickle('./tweets_df.pkl')