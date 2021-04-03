import numpy as np
import pandas as pd
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics 
from sklearn.metrics import make_scorer, accuracy_score, classification_report
from sklearn.metrics import f1_score

# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.corpus import stopwords

from textblob import TextBlob
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import spacy

# Import data
path = os.getcwd()
path_train = os.path.join(path, 'data', 'semeval-2017-train.csv')
path_test = os.path.join(path, 'data', 'tweets_df.pkl')

train = pd.read_csv(path_train, sep='\t')
train = train.sample(frac=1, random_state=42).reset_index(drop=True)

test = pd.read_pickle(path_test)
test = test['tweet']



# TextBlob Sentiment Analysis
test_tweets = train.iloc[:1000].copy()

test_tweets['polarity'] = [TextBlob(x).sentiment.polarity for x in test_tweets['text']]
test_tweets['subjectivity'] = [TextBlob(x).sentiment.subjectivity for x in test_tweets['text']]
test_tweets = test_tweets.drop(columns=['text'])

stratify = test_tweets['label']
df_train, df_val = train_test_split(test_tweets, test_size=0.1, random_state=42, stratify=stratify)

# Using Textblob sentiment(polarity, subjectivity)
X_train = df_train[['polarity', 'subjectivity']].to_numpy()
X_valid = df_val[['polarity', 'subjectivity']].to_numpy()

y_train = df_train['label'].to_numpy()
y_valid = df_val['label'].to_numpy()




# Expanding the tweet using count vectorizer
# stop = list(stopwords.words('english'))
# vectorizer = CountVectorizer(decode_error = 'replace',stop_words = stop)

# X_train = vectorizer.fit_transform(df_train['text'].values)
# X_valid = vectorizer.fit_transform(df_val['text'].values)

# y_train = df_train['label'].values
# y_valid = df_val['label'].values


# scikit-learn models
svc = SVC()

svc.fit(X_train, y_train)

svc_prediction = svc.predict(X_valid)
svc_accuracy = accuracy_score(y_valid, svc_prediction)
print("SVC Training accuracy Score  : ",svc.score(X_train,y_train))
print("SVC Validation accuracy Score: ",svc_accuracy )
print(classification_report(y_valid, svc_prediction))

# naiveByes_clf = MultinomialNB()

# naiveByes_clf.fit(X_train,y_train)

# NB_prediction = naiveByes_clf.predict(X_valid)
# NB_accuracy = accuracy_score(y_valid,NB_prediction)
# print("Bayes Training accuracy Score   : ",naiveByes_clf.score(X_train,y_train))
# print("Bayes Validation accuracy Score : ",NB_accuracy )
# print(classification_report(NB_prediction,y_valid))







# # Spacy processing for tweet analysis
# spacy.prefer_gpu()
# nlp = spacy.load("en_core_web_sm")

# # Modify token matching for hashtags
# re_token_match = spacy.tokenizer._get_regex_pattern(nlp.Defaults.token_match)
# re_token_match = f"({re_token_match}|#\\w+)"
# nlp.tokenizer.token_match = re.compile(re_token_match).match

# def spacy_process(s, nlp, features):
#     s = s.lower()
#     doc = nlp(s)
#     lemmas = []
#     for token in doc:
#         lemmas.append(token.lemma_)
        
#     features |= set(lemmas)
    
#     freq = {'#': 0, '@': 0, 'URL': 0}
#     for word in lemmas:
#         freq[str(word)] = 0
#     for token in doc:
#         if '#' in str(token): freq['#'] += 1 # count num of hashtags
#         if '@' in str(token): freq['@'] += 1 # count num of mentions
#         if 'http://' in str(token): freq['URL'] += 1 # count num of URLs
#         freq[str(token.lemma_)] += 1
        
#     return features, freq

# preprocess_df = train
# features = set({'#', '@', 'URL'})

# bow_array = []
# for i in range(len(preprocess_df)):
#     features, freq = spacy_process(preprocess_df.iloc[i]['text'], nlp, features)
#     bow_array.append(freq)
    
# bow = pd.DataFrame('0', columns=features, index=range(len(preprocess_df)))
# bow['id'] = preprocess_df.index
# bow.set_index('id', drop=True, inplace=True)

# for i in range(len(preprocess_df)):
#     freq = bow_array[i]
#     for f in freq:
#         bow.loc[i+1, f] = freq[f]
        
# preprocess_df = preprocess_df.join(bow, lsuffix='_data') 

# y = preprocess_df['label']
# df_train, df_val = train_test_split(preprocess_df, test_size=0.1, random_state=42, stratify=y)

# print(df_train.shape, df_val.shape)

# def cal_sum(df, describe_which):
#     ''' 
#     Check the balance between original, test, and val using this function.
#     Prints results
#     '''
#     pos_sum = np.sum(df['label']==1)
#     neu_sum = np.sum(df['label']==0)
#     neg_sum = np.sum(df['label']==-1)
#     tot_sum = pos_sum + neu_sum + neg_sum
    
#     print(describe_which, ' Pos: ', pos_sum / tot_sum)
#     print(describe_which, ' Neg: ', neu_sum / tot_sum)
#     print(describe_which, ' Neu: ', neg_sum / tot_sum)
#     return

# cal_sum(preprocess_df, 'Original')
# cal_sum(df_train, 'Train')
# cal_sum(df_val, 'Validation')

# X_train = df_train.drop(columns=['label', 'text']).to_numpy()
# X_valid = df_val.drop(columns=['label', 'text']).to_numpy()

# y_train = df_train['label'].to_numpy()
# y_valid = df_val['label'].to_numpy()

















'''
# TextBlob Sentiment Analysis
test_tweets = train

test_tweets['polarity'] = [TextBlob(x).sentiment.polarity for x in test_tweets['text']]
test_tweets['subjectivity'] = [TextBlob(x).sentiment.subjectivity for x in test_tweets['text']]
test_tweets['predict'] = [1 if (x > 0.1 and y > 0.6) else (-1 if (x < -0.05 and y > 0.6) else 0) for x, y in zip(test_tweets['polarity'], test_tweets['subjectivity'])]
test_tweets['correct'] = [1 if x==y else 0 for x, y in zip(test_tweets['label'], test_tweets['predict'])]

num_correct = test_tweets['correct'].value_counts()[1]

print(f'Predict accuracy: {num_correct / len(test_tweets) * 100}')
'''



'''
train['text'] = process_tweets(train['text'])
# test = process_tweets(test)

# Sentiment Analysis Using Vader
analyzer = SentimentIntensityAnalyzer()
vs_results = pd.DataFrame()

vs_results['vs'] = [analyzer.polarity_scores(x) for x in train['text']]

vs_results = vs_results.join(pd.json_normalize(vs_results.vs))

vs_results.drop(columns=['vs'], inplace=True)
vs_results['label'] = train['label']

len_results = len(vs_results)
neg_count = len(vs_results[vs_results['neg'] >= 0.1])
neu_count = len(vs_results[vs_results['neu'] > 0.8])
pos_count = len(vs_results[vs_results['pos'] >= 0.1])

neg_label = len(vs_results[vs_results['label'] == -1])
neu_label = len(vs_results[vs_results['label'] == 0])
pos_label = len(vs_results[vs_results['label'] == 1])

print('neg count', neg_label)
print('neu count', neu_label)
print('pos count', pos_label)

print(vs_results.head(30))
print(vs_results.sample(n=30))
vs_results['predict'] = [1 if (x > 0.1 and y < 0.1) else (-1 if (y >= 0.1 and x <= 0.1) else 0) for x, y in zip(vs_results['pos'], vs_results['neg'])]
vs_results['predict_compound'] = [1 if x > 0.3 else (-1 if x < -0.3 else 0) for x in vs_results['compound']]

vs_results['correct'] = [1 if x==y else 0 for x, y in zip(vs_results['label'], vs_results['predict'])]
vs_results['correct_compound'] = [1 if x==y else 0 for x, y in zip(vs_results['label'], vs_results['predict_compound'])]

num_correct = vs_results['correct'].value_counts()[1]
num_correct_compound = vs_results['correct_compound'].value_counts()[1]

print(f'Predict accuracy: {num_correct / len_results * 100}')
print(f'Predict accuracy compound: {num_correct_compound / len_results * 100}')



sns.countplot(data=vs_results, x='label', hue='correct')
plt.show()
sns.countplot(data=vs_results, x='label', hue='correct_compound')

plt.show()





sample_size = min(len(train_pos), len(train_neg))

raw = np.concatenate((train_pos['text'].values[:sample_size], 
                 train_neg['text'].values[:sample_size]), axis=0)
labels = [1]*sample_size + [0]*sample_size

'''






















# # Constants for Regex
# url_regex = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
# twitter_handle_regex = r'@([a-zA-Z0-9_]+)'
# hashtag_regex = r'#([^\s]+)'
# emoticon_regex = r':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:'
# contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'), (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
# multiexclamation_regex = r'(\!)\1+'
# multiquestion_regex = r'(\?)\1+'
# multistop_regex = r'(\.)\1+'







# # Clean tweets for urls, @, etc.
# def replace_contractions(text):
#     patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
#     for (pattern, repl) in patterns:
#         (text, count) = re.subn(pattern, repl, text)
#     return text

# def process_tweets(tweets):
#     tweets = tweets.str.replace(url_regex, 'url', regex=True)
#     tweets = tweets.str.replace(twitter_handle_regex, 'user', regex=True)
#     tweets = tweets.str.replace(hashtag_regex, '', regex=True)
#     tweets = pd.Series([replace_contractions(x) for x in tweets])
#     tweets = tweets.str.lower()
#     tweets = tweets.str.replace(emoticon_regex, '', regex=True)
#     tweets = tweets.str.replace('  ', ' ')
#     tweets = tweets.str.replace(multiexclamation_regex, ' multiExclamation ', regex=True)
#     tweets = tweets.str.replace(multiquestion_regex, ' multiQuestion ', regex=True)
#     tweets = tweets.str.replace(multistop_regex, ' multiStop ', regex=True)
#     # tweets = tweets.str.replace(extra_regex, '', regex=True)
    
#     return tweets
