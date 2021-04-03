import pandas as pd
import spacy
import os
import re

# Pickle Name Constant
PICKLE_NAME = 'clean_spacy_df.pkl'
PATH_TO_BE_PICKLED = 'semeval-2017-train.csv'
COL_OF_TEXT = 'text'

# Paths for pandas
path = os.getcwd()
path_df = os.path.join(path, 'data', PATH_TO_BE_PICKLED)
path_pickle_nlp = os.path.join(path, 'data', PICKLE_NAME)

# Import data
df = pd.read_csv(path_df, sep='\t')

# Spacy settings
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

# Regex for pattern matching
unicode_regex_1 = r'(\\u[0-9A-Fa-f]+)'
unicode_regex_2 = r'[^\x00-\x7f]'
url_regex = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'

# Add token for hashtags
re_token_match = spacy.tokenizer._get_regex_pattern(nlp.Defaults.token_match)
re_token_match = f"({re_token_match}|#\\w+)"
nlp.tokenizer.token_match = re.compile(re_token_match).match

def clean_text(df, column):
    # Unicode
    df[column] = df[column].str.replace(unicode_regex_1, r' ', regex=True)
    df[column] = df[column].str.replace(unicode_regex_2, r' ', regex=True)
    
    # Urls
    df[column] = df[column].str.replace(url_regex, 'url', regex=True)
    
    # Spacy
    df[column] = list(nlp.pipe(df[column].values))
    df[column] = [[token.lemma_ for token in x if not token.is_stop] for x in df[column]]
    df[column] = [' '.join(x) for x in df[column]]
    
    return df

# Apply spacy NLP
df_NLP = clean_text(df, COL_OF_TEXT)

# Pickle results
df_NLP.to_pickle(path_pickle_nlp)
