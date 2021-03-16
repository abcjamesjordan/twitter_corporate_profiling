# Twitter Analysis

This project is aimed developing code to scrape, analyize, and visualize tweets for corporate accounts(companies).

## About

I am currently applying to various companies. For each job I much do various research to see if I think it might be a good fit for me. In the process of this research typically
I will checkout their social media platforms. Thus, I have automated this process to scrape, compile, and analyize twitter tweets from each company I am seriously considering.

## Requirements

Python Libraries

1. Twint (python library) to scrape twitter tweets (currently only able to install using the dockerfile)
2. Pandas to organize the tweets
3. Matplotlib / Seaborn to visualize the tweets
4. Scikit-learn to organize parts of the data
5. Wordcloud: to create visual wordclouds of the mentions and hashtags
6. FPDF: to compile it all into a easy to consume PDF Analysis Report

## Methods Used

1. Scrape tweets using twint
2. Organize data using various pandas and scikit-learn techniques
3. Plot data points of interest to see some interesting trends in tweets
4. Compare tweets with top 3 mentions within all of the companies tweets
5. Export charts to a PDF created within python
6. More...
