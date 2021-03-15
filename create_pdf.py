from datetime import datetime
from fpdf import FPDF

WIDTH = 210
HEIGHT = 297

def create_title(day, company, pdf):
    pdf.set_font('Arial', 'B', 24)
    pdf.ln(50)
    pdf.write(5, f'{company} Twitter Tweet Analysis')
    pdf.ln(10)
    pdf.set_font('Arial', '', 16)
    pdf.write(5, f'{day}')
    pdf.ln(5)
    

def create_pdf(day, company, filename='twitter_analysis.pdf'):
    pdf = FPDF()
    ''' PAGE 1 '''
    pdf.add_page()
    pdf.image('images/header_image.png', 0, 0, WIDTH)
    create_title(day, company, pdf)
    
    # Tweets Over Time
    pdf.image('images/total_tweets.png', 10, 90, WIDTH-25)
    # Engagements vs tweets
    pdf.image('images/engagements_vs_tweets.png', 10, 190, WIDTH-20)
    
    ''' PAGE 2 '''
    pdf.add_page()
    # Tweets by weekday
    pdf.image('images/tweets_month_per_day_avg.png', 10, 35, WIDTH-30)
    
    # Mentions Weekdays
    pdf.image('images/tweets_year_per_day_avg.png', 10, 155, WIDTH-30)
    
    ''' PAGE 3 '''
    pdf.add_page()
    # Mentions Wordcloud
    pdf.set_font('Arial', 'B', 18)
    pdf.write(25, f'Twitter Mentions Wordcloud')
    pdf.image('images/mentions_wordcloud.png', 15, 35, WIDTH-30)
    pdf.ln(1)
    
    # Hashtags Wordcloud
    pdf.set_font('Arial', 'B', 18)
    pdf.write(260, f'Twitter Hashtags Wordcloud')
    pdf.image('images/hashtags_wordcloud.png', 15, 155, WIDTH-30)
    
    ''' PAGE 4 '''
    pdf.add_page()
    # Mentions Tweets Over Time
    pdf.image('images/top10_mentions.png', 10, 35, WIDTH-30)
    
    # Mentions Weekdays
    pdf.image('images/mentions_over_time.png', 10, 165, WIDTH-30)
    
    ''' PAGE 5 '''
    pdf.add_page()
    # Mentions Tweets Over Time
    pdf.image('images/tweets_over_time.png', 10, 35, WIDTH-30)
    
    # Mentions Weekdays
    pdf.image('images/tweets_weekday_all.png', 10, 155, WIDTH-30)
    
    ''' PAGE 6 '''
    pdf.add_page()
    # Mentions Tweets Over Time
    pdf.image('images/engagements_vs_mentions.png', 15, 35, WIDTH-30)
    
    # pdf.ln(155)
    # pdf.set_font('Arial', '', 12)
    # conclusion = f'That concludes the twitter tweet analysis. Please see my website for the most up-to-date version of this Twitter Analytics PDF. It is constantly being updated and can be followed on my website at jamesejordan.com'
    # pdf.multi_cell(WIDTH-15, 5, conclusion, align='L')
    
    
    pdf.output('twitter_analysis.pdf', 'F')
    
    return

now = datetime.today().strftime('%m/%d/%Y')
current_company = 'Veeva'

create_pdf(now, current_company)