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
    

def create_pdf(day, company, filename='tuto1.pdf'):
    pdf = FPDF()
    ''' PAGE 1 '''
    pdf.add_page()
    pdf.image('header_image.png', 0, 0, WIDTH)
    create_title(day, company, pdf)
    
    # Tweets Over Time
    pdf.image('total_tweets.png', 10, 90, WIDTH-25)
    # Engagements vs tweets
    pdf.image('engagements_vs_tweets.png', 10, 190, WIDTH-20)
    
    ''' PAGE 2 '''
    pdf.add_page()
    # Mentions Wordcloud
    pdf.set_font('Arial', 'B', 18)
    pdf.write(25, f'Twitter Mentions')
    pdf.image('mentions_wordcloud.png', 15, 35, WIDTH-30)
    pdf.ln(1)
    
    # Hashtags Wordcloud
    pdf.set_font('Arial', 'B', 18)
    pdf.write(260, f'Twitter Hashtags')
    pdf.image('hashtags_wordcloud.png', 15, 155, WIDTH-30)
    
    ''' PAGE 3 '''
    pdf.add_page()
    # Tweets by weekday
    pdf.image('tweets_month_per_day_avg.png', 10, 35, WIDTH-30)
    
    # Mentions Weekdays
    pdf.image('tweets_year_per_day_avg.png', 10, 155, WIDTH-30)
    
    ''' PAGE 4 '''
    pdf.add_page()
    # Mentions Tweets Over Time
    pdf.image('tweets_over_time.png', 10, 35, WIDTH-30)
    
    # Mentions Weekdays
    pdf.image('tweets_weekday_all.png', 10, 155, WIDTH-30)

    pdf.output('tuto1.pdf', 'F')
    
    return

now = datetime.today().strftime('%m/%d/%Y')
current_company = 'Veeva'

create_pdf(now, current_company)