
# Scrape news from https://www.news.com.au/world/breaking-news


## Load python libraries

import requests
from bs4 import BeautifulSoup
import pandas as pd
import random


## Create a data frame

data = pd.DataFrame(columns = 
        ['title', 'desc', 'topics', 'content', 'news_agency', 'location', 'date_published'])


## Define parameters 

article_links = []

base_url = 'https://www.news.com.au/world/breaking-news'

headers_list = [
    {
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
    },
    {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:76.0) Gecko/20100101 Firefox/76.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
]

## Scrape and obtain the article links

headers = random.choice(headers_list)
cur_url = base_url

r = requests.get(cur_url, headers = headers)
soup = BeautifulSoup(r.content)
    
links = soup.find('div', attrs = {'class':"module-content"}).find_all('a')
    
# for some reason same link was coming twice
links = set(links)
    
for link in links:
    href = link.get('href')
    if 'page' in href:
        continue
    else:
        article_links.append(href)
        

## Create empty lists for attributes of interest

title = []
desc = []
topics = []
content = []
news_agency = []
location = []
date_published = []


## Extract attributes from the scraped link

for link in article_links:

    #headers = random.choice(headers_list)

    # Without using headers, all of the content is not fetched but only a part of it
    response = requests.get(link)#, headers=headers)
    soup = BeautifulSoup(response.content)
    #print(link)

    # scrape news title
    title.append(soup.title.text)
    #print(link)
    
    # scrape news description
    descText1 = soup.find('p', attrs={'class':"intro"}).text
    descText2 = soup.find('p', attrs={'class':"description"}).text
    descText = str(descText1) + ' ' + str(descText2)
    desc.append(descText)
    
    # scrape topics
    tag1 = []
    tag2 = []
    
    ul = soup.find_all('ul', attrs={'id':'breadcrumbs'})
    for x in ul:
        tag1 = [tag for tag in x.find_all('a')]
        tag1 = [tag.text for tag in tag1]
    
    div = soup.find_all('div', attrs={'class':'topic-list'})
    for x in div:
        tag2 = [tag for tag in x.find_all('a')]
        tag2 = [tag.text for tag in tag2]
    
    topicTags = str(tag1) + ' ' + str(tag2)
    topics.append(topicTags)
    
    # news agency
    news_agency.append(soup.find('span', attrs={'class':"source"}).text)
    
    # date published
    date_published.append(soup.find('span', attrs={'class':"date"}).text)

    # content
    text = ""
    p = soup.find('div', attrs={'class':'story-content'})
    for x in p.find_all('p'):
        text += str(x.text) + " "
    content.append(text)


## Update data frame with scraped and processed data

data['title'] = title
data['desc'] = desc
data['topics'] = topics
data['content'] = content
data['news_agency'] = news_agency
#data['location'] = location
data['date_published'] = date_published


## Save the data frame as a csv file

data.to_csv('news.com.au.csv')
