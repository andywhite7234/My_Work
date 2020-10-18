# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:23:25 2020

@author: andy_white
"""

import requests
from bs4 import BeautifulSoup
import pprint
import pandas as pd
import numpy as np
from newsapi import NewsApiClient   #this pacakge only allows for 260 characters per news article - it can suffice but will try the bloomberg api news package
from GoogleNews import GoogleNews
#googlenews = GoogleNews()

#for news api:
api_key ='056d11e98daa4776a5a24cd42e773374'

# Init
newsapi = NewsApiClient(api_key=api_key)
EVERYTHING_URL = "https://newsapi.org/v2/everything"

#for NYT api
from topstories import TopStoriesAPI
api = TopStoriesAPI('wdodAoXkj2okStQX8peDHrrvfJ32gfk4')
#load required code for google news:
#googlenews = GoogleNews()
#googlenews = GoogleNews('en','d')
#googlenews.setlang('en')
#googlenews.setperiod('d')
#googlenews.search('APPL')


stories = api.get_stories("business") # list of story dicts
stories
url_nyt = []
author =[]
pubdate = []
title =[]
abstract = []
section=[]
for i in stories:
    print(i['abstract'])
    print(i['url'])
    url_nyt.append(i['url'])
    author.append(i['byline'])
    pubdate.append(i['published_date'])
    title.append(i['title'])
    abstract.append(i['abstract'])
    section.append(i['section'])
df = pd.DataFrame({'author':author,'pubdate':pubdate,'title':title,'abstract':abstract,'section':section})
#df['bus or opin'] = 'business'
#df.iloc[34:,4]='opinion'
#df
#now we need to convert each item in list to str in order to concatenate and then run the regex code
def remove_html(script):
    new_list = []
    seperator =' '
    #this convers each item in list to str in order to concatenate and then run the regex code
    #this is done because the initial script from BS4 is not a list, so it needs to be converted
    for i in script:
    #print(i)
        new_list.append(str(i))
    #The following concatenates the article and includes spaces between each list item. and then runs the regex code
    new_list = seperator.join(new_list)
    pattern='\<.*?\>'
    clean_text = re.sub(pattern, '', new_list)
    return(clean_text)

#### Now for the parsing:

import re
import urllib 
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup
import requests
import pprint
import pandas as pd

#lets see if we can loop through NYT URLs and grab text:

all_text = []
for link in url_nyt:
    uClient=uReq(link)                    #this takes each link and requests from the url associated
    page_html = uClient.read()              #reads in the the html code. 
    uClient.close()                         #make sure to close the connection
    page_soup = BeautifulSoup(page_html,"html.parser")   #parse the code
    nyt_script_body = page_soup.findAll("div",{"class":"css-1fanzo5 StoryBodyCompanionColumn"}) # intersting thing about NYT - this is the pattern that holds the text
    nyt_script_txt = remove_html(nyt_script_body) #see above for function descript, but this cleans
    nyt_script_txt = nyt_script_txt.lower()
    all_text.append(nyt_script_txt)       #now we have all text without html.

len(all_text)

import nltk
regex_pattern = r''' (?x) 		# set flag to allow verbose regexps
         \b[A-Z][a-z]+\.(?=\s)        
        | (?:[\£\$\€]{1}[,\d]+.?\d*)  # currency and percentages, $12.40
        | \$?\d+(?:\.\d+)?%?  # currency and percentages, $12.40, 50%
        | (?:[A-Z]\.)+    	# abbreviations, e.g. U.S.A.
        | \w+(?:-\w+)*  	# words with internal hyphens
        | \.\.\.        	# ellipsis
        | [][.,;"'?():-_%#']  # separate tokens
        '''
#testing on some numbers and Ms. ect. 
test_text = "mr. Fun their are 50% educate each student is, say, $38,000, your $45,000 can still help the students who can afford to pay only $25,000."
nltk.regexp_tokenize(test_text,regex_pattern)

#looks like it worked, now lets tokenize the corpora
tokenized_sents = [nltk.regexp_tokenize(i,regex_pattern) for i in all_text]

#lets also tokenize as one massive corpus
seperator = ' '
all_text2 = seperator.join(all_text)
len(all_text2)
#now tokenize
business_tok = nltk.regexp_tokenize(all_text2,regex_pattern)

#Begin with the stop words dictionary:
nltkstopwords = nltk.corpus.stopwords.words('english')
morestopwords = ['could','would','might','must','need','sha','wo','y',"'s","'d","'ll","'t","'m","'re","'ve",".",",","said","?",':',';']
stopwords= nltkstopwords + morestopwords

stopped_business_words = [w for w in business_tok if not w in stopwords]
from nltk import FreqDist
ndist = FreqDist(stopped_business_words)
bus_items = ndist.most_common(50)

bus_word=[]
bus_word_freq=[]
for item in bus_items:
    print(item)
    bus_word.append(item[0])
    bus_word_freq.append(item[1])
bus_word
bus_word_freq

import matplotlib.pyplot as plt
plt.title("Word Frequencies in Business")
plt.ylabel("Total Number of Occurrences")
plt.xlabel("Rank of Word")
plt.bar(bus_word,bus_word_freq)
plt.xticks(rotation=90)
plt.show()

#Now lets get after the bigrams
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(stopped_business_words)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:50]:
    print (bscore)


#better way is to apply a frequency filter
finder2 = BigramCollocationFinder.from_words(stopped_business_words)
scored2 = finder2.score_ngrams(bigram_measures.pmi)
finder2.apply_freq_filter(5)
scored2 = finder2.score_ngrams(bigram_measures.pmi)
for bscore in scored2[:50]:
    print (bscore)

#try some stemming:
porter = nltk.PorterStemmer()
stopped_business_words_stem = [porter.stem(t) for t in stopped_business_words]
print(stopped_business_words_stem[:200])

finder3 = BigramCollocationFinder.from_words(stopped_business_words_stem)
scored3 = finder3.score_ngrams(bigram_measures.pmi)
finder3.apply_freq_filter(5)
scored3 = finder3.score_ngrams(bigram_measures.pmi)
for bscore in scored3[:50]:
    print (bscore)


########################## Now to Tokenize the opinion and run same analysis: ##########################################
all_text_opin = []
#url_nyt = url_nyt[34:]  #need to feed only the opinion text into this
nyt_script_body=[]
nyt_script_txt = []
for link in url_nyt:
    uClient=uReq(link)                    #this takes each link and requests from the url associated
    page_html = uClient.read()              #reads in the the html code. 
    uClient.close()                         #make sure to close the connection
    page_soup = BeautifulSoup(page_html,"html.parser")   #parse the code
    nyt_script_body = page_soup.findAll("div",{"class":"css-1fanzo5 StoryBodyCompanionColumn"}) # intersting thing about NYT - this is the pattern that holds the text
    nyt_script_txt = remove_html(nyt_script_body) #see above for function descript, but this cleans
    nyt_script_txt = nyt_script_txt.lower()
    all_text_opin.append(nyt_script_txt)   

# we will have to update the regex pattern to remove the tail of the opinion section:
"the times is committed to publishing"

regex_pattern_opin = r''' (?x) 		# set flag to allow verbose regexps
         \b[A-Z][a-z]+\.(?=\s)   
        | (?:[\£\$\€]{1}[,\d]+.?\d*)  # currency and percentages, $12.40
        | \$?\d+(?:\.\d+)?%?  # currency and percentages, $12.40, 50%
        | (?:[A-Z]\.)+    	# abbreviations, e.g. U.S.A.
        | \w+(?:-\w+)*  	# words with internal hyphens
        | \.\.\.        	# ellipsis
        | [][.,;"'?():-_%#']  # separate tokens
        '''
test_text = 'of man, the kind that both dogs and humans — all of us — might look to with affection and respect. for over three years now many americans have been anxiously waiting for mr. trump to grow into the job, to show that he understands he is the leader of the whole country and not just his core supporters. for a while, we thought, national moments of mourning, from charlottesville to el paso, might engender a new trump, showing us a man governing — just once — from his heart, rather than his spleen. donald trump has failed to be that man. now, with tens of thousands of americans dead from the coronavirus and an economy in ruins, he’s the man who boasts that his tv ratings are as high as “the bachelor” finale’s. the fala speech, 76 years ago, generated humor and good will during a dark time. surely right about now we could use generous portions of both. oh, i know full well that the odds of the president’s becoming a different sort of man at this hour are slim. he’s an old dog. the era of new tricks is over. and yet, the seven dogs with whom i have shared my life — playboy, sausage, matt the mutt, brown, alex, lucy and ranger — have made me into an optimist. they have taught me to have hope. they have shown me what it means to be loved. it is impossible for me not to wish that the leader of the free world could feel this too.mr. president, i want to believe that somewhere deep inside you, there is a good boy, still waiting to be born.the times is committed to publishing a diversity of letters to the editor. we’d like to hear what you think about this or any of our articles. here are some tips. and here’s our email: letters@nytimes.com.follow the new york times opinion section on facebook, twitter (@nytopinion) and instagram.'

test_text = re.sub('\b*the times is committed to publishing\b*(.*)','',test_text)
test_text
#ok that worked - lets apply to the entire list
   #nltk.regexp_tokenize(test_text,regex_pattern_opin)
tokenized_sents = [nltk.regexp_tokenize(i,regex_pattern) for i in all_text]

all_text_opin = [re.sub('\b*the times is committed to publishing\b*(.*)','',i) for i in all_text_opin]

#lets also tokenize as one massive corpus
seperator = ' '
all_text_opin2 = seperator.join(all_text_opin)
len(all_text_opin2)
#now tokenize
opinion_tok = nltk.regexp_tokenize(all_text_opin2,regex_pattern_opin)

#Begin with the stop words dictionary:
nltkstopwords = nltk.corpus.stopwords.words('english')
morestopwords = ['could','would','might','must','need','sha','wo','y',"'s","'d","'ll",
                 "'t","'m","'re","'ve",".",",","said","?",':',';','(',')']
stopwords= nltkstopwords + morestopwords

stopped_opinion_words = [w for w in opinion_tok if not w in stopwords]
from nltk import FreqDist
ndist = FreqDist(stopped_opinion_words)
opin_items = ndist.most_common(50)

opin_word=[]
opin_word_freq=[]
for item in opin_items:
    print(item)
    opin_word.append(item[0])
    opin_word_freq.append(item[1])
opin_word
opin_word_freq

import matplotlib.pyplot as plt
plt.title("Word Frequencies in Opinion Columns")
plt.ylabel("Total Number of Occurrences")
plt.xlabel("Rank of Word")
plt.bar(opin_word,opin_word_freq)
plt.xticks(rotation=90)
plt.show()

#Now lets get after the bigrams
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(stopped_opinion_words)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:50]:
    print (bscore)


#better way is to apply a frequency filter
finder2 = BigramCollocationFinder.from_words(stopped_opinion_words)
scored2 = finder2.score_ngrams(bigram_measures.pmi)
finder2.apply_freq_filter(5)
scored2 = finder2.score_ngrams(bigram_measures.pmi)
for bscore in scored2[:50]:
    print (bscore)

#try some stemming:
porter = nltk.PorterStemmer()
stopped_opinion_words_stem = [porter.stem(t) for t in stopped_opinion_words]
print(stopped_opinion_words_stem[:200])

finder3 = BigramCollocationFinder.from_words(stopped_opinion_words_stem)
scored3 = finder3.score_ngrams(bigram_measures.pmi)
finder3.apply_freq_filter(5)
scored3 = finder3.score_ngrams(bigram_measures.pmi)
for bscore in scored3[:50]:
    print (bscore)




#the following is for keeping each article intact - may come back to this at a later date
filtered_words = []
for i in tokenized_sents:
    filtered_words.append([w for w in i if not w in stopwords])
    
filtered_words
from nltk import FreqDist
ndist = FreqDist(filtered_words[0])
nitems = ndist.most_common(5)
dicts ={}
for item in nitems:
    keys=item[1]
    values=[0] 
#    for i in keys:
#        dicts[i]=values[i]


nltk.word_tokenize(all_text[2])
#old/test code

my_url ='https://www.nytimes.com/2020/04/25/business/economy/coronavirus-economy-reopening.html'

#opening up connection to grab the page
uClient =uReq(my_url)
#offloads content into variable and then close
page_html =uClient.read()
uClient.close()

page_soup = BeautifulSoup(page_html,"html.parser") #website is an html 
page_soup.h1  #grabs header

#here we grab the body of the text - Css-1fanzo is similar for all instances in NYT. Although this isn't completely vetted
#nyt_script = page_soup.findAll("section",{"name":"articleBody"})   
nyt_script_body = page_soup.findAll("div",{"class":"css-1fanzo5 StoryBodyCompanionColumn"})

#now we need to convert each item in list to str in order to concatenate and then run the regex code
def remove_html(script):
    new_list = []
    seperator =' '
    #this convers each item in list to str in order to concatenate and then run the regex code
    #this is done because the initial script from BS4 is not a list, so it needs to be converted
    for i in script:
    #print(i)
        new_list.append(str(i))
    #The following concatenates the article and includes spaces between each list item. and then runs the regex code
    new_list = seperator.join(new_list)
    pattern='\<.*?\>'
    clean_text = re.sub(pattern, '', new_list)
    return(clean_text)

remove_html(nyt_script_body)
#new_list = seperator.join(new_list)




































































