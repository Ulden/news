import csv
import jieba
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import os
import re

textdir = './live_text/'
news_csvdir='./news_csv/'
newsdir = './standard_news/'
live_text_files = os.listdir(textdir)
news_files = os.listdir(newsdir)
jieba.load_userdict('user_dict.txt')

for csvfile in live_text_files:
    if csvfile != '.DS_Store':
        live_text = pd.read_csv(textdir+csvfile)
        tmp = []
        live_text = live_text.dropna()
        for sentence in live_text.message:
            words = jieba.cut(sentence)
            tmp.append('/'.join(words))
        #print(csvfile)
        
        live_text['word_sequence'] = tmp
        live_text.to_csv(textdir+csvfile,index=False)

for newsfile in news_files:
    if newsfile != '.DS_Store':
        with open(newsdir+newsfile, 'r') as f:
            content = f.read()
            sentence_list=re.split(r'[\s\n\!\。\！\.]+',content)
        newslist = []
        for sentence in sentence_list:
            words = jieba.cut(sentence)
            newslist.append({'news':sentence,'word_sequence':'/'.join(words)})
        
        df = DataFrame(newslist)
        df.to_csv(news_csvdir+re.match(r'(.+)\.md',newsfile).group(1)+'.csv',index=False)



            