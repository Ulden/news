import csv
import jieba
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer

textdir = './live_text/'
news_csvdir='./news_csv/'
live_text_files = os.listdir(textdir)
news_files = os.listdir(news_csvdir)
jieba.load_userdict('user_dict.txt')

for csvfile in live_text_files:
    if csvfile != '.DS_Store':
        live_text = pd.read_csv(textdir+csvfile)
        #live_text = news_csv.dropna()
        vectorizer=TfidfVectorizer(min_df=1,tokenizer=jieba.cut)
        tfidf=vectorizer.fit_transform(live_text['message'])
        tfidf_matrix=tfidf.toarray()
        tfidf_vectors = []
        for item in tfidf_matrix:
            tfidf_vectors.append('#'.join(map(str,item)))
        live_text['tfidf_vector'] = tfidf_vectors
        live_text.to_csv(textdir+csvfile,index=False)

for csvfile in news_files:
    if csvfile != '.DS_Store':
        news_csv = pd.read_csv(news_csvdir+csvfile)
        news_csv = news_csv.dropna()
        vectorizer=TfidfVectorizer(min_df=1,tokenizer=jieba.cut)
        tfidf=vectorizer.fit_transform(news_csv['news'])
        tfidf_matrix=tfidf.toarray()
        tfidf_vectors = []
        for item in tfidf_matrix:
            tfidf_vectors.append('#'.join(map(str,item)))
        news_csv['tfidf_vector'] = tfidf_vectors
        news_csv.to_csv(news_csvdir+csvfile,index=False)