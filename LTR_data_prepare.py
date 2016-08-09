import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import re

textdir = './data_collecting/live_text/'
newsdir = './data_collecting/news_csv/'
news_files = os.listdir(newsdir)


def merge_dataset(files):
    test = DataFrame(columns = ['file'])
    for csvfile in files:
        if csvfile != '.DS_Store':
            dataframe = pd.read_csv(textdir + csvfile)
            test = pd.concat([test, dataframe], ignore_index = True)
            test = test.fillna(np.float(re.match(r'(.+)\.csv', csvfile).group(1)))
    test = test[
        ['file', 'position', 'number_of_stopwords', 'length', 'highlight_marker', 'sum_of_word_weigth', 'scoreline_s1',
         'scoreline_s2', 'scoreline_s3', 'specific_timestamp', 'similarity1', 'similarity2', 'f_score']]
    return test
