import csv
import jieba
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import os
import re

textdir = './live_text/'
live_text_files = os.listdir(textdir)

for csvfile in live_text_files:
    if csvfile != '.DS_Store':
        if re.match(r'(.+)\.csv',csvfile).group(1) in map(str,np.arange(139180,139231)):
            #print(csvfile+'in block1')
            live_text = pd.read_csv(textdir+csvfile)
            live_text['time'] = (live_text['time']/60).astype(np.int64) + 1
            tmp = live_text[live_text['period']==1]['time'].max()
            index = live_text['period'] == 2 
            live_text.loc[index,'time'] = tmp + live_text.loc[index,'time']
            index = live_text['period'] == 5
            live_text.loc[index,'time'] = tmp + live_text.loc[index,'time']
        elif re.match(r'(.+)\.csv',csvfile).group(1) in map(str,np.arange(1596898,1596951)):
            #print(csvfile+'in block2')
            live_text = pd.read_csv(textdir+csvfile)
            period_map = {'上半场':1,'未开始':0,'下半场':2,'半场':6,'完赛':5,'加时赛':7}
            live_text['period'] = live_text['period'].map(period_map)
            tmp = []
            for item in live_text['time']:
                if type(item) is not np.int64:
                    if re.match(r'\d+\+\d+',item):
                        item = int(re.match(r'(\d+)\+(\d+)',item).group(1)) + int(re.match(r'(\d+)\+(\d+)',item).group(2))
                    elif re.match(r'\d+\’',str(item)):
                        item = int(re.match(r'(\d+)\’',item).group(1))
                    elif item in ['未赛','完赛','中场','半场','常规时间结束','休息','点球大战',' 点球大战','加时赛']:
                        item = 0
                tmp.append(item)
            live_text['time'] = tmp
            live_text['time'] = live_text['time'].astype(np.int64)
            index = live_text['period'] == 5
            live_text.loc[index,'time'] = live_text[live_text['period']==2]['time'].max()
            index = live_text['period'] == 6
            live_text.loc[index,'time'] = live_text[live_text['period']==1]['time'].max()
        
        live_text.to_csv(textdir+csvfile,index=False)
        
