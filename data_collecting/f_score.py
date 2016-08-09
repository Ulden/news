import numpy as np
import pandas as pd
import os
import re
from nltk.translate.bleu_score import modified_precision

from pyrouge import Rouge155
from pprint import pprint

from PythonROUGE.PythonROUGE import PythonROUGE

textdir = './live_text/'
newsdir = './news_csv/'
tmpdir = './tmp_dir/'
stop_word = pd.read_csv('stopword.csv')


def use_rouge(filename):
    textframe = pd.read_csv(textdir + filename)
    newsframe = pd.read_csv(newsdir + filename)
    i = 0
    for item in newsframe['word_sequence']:
        if type(item) is not np.float:
            sentence = re.split(r'[//]', item)
            with open(tmpdir+'ref/'+str(filename)+'_'+str(i), mode = 'w') as f:
                f.write(' '.join(sentence))
            i += 1
    i = 0
    for item in textframe['word_sequence']:
        sentence = re.split(r'[//]', item)
        with open(tmpdir+'gus/'+str(filename)+'_'+str(i), mode = 'w') as f:
                f.write(' '.join(sentence))
        i += 1
    reflist = [[os.path.join(tmpdir+'ref/', f) for f in os.listdir(tmpdir+'ref/')]]
    guslist = [os.path.join(tmpdir+'gus/', f) for f in os.listdir(tmpdir+'gus/')]

    guess_summary_list = ['tmp_dir/gus/139180.csv_0']
    ref_summ_list = [['tmp_dir/ref/139180.csv_0', 'tmp_dir/ref/139180.csv_1','tmp_dir/ref/139180.csv_3', 'tmp_dir/ref/139180.csv_4']]
    recall_list, precision_list, F_measure_list = PythonROUGE(guess_summ_list = guess_summary_list, ref_summ_list = ref_summ_list)

    print('recall = ' + str(recall_list))
    print('precision = ' + str(precision_list))
    print('F = ' + str(F_measure_list))

    [os.remove(os.path.join(tmpdir+'ref/',f)) for f in os.listdir(tmpdir+'ref/')]
    [os.remove(os.path.join(tmpdir+'gus/',f)) for f in os.listdir(tmpdir+'gus/')]


def mark_text(filename):
    print(filename)
    textframe = pd.read_csv(textdir + filename)
    newsframe = pd.read_csv(newsdir + filename)
    newstext = []
    mark = []
    for item in newsframe['word_sequence']:
        if type(item) is not np.float:
            sentence = re.split(r'[//]', item)
            newstext.append(sentence)
    for item in textframe['word_sequence']:
        sentence = re.split(r'[//]', item)
        try:
            mark.append(float(modified_precision(references = newstext, hypothesis = sentence, n = 2)))
        except ZeroDivisionError:
            mark.append(0)
    textframe['f_score'] = mark
    # print(textframe[textframe['f_score'] != 0]['f_score'])
    textframe.to_csv(textdir + filename, index = False)


live_text_files = os.listdir(textdir)
for textfile in live_text_files:
    if textfile != '.DS_Store':
        mark_text(textfile)

'''ref_texts = {'A': "Poor nations pressurise developed countries into granting trade subsidies.",
             'B': "Developed countries should be pressurized. Business exemptions to poor nations.",
             'C': "World's poor decide to urge developed nations for business concessions."}
summary_text = "Poor nations demand trade subsidies from developed nations."


rouge = Rouge155(n_words=100)
score = rouge.score_summary(summary_text, ref_texts)
pprint(score)'''
