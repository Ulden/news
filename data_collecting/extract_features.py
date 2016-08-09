import numpy as np
import pandas as pd
import os
import re


def length_stopword_highlight(dataframe):
    stop_word = pd.read_csv('stopword.csv')
    highlight_marker_list = pd.read_csv('highlight_marker.csv')
    adjusted_length_vector = []
    stop_word_num_vector = []
    highlight_marker_vector = []
    for item in dataframe['word_sequence']:
        stop_word_num = 0
        adjusted_length = 0
        highlight_marker = 0
        sentence_splited = re.split(r'[//]+', item)
        for word in sentence_splited:
            if word in list(stop_word['stop_word']):
                stop_word_num += 1
            else:
                adjusted_length += 1
            if word in list(highlight_marker_list['highlight_marker']):
                highlight_marker += 1
        stop_word_num_vector.append(stop_word_num)
        adjusted_length_vector.append(adjusted_length)
        highlight_marker_vector.append(highlight_marker)
    return stop_word_num_vector, adjusted_length_vector, highlight_marker_vector


def scoreline_features(dataframe):
    last_flag_change = -1000
    scoreline_s1 = []
    scoreline_s2 = []
    scoreline_s3 = []
    for index in range(len(dataframe)):
        s1 = s2 = s3 = False
        if index != 0:
            if dataframe['score1'][index] != dataframe['score1'][index - 1] or dataframe['score2'][index] != \
                    dataframe['score2'][index - 1]:
                s1 = True
                last_flag_change = index
            elif index - last_flag_change < 6:
                s2 = True
                if dataframe['score1'][last_flag_change] == dataframe['score2'][last_flag_change]:
                    s3 = True
        scoreline_s1.append(s1)
        scoreline_s2.append(s2)
        scoreline_s3.append(s3)
    return scoreline_s1, scoreline_s2, scoreline_s3


def cosin_similarity(vector1, vector2):
    Lx = np.sqrt(vector1.dot(vector1))
    Ly = np.sqrt(vector2.dot(vector2))
    cos = vector1.dot(vector2) / (Lx * Ly)
    return cos


def avg(num1, num2):
    return (num1 + num2) / 2


def sentence_similarity(dataframe):
    # cosine similarity
    similarity1 = []
    similarity2 = []
    vectors = [np.array(re.split(r'[#]+', item)).astype(np.float) for item in dataframe['tfidf_vector']]
    for index in range(len(vectors)):
        if index != 0 and index != 1 and index != len(vectors) - 1 and index != len(vectors) - 2:
            similarity1.append(avg(cosin_similarity(vectors[index], vectors[index + 1]),
                                   cosin_similarity(vectors[index], vectors[index - 1])))
            similarity2.append(avg(cosin_similarity(vectors[index], vectors[index + 2]),
                                   cosin_similarity(vectors[index], vectors[index - 2])))
        elif index == 0:
            similarity1.append(cosin_similarity(vectors[index], vectors[index + 1]))
            similarity2.append(cosin_similarity(vectors[index], vectors[index + 2]))
        elif index == 1:
            similarity1.append(avg(cosin_similarity(vectors[index], vectors[index + 1]),
                                   cosin_similarity(vectors[index], vectors[index - 1])))
            similarity2.append(cosin_similarity(vectors[index], vectors[index + 2]))
        elif index == len(vectors) - 1:
            similarity1.append(cosin_similarity(vectors[index], vectors[index - 1]))
            similarity2.append(cosin_similarity(vectors[index], vectors[index - 2]))
        elif index == len(vectors) - 2:
            similarity1.append(avg(cosin_similarity(vectors[index], vectors[index + 1]),
                                   cosin_similarity(vectors[index], vectors[index - 1])))
            similarity2.append(cosin_similarity(vectors[index], vectors[index - 2]))
    return similarity1, similarity2


def live_text_feature_vector(dataframe):
    # basic features
    dataframe['position'] = 1 - (dataframe.index - 1) / len(dataframe)
    dataframe['number_of_stopwords'], dataframe['length'], dataframe['highlight_marker'] = length_stopword_highlight(
        dataframe)
    dataframe['sum_of_word_weigth'] = [sum(map(float, re.split(r'[#]+', item))) for item in dataframe['tfidf_vector']]
    dataframe['similarity1'], dataframe['similarity2'] = sentence_similarity(dataframe)
    # specification features
    dataframe['scoreline_s1'], dataframe['scoreline_s2'], dataframe['scoreline_s3'] = scoreline_features(dataframe)
    dataframe['specific_timestamp'] = dataframe['time'] / dataframe['time'].max()
    # dataframe['player_popularity'] =
    return dataframe


textdir = './live_text/'
live_text_files = os.listdir(textdir)
for csvfile in live_text_files:
    if csvfile != '.DS_Store':
        live_text = pd.read_csv(textdir + csvfile)
        live_text_feature_vector(live_text).to_csv(textdir + csvfile, index = False)
