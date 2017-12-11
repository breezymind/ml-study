# -*- coding: utf-8 -*-

from konlpy.utils import pprint 
from konlpy.tag import Mecab

mecab = Mecab()

corpus_file = '../crawler/data/daily_body.txt'

import time

def flat(content):
    return ["{}/{}".format(word, tag) for word, tag in mecab.pos(content)]

import pickle
news_fp = open('./data/daily_news_test','wb')
temp_fp = open(corpus_file,'r')

tagging_filter = []
while True:
    line = temp_fp.readline()
    if not line: break

    tmp = []
    for w in flat(line): 
        if w.split('/')[1] in ['NNG','NNP','VA+ETM','VV+EP','NP','VV','VV+EC','MAG']:
            tmp.append(w)

    if len(tmp) > 0:
        tagging_filter.append(tmp)
            
pickle.dump(tagging_filter, news_fp)

temp_fp.close()
news_fp.close()

with open('./data/daily_news_test','rb') as tagging:
    loadpos = pickle.load(tagging)

print('done')