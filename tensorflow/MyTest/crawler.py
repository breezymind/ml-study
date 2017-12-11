#!/usr/bin/env python
# -*- coding: utf-8 -*-

# '17 0310, 수지구 아파트 매매가 (평수, 가격)

import urllib.request
import simplejson as json
import pprint
import pickle
import math

page_start = 1
# page_end = 276
page_end = 2

estate_list = []

target = 'https://m.land.naver.com/cluster/ajax/articleList?itemId=&mapKey=&lgeo=&showR0=&rletTpCd=APT&tradTpCd=A1&z=7&lat=37.322167&lon=127.097962&btm=37.2960664&lft=126.8677524&top=37.3478202&rgt=127.3283336&totCnt=5505&cortarNo=4146500000&sort=rank&page='

print('''
crawling start!!
''')
while page_start <= page_end:
    
    target_url = target+str(page_start)

    print("\n %s \n" % target_url)

    # request 
    fp = urllib.request.urlopen(target_url)
    rawdata = fp.read()
    fp.close()

    # print("\n %s \n" % str(rawdata)[2:-1])

    parsed = json.loads(rawdata)

    for item in parsed['body']:
        estate_list.append({
            'name' : item['atclNm'], 
            'price': item['prc'], 
            'space': math.floor(item['spc1']*0.3025),
            })

    page_start+=1

estate_file = open('./data/estate_price.dat', 'wb')
pickle.dump(estate_list, estate_file)
estate_file.close()

print('''
crawling done!!
''')