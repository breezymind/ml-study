#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pprint
import pickle

estate_file = open('./data/estate_price.dat','rb')
estate_price = pickle.load(estate_file)

print(len(estate_price))
# print(estate_price[0])

# for item in estate_price:
#     print(item)

