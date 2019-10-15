#! /usr/bin/env python3
# -*- coding:utf-8 -*-
'''
模块功能描述：

@author Liu Mingxing
@date 2019/10/14
'''

import pandas as pd
df = pd.read_csv('example/data/train.csv')
ps_ind_06_bin_val = df['ps_ind_06_bin'].unique()
print(ps_ind_06_bin_val)