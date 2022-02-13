#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:44:08 2021

@author: melikedila
"""

import numpy as np
import pandas as pd


name = ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas']
score = [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19]
attempts = [1, 3, 2, 3, 2, 3, 1, 1, 2, 1]
qualify =  ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']

exam_data  = {'name':name, 'score':score, 'attempts':attempts, 'qualify':qualify}
labels = list('abcdefghij')

df = pd.DataFrame(exam_data , index=labels)

def print_cols(df):
    for col in df.columns:
        #print(col)
        print(df[col])
        print('_'*80)
        
print_cols(df)        
        
def has_nan(df, col_name):
    return np.any(df[col_name].isna())

def get_columns_with_missing_values(df):
    return [col for col in df.columns if np.any(df[col].isna())]       



np.any(df['score'].isna())
