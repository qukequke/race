import pandas as pd
import numpy as np 
import os


df = pd.read_csv('submit.csv')
filenames = list(df['filename'])
index = list(df['index'])

list_ = list(map(lambda x,y:x+'\t'+y + '\n', filenames, index))
with open('submit11.txt', 'w') as f:
    f.writelines(list_)
