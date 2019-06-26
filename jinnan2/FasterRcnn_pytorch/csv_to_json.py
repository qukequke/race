import glob
import os
import pandas as pd
import json
import numpy as np

df = pd.read_csv('../submit/larger0.5.csv',header=None)
len_file = len(set(df[0]))
print(len_file)
# print(df)
fn_to_ana = {}
for i in (df.index):
    fn = df.loc[i,0]
    loc = df.loc[i,1].split(' ')
    loc = list(map(lambda x:int(x), loc))
    label = int(df.loc[i,2])
    confidence = np.float(df.loc[i,3])
    loc.append(label)
    loc.append(confidence)
    if fn not in fn_to_ana:
        fn_to_ana[fn] = [loc]
    else:
        fn_to_ana[fn].append(loc)
print(len(fn_to_ana))


final_dict = {}
fns = (glob.glob('../test_data/*.jpg'))
pics = []
for fn in fns:
    fn = os.path.basename(fn)
    ana = fn_to_ana.get(fn)
    rects = []
    pic_i = {}
    if ana == None:
        pass
    else:
        for i in range(len(ana)):
            # print(ana)
            bbox = {}
            bbox['xmin'] = ana[i][0]
            bbox['xmax'] = ana[i][2]
            bbox['ymin'] = ana[i][1]
            bbox['ymax'] = ana[i][3]
            bbox['label'] = ana[i][4]+1
            bbox['confidence'] = ana[i][5]
            rects.append(bbox.copy())
    pic_i['filename'] = fn
    pic_i['rects'] = rects
    pics.append(pic_i)
print(len(pics))
final_dict['results']=pics
with open('../submit.json', 'w') as f:
    json.dump(final_dict, f)

