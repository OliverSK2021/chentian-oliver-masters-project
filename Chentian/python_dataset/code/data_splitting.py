import pandas as pd
import json

import random

vlist=['command_injection','open_redirect','path_disclosure','remote_code_execution','sql','xsrf','xss']

random.seed(2022)

def train_test_split(df,test_size):
    length = df.shape[0]
    alist = range(int(length/2))
    num_sample = int(length/2*test_size)
    indexs = random.sample(alist, num_sample)

    test_index = []
    for i in indexs:
        test_index.append(i*2)
        test_index.append(i*2+1)
    test = df.iloc[test_index]
    train_index = []
    indexs = list(set(alist).difference(set(indexs)))
    for i in indexs:
        train_index.append(i*2)
        train_index.append(i*2+1)
    train = df.loc[train_index]

    assert train.shape[0]+test.shape[0] == df.shape[0]
    return train, test

def index_generater(uindexs):
    indexs = []
    for i in uindexs:
        indexs.append(2*i)
        indexs.append(2*i+1)
    return indexs

def index_diff(larger_indexs, smaller_indexs):
    return list(set(larger_indexs).difference(set(smaller_indexs)))

def datasplit(df, train_size, test_size):
    ulength = df.shape[0]/2
    ulist = range(int(ulength))
    utrain_number = int(ulength*train_size)
    utest_number = int(ulength*test_size)
    utrain_indexs = random.sample(ulist, utrain_number)
    ures_indexs = index_diff(ulist, utrain_indexs)
    utest_indexs = random.sample(ures_indexs, utest_number)
    uvalid_indexs = index_diff(ures_indexs,utest_indexs)
    train_indexs = index_generater(utrain_indexs)
    train=df.loc[train_indexs]

    test_indexs = index_generater(utest_indexs)
    test = df.loc[test_indexs]

    valid_indexs = index_generater(uvalid_indexs)
    valid = df.loc[valid_indexs]

    assert train.shape[0]+test.shape[0]+valid.shape[0] == df.shape[0]

    return train, test, valid


for vtype in vlist:
    df = pd.read_csv('../python_data/'+vtype+'_df.csv')
    train, test, valid = datasplit(df, train_size=0.7,test_size=0.15)
    train.to_csv('../python_data/split_data/'+vtype+'_train.csv')
    test.to_csv('../python_data/split_data/'+vtype+'_test.csv')
    valid.to_csv('../python_data/split_data/'+vtype+'_valid.csv')

for ds in ['train','test','valid']:
    locals()[ds+'_df'] = pd.DataFrame()
    for vtype in vlist:
        df = pd.read_csv('../python_data/split_data/'+vtype+'_'+ds+'.csv').drop(columns = ['Unnamed: 0'])
        df['vtype'] = [vtype]*df.shape[0]
        locals()[ds+'_df'] = locals()[ds+'_df'].append(df)
    locals()[ds+'_df'].to_csv('../python_data/split_data/'+ds+'.csv')