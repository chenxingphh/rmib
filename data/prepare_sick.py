import re
import os
import json
from tqdm import tqdm
import pandas as pd

os.makedirs('sick', exist_ok=True)

label_map = {
    'ENTAILMENT': 0,
    'NEUTRAL': 1,
    'CONTRADICTION': 2,
}


def label_parse(label):
    return label_map[label]


df = pd.read_csv('orig/SICK/SICK.txt', sep='\t')
df_train = df[df['SemEval_set'] == 'TRAIN']
df_dev = df[df['SemEval_set'] == 'TRIAL']
df_test = df[df['SemEval_set'] == 'TEST']

print('shape of train:', df_train.shape)
print('shape of dev:', df_dev.shape)
print('shape of test:', df_test.shape)

df_train['label'] = df_train['entailment_label'].apply(label_parse)
df_dev['label'] = df_dev['entailment_label'].apply(label_parse)
df_test['label'] = df_test['entailment_label'].apply(label_parse)

df_train = df_train[['sentence_A', 'sentence_B', 'label']]
df_dev = df_dev[['sentence_A', 'sentence_B', 'label']]
df_test = df_test[['sentence_A', 'sentence_B', 'label']]

df_train.to_csv('sick/train.txt', index=False, header=None, sep='\t')
df_dev.to_csv('sick/dev.txt', index=False, header=None, sep='\t')
df_test.to_csv('sick/test.txt', index=False, header=None, sep='\t')
