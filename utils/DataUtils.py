#endcoding = 'utf-8'
from __future__ import print_function

from collections import *
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation #, Merge
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.models import model_from_json
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import jieba
import random


def loadData(corpus_path,corpus_train,corpus_dev,maxlen,separator="_label_",label_separator=" ",sentence_separator="_sentence_",label='single',muti_sentence=False):
    tokenizer = Tokenizer(filters=u'!"#$%&()*+,-./:;<=>?@[\\]【】。！，《》^_`，。{|}~\t\n',
                         lower=True,
                         split=" ",
                         char_level=False)
    if label=='single' and not muti_sentence:
        Y,X = zip(*[(i.split(separator)[0].strip(),i.split(separator)[1]) for i in open(corpus_path+corpus_train,encoding='utf-8').readlines()])
        _Y,_X = zip(*[(i.split(separator)[0].strip(),i.split(separator)[1]) for i in open(corpus_path+corpus_dev,encoding='utf-8').readlines()])
        
        lb = preprocessing.LabelBinarizer()
        lb.fit(Y)
        Y_DATA=lb.transform(Y+_Y)
    elif label=='multiple' and not muti_sentence:
        Y,X = zip(*[(i.split(separator)[0].strip().split(label_separator),i.split(separator)[1]) for i in open(corpus_path+corpus_train,encoding='utf-8').readlines()])
        _Y,_X = zip(*[(i.split(separator)[0].strip().split(label_separator),i.split(separator)[1]) for i in open(corpus_path+corpus_dev,encoding='utf-8').readlines()])
        
        lb = MultiLabelBinarizer()
        lb.fit(Y)
        Y_DATA = lb.transform(Y+_Y)
    elif label=='single' and muti_sentence:
        pass
    elif label=='mutiple' and muti_sentence:
        pass
    else:
        lb=None
        Y_DATA,Y,X,_Y,_X=['','','','','']
    
    tokenizer.fit_on_texts(X)
    X_DATA =sequence.pad_sequences(tokenizer.texts_to_sequences(X+_X),maxlen=maxlen)
    
    index2word = dict(zip(tokenizer.word_index.values(),tokenizer.word_index.keys()))
    word2index = tokenizer.word_index
    
    return (X_DATA[:len(X)],Y_DATA[:len(X)]),(X_DATA[len(X):],Y_DATA[len(X):]),(word2index,index2word),(list(lb.classes_))