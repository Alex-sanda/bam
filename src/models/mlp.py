'''
Author Alex
mail: 908337832@qq.com

'''
from __future__ import print_function
import keras
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, merge, Bidirectional,Input,Lambda
from keras.layers import Conv1D, GlobalMaxPooling1D,Concatenate,Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping,TensorBoard
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers.recurrent import GRU,LSTM
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
import numpy as np
from custom_layer.attention_visualize import Attention
import numpy
import jieba
from collections import *
from gensim.models.keyedvectors import KeyedVectors
import json
from utils import metric
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from keras.datasets import imdb
import keras.backend as K
from keras.optimizers import Adam
from utils import metric
from sklearn.svm import SVC
import sklearn
from tqdm import tqdm
import gc
import pickle


class MLP:
    def __init__(self,args,textData):

        self.use_vocab = 3000
        self.args = args
        self.textData = textData
        self.feature = 'tf-idf' #'BoW'

        print("-----------------------")
        print("Model : MLP-{}".format(self.feature))
        print("-----------------------")
        #build the network in keras code
        self.network = None
        self.callbacks = None
        self.buildModel()



    def buildModel(self):
        #self.network = SVC(gamma='auto',C=200) #sklearn.linear_model.LogisticRegression()
        if self.feature == 'BoW' or self.feature=='tf-idf':
            model_input = Input(shape=(self.use_vocab,))
        else:
            pass

        feature = Dense(self.args.hidden_dims,activation='relu')(model_input)
        feature = Dense(self.args.hidden_dims,activation='tanh')(feature)

        metrics = ['accuracy', metric.precision, metric.recall, metric.f1]
        adam_optimizer = Adam(lr=self.args.learningRate)

        if self.args.num_class==2:
            emo_predtion = Dense(1, activation='sigmoid')(feature)
            emo_pred_model = Model(inputs=model_input, outputs=emo_predtion)
            emo_pred_model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=metrics)
        else:
            emo_predtion = Dense(self.args.num_class,activation='softmax')(feature)
            emo_pred_model = Model(inputs=model_input,outputs=emo_predtion)
            emo_pred_model.compile(loss='categorical_crossentropy',optimizer=adam_optimizer, metrics=metrics)
        self.network = emo_pred_model



    def fit(self,X_train,Y_train,X_test,Y_test,*argument,**name_arguments):
        reindex_dict, reverse_reindex = self.build_reindex_dict(self.use_vocab)
        for e in range(self.args.numEpochs):
            print("-----------------------Epoch :{} -----------------------------".format(e))
            for i in range(X_train.shape[0] // self.args.batchSize):
                print("l",end='')
                train_batch = X_train[i * self.args.batchSize:(i + 1) * self.args.batchSize]
                if self.args.num_class==2:
                    train_y = Y_train[i * self.args.batchSize:(i + 1) * self.args.batchSize]
                else:
                    train_y = np.argmax(Y_train[i * self.args.batchSize:(i + 1) * self.args.batchSize], axis=1)
                if self.feature=='BoW':
                    down_train_batch = self.down_vocab(train_batch, keep=self.use_vocab, reindex_dict=reindex_dict)
                    train_batch = self.Id2Bow(down_train_batch, use_vocab=self.use_vocab)
                elif self.feature=='tf-idf':
                    down_train_batch = self.down_vocab(train_batch, keep=self.use_vocab, reindex_dict=reindex_dict)
                    train_batch = self.Id2tfidf(down_train_batch,use_vocab=self.use_vocab)
                # train step
                self.network.train_on_batch(train_batch, train_y)

                del train_batch
                del down_train_batch
                gc.collect()
            if self.feature=='BoW':
                down_test_batch = self.down_vocab(X_test,keep=self.use_vocab,reindex_dict=reindex_dict)
                test_batch = self.Id2Bow(down_test_batch, use_vocab=self.use_vocab)
            elif self.feature=='tf-idf':
                down_test_batch = self.down_vocab(X_test,keep=self.use_vocab,reindex_dict=reindex_dict)
                test_batch = self.Id2tfidf(down_test_batch, use_vocab=self.use_vocab)
            y_test_pred = self.network.predict(test_batch)
            if self.args.num_class==2:
                y_test_true = Y_test
                y_test_pred[y_test_pred>0.5]=1
                y_test_pred[y_test_pred<0.5]=0
            else:
                y_test_true = np.argmax(Y_test,axis=1)
            acc = np.mean(y_test_pred==y_test_true)
            print("----epoch : {}  , test acc: {}----".format(e,acc))



    def inference(self,data,classes,labelOneHot=None):
        """

        :param data:
        :param labelOneHot:
        :param classes:
        :param model:
        :return:
        """
        posi2label = dict(zip(range(len(classes)),classes))
        probability = self.network.predict(data)
        classPred = np.argmax(probability,axis=1)
        preLabel = [posi2label[c] for c in classPred]
        if labelOneHot is not None:
            trueLabel = [posi2label[c] for c in np.argmax(labelOneHot,axis=1)]
            return preLabel,trueLabel,probability
        else:
            return preLabel,None,probability

    # 构建截取高频次后，原词表与新词表的映射
    def build_reindex_dict(self,keep):
        # keep = 3000
        reindex_dict = {}
        wc = sorted(self.textData.idCount.items(), key=lambda x: x[1], reverse=True)
        wc = wc[:keep - 1]
        reindex_dict[self.textData.padToken] = self.textData.padToken
        for i, (w, c) in enumerate(wc):
            reindex_dict[w] = i + 1
        reverse_reindex = dict(zip(reindex_dict.values(), reindex_dict.keys()))
        return reindex_dict, reverse_reindex

    # 只截取高频词，加速训练
    def down_vocab(self,inputs, keep=3000, reindex_dict=None):
        if reindex_dict is None:
            reindex_dict, reverse_reindex = self.build_reindex_dict(keep)
        reindex_inputs = []
        # word = []
        for t in inputs:
            # w = [td.id2word[u] for u in t]
            t = [reindex_dict.get(idx, 0) for idx in t]
            reindex_inputs.append(t)
        return reindex_inputs

    def Id2Bow(self,inputs, use_vocab=None):
        if use_vocab is None:
            use_vocab = self.textData.getSamepleSize()
        out = np.zeros((len(inputs), use_vocab))
        for i in range(len(inputs)):
            out[i, inputs[i]] = 1
        return out

    def Id2tfidf(self,inputs, use_vocab=None):
        if use_vocab is None:
            use_vocab = self.textData.getSamepleSize()
        inputs = np.array(inputs)
        out = np.zeros((len(inputs), use_vocab))
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                out[i, inputs[i, j]] += self.textData.id2idf[inputs[i, j]]
        return out

    def step(self,batch):
        pass