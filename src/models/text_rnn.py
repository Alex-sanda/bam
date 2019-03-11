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

class Text_RNN:
    def __init__(self,args,textData):
        print("-----------------------")
        print("Model : Text-RNN")
        print("-----------------------")
        self.args = args
        self.textData = textData

        #build the network in keras code
        self.network = None
        self.callbacks = None
        self.buildModel()

    def buildModel(self):

        #word2vec settings
        def useword2vec(word_index):
            # KeyedVectors.load_word2vec_format("/data/word2vec/word2vec/GoogleNews-vectors-negative300.bin", binary=True)
            word_vectors = KeyedVectors.load_word2vec_format(self.args.word2vec_path)
            embedding_matrix = np.zeros((len(word_index) + 1, self.args.embedding_dims))
            mu, sigma = 0, 0.1  # 均值与标准差
            rarray = numpy.random.normal(mu, sigma, self.args.embedding_dims)
            for word, i in word_index.items():
                if word in word_vectors.vocab:
                    embedding_vector = word_vectors[word]
                else:
                    embedding_vector = rarray
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector[:self.args.embedding_dims]

            return embedding_matrix

        if self.args.using_word2vec:
            print('Using word2vec to initialize embedding layer ... ')
            embedding_mat = useword2vec(self.textData.word2id)

            embedding_layer = Embedding(embedding_mat.shape[0],
                                        embedding_mat.shape[1],
                                        weights=[embedding_mat],
                                        mask_zero=False,
                                        trainable=False,
                                        input_length=self.args.maxLength)

        else:
            embedding_layer = Embedding(len(self.textData.getVocabularySize()),  # max_features,
                                        self.args.embedding_dims,
                                        mask_zero=False,
                                        trainable=False,
                                        input_length=self.args.maxLength)

        #inputs
        model_input = Input(shape=(self.args.maxLength,))
        #Embedding
        sequence_embedding = embedding_layer(model_input)
        # BI-RNN : Structure for Text-RNN
        rnn_features = Bidirectional(LSTM(self.args.hidden_dims, dropout=self.args.dropout, return_sequences=True), merge_mode='concat')(sequence_embedding)
        sentence_feature = Lambda(lambda x:x[:,-1,:])(rnn_features)
        #sentence_feature = Lambda(lambda x:np.mean(x,axis=1))(rnn_features)
        feature_map = Dense(self.args.hidden_dims,activation='tanh',name='feature_map')(sentence_feature)
        if self.args.num_class==2:
            emotion_predict = Dense(1, activation='sigmoid', name='emotion_predict')(feature_map)
            loss = 'binary_crossentropy'
        else:
            emotion_predict = Dense(self.args.num_class, activation='softmax', name='emotion_predict')(feature_map)
            loss = 'categorical_crossentropy'
        model = Model(inputs=model_input, outputs=emotion_predict)



        metrics = ['accuracy', metric.precision, metric.recall, metric.f1]
        adam_optimizer = Adam(lr=self.args.learningRate)
        model.compile(loss=loss,optimizer=adam_optimizer, metrics=metrics)

        self.network = model
        print(self.network.summary())

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

    def step(self,batch):
        pass