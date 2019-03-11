'''
Author Alex
mail: 908337832@qq.com

'''

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, merge, Bidirectional,Input,Lambda
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping,TensorBoard
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers.recurrent import GRU
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
from utils import metric
from keras.optimizers import Adam

class GRU_Attention:
    def __init__(self,args,textData):
        print("-----------------------")
        print("Model : GRU-Attention")
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

        def slice_front(x, index):
            return x[:, :index]

        def slice_back(x, index):
            return x[:, index:]

        if self.args.using_word2vec:
            print('Using word2vec to initialize embedding layer ... ')
            embedding_mat = useword2vec(self.textData.word2id)

            embedding_layer = Embedding(embedding_mat.shape[0],
                                        embedding_mat.shape[1],
                                        weights=[embedding_mat],
                                        mask_zero=True,
                                        trainable=True,
                                        input_length=self.args.maxLength)

        else:
            embedding_layer = Embedding(len(self.textData.word2id.keys()) + 1,  # max_features,
                                        self.args.embedding_dims,
                                        mask_zero=True,
                                        trainable=True,
                                        input_length=self.args.maxLength)

        #training settings
        loss = 'categorical_crossentropy'
        metrics = ['accuracy', metric.precision, metric.recall, metric.f1]

        # model structure
        model_input = Input(shape=(self.args.maxLength,))
        sequence_embedding = embedding_layer(model_input)
        #dropout_tense = Dropout(dropout)(sequence_embedding)
        rnn_features = Bidirectional(GRU(50, dropout=0.5, return_sequences=True), merge_mode='concat')(sequence_embedding)
        mixed_attention_features = Attention(self.args.attention_size)(rnn_features)
        attention_feature = Lambda(slice_front, output_shape=(100,), arguments={'index': 100})(mixed_attention_features)
        attention_weights = Lambda(slice_back, output_shape=(self.args.maxLength,), arguments={'index': 100},
                                   name='attention_weights')(mixed_attention_features)
        feature_map = Dense(self.args.hidden_dims, activation='tanh')(attention_feature)

        if self.args.num_class==2:
            emotion_predict = Dense(1, activation='sigmoid', name='emotion_predict')(feature_map)
            loss = 'binary_crossentropy'
        else:
            emotion_predict = Dense(self.args.num_class, activation='softmax', name='emotion_predict')(feature_map)
            loss = 'categorical_crossentropy'

        model = Model(input=model_input, output=[emotion_predict, attention_weights])
        adam_optimizer = Adam(lr=self.args.learningRate)
        model.compile(loss={'emotion_predict': loss, 'attention_weights': 'mean_squared_error'},
                      optimizer=adam_optimizer, metrics=metrics, loss_weights=[1.0, 0])

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
        probability,attention_weights = self.network.predict(data)
        classPred = np.argmax(probability,axis=1)
        preLabel = [posi2label[c] for c in classPred]
        if labelOneHot is not None:
            trueLabel = [posi2label[c] for c in np.argmax(labelOneHot,axis=1)]
            return preLabel,trueLabel,probability,attention_weights
        else:
            return preLabel,None,probability,attention_weights

    def visualizeAttention(self,data,preLabel,trueLabel,probability,attention):
        """

        :param data:
        :param preLabel:
        :param trueLabel:
        :param probability:
        :param attention:
        :return:
        """
        result = []
        for i in range(data.shape[0]):
            attention_w = [round(n,2) for n in attention[i].tolist()]
            wordIndexs = data[i].tolist()
            dataStr = [self.textData.id2word[idx] if idx in self.textData.id2word.keys() else '<unknown>' for idx in wordIndexs]
            pairs = [(w,at) for w,at in zip(dataStr,attention_w) if w!='<pad>']
            dataStr = [w for w in dataStr if w!='<pad>']
            sentence = "".join(dataStr)
            prob = [round(n,2) for n in probability[i].tolist()]
            result.append([sentence,preLabel[i],trueLabel[i],pairs,prob])
        return result

    def step(self,batch):
        pass