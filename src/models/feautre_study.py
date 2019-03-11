'''
Author Alex
mail: 908337832@qq.com

'''

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, merge, Bidirectional,Input,Lambda
from keras import regularizers
from keras.layers import Input, Dense, Lambda, Activation, Dropout, Flatten, Bidirectional, Conv2D, MaxPool2D, Reshape, BatchNormalization, Layer, Embedding, dot
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model, Progbar, normalize
from keras.layers.recurrent import LSTM
from keras.layers.merge import add, concatenate
from datetime import datetime
import os
import sys
import json
import pickle
import gensim
from sklearn.metrics import f1_score, accuracy_score
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
from tqdm import tqdm
import gc
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
from custom_layer.topic_attention import Attention_Over_Latent_Topic
from custom_layer.attention_visualize import Attention
from keras.optimizers import Adam

#from keras.utils.vis_utils import plot_model

#  settings for plot_model
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'  # 安装graphviz的路径

class FeatureStudy:
    def __init__(self,args,textData):
        self.args = args
        self.textData = textData

        self.model_name = 'seq-attention'
        self.feature = 'NTM'
        self.ntm_only = False

        print("------------------------------------------------------")
        print("Feature study : Model-{}  Feature-{}".format(self.model_name,self.feature))
        print("------------------------------------------------------")

        # build the network in keras code
        self.network = None
        self.neural_topic_model = None
        self.visualize_model = None
        self.topic_emb = None
        self.callbacks = None
        self.word2vec_method ="fasttext"
        print("Word2Vec method : {}".format(self.word2vec_method))

        if self.feature == 'NTM':
            # Model Configuration
            self.HIDDEN_NUM = [100, 100]  # hidden layer size
            self.TOPIC_NUM = 40
            self.CATEGORY = self.args.num_class
            self.TOPIC_EMB_DIM = 50  # topic memory size
            # self.MAX_SEQ_LEN = 24  # clip length for a text
            # self.BATCH_SIZE = 32
            # MAX_EPOCH = 800
            self.MIN_EPOCH = 0
            self.PATIENT = 10
            self.PATIENT_GLOBAL = 60
            self.PRE_TRAIN_EPOCHS = 50
            self.ALTER_TRAIN_EPOCHS = 50
            self.TARGET_SPARSITY = 0.75
            self.KL_GROWING_EPOCH = 0
            self.MAX_NTM = 100
            self.SHORTCUT = True
            self.TRANSFORM = None  # 'relu'|'softmax'|'tanh'
            self.l1_strength = None
            self.kl_strength = None

            self.optimize_ntm = True
            self.first_optimize_ntm = True

        if self.word2vec_method =="word2vec":
            pass
        elif self.word2vec_method == 'fasttext':
            self.args.word2vec_path="./stc_nlpcc_fasttext50dim.txt"
        elif self.word2vec_method == 'GLoVe':
            self.args.word2vec_path = "./stc_nlpcc_GloVe50dim.txt"


        self.buildModel()

    def buildModel(self):
        #################### functions ###############################
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
        ##################### Inputs #################################
        seq_input = Input(shape=(self.args.maxLength,),dtype='int32',name='seq_input')
        if self.feature == 'NTM':
            bow_input = Input(shape=(self.textData.getVocabularySize(),), name='bow_input')
            psudo_input = Input(shape=(self.TOPIC_NUM,),dtype='int32',name='psudo_input')
        ########## Word Level Feature Representation #################
        seq_emb = None
        seq_topic_emb = None
        word_level_features = []

        # word2vec
        if self.args.using_word2vec:
            print('Using word2vec to initialize embedding layer ... ')
            embedding_mat = useword2vec(self.textData.word2id)

            seq_emb = Embedding(embedding_mat.shape[0],
                                embedding_mat.shape[1],
                                weights=[embedding_mat],
                                mask_zero=False,
                                trainable=False,
                                input_length=self.args.maxLength)
        else:
            seq_emb = Embedding(len(self.textData.word2id.keys()) + 1,  # max_features,
                                        self.args.embedding_dims,
                                        mask_zero=False,
                                        trainable=False,
                                        input_length=self.args.maxLength)

        sequence_embedding = seq_emb(seq_input)
        word_level_features.append(sequence_embedding)

        # word-topic embedding
        if self.feature == 'NTM' :
            topic_emb = Embedding(self.TOPIC_NUM,self.textData.getVocabularySize(),input_length=self.TOPIC_NUM,trainable=False)
            self.topic_emb = topic_emb
            p_x_given_h, represent_mu, ntm_model = self.build_ntm(bow_input)
            self.neural_topic_model = ntm_model

            topic_attention = Attention_Over_Latent_Topic(self.args.embedding_dims, self.TOPIC_EMB_DIM,combine_method='mul')  # Attention mechanism over latent topic to extract richer meaning for given word/context
            s_trans = Dense(self.TOPIC_EMB_DIM, activation='relu')
            v_trans = Dense(self.TOPIC_EMB_DIM, activation='relu')

            # latent topic
            topic_matric = topic_emb(psudo_input)
            S = s_trans(topic_matric)
            V = v_trans(topic_matric)

            # sequence latent topic feature
            seq_topic_embeding,P = topic_attention([sequence_embedding, represent_mu, S, V])
            word_level_features.append(seq_topic_embeding)
            if self.ntm_only:
                word_level_features = [seq_topic_embeding]

        if len(word_level_features)>1:
            word_level_representation = concatenate(word_level_features, axis=-1)
        else:
            word_level_representation = word_level_features[0]

        ##############################  CLF MODEL  ###############################
        if self.model_name == 'seq-attention':
            emotion_predict,seq_attention = self.seq_attention_model(word_level_representation)
        elif self.model_name == 'mcnn':
            emotion_predict = self.text_cnn_model(word_level_representation)


        ######################## build and compile model #########################
        #metrics = ['accuracy', metric.precision, metric.recall, metric.f1]
        adam_optimizer = Adam(lr=self.args.learningRate)

        """
        clf_model = Model(input=[bow_input,seq_input,psudo_input], output=[emotion_predict, seq_attention])
        clf_model.compile(loss={'emotion_predict': loss, 'attention_weights': 'mean_squared_error'},
                      optimizer=adam_optimizer, metrics=['accuracy'], loss_weights=[1.0, 0])
        """

        if self.feature=='NTM':
            clf_model = Model(input=[bow_input, seq_input, psudo_input], output=emotion_predict)
        elif self.feature=='word2vec':
            clf_model = Model(input=seq_input,output=emotion_predict)


        if self.args.num_class==2:
            clf_model.compile(loss={'emotion_predict': 'binary_crossentropy'},
                              optimizer=adam_optimizer, metrics=['accuracy'])
        else:
            clf_model.compile(loss={'emotion_predict': 'categorical_crossentropy'},
                              optimizer=adam_optimizer, metrics=['accuracy'])

        self.network = clf_model
        #plot_model(self.network,'hierachical_attention.png')
        #plot_model(self.neural_topic_model,'neural_topic_model.png')
        print(clf_model.summary())

    def build_ntm(self, bow_input):
        ######################## function ##########################
        class CustomizedL1L2(regularizers.L1L2):
            def __init__(self, l1=0., l2=0.):
                self.l1 = K.variable(K.cast_to_floatx(l1))
                self.l2 = K.variable(K.cast_to_floatx(l2))

        def sampling(args):
            mu, log_sigma = args
            epsilon = K.random_normal(shape=(self.TOPIC_NUM,), mean=0.0, stddev=1.0)
            return mu + K.exp(log_sigma / 2) * epsilon
        ######################## build ntm #########################
        # encoder
        e1 = Dense(self.HIDDEN_NUM[0], activation='relu')
        e2 = Dense(self.HIDDEN_NUM[1], activation='relu')
        e3 = Dense(self.TOPIC_NUM)
        e4 = Dense(self.TOPIC_NUM)

        h = e1(bow_input)
        h = e2(h)
        if self.SHORTCUT:
            es = Dense(self.HIDDEN_NUM[1], use_bias=False)
            h = add([h, es(bow_input)])

        z_mean = e3(h)
        z_log_var = e4(h)

        # sample
        z = Lambda(sampling, output_shape=(self.TOPIC_NUM,))([z_mean, z_log_var])  # z = z_mean + eps * z_var

        # generator
        g1 = Dense(self.TOPIC_NUM, activation='tanh')
        g2 = Dense(self.TOPIC_NUM, activation='tanh')
        g3 = Dense(self.TOPIC_NUM, activation='tanh')
        g4 = Dense(self.TOPIC_NUM)

        def generate(h):
            tmp = g1(h)
            tmp = g2(tmp)
            tmp = g3(tmp)
            tmp = g4(tmp)
            if self.SHORTCUT:
                r = add([Activation('tanh')(tmp), h])
            else:
                r = tmp
            if self.TRANSFORM is not None:
                r = Activation(self.TRANSFORM)(r)
                return r
            else:
                return r

        represent = generate(z)
        represent_mu = generate(z_mean)

        # decoder
        self.l1_strength = CustomizedL1L2(l1=0.001)
        d1 = Dense(self.textData.getVocabularySize(), activation='sigmoid', kernel_regularizer=self.l1_strength,
                   name='p_x_given_h')
        p_x_given_h = d1(represent)

        def kl_loss(x_true, x_decoded):
            kl_term = - 0.5 * K.sum(
                1 - K.square(z_mean) + z_log_var - K.exp(z_log_var),
                axis=-1)
            return kl_term

        def nnl_loss(x_true, x_decoder):
            nnl_term = - K.sum(x_true * K.log(x_decoder + 1e-32), axis=-1)
            return nnl_term

        # neural topic model
        self.kl_strength = K.variable(1.0)
        ntm_model = Model(bow_input, [represent_mu, p_x_given_h])
        ntm_model.compile(loss=[kl_loss, nnl_loss], loss_weights=[self.kl_strength, 1.0], optimizer="adagrad")

        return p_x_given_h,represent_mu,ntm_model

    def seq_attention_model(self,word_level_features):
        #---- Attention Over Sequence -----#
        def slice_front(x, index):
            return x[:, :index]

        def slice_back(x, index):
            return x[:, index:]

        GRU_HIDDEN = 50
        context_modeling = Bidirectional(GRU(GRU_HIDDEN, dropout=0.5, return_sequences=True), merge_mode='concat',name='context_modeling')
        sequence_attention = Attention(self.args.attention_size)

        context_embedding = context_modeling(word_level_features)
        division_dim = context_embedding._keras_shape[-1]
        mixed_attention_features = sequence_attention(context_embedding)
        attention_feature = Lambda(slice_front, output_shape=(division_dim,), arguments={'index': division_dim})(mixed_attention_features)
        attention_weights = Lambda(slice_back, output_shape=(self.args.maxLength,), arguments={'index': division_dim},name='attention_weights')(mixed_attention_features)
        feature_map = Dense(self.args.hidden_dims, activation='tanh')(attention_feature)
        if self.args.num_class==2:
            emotion_predict = Dense(1, activation='sigmoid', name='emotion_predict')(feature_map)
        else:
            emotion_predict = Dense(self.args.num_class, activation='softmax', name='emotion_predict')(feature_map)
        return emotion_predict,attention_weights

    def text_cnn_model(self,word_level_features):
        def text_cnn_conv_layer(inputs, kernel_size, num_filters, padding='same'):
            layter_output = []
            for ks, f in zip(kernel_size, num_filters):
                conv_tensor = Conv1D(f, ks, padding=padding, kernel_initializer='normal', activation='relu')(inputs)
                max_pool_tensor = GlobalMaxPooling1D()(conv_tensor)
                layter_output.append(max_pool_tensor)
            return layter_output
        conv_features = text_cnn_conv_layer(word_level_features, [3, 4, 5], [100, 100, 100])
        conv_features = concatenate(conv_features,axis=-1)
        feature_map = Dense(self.args.hidden_dims, activation='tanh')(conv_features)
        if self.args.num_class==2:
            emotion_predict = Dense(1, activation='sigmoid', name='emotion_predict')(feature_map)
        else:
            emotion_predict = Dense(self.args.num_class, activation='softmax', name='emotion_predict')(feature_map)
        return emotion_predict

    def fit(self,trainX,trainY,testX,testY,optimize_ntm=True):
        self.optimize_ntm = optimize_ntm
        min_val_loss = np.inf  # save best only
        min_bound_cls = -np.inf
        min_bound_ntm = np.inf
        epoch_since_improvement = 0
        epoch_since_improvement_global = 0
        if self.feature == 'word2vec':
            callbacks = [
                EarlyStopping(monitor='val_acc', patience=self.args.early_stopping_step, mode='max', verbose=2),
                ModelCheckpoint(self.args.kfold_weights_path, monitor='val_acc', save_best_only=True, mode='max',
                                verbose=2, save_weights_only=True),
                TensorBoard(log_dir=self.args.model_save_path + "log/", histogram_freq=3)
            ]
            self.network.fit(trainX, trainY,
                batch_size=self.args.batchSize, epochs=self.args.numEpochs, shuffle=True, verbose=self.args.verbose,
                validation_data=[testX, testY],
                callbacks=callbacks)

        else:
            for e in range(self.args.numEpochs):
                indices = np.arange(trainX.shape[0])
                np.random.shuffle(indices)

                batch_size = self.args.batchSize
                num_batch = trainX.shape[0]//batch_size

                epoch_train = []
                epoch_test = []
                if self.feature=='NTM' and self.optimize_ntm:
                    print('Epoch {}/{} training {}'.format(e,self.args.numEpochs, "ntm"))
                    for i in tqdm(range(num_batch)):
                        seq_train_batch = trainX[indices[i*batch_size:(i+1)*batch_size]]
                        bow_train_batch = self.Id2Bow(seq_train_batch)
                        info = self.neural_topic_model.train_on_batch(
                            bow_train_batch,[np.zeros([seq_train_batch.shape[0],self.TOPIC_NUM]),bow_train_batch]
                        )
                        epoch_train.append(info)

                        del seq_train_batch
                        del bow_train_batch
                        del info
                        gc.collect()
                        if i%1000==0:
                            [train_loss, train_kld, train_nnl] = np.mean(epoch_train, axis=0)
                            print("ntm train loss: %.4f, kld: %.4f, nnl: %.4f" % (train_loss, train_kld, train_nnl))

                    [train_loss, train_kld, train_nnl] = np.mean(epoch_train, axis=0)
                    print("ntm train loss: %.4f, kld: %.4f, nnl: %.4f" % (train_loss, train_kld, train_nnl))
                    # check sparsity
                    sparsity = self.check_sparsity(self.neural_topic_model)
                    self.update_l1(self.l1_strength, sparsity, self.TARGET_SPARSITY)

                    # test
                    num_test = testX.shape[0]//batch_size
                    for i in range(num_test):
                        seq_test_batch = testX[i*batch_size:(i+1)*batch_size]
                        bow_test_batch = self.Id2Bow(seq_test_batch)
                        info = self.neural_topic_model.evaluate(
                            bow_test_batch,[np.zeros([seq_test_batch.shape[0],self.TOPIC_NUM]),bow_test_batch],
                            verbose=0
                        )
                        epoch_test.append(info)
                    [val_loss, kld, nnl] = np.mean(epoch_test, axis=0)
                    bound = val_loss #np.exp(val_loss / np.mean(np.sum(testX>0,axis=1)))
                    print("ntm estimated perplexity upper bound on validation set: %.3f" % bound)
                    print("ntm val loss: %.4f, kld: %.4f, nnl: %.4f" % (val_loss, kld, nnl))
                    if bound < min_bound_ntm and e >= self.KL_GROWING_EPOCH:
                        print("New best val bound: %.3f in %d epoch" % (bound, e))
                        min_bound_ntm = bound
                        if self.first_optimize_ntm:
                            print("Saving model")
                            # ntm_model.save(Model_fn)
                            restorePath = self.args.rootDir + '/checkpoint/'
                            restorePath = restorePath + "{}_ntm_bound_{}_.h5".format(self.args.corpus, str(round(bound, 2)))
                            self.neural_topic_model.save_weights(restorePath)
                        epoch_since_improvement = 0
                        epoch_since_improvement_global = 0
                    elif bound >= min_bound_ntm:
                        epoch_since_improvement += 1
                        epoch_since_improvement_global += 1
                        print("No improvement in epoch %d" % e)
                    if e < self.KL_GROWING_EPOCH:
                        print("Growing kl strength %.3f" % K.get_value(self.kl_strength))
                    if (epoch_since_improvement > self.PATIENT and e > self.MIN_EPOCH) or e>self.MAX_NTM:
                        self.optimize_ntm = False
                        self.first_optimize_ntm = False
                        epoch_since_improvement = 0
                        beta_exp = np.exp(self.neural_topic_model.get_weights()[-2])
                        beta = beta_exp / (np.sum(beta_exp, 1)[:, np.newaxis])
                        self.topic_emb.set_weights([beta])  # update topic-word matrix
                        # min_bound_ntm += 2    # relax ntm bound a bit
                    if epoch_since_improvement_global > self.PATIENT_GLOBAL:
                        break
                elif self.feature=='NTM':
                    print('Epoch {}/{} training {}'.format(e,self.args.numEpochs, "clf"))
                    if self.args.set_weights:
                        print("set weights for topic_emb")
                        self.args.set_weights=False
                        epoch_since_improvement = 0
                        beta_exp = np.exp(self.neural_topic_model.get_weights()[-2])
                        beta = beta_exp / (np.sum(beta_exp, 1)[:, np.newaxis])
                        self.topic_emb.set_weights([beta])  # update topic-word matrix
                        # min_bound_ntm += 2    # relax ntm bound a bit
                    for i in tqdm(range(num_batch)):
                        seq_train_batch = trainX[indices[i*batch_size:(i+1)*batch_size]]
                        bow_train_batch = self.Id2Bow(seq_train_batch)
                        psudo_batch = np.repeat(np.expand_dims(np.arange(self.TOPIC_NUM), axis=0), seq_train_batch.shape[0], axis=0)
                        label_batch = trainY[indices[i*batch_size:(i+1)*batch_size]]
                        info = self.network.train_on_batch(
                            [bow_train_batch,seq_train_batch,psudo_batch],label_batch
                        )
                        epoch_train.append(info)

                        del seq_train_batch
                        del bow_train_batch
                        del psudo_batch
                        del label_batch
                        del info
                        gc.collect()
                        if i%1000==0:
                            train_loss, train_acc = np.mean(epoch_train, axis=0)
                            print("cls train loss: %.4f, train acc: %2f" % (train_loss, train_acc))
                            #print("cls train loss: %.4f, train info %2f,  %2f, %2f, %2f" % (train_loss, info1,info2,info3,info4))

                    train_loss, train_acc = np.mean(epoch_train, axis=0)
                    print("cls train loss: %.4f, train acc: %2f" % (train_loss,train_acc))
                    #print("cls train loss: %.4f, train info %2f,  %2f, %2f, %2f" % (train_loss, info1, info2, info3, info4))

                    # test
                    num_test = testX.shape[0]//batch_size
                    y_test_pred = []

                    for i in range(num_test):
                        seq_test_batch = testX[i*batch_size:(i+1)*batch_size]
                        bow_test_batch = self.Id2Bow(seq_test_batch)
                        psudo_test_batch = np.repeat(np.expand_dims(np.arange(self.TOPIC_NUM), axis=0), seq_test_batch.shape[0], axis=0)
                        y_test_pred_batch = self.network.predict(
                            [bow_test_batch,seq_test_batch,psudo_test_batch]
                        )
                        y_test_pred.append(y_test_pred_batch)
                    y_pred = np.concatenate(y_test_pred,axis=0)
                    if self.args.num_class==2:
                        y_pred[y_pred>0.5]=1
                        y_pred[y_pred<=0.5]=0
                        y_pred_label = y_pred
                        y_true_label = testY[:y_pred_label.shape[0]]
                    else:
                        y_pred_label = np.argmax(y_pred, axis=1)
                        y_true_label = np.argmax(testY[:y_pred_label.shape[0]], axis=1)
                    test_acc = accuracy_score(y_true_label, y_pred_label)
                    test_f1 = f1_score(y_true_label, y_pred_label, average="weighted")
                    print("cls val acc: {}, val f1:{}".format(test_acc,test_f1))
                    if test_acc > min_bound_cls:
                        min_bound_cls = test_acc
                        print( "New best val acc: %.4f, f1: %.4f in %d epoch" % (min_bound_cls, test_f1, e))
                        restorePath = self.args.rootDir + '/checkpoint/'
                        clf_restorePath = restorePath + "{}_cls_{}_feature_{}_acc{}_use_ntm_{}.h5".format(self.args.corpus,self.model_name,self.feature,round(test_acc,2),str(not self.args.hierachical_attention_without_ntm))
                        ntm_restorePath = restorePath + "{}_ntm_{}_feature_{}_acc{}_use_ntm_{}.h5".format(self.args.corpus,self.model_name,self.feature,round(test_acc,2),str(not self.args.hierachical_attention_without_ntm))
                        self.network.save_weights(clf_restorePath)
                        self.neural_topic_model.save_weights(ntm_restorePath)
                        epoch_since_improvement = 0
                        epoch_since_improvement_global = 0
                    else:
                        epoch_since_improvement += 1
                        epoch_since_improvement_global += 1
                        print("No improvement in epoch %d with val acc %.4f, f1 %.4f" % (e, test_acc, test_f1))
                        if epoch_since_improvement % 10 ==0:
                            self.args.learningRate *= 0.1
                    if epoch_since_improvement > self.PATIENT and e > self.MIN_EPOCH:
                        break



    def inference(self,data,classes,labelOneHot=None):
        pass

    def visualizeAttention(self,data,preLabel,trueLabel,probability,attention):
        pass

    def step(self,batch):
        pass

    def Id2Bow(self,inputs, use_vocab=None,**arguments):
        if use_vocab is None:
            use_vocab = self.textData.getVocabularySize()
        out = np.zeros((len(inputs), use_vocab))
        for i in range(inputs.shape[0]):
            out[i, inputs[i]] = 1
        return out

    def check_sparsity(self,model, sparsity_threshold=1e-3):
        kernel = model.get_weights()[-2]
        num_weights = kernel.shape[0] * kernel.shape[1]
        num_zero = np.array(np.abs(kernel) < sparsity_threshold, dtype=float).sum()
        return num_zero / float(num_weights)

    def update_l1(self,cur_l1, cur_sparsity, sparsity_target):
        current_l1 = K.get_value(cur_l1.l1)
        diff = sparsity_target - cur_sparsity
        new_l1 = current_l1 * 2.0 ** diff
        K.set_value(cur_l1.l1, K.cast_to_floatx(new_l1))
