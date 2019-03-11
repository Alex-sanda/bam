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

class Hierarchical_Attention_On_Seq_And__Topic:
    def __init__(self,args,textData):
        print("------------------------------------------------------")
        print("Model : Hierarchical Attention On Seq And Latent Topic")
        print("------------------------------------------------------")
        if args.hierachical_attention_without_ntm:
            print("train model without Neural Topic Model: Equal to Gru-Attention")
        self.args = args
        self.textData = textData

        # build the network in keras code
        self.network = None
        self.neural_topic_model = None
        self.visualize_model = None
        self.topic_emb = None
        self.callbacks = None

        # Model Configuration
        self.HIDDEN_NUM = [100, 100]  # hidden layer size
        self.TOPIC_NUM = 40
        self.CATEGORY = self.args.num_class
        self.TOPIC_EMB_DIM = 50  # topic memory size
        #self.MAX_SEQ_LEN = 24  # clip length for a text
        #self.BATCH_SIZE = 32
        #MAX_EPOCH = 800
        self.MIN_EPOCH = 0
        self.PATIENT = 50
        self.PATIENT_GLOBAL = 60
        self.PRE_TRAIN_EPOCHS = 50
        self.ALTER_TRAIN_EPOCHS = 50
        self.TARGET_SPARSITY = 0.85
        self.KL_GROWING_EPOCH = 0
        self.MAX_NTM = 200
        self.SHORTCUT = True
        self.TRANSFORM = 'softmax'  # 'relu'|'softmax'|'tanh'
        self.l1_strength = None
        self.kl_strength = None

        self.optimize_ntm = False
        self.first_optimize_ntm = False
        self.optimize_clf = True

        self.set_weights = True
        self.joint_training = False
        self.model_context_in_word_level = False
        self.model_context_in_sequence_level = True
        self.combine_method = 'mul'
        self.fix_ntm_while_training_clf = True


        self.partial_trainabile = True


        if self.joint_training:
            print("--------------Joint Training------------------")
        # fix ntm parameters while training clf model,to gaerentee the ntm works

        if self.optimize_clf and not self.optimize_ntm:
            if self.fix_ntm_while_training_clf and not self.optimize_ntm:
                self.partial_trainabile = False
                print("Trainging Setup:: fix NTM while training clf model")
            else:
                self.partial_trainabile = True

        self.buildModel()

    def buildModel(self):
        #################### functions ###############################
        class CustomizedL1L2(regularizers.L1L2):
            def __init__(self, l1=0., l2=0.):
                self.l1 = K.variable(K.cast_to_floatx(l1))
                self.l2 = K.variable(K.cast_to_floatx(l2))

        def sampling(args):
            mu, log_sigma = args
            epsilon = K.random_normal(shape=(self.TOPIC_NUM,), mean=0.0, stddev=1.0)
            return mu + K.exp(log_sigma / 2) * epsilon

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
        bow_input = Input(shape=(self.textData.getVocabularySize(),),name='bow_input')
        seq_input = Input(shape=(self.args.maxLength,),dtype='int32',name='seq_input')
        psudo_input = Input(shape=(self.TOPIC_NUM,),dtype='int32',name='psudo_input')

        # Embeddings
        topic_emb = Embedding(self.TOPIC_NUM,self.textData.getVocabularySize(),input_length=self.TOPIC_NUM,trainable=False)
        self.topic_emb = topic_emb
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
        ######################## build ntm #########################
        # encoder
        e1 = Dense(self.HIDDEN_NUM[0],activation='relu',trainable=self.partial_trainabile)
        e2 = Dense(self.HIDDEN_NUM[1],activation='relu',trainable=self.partial_trainabile)
        e3 = Dense(self.TOPIC_NUM,trainable=self.partial_trainabile)
        e4 = Dense(self.TOPIC_NUM,trainable=self.partial_trainabile)

        h = e1(bow_input)
        h = e2(h)
        if self.SHORTCUT:
            es = Dense(self.HIDDEN_NUM[1],use_bias=False,trainable=self.partial_trainabile)
            h = add([h,es(bow_input)])

        z_mean = e3(h)
        z_log_var = e4(h)

        # sample
        z = Lambda(sampling,output_shape=(self.TOPIC_NUM,))([z_mean,z_log_var]) # z = z_mean + eps * z_var

        # generator
        g1 = Dense(self.TOPIC_NUM, activation='tanh',trainable=self.partial_trainabile)
        g2 = Dense(self.TOPIC_NUM, activation='tanh',trainable=self.partial_trainabile)
        g3 = Dense(self.TOPIC_NUM, activation='tanh',trainable=self.partial_trainabile)
        g4 = Dense(self.TOPIC_NUM)

        def generate(h):
            tmp = g1(h)
            tmp = g2(tmp)
            tmp = g3(tmp)
            tmp = g4(tmp)
            if self.SHORTCUT:
                r = add([Activation('tanh')(tmp),h])
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
        d1 = Dense(self.textData.getVocabularySize(),activation='sigmoid',kernel_regularizer=self.l1_strength,name='p_x_given_h',trainable=self.partial_trainabile)
        #d1 = Dense(self.textData.getVocabularySize(), activation='softmax', name='p_x_given_h')
        p_x_given_h = d1(represent)

        #################  Hierarchical_Attention  ####################
        GRU_HIDDEN = 50
        if self.model_context_in_word_level:
            word_level_context_emb_size = GRU_HIDDEN * 2
            word_context_modeling = Bidirectional(GRU(GRU_HIDDEN, dropout=0.5, return_sequences=True),merge_mode='concat', name='word_context_modeling')
            topic_attention = Attention_Over_Latent_Topic(word_level_context_emb_size,self.TOPIC_EMB_DIM,combine_method=self.combine_method) # Attention mechanism over latent topic to extract richer meaning for given word/context
        else:
            topic_attention = Attention_Over_Latent_Topic(self.args.embedding_dims, self.TOPIC_EMB_DIM,combine_method=self.combine_method)  # Attention mechanism over latent topic to extract richer meaning for given word/context
        seq_context_modeling = Bidirectional(GRU(GRU_HIDDEN, dropout=0.5, return_sequences=True), merge_mode='concat',name='seq_context_modeling')
        sequence_attention = Attention(self.args.attention_size)
        s_trans = Dense(self.TOPIC_EMB_DIM,activation='relu')
        v_trans = Dense(self.TOPIC_EMB_DIM,activation='relu')

        #--- Attention Over Latent Topic ---#
        # context
        sequence_embedding = seq_emb(seq_input)
        if self.model_context_in_word_level:
            word_level_context_feature = word_context_modeling(sequence_embedding)

        # latent topic
        topic_matric = topic_emb(psudo_input)
        S = s_trans(topic_matric)
        V = v_trans(topic_matric)

        # sequence latent topic feature
        if self.model_context_in_word_level:
            seq_latent_topic_feature,att_topic_P,et_raw = topic_attention([word_level_context_feature,represent_mu,S,V])
        else:
            seq_latent_topic_feature, att_topic_P, et_raw = topic_attention([sequence_embedding, represent_mu, S, V])

        #---- Attention Over Sequence -----#
        def slice_front(x, index):
            return x[:, :index]

        def slice_back(x, index):
            return x[:, index:]


        if self.args.hierachical_attention_without_ntm:
            if self.model_context_in_word_level:
                enrich_feature = word_level_context_feature
                division_dim = word_level_context_emb_size
            else:
                enrich_feature = sequence_embedding
                division_dim = self.args.embedding_dims

        else:
            if self.model_context_in_word_level:
                enrich_feature = concatenate([word_level_context_feature,seq_latent_topic_feature],axis=-1)
                division_dim = word_level_context_emb_size + self.TOPIC_EMB_DIM
            else:
                enrich_feature = concatenate([sequence_embedding,seq_latent_topic_feature],axis=-1)
                division_dim = self.args.embedding_dims + self.TOPIC_EMB_DIM

        if self.model_context_in_sequence_level:
            enrich_feature = seq_context_modeling(enrich_feature)
            division_dim = GRU_HIDDEN * 2


        mixed_attention_features = sequence_attention(enrich_feature)
        attention_feature = Lambda(slice_front, output_shape=(division_dim,), arguments={'index': division_dim})(mixed_attention_features)
        attention_weights = Lambda(slice_back, output_shape=(self.args.maxLength,), arguments={'index': division_dim},name='attention_weights')(mixed_attention_features)
        feature_map = Dense(self.args.hidden_dims, activation='tanh')(attention_feature)
        if self.args.num_class==2:
            emotion_predict = Dense(1, activation='sigmoid', name='emotion_predict')(feature_map)
            loss = 'binary_crossentropy'
        else:
            emotion_predict = Dense(self.args.num_class, activation='softmax', name='emotion_predict')(feature_map)
            loss = 'categorical_crossentropy'


        ######################## build and compile model #########################
        metrics = ['accuracy', metric.precision, metric.recall, metric.f1]

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


        adam_optimizer = Adam(lr=self.args.learningRate)
        clf_model = Model(input=[bow_input,seq_input,psudo_input], output=[emotion_predict, attention_weights])
        clf_model.compile(loss={'emotion_predict': loss, 'attention_weights': 'mean_squared_error'},
                      optimizer=adam_optimizer, metrics=['accuracy'], loss_weights=[1.0, 0])

        vis_model = Model(inputs=[bow_input, seq_input, psudo_input],output=[emotion_predict,feature_map ,attention_weights, att_topic_P,represent_mu,et_raw])

        joint_model = Model(inputs=[bow_input, seq_input, psudo_input],output=[emotion_predict,represent_mu,p_x_given_h])
        joint_model.compile(loss=[loss,kl_loss, nnl_loss], loss_weights=[5.0,self.kl_strength, 1.0],metrics=['accuracy'], optimizer=adam_optimizer)

        if self.joint_training == True:
            self.network = joint_model
        else:
            self.network = clf_model
        self.neural_topic_model = ntm_model
        self.visualize_model = vis_model
        #plot_model(self.network,'hierachical_attention.png')
        #plot_model(self.neural_topic_model,'neural_topic_model.png')


        print(clf_model.summary())

    def fit(self,trainX,trainY,testX,testY,optimize_ntm=True):
        #self.optimize_ntm = optimize_ntm
        min_val_loss = np.inf  # save best only
        min_bound_cls = -np.inf
        min_bound_ntm = np.inf
        epoch_since_improvement = 0
        epoch_since_improvement_global = 0
        if self.joint_training:
            print("--------------Joint Training-----------------")

        for e in range(self.args.numEpochs):
            indices = np.arange(trainX.shape[0])
            np.random.shuffle(indices)

            batch_size = self.args.batchSize
            num_batch = trainX.shape[0]//batch_size

            epoch_train = []
            epoch_test = []

            if self.joint_training and not self.first_optimize_ntm:
                print('Epoch {}/{} training {}'.format(e,self.args.numEpochs, "clf"))
                if self.set_weights and not self.first_optimize_ntm:
                    self.set_weights = False
                    print("set weights for topic_emb")
                    #self.args.set_weights=False
                    epoch_since_improvement = 0
                    beta = self.neural_topic_model.get_weights()[-2]
                    #beta_exp = np.exp(self.neural_topic_model.get_weights()[-2])
                    #beta = beta_exp / (np.sum(beta_exp, 1)[:, np.newaxis])
                    self.topic_emb.set_weights([beta])  # update topic-word matrix
                for i in tqdm(range(num_batch)):
                    seq_train_batch = trainX[indices[i * batch_size:(i + 1) * batch_size]]
                    bow_train_batch = self.Id2Bow(seq_train_batch)
                    psudo_batch = np.repeat(np.expand_dims(np.arange(self.TOPIC_NUM), axis=0), seq_train_batch.shape[0],
                                            axis=0)
                    label_batch = trainY[indices[i * batch_size:(i + 1) * batch_size]]
                    info = self.network.train_on_batch(
                        [bow_train_batch, seq_train_batch, psudo_batch],
                        [label_batch, np.zeros([seq_train_batch.shape[0],self.TOPIC_NUM]),bow_train_batch]
                    )
                    epoch_train.append(info)

                    del seq_train_batch
                    del bow_train_batch
                    del psudo_batch
                    del label_batch
                    del info
                    gc.collect()
                    if (i+1) % 1000 == 0:
                        train_loss, clf_loss, clf_acc, train_kld,_, train_nnl,_ = np.mean(epoch_train, axis=0)
                        print("train_loss: {}  , clf_loss: {}  , clf_acc: {}  , train_kld: {} , train_nnl".format(train_loss, clf_loss, clf_acc, train_kld, train_nnl))
                        # print("cls train loss: %.4f, train info %2f,  %2f, %2f, %2f" % (train_loss, info1,info2,info3,info4))

                # check sparsity
                sparsity = self.check_sparsity(self.neural_topic_model)
                self.update_l1(self.l1_strength, sparsity, self.TARGET_SPARSITY)

                train_info = np.mean(epoch_train, axis=0)
                for q in range(len(train_info)):
                    train_info[q] = np.round(train_info[q],3)
                train_loss, clf_loss,train_kld,train_nnl, clf_acc,_,_ = train_info
                print("train_loss: {}  , clf_loss: {}  , train_kld: {}  , train_nnl: {} , clf_acc: {}".format(train_loss,clf_loss,train_kld,train_nnl,clf_acc))
                # print("cls train loss: %.4f, train info %2f,  %2f, %2f, %2f" % (train_loss, info1, info2, info3, info4))

                # test
                num_test = testX.shape[0] // batch_size
                y_test_pred = []
                test_info = []
                for i in range(num_test):
                    seq_test_batch = testX[i * batch_size:(i + 1) * batch_size]
                    bow_test_batch = self.Id2Bow(seq_test_batch)
                    psudo_test_batch = np.repeat(np.expand_dims(np.arange(self.TOPIC_NUM), axis=0),
                                                 seq_test_batch.shape[0], axis=0)
                    if self.args.num_class == 2:
                        y_true_label = testY[i * batch_size:(i + 1) * batch_size]
                    else:
                        y_true_label = np.argmax(testY[i * batch_size:(i + 1) * batch_size], axis=1)

                    test_info_batch = self.network.evaluate(
                        [bow_test_batch, seq_test_batch, psudo_test_batch],
                        [y_true_label,np.zeros([seq_test_batch.shape[0],self.TOPIC_NUM]),bow_test_batch],
                        verbose=0
                    )
                    test_info.append(test_info_batch)
                test_info = np.mean(test_info,axis=0)
                for q in range(len(test_info)):
                    test_info[q] = round(test_info[q],3)
                test_loss, test_clf_loss, test_kld, test_nnl, test_acc,_ , _ = test_info
                print("test_loss: {}, test_clf_loss: {}, test_kld: {}, test_nnl: {}, test_acc: {}".format(test_loss, test_clf_loss, test_kld, test_nnl,test_acc))

                if test_acc > min_bound_cls:
                    min_bound_cls = test_acc
                    print("New best val acc: %.4f in %d epoch" % (min_bound_cls, e))
                    restorePath = self.args.rootDir + '/checkpoint/'
                    restorePath = restorePath + "{}_hierachical_joint_training_cls_acc{}_TOPICNUM_{}_A_{}_firstOptimizeNTM_{}_model_word_context_{}_model_seq_context_{}.h5".format(
                        self.args.corpus, round(test_acc, 2),
                        self.TOPIC_NUM, self.args.attention_size,'False' if not self.first_optimize_ntm else self.MAX_NTM,
                        str(self.model_context_in_word_level),str(self.model_context_in_sequence_level)
                    )
                    print("save joint model")
                    self.network.save_weights(restorePath)
                    epoch_since_improvement = 0
                    epoch_since_improvement_global = 0
                else:
                    epoch_since_improvement += 1
                    epoch_since_improvement_global += 1
                    print("No improvement in epoch %d with val acc %.4f, f1 " % (e, test_acc))
                if epoch_since_improvement > self.PATIENT and e > self.MIN_EPOCH:
                    break
            elif self.optimize_ntm:
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
                        restorePath = restorePath + "{}_ntm_bound_{}_num_Topic_{}_A_{}.h5".format(self.args.corpus, str(round(bound, 2)),self.TOPIC_NUM,self.args.attention_size)
                        self.neural_topic_model.save_weights(restorePath)
                        print("set weights for topic_emb")
                        beta = self.neural_topic_model.get_weights()[-2]
                        self.topic_emb.set_weights([beta])  # update topic-word matrix
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
                    #beta_exp = np.exp(self.neural_topic_model.get_weights()[-2])
                    #beta = beta_exp / (np.sum(beta_exp, 1)[:, np.newaxis])
                    #beta = self.neural_topic_model.get_weights()[-2]
                    #self.topic_emb.set_weights([beta])  # update topic-word matrix
                    # min_bound_ntm += 2    # relax ntm bound a bit
                    if not self.optimize_clf:
                        print("exit training after optimizing NTM")
                        exit()
                if epoch_since_improvement_global > self.PATIENT_GLOBAL:
                    break
            else:
                print('Epoch {}/{} training {}'.format(e,self.args.numEpochs, "clf"))
                if self.set_weights:
                    print("set weights for topic_emb")
                    self.set_weights=False
                    epoch_since_improvement = 0
                    #beta_exp = np.exp(self.neural_topic_model.get_weights()[-2])
                    #beta = beta_exp / (np.sum(beta_exp, 1)[:, np.newaxis])
                    print("set weights for topic_emb")
                    beta = self.neural_topic_model.get_weights()[-2]
                    self.topic_emb.set_weights([beta])  # update topic-word matrix
                    # min_bound_ntm += 2    # relax ntm bound a bit
                for i in tqdm(range(num_batch)):
                    seq_train_batch = trainX[indices[i*batch_size:(i+1)*batch_size]]
                    bow_train_batch = self.Id2Bow(seq_train_batch)
                    psudo_batch = np.repeat(np.expand_dims(np.arange(self.TOPIC_NUM), axis=0), seq_train_batch.shape[0], axis=0)
                    label_batch = trainY[indices[i*batch_size:(i+1)*batch_size]]
                    info = self.network.train_on_batch(
                        [bow_train_batch,seq_train_batch,psudo_batch],[label_batch,np.zeros((seq_train_batch.shape[0],self.args.maxLength))]
                    )
                    epoch_train.append(info)

                    del seq_train_batch
                    del bow_train_batch
                    del psudo_batch
                    del label_batch
                    del info
                    gc.collect()
                    if i%1000==0:
                        train_loss, info1, info2, train_acc, info4 = np.mean(epoch_train, axis=0)
                        print("cls train loss: %.4f, train acc: %.3f" % (train_loss, train_acc))
                        #print("cls train loss: %.4f, train info %2f,  %2f, %2f, %2f" % (train_loss, info1,info2,info3,info4))

                train_loss, info1, info2, train_acc, info4 = np.mean(epoch_train, axis=0)
                print("cls train loss: %.4f, train acc: %.3f" % (train_loss,train_acc))
                #print("cls train loss: %.4f, train info %2f,  %2f, %2f, %2f" % (train_loss, info1, info2, info3, info4))

                # test
                num_test = testX.shape[0]//batch_size
                y_test_pred = []

                for i in range(num_test):
                    seq_test_batch = testX[i*batch_size:(i+1)*batch_size]
                    bow_test_batch = self.Id2Bow(seq_test_batch)
                    psudo_test_batch = np.repeat(np.expand_dims(np.arange(self.TOPIC_NUM), axis=0), seq_test_batch.shape[0], axis=0)
                    y_test_pred_batch,_ = self.network.predict(
                        [bow_test_batch,seq_test_batch,psudo_test_batch]
                    )
                    y_test_pred.append(y_test_pred_batch)
                y_pred = np.concatenate(y_test_pred,axis=0)
                if self.args.num_class == 2:
                    y_pred_label = np.zeros((y_pred.shape[0],1))
                    y_pred_label[y_pred>0.5]=1
                    y_true_label = testY[:y_pred_label.shape[0]]
                else:
                    y_pred_label = np.argmax(y_pred, axis=1)
                    y_true_label = np.argmax(testY[:y_pred_label.shape[0]], axis=1)
                test_acc = accuracy_score(y_true_label, y_pred_label)
                test_f1 = f1_score(y_true_label, y_pred_label, average="weighted")
                print("cls val acc: %.2f, val f1: %.2f" % (test_acc,test_f1))
                if test_acc > min_bound_cls:
                    min_bound_cls = test_acc
                    print( "New best val acc: %.4f, f1: %.4f in %d epoch" % (min_bound_cls, test_f1, e))
                    restorePath = self.args.rootDir + '/checkpoint/'
                    clf_restorePath = restorePath + "{}_hierachical_cls_acc{}_use_ntm_{}_TOPICNUM_{}_A_{}_firstNTM_{}_model_word_context_{}_model_seq_context_{}.h5".format(self.args.corpus,
                                                                                                                                                                            round(test_acc,2),
                                                                                                                                                                            str(not self.args.hierachical_attention_without_ntm),
                                                                                                                                                                            self.TOPIC_NUM,
                                                                                                                                                                            self.args.attention_size,
                                                                                                                                                                            self.MAX_NTM,
                                                                                                                                                                            str(self.model_context_in_word_level),
                                                                                                                                                                            str(self.model_context_in_sequence_level)
                                                                                                                                                                            )
                    ntm_restorePath = restorePath + "{}_hierachical_ntm_acc{}_use_ntm_{}_TOPICNUM_{}_A_{}_firstNTM_{}_model_word_context_{}_model_seq_context_{}.h5".format(self.args.corpus,
                                                                                                                                                                            round(test_acc,2),
                                                                                                                                                                            str(not self.args.hierachical_attention_without_ntm),
                                                                                                                                                                            self.TOPIC_NUM,
                                                                                                                                                                            self.args.attention_size,
                                                                                                                                                                            self.MAX_NTM,
                                                                                                                                                                            str(self.model_context_in_word_level),
                                                                                                                                                                            str(self.model_context_in_sequence_level)
                                                                                                                                                                            )
                    self.network.save_weights(clf_restorePath)
                    self.neural_topic_model.save_weights(ntm_restorePath)
                    epoch_since_improvement = 0
                    epoch_since_improvement_global = 0
                else:
                    epoch_since_improvement += 1
                    epoch_since_improvement_global += 1
                    print("No improvement in epoch %d with val acc %.4f, f1 %.4f" % (e, test_acc, test_f1))
                if epoch_since_improvement > self.PATIENT and e > self.MIN_EPOCH:
                    break

    def inference(self,data,classes,labelOneHot=None):
        pass

    def predict_and_return_Attention(self,data,preLabel,trueLabel,probability,attention):
        seq_data = data
        bow_data = self.Id2Bow(data)
        psudo_data = np.repeat(np.expand_dims(np.arange(self.TOPIC_NUM), axis=0), data.shape[0], axis=0)
        pred,seq_att,topic_att = self.visualize_model.predict([bow_data, seq_data, psudo_data])
        return data,pred,seq_att,topic_att

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
        new_l1 = current_l1 * np.exp(diff)
        print("current sparsity : {}".format(cur_sparsity))
        print("update l2 to {}".format(new_l1))
        K.set_value(cur_l1.l1, K.cast_to_floatx(new_l1))