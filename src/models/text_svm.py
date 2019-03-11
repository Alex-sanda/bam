'''
Author Alex
mail: 908337832@qq.com

'''

from __future__ import print_function
import numpy as np
from sklearn.svm import SVC
import sklearn
from tqdm import tqdm
import gc
import pickle


class Text_SVM:
    def __init__(self,args,textData):
        print("-----------------------")
        print("Model : SVM-{}".format('BoW'))
        print("-----------------------")
        self.use_vocab = 3000
        self.args = args
        self.textData = textData

        #build the network in keras code
        self.network = None
        self.callbacks = None
        self.buildModel()

    def buildModel(self):
        self.network = SVC(gamma='auto',C=200) #sklearn.linear_model.LogisticRegression()

    def fit(self,X_train,Y_train,X_test,Y_test,*argument,**name_arguments):
        reindex_dict, reverse_reindex = self.build_reindex_dict(self.use_vocab)
        for e in range(self.args.numEpochs):
            print("-----------------------Epoch :{} -----------------------------".format(e))
            for i in range(X_train.shape[0] // self.args.batchSize):
                print("l",end='')
                train_batch = X_train[i * self.args.batchSize:(i + 1) * self.args.batchSize]
                down_train_batch = self.down_vocab(train_batch, keep=self.use_vocab, reindex_dict=reindex_dict)
                if self.args.num_class==2:
                    train_y = Y_train[i * self.args.batchSize:(i + 1) * self.args.batchSize]
                else:
                    train_y = np.argmax(Y_train[i * self.args.batchSize:(i + 1) * self.args.batchSize], axis=1)
                bow_train_batch = self.Id2Bow(down_train_batch, use_vocab=self.use_vocab)
                # train step
                self.network.fit(bow_train_batch, train_y)
                # test step
                if i % 100 == 0:
                    down_test_batch = self.down_vocab(X_test,keep=self.use_vocab,reindex_dict=reindex_dict)
                    bow_test_batch = self.Id2Bow(down_test_batch, use_vocab=self.use_vocab)
                    y_test_pred = self.network.predict(bow_test_batch)
                    if self.args.num_class==2:
                        y_test_true = Y_test
                    else:
                        y_test_true = np.argmax(Y_test,axis=1)
                    acc = np.mean(y_test_pred==y_test_true)
                    print("----iter : {}  , test acc: {}----".format(i,acc))
                if i%1000==0:
                    print('save model wait for a few second')
                    restorePath = self.args.rootDir + '/checkpoint/'
                    restorePath = restorePath + 'svm_bow.pkl'
                    if self.args.model == 'text_svm':
                        with open(restorePath, 'wb') as f:
                            pickle.dump(self.network,f)

                del train_batch
                del down_train_batch
                del bow_train_batch
                gc.collect()

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
        out = np.zeros((len(inputs), use_vocab))
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                out[i, inputs[i, j]] += self.textData.Id2tfidf[inputs[i, j]]
        return out

    def step(self,batch):
        pass