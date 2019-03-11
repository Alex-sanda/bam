import tensorflow as tf
import math
import easydict
import jieba
from tqdm import tqdm
import os
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping,TensorBoard
from keras.preprocessing import sequence
import pickle

from src.textdata import TextData
from src.models.gru_attention import GRU_Attention
from src.models.text_cnn import Text_CNN
from src.models.text_rnn import Text_RNN
from src.models.text_rcnn import Text_RCNN
from src.models.vdcnn import VDCNN
from src.models.text_svm import Text_SVM
from src.models.tmn import TopicMemoryNetwork
from src.models.hierachical_attention import Hierarchical_Attention_On_Seq_And__Topic
from src.models.mlp import MLP
from src.models.feautre_study import FeatureStudy
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#arguments of the project
args = easydict.EasyDict({
    "batchSize": 256,
    'corpus':'tansongbo_restaurant',
    'createDataset':False,
    'datasetTag':"",
    'debug':False,
    'device':'gpu',
    'dropout':0.5,
    'filterVocab':1,
    'hidden_dims':50,
    'initEmbeddings':False,
    'keepAll':False,
    'learningRate':0.001,
    'maxLength':100,
    'modelTag':None,
    'numEpochs':300,
    'numLayers':2,
    'playDataset':None,
    'rootDir':'./',
    'saveEvery':2000,
    'seed':None,
    'skipLines':False,
    'softmaxSamples':1000,
    'test': False,
    'vocabularySize':40000,
    'watsonMode':False,
    '50':300,
    'base_path':'./data/',
    'reProduceSentence':False,
    'word2vec_path' :'./stc_nlpcc_fasttext50dim.txt',            #'./model/zimu_title_wiki_bulletscreen_googleNews300d.vector.txt',
    'embedding_dims':50,
    'using_word2vec': True,
    'attention_size': 100,
    'num_class':2,
    'early_stopping_step':10,
    'model_save_path':'./model/',
    'restore':True,
    'verbose':2,
    'visualizePath':"./visualizeAttention.txt",
    'divideCorpus':False,
    'divOutPath':"./",
    'attention_thr':0.1,
    'load_stc':False,
    'svm_feature':'Bow',   #'Bow',#TF-IDF
    'hierachical_attention_without_ntm':False,
    'optimize_ntm': True,
    'set_weights':True,
    'model': 'hierachical_attention',    #'text_rcnn' #'text_cnn'  #'gru_attention'
})

TASK_NAME='emotion_recognizer'
using_mutiple_kernels = True
if args.model == 'gru_attention':
    args.kfold_weights_path = './checkpoint/%s-%s-%s.{epoch:02d}-{val_emotion_predict_f1:.2f}-using_mutiple_kernels_%s-using_word2vec_%s.hdf5' %(TASK_NAME,args.model,args.corpus,using_mutiple_kernels,args.using_word2vec)
else:
    args.kfold_weights_path = './checkpoint/%s-%s-%s.{epoch:02d}-{val_acc:.2f}-using_mutiple_kernels_%s-using_word2vec_%s.hdf5' %(TASK_NAME,args.model,args.corpus,using_mutiple_kernels,args.using_word2vec)


Model_Select={
    'gru_attention':GRU_Attention,
    'text_cnn':Text_CNN,
    'text_rnn':Text_RNN,
    'text_rcnn':Text_RCNN,
    'vdcnn':VDCNN,
    'text_svm':Text_SVM,
    'tmn':TopicMemoryNetwork,
    'hierachical_attention':Hierarchical_Attention_On_Seq_And__Topic,
    'mlp':MLP,
    'feature_study':FeatureStudy,
}

class EmotionRecognizer:
    def __init__(self):
        self.args = None
        self.args = args
        self.globalStep = 0
        self.modelName = self.args.rootDir + "/save/" + "model_save_{}_".format(self.args.corpus)

    def main(self,args=None):
        print("")
        print("This is Emotion Recognizer : Inference and Separate")
        print("")

        if self.args.test is not None:
            print('mode: %s.' % (self.args.test))
        else:
            print('mode: train')

        #load corpus data
        self.textData = TextData(self.args)

        #build computational graph
        print("build computational graph...")
        with tf.device(self.getDevice()):
            if self.args.model == 'gru_attention':
                self.model = GRU_Attention(self.args,self.textData)
            else:
                self.model = Model_Select[self.args.model](self.args,self.textData)

        # restore previous model or something
        self.managePreviousModel()

        #call backs
        if self.args.model =='gru_attention':
            self.model.callbacks = [
                EarlyStopping(monitor='val_emotion_predict_f1', patience=self.args.early_stopping_step, mode='max', verbose=2),
                ModelCheckpoint(self.args.kfold_weights_path, monitor='val_emotion_predict_f1', save_best_only=True, mode='max',verbose=2,save_weights_only=True),
                TensorBoard(log_dir=self.args.model_save_path + "log/", histogram_freq=3)
            ]
        else:
            print("ca")
            self.model.callbacks = [
                EarlyStopping(monitor='val_acc', patience=self.args.early_stopping_step, mode='max', verbose=2),
                ModelCheckpoint(self.args.kfold_weights_path, monitor='val_acc', save_best_only=True, mode='max',verbose=2,save_weights_only=True),
                TensorBoard(log_dir=self.args.model_save_path + "log/", histogram_freq=3)
            ]


        if self.args.test:
            if self.args.model == 'gru_attention':
                if self.args.divideCorpus:
                    # divide stc corpus and make knowledge base
                    stcTrainInId_post = self.textData.stc_train_post
                    stcTrainInId_res = self.textData.stc_train_res
                    stcTestInId_post = self.textData.stc_test_post
                    stcTestInId_res = self.textData.stc_test_res
                    ecmInId = self.textData.ecm_test

                    self.constructFullKnowledgeBase(stcTrainInId_post + stcTestInId_post,
                                               stcTrainInId_res + stcTestInId_res,
                                               [0, 1, 2, 3, 4, 5],
                                               "./FullKnowledgeBase.txt")
                    self.create_TopicKnowledge_And_ExpressionKnowledge_FromFullBase("FullKnowledgeBase.txt","topicKnowLedgeBase.txt","emotionExpressionKnowledge")
                else:
                    self.mainTest()
        else:
            self.mianTrain()

    def create_TopicKnowledge_And_ExpressionKnowledge_FromFullBase(self,fullBasePath,TKOutpath,EKOutBasePath):
        idx = []
        post_sentence = []
        post_topic = []
        post_emo = []
        post_PreL = []

        res_sentence = []
        res_topic = []
        res_emo = []
        res_PreL = []
        with open(fullBasePath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                idd, ps, pt, pe, pp, rs, rt, re, rp = line.strip().split('\t')
                idx.append(int(idd))

                post_sentence.append(ps)
                post_topic.append(pt)
                post_emo.append(pe)
                post_PreL.append(pp)

                res_sentence.append(rs)
                res_topic.append(rt)
                res_emo.append(re)
                res_PreL.append(rp)
        with open(TKOutpath, 'w', encoding='utf-8') as tkp:
            for idd, ps, rt in zip(idx, post_sentence, res_topic):
                tkp.write(str(idd) + '\t' + ps + '\t' + rt + '\n')
        select = {}
        for pt, pe, pp in zip(post_topic, post_emo, post_PreL):
            if pp in select.keys():
                select[pp].append((str(len(select[pp])), pt, pe))
            else:
                select[pp] = [(str(0), pt, pe)]

        for rt, re, rp in zip(res_topic, res_emo, res_PreL):
            if rp in select.keys():
                select[rp].append((str(len(select[rp])), rt, re))
            else:
                select[rp] = [(str(0), rt, re)]

        for k in select.keys():
            with open(EKOutBasePath + "_label_" + k + ".txt", 'w', encoding='utf-8') as f:
                for ek_item in select[k]:
                    f.write('\t'.join(ek_item) + '\n')


    def mianTrain(self):
        print("")
        print("Start Training")
        print("")

        try:
            # restore training data
            (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST), classes = self.textData.getTrainAndTest()

            # training process
            if self.args.model == 'gru_attention':
                self.model.network.fit(X_TRAIN, {'emotion_predict': Y_TRAIN,
                    'attention_weights': np.zeros((X_TRAIN.shape[0], self.args.maxLength))},
                    batch_size=256, epochs=self.args.numEpochs, shuffle=True, verbose=self.args.verbose,
                    validation_data=[X_TEST, [Y_TEST, np.zeros((X_TEST.shape[0], self.args.maxLength))]],
                    callbacks=self.model.callbacks)
            elif self.args.model == 'vdcnn':
                X_TRAIN = sequence.pad_sequences(X_TRAIN,maxlen=64,value=self.textData.padToken,padding='pre')
                X_TEST = sequence.pad_sequences(X_TEST, maxlen=64, value=self.textData.padToken, padding='pre')
                self.model.network.fit(X_TRAIN, Y_TRAIN,
                    batch_size=256, epochs=self.args.numEpochs, shuffle=True, verbose=self.args.verbose,
                    validation_data=[X_TEST, Y_TEST],
                    callbacks=self.model.callbacks)
            elif self.args.model == 'text_svm' or self.args.model=='tmn' or self.args.model == 'hierachical_attention' or self.args.model=='mlp' or self.args.model=='feature_study':
                self.model.fit(X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,optimize_ntm=self.args.optimize_ntm)
            else:
                self.model.network.fit(X_TRAIN, Y_TRAIN,
                    batch_size=256, epochs=self.args.numEpochs, shuffle=True, verbose=self.args.verbose,
                    validation_data=[X_TEST, Y_TEST],
                    callbacks=self.model.callbacks)

        except(KeyboardInterrupt, SystemExit):  # Ctrl +c 退出训练
            print("Interrupt from System or User,Exiting the program")

        self._saveModel() #保存训练进度

    def mainTest(self):
        print("")
        print("Start Testing on test set")
        print("")
        (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST), classes = self.textData.getTrainAndTest()

        print("Doing inference on test set")
        preLabel,trueLabel,emoTestPred,emoTestAttentionWeights = self.model.inference(X_TEST,classes,labelOneHot=Y_TEST)

        #visualizePath = "./visualizeAttention.txt"
        print("visualize attention : ouput file :{}".format(self.args.visualizePath))
        visualizeRes = self.model.visualizeAttention(X_TEST,preLabel,trueLabel,emoTestPred,emoTestAttentionWeights)
        with open(self.args.visualizePath,'w',encoding='utf-8') as f:
            for res in visualizeRes:
                f.write("\t".join([str(e) for e in res])+"\n")

    def constructFullKnowledgeBase(self,post, response, classes, outPath):
        # 为了得到原始的字符串
        f_post = open(os.path.join(self.textData.stcDir, "repos/repository/stc2-repos-id-post"), 'r', encoding='utf-8')
        f_res = open(os.path.join(self.textData.stcDir, "repos/repository/stc2-repos-id-cmnt"), 'r', encoding='utf-8')
        f_post.seek(0)
        f_res.seek(0)

        with open(outPath, 'w', encoding='utf-8') as f:
            batch_size = self.args.batchSize
            numBatches = math.ceil(len(post) / batch_size)
            for i in tqdm(range(numBatches)):
                if i == numBatches - 1:
                    post_batch = post[i * batch_size:]
                    res_batch = response[i * batch_size:]
                else:
                    post_batch = post[i * batch_size:(i + 1) * batch_size]
                    res_batch = response[i * batch_size:(i + 1) * batch_size]
                post_batch = sequence.pad_sequences(post_batch, maxlen=self.args.maxLength, value=self.textData.padToken,padding='pre')
                res_batch = sequence.pad_sequences(res_batch, maxlen=self.args.maxLength, value=self.textData.padToken, padding='pre')

                # get raw string for post/res batch
                post_str_batch = []
                res_str_batch = []
                for p in range(len(post_batch)):
                    post_str_batch.append(f_post.readline().strip().split('\t')[1])
                    res_str_batch.append(f_res.readline().strip().split('\t')[1])

                # predict data and get attention
                post_PreL, _, _, post_Attention = self.model.inference(post_batch, classes)
                res_PreL, _, _, res_Attention = self.model.inference(res_batch, classes)

                # divid corpus with attention
                post_topic, post_sentence, post_emo = self.divCorpusWithAttention(post_batch, post_str_batch, post_Attention)
                res_topic, res_sentence, res_emo = self.divCorpusWithAttention(res_batch, res_str_batch, res_Attention)

                for j in range(len(post_batch)):
                    idx = i * batch_size + j
                    f.write(str(idx) + '\t' + post_sentence[j] + '\t' + post_topic[j] + '\t' + post_emo[j] + '\t' + str(post_PreL[j])+
                                        '\t'+ res_sentence[j] + '\t' + res_topic[j] + '\t' + res_emo[j] + '\t' + str(res_PreL[j]) + '\n'
                    )

    def divCorpusPipline(self,data,classes,outPath,label=None):
        selection = {}
        batch_size = self.args.batchSize
        numBatches = math.ceil(data.shape[0]/batch_size)
        for i in tqdm(range(numBatches)):
            batchData = None
            batchLabel= None
            if i== numBatches-1:
                batchData = data[i*batch_size:]
                if label is not None:
                    batchLabel = label[i*batch_size:]
            else:
                batchData = data[i*batch_size:(i+1)*batch_size]
                if label is not None:
                    batchLabel = label[i*batch_size:(i+1)*batch_size]

            # predict data and get attention
            batchPreL, batchLabel, batchtPred, batchAttention = self.model.inference(batchData, classes, labelOneHot=batchLabel)

            #divide sentences into topic and emotion expression
            batchTopic,batchSentence,batchEmoExp = self.divCorpusWithAttention(batchData,batchAttention,self.args.attention_thr)

            #group sentences by label
            if batchLabel is not None:
                self.groupSentenceByLabel(selection, batchTopic, batchSentence, batchEmoExp, batchLabel) #true label is given
            else:
                self.groupSentenceByLabel(selection, batchTopic, batchSentence, batchEmoExp, batchPreL) #use predict label

        #output division result
        for key,texts in selection.items():
            with open(outPath+"emotion."+str(key)+".txt",'w',encoding='utf-8') as f:
                for t in texts:
                    f.write(t+"\n")
        return

    def divCorpusWithAttention(self,dataInId, dataStr, attention, thr=1.2):
        topic = []
        emotionExpression = []
        sentence = []

        for i in range(dataInId.shape[0]):
            att_w = attention[i].tolist()
            sentence_idx = dataInId[i].tolist()
            sentence_raw = dataStr[i]

            # some what preprocessing on attention weights
            att_w = self.processAttention(att_w)

            # divide sentence with attention_weights
            try:
                s_topic, s_raw, s_emoexp = self.divSentenceWithAttention(sentence_idx, sentence_raw, att_w, thr)
            except:
                print("error in divSentenceWithAttention")
                continue
            # store
            topic.append(s_topic)
            emotionExpression.append(s_emoexp)
            sentence.append(s_raw)

        return topic, sentence, emotionExpression

    def divSentenceWithAttention(self,sentenceInId, sentenceStr, att, thr):
        """
        :param sentence:
        :param att:
        :param thr:
        :return:
        """
        assert len(sentenceInId) == len(att)
        topicStr = []
        emotionExpStr = []

        sentence = [w for w in jieba.cut(sentenceStr) if w != ' ']
        # 过滤 padToken
        sent_posi = 0
        for wid, at in zip(sentenceInId, att):
            if self.textData.padToken == wid:
                continue
            word = sentence[sent_posi]
            if at > thr/len(sentence):
                emotionExpStr.append(word)
            else:
                topicStr.append(word)
            sent_posi += 1

        topic = " ".join(topicStr)
        emo_expression = " ".join(emotionExpStr)
        sentenceStr = " ".join(sentence)
        return topic, sentenceStr, emo_expression

    def divCorpusInIDWithAttention(self,data,attention,thr = 0.1):
        topic = []
        emotionExpression = []
        sentence = []

        for i in range(data.shape[0]):
            att_w = attention[i].tolist()
            sentence_idx = data[i].tolist()

            #some what preprocessing on attention weights
            att_w = self.processAttention(att_w)

            # divide sentence with attention_weights
            try:
                s_topic,s_raw,s_emoexp = self.divSentenceWithAttention(sentence_idx,att_w,thr)
            except:
                print("error in divSentenceWithAttention")
                continue
            #store
            topic.append(s_topic)
            emotionExpression.append(s_emoexp)
            sentence.append(s_raw)

        return topic,sentence,emotionExpression

    def divSentenceInIDWithAttention(self,sentenceInId,att,thr):
        assert len(sentenceInId)==len(att)

        sentenceStr = []
        topicStr = []
        emotionExpStr = []

        for wid,at in zip(sentenceInId,att):
            if self.textData.padToken == wid:
                continue
            word = self.textData.id2word[wid] if wid in self.textData.id2word.keys() else '<unknown>'
            if at > thr:
                emotionExpStr.append(word)
            else:
                topicStr.append(word)
            sentenceStr.append(word)

        sentence = " ".join(sentenceStr)
        topic =  " ".join(topicStr)
        emo_expression = " ".join(emotionExpStr)
        return topic,sentence,emo_expression


    def processAttention(self,attention):
        return attention

    def groupSentenceByLabel(self,selection,topic,sentence,emo_expression,labels):
        assert len(topic)==len(sentence)
        assert len(topic)==len(emo_expression)
        assert len(topic)==len(labels)

        for t,s,e,l in zip(topic,sentence,emo_expression,labels):
            if l in selection.keys():
                selection[l].append(t+"\t"+s+"\t"+e+"\t"+str(l))
            else:
                selection[l] = [t+"\t"+s+"\t"+e+"\t"+str(l)]
        return

    def managePreviousModel(self):
        #implement restore only
        restorePath = self.args.rootDir + '/checkpoint/'
        if os.path.exists(restorePath):
            print("Found Previous Trained Model from {}".format(restorePath))
            try:
                print("This are model weights to be restore :")
                print(os.listdir(restorePath))

                if self.args.model=='text_svm':
                    restoreModelPath = restorePath + input(">").strip()
                    with open(restoreModelPath,'rb') as f:
                        self.model.network = pickle.load(f)
                elif self.args.model == 'tmn' or self.args.model=='hierachical_attention' or self.args.model=='feature_study':
                    if self.args.model=='hierachical_attention' and not self.model.joint_training :
                        ntm_restorePath = restorePath + input("Neural Topic Model weights>")
                        self.model.neural_topic_model.load_weights(ntm_restorePath)
                    elif self.args.model == 'feature_study' and self.model.feature =='NTM':
                        ntm_restorePath = restorePath + input("Neural Topic Model weights>")
                        self.model.neural_topic_model.load_weights(ntm_restorePath)
                    network_restorePath = restorePath + input("CLF Model weights>")
                    self.model.network.load_weights(network_restorePath)
                else:
                    restoreModelPath = restorePath + input(">").strip()
                    self.model.network.load_weights(restoreModelPath)
            except Exception as e:
                print("fall to restore the previous model,check the corresponding error")
                print(e)
        else:
            print("Pretrained Model Not Found,Continue")

    def _saveModel(self):
        print("Saving the model please wait")
        restorePath = self.args.rootDir + '/checkpoint/'
        if self.args.model == 'text_svm':
            restorePath = restorePath + input("请输入模型信息>") + ".h5"
            with open(restorePath,'wb') as f:
                pickle.dump(self.model.network,f)
        elif self.args.model == 'tmn'or self.args.model=='hierachical_attention' or self.args.model=='feature_study':
            ntm_restorePath = restorePath + input("请输入Neural Topic Model信息>") + ".h5"
            self.model.neural_topic_model.save_weights(ntm_restorePath)
            network_restorePath = restorePath + input("请输入CLF Model信息>") + ".h5"
            self.model.network.save_weights(network_restorePath)
        else:
            restorePath = restorePath + input("请输入模型信息>") + ".h5"
            self.model.network.save_weights(restorePath)
        print("model saved")

    def getDevice(self):
        if self.args.device == 'cpu':
            return '/cpu:0'
        elif self.args.device == 'gpu':
            return '/gpu:0'
        elif self.args.device is None:  # No specified device (default)
            return None
        else:
            print('Warning: Error in the device name: {}, use the default device'.format(self.args.device))
            return None
