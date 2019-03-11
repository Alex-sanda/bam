"""
Loads the dialogue corpus, builds the vocabulary
"""

import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random
import string
import collections
import json
import jieba
from keras.preprocessing import sequence
from sklearn import preprocessing

#load corpus
from src.corpus.nlpcc import NLPCC
from src.corpus.iqiyi import IQIYI
from src.corpus.stc   import STC
from src.corpus.tansongbo_restaurant import Restaurant

class Batch:
    def __init__(self):
        #self.encoderSeqs=[]
        #self.contentSeqs  = []
        #self.attributeSeqs= []
        self.mergedSeqs   = []
        self.sentenceSeqs = []
        self.targetSeqs   = []
        self.weights      = []


class TextData:
    """Dataset class
    Warning: No vocabulary limit
    """
    availableCorpus = collections.OrderedDict([  # OrderedDict because the first element is the default choice
        ('nlpcc', NLPCC),
        ('iqiyi', IQIYI),
        ('tansongbo_restaurant',Restaurant),
    ])

    corpus_language = {
        'nlpcc':'chinese',
        'iqiyi':'chinese',
        'stc'  :'chinese',
        'tansongbo_restaurant':'chinese'
    }

    # init textdata
    def __init__(self, args):
        self.args = args
        self.corpusDir = os.path.join(self.args.rootDir, 'data', self.args.corpus)
        self.stcDir = os.path.join(self.args.rootDir,'data','stc')
        basePath = self._constructBasePath()
        self.fullSamplesPath = basePath + '.pkl'  # Full sentences length/vocab
        self.filteredSamplesPath = basePath + '-length{}-filter{}-vocabSize{}.pkl'.format(
            self.args.maxLength,
            self.args.filterVocab,
            self.args.vocabularySize,
        )  # Sentences/vocab filtered for this model

        self.padToken = -1  # Padding
        self.goToken = -1  # Start of sequence
        self.eosToken = -1  # End of sequence
        self.unknownToken = -1  # Word dropped from vocabulary
        self.mergedToken  = -1 #merge symbol of two sub sequence

        self.trainX = []  # 2d array containing each question and his answer [[input,target]]
        self.trainY = []
        self.testX  = []
        self.testY  = []

        self.stc_train_post= []
        self.stc_train_res = []
        self.stc_test_post = []
        self.stc_test_res  = []
        self.ecm_test = []

        self.word2id = {}
        self.id2word = {}  # For a rapid conversion (Warning: If replace dict by list, modify the filtering to avoid linear complexity with del)
        self.idCount = {}  # Useful to filters the words (TODO: Could replace dict by list or use collections.Counter)
        self.id2idf = None
        self.loadCorpus()

    def loadCorpus(self):
        """
        :load/create the conversation data
        """
        print("*****************")
        print("load corpus from", self.args.corpus)
        print("*****************")
        # datasetExist = os.path.
        dataExists = os.path.exists(self.fullSamplesPath)
        if not dataExists:  # 构造数据集
            print("construct full dataset..")
            corpusData = TextData.availableCorpus[self.args.corpus](self.corpusDir)
            trainSamples,testSamples = corpusData.getSamples()
            self.createFullCorpus(trainSamples,testSamples)
            self.saveDataset(self.fullSamplesPath)
        else:
            self.loadDataset(self.fullSamplesPath)
        if self.args.load_stc:
            print("Load stc dataset with nlpcc textData")
            stcExists = os.path.exists("./data/samples/nlpcc-recognize-stc.pkl")
            if not stcExists:
                stcData = STC(self.stcDir)
                stcTrainSamples, stcTestSamples = stcData.getSamples()
                ecmData = stcData.get_ecm_test_data()
                self.createSTCCorpus(stcTrainSamples, stcTestSamples, ecmData)
                self.saveSTCDataset()
            else:
                self.loadSTCDataset()
        if self.args.reProduceSentence:
            print("Train a model to memorize a sentence and reproduce it")
            def reProduce(samples):
                res = []
                for sample in samples:
                    res.append([sample[2],sample[1],sample[2],sample[3]])
                return res
            self.trainingSamples = reProduce(self.trainingSamples)

        # build idf dict
        self.id2idf = self.buildIdfDict()

    """
        def getSTCRawSentence(self):
            stcData = STC(self.stcDir)
            stcTrainSamples, stcTestSamples = stcData.getSamples()
            ecmData = stcData.get_ecm_test_data()
            train_raw,test_raw,ecm_raw = self.extract_raw_sentence(stcTrainSamples,stcTestSamples,ecmData)
            return train_raw,test_raw,ecm_raw
    
        def extract_raw_sentence(self,train_samples,test_samples,ecm_samples):
            train_raw = []
            test_raw = []
            ecm_raw = []
            for sample in tqdm(train_samples, desc='Extract conversations Train'): #training set
                train_raw.append(self.get_raw_sentence_from_sample(sample,type='train'))
            for sample in tqdm(test_samples, desc='Extract conversations Test'):  #test set
                test_raw.append(self.get_raw_sentence_from_sample(sample,type='test'))
            for sample in tqdm(ecm_samples, desc='Extract conversations ECM'): #training set
                ecm_raw.append(self.get_raw_sentence_from_sample(sample,type='ecm'))
            return train_raw,test_raw,ecm_raw
    
        def get_raw_sentence_from_sample(self,sample,type='train'):
            post = []
            res  = []
            if type == 'train' or type=='test':
                pText = sample['lines'][0]['text']
                rText = sample['lines'][1]['text']
    
                post = [w for w in jieba.cut(pText) if w!=' ']
                res  = [w for w in jieba.cut(rText) if w!=' ']
            elif type=='ecm':
                pText = sample['lines'][0]['text']
                post = [w for w in jieba.cut(pText) if w != ' ']
            return post,res
    
    """

    def createSTCCorpus(self,train,test,ecm):
        for sample in tqdm(train, desc='Extract conversations Train'): #training set
            self.extractSTCSample(sample,type='train')
        for sample in tqdm(test, desc='Extract conversations Test'):  #test set
            self.extractSTCSample(sample,type='test')
        for sample in tqdm(ecm, desc='Extract conversations ECM'): #training set
            self.extractSTCSample(sample,type='ecm')

    def extractSTCSample(self,samples,type='train'):
        # extract materials from conversation object
        if type == 'train' or type=='test':
            pText = samples['lines'][0]['text']
            pLabel = samples['lines'][0]['post_label']
            rText = samples['lines'][1]['text']
            rLabel  = samples['lines'][1]["response_label"]

            # word2id for content/attribute/sentence
            postText = self.extractText(pText,create=False)
            resText = self.extractText(rText,create=False)

        elif type == 'ecm':
            pText = samples['lines'][0]['text']
            pLabel = samples['lines'][0]['post_label']
            postText = self.extractText(pText,create=False)
            resText = postText


        if postText and resText:  # Filter wrong samples (if one of the list is empty)
            if type == 'train':
                self.stc_train_post.append(postText)
                self.stc_train_res.append(resText)
            elif type == 'test':
                self.stc_test_post.append(postText)
                self.stc_test_res.append(resText)
            elif type == 'ecm':
                self.ecm_test.append(postText)
            else:
                print("false configuration in extractSample function")

    def saveSTCDataset(self):
        with open("./data/samples/nlpcc-recognize-stc.pkl", 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'word2id': self.word2id,
                'id2word': self.id2word,
                'idCount': self.idCount,
                'stc_train_post' : self.stc_train_post,
                'stc_train_res': self.stc_train_res,
                'stc_test_post':self.stc_test_post,
                'stc_test_res' : self.stc_test_res,
                'ecm_test'  : self.ecm_test,
            }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadSTCDataset(self):
        stc_dataset_path = "./data/samples/nlpcc-recognize-stc.pkl"
        print('Loading dataset from {}'.format(stc_dataset_path))
        with open(stc_dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.stc_train_post = data['stc_train_post']
            self.stc_train_res = data['stc_train_res']
            self.stc_test_post = data['stc_test_post']
            self.stc_test_res  = data['stc_test_res']
            self.ecm_test = data['ecm_test']

    def _constructBasePath(self):
        """Return the name of the base prefix of the current dataset
        """
        path = os.path.join(self.args.rootDir, 'data' + os.sep + 'samples' + os.sep)
        path += 'dataset-{}'.format(self.args.corpus)
        if self.args.datasetTag:
            path += '-' + self.args.datasetTag
        return path

    def createFullCorpus(self, trainSample,testSample):
        """Extract all data from the given vocabulary.
        Save the data on disk. Note that the entire corpus is pre-processed
        without restriction on the sentence length or vocab size.
        """
        # Add standard tokens
        self.padToken = self.getWordId('<pad>')  # Padding (Warning: first things to add > id=0 !!)
        self.goToken = self.getWordId('<go>')  # Start of sequence
        self.eosToken = self.getWordId('<eos>')  # End of sequence
        self.unknownToken = self.getWordId('<unknown>')  # Word dropped from vocabulary
        self.mergedToken =  self.getWordId('<merge>')

        # Preprocessing data
        for sample in tqdm(trainSample, desc='Extract conversations'): #training set
            self.extractSample(sample,saveTo='train')
        for sample in tqdm(testSample, desc='Extract conversations'):  #test set
            self.extractSample(sample,saveTo='test')

    def getWordId(self, word, create=True):
        """Get the id of the word (and add it to the dictionary if not existing). If the word does not exist and
        create is set to False, the function will return the unknownToken value
        Args:
            word (str): word to add
            create (Bool): if True and the word does not exist already, the world will be added
        Return:
            int: the id of the word created
        """
        # english words to lower case
        if self.corpus_language[self.args.corpus]:
            word = word.lower()  # Ignore case

        # At inference, we simply look up for the word
        if not create:
            wordId = self.word2id.get(word, self.unknownToken)
        # Get the id if the word already exist
        elif word in self.word2id:
            wordId = self.word2id[word]
            self.idCount[wordId] += 1
        # If not, we create a new entry
        else:
            wordId = len(self.word2id)
            self.word2id[word] = wordId
            self.id2word[wordId] = word
            self.idCount[wordId] = 1

        return wordId

    def extractSample(self, samples,saveTo='train',type='train'):
        """Extract the sample lines from the conversations
        Args:
            conversation (Obj): a conversation object containing the lines to extract
            convObj['lines'] = {"content":content,"attribute":attribute,"sentence":sentence,"num":1.0}
        """
        # extract materials from conversation object
        if type=='train' or type=='test':
            try:
                label = samples['lines'][0]['label']
                text  = samples['lines'][0]['text']
            except:
                return

        # word2id for content/attribute/sentence
        inputText = self.extractText(text)


        if inputText:  # Filter wrong samples (if one of the list is empty)
            if saveTo == 'train':
                self.trainX.append(inputText)
                self.trainY.append(label)
            elif saveTo == 'test':
                self.testX.append(inputText)
                self.testY.append(label)
            else:
                print("false configuration in extractSample function")


    def extractText(self, line,create=True):
        """Extract the words from a sample lines
        Args:
            line (str): a line containing the text to extract
        Return:
            list<int>: the list of word ids of the sentence
        """
        sentence = []
        #cleaning job
        line = line.strip()
        #pattern = re.compile()

        # Tokenize
        if self.corpus_language[self.args.corpus] == 'english':
            sentenceToken = nltk.word_tokenize(line)
        elif self.corpus_language[self.args.corpus] == 'chinese':
            sentenceToken = [w for w in jieba.cut(line) if w != ' ']
        else:
            print("unkown language")
            return

        # word2id
        for i in range(len(sentenceToken)):
            sentence.append(self.getWordId(sentenceToken[i],create=create))

        return sentence

    def saveDataset(self, filename):
        with open(os.path.join(filename), 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'word2id': self.word2id,
                'id2word': self.id2word,
                'idCount': self.idCount,
                'trainX' : self.trainX,
                'trainY' : self.trainY,
                'testX'  : self.testX,
                'testY'  : self.testY
            }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def buildIdfDict(self):
        print("build idf dict")
        idf_vocab = {}
        total_corpus = self.getSamepleSize()
        for k,v in self.idCount.items():
            idf = math.log(total_corpus/ v)
            idf_vocab[k] = idf

        # mask down pad/go/eos/merge tokens
        idf_vocab[self.padToken] = 0
        idf_vocab[self.goToken] = 0
        idf_vocab[self.eosToken] = 0
        idf_vocab[self.mergedToken] = 0
        return idf_vocab

    def getIdfWeights(self,data):
        if self.id2idf is None:
            print('build tf-idf dict')
            self.id2idf = self.buildIdfDict()
        IDF_weights = []
        for sample in data:
            sample_w = [self.id2idf[w] for w in sample]
            IDF_weights.append(sample_w)
        return np.array(IDF_weights)

    def loadDataset(self,filename):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2id = data['word2id']
            self.id2word = data['id2word']
            self.idCount = data.get('idCount', None)
            self.trainX = data['trainX']
            self.trainY = data['trainY']
            self.testX  = data['testX']
            self.testY  = data['testY']

            self.padToken = self.word2id['<pad>']
            self.goToken = self.word2id['<go>']
            self.eosToken = self.word2id['<eos>']
            self.unknownToken = self.word2id['<unknown>']  # Restore special words
            self.mergedToken = self.word2id['<merge>']

    def getTrainAndTest(self):
        X_TRAIN = sequence.pad_sequences(self.trainX,maxlen=self.args.maxLength,value=self.padToken)
        X_TEST =  sequence.pad_sequences(self.testX,maxlen=self.args.maxLength,value=self.padToken)
        #process label to one-hot format
        lb = preprocessing.LabelBinarizer()
        lb.fit(self.trainY+self.testY)
        Y_TRAIN = lb.transform(self.trainY)
        Y_TEST  = lb.transform(self.testY)



        return (X_TRAIN,Y_TRAIN),(X_TEST,Y_TEST),(list(lb.classes_))

    def sentence2enco(self,sentence):
        if sentence == '' :
            return None
        #Tokenize the sentence
        if self.corpus_language[self.args.corpus] == 'english':
            sentenceToken = nltk.word_tokenize(sentence)
        elif self.corpus_language[self.args.corpus] == 'chinese':
            sentenceToken = [w for w in jieba.cut(sentence) if w!=' ']
        else:
            print("Error in sentence2enco : corpus language error")

        #确保句子长度不会超长
        if len(sentenceToken)>self.args.maxLength:
            print("Question too long! I could only get part of it")
            sentenceToken = sentenceToken[:self.args.maxLength]


        sentenceIDSeq = [self.getWordId(w,create=False) for w in sentenceToken]

        return sentenceIDSeq

    def sample2str(self,sample):
        resStr = []
        if sample is None or type(sample)!=list or len(sample)!=4:
            print("Format error : please check the sample input follow the format [[content],[attribute],[sentence],num]")
            return resStr
        else:
            field = ["content","attribute","sentence","num"]
            for i,name in enumerate(field):
                if i==3:
                    resStr.append((name,sample[i]))
                else:
                    resStr.append((name,[self.id2word[w] for w in sample[i]]))
        return resStr

    def visualizeBatch(self,batch):
        """"
        to print a batch
        """
        pass


    def deco2sentence(self,decoderOutputs):
        """Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>):
        """
        sequence = []

        #Choose the words with the highest prediction score
        for out in decoderOutputs:
            sequence.append(np.argmax(out))
        return sequence

    def sequence2str(self,sequence):
        """Convert a list of integer into a human readable string

        :param sequence: list of integer
        :return: string
        """
        if not sequence:
            return ''

        # Do ID to word
        sentence = []
        for wordID in sequence:
            if wordID == self.eosToken:
                break
            elif wordID != self.padToken and wordID != self.goToken:
                sentence.append(self.id2word[wordID])

        outString = None
        # detokenize
        if self.corpus_language[self.args.corpus] == 'chinese':
            outString = ''.join(sentence)
        elif self.corpus_language[self.args.corpus] == 'english':
            outString = ' '.join(sentence)

        return outString

    def getSamepleSize(self):
        return len(self.trainX)

    def getVocabularySize(self):
        return len(list(self.word2id.keys()))