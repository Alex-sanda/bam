import os
import random


class Restaurant:
    def __init__(self, dirName):
        self.trainSamples = []
        self.testSamples = []

        RESTAURANT_FIELDS = ["label", "text"]

        self.numTestSamples = 1000
        self.trainSamples,self.testSamples = self.loadSamples(dirName, RESTAURANT_FIELDS)

    def loadSamples(self, dirName, field):
        train_conversation = []
        test_conversation = []
        pos_conversations = []
        neg_conversations = []
        # readPos data
        with open(os.path.join(dirName,'pos.txt'), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                sample = line.strip().split('    ')
                sample[0] = 1

                convObj = {}  #add post
                convObj["lines"] = []
                convObj["lines"].append(dict(zip(field, sample)))
                pos_conversations.append(convObj)

        # readNeg data
        with open(os.path.join(dirName,'neg.txt'), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                sample = line.strip().split('    ')
                sample[0] = 0

                convObj = {}  #add post
                convObj["lines"] = []
                convObj["lines"].append(dict(zip(field, sample)))
                neg_conversations.append(convObj)

        #split train and test
        train_conversation = pos_conversations[:-round(self.numTestSamples*0.7)]+neg_conversations[:-round(self.numTestSamples*0.3)]
        test_conversation  = pos_conversations[-round(self.numTestSamples*0.7):]+neg_conversations[-round(self.numTestSamples*0.3):]

        random.shuffle(train_conversation)
        random.shuffle(test_conversation)

        return train_conversation,test_conversation

    def getSamples(self):
        return self.trainSamples,self.testSamples