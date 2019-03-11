import os
import json


class IQIYI:
    def __init__(self, dirName):
        self.trainSamples = []
        self.testSamples = []

        IQIYI_FIELDS = ["label","text"]

        self.trainSamples = self.loadSamples(os.path.join(dirName, "train.txt"), IQIYI_FIELDS)
        self.testSamples  = self.loadSamples(os.path.join(dirName, "test.txt"), IQIYI_FIELDS)

    def loadSamples(self, fileName, field):
        conversations = []
        with open(fileName, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # The training dataset looks like: [[[post,post_label],
                # [response,response_label]],[[post,post_label],[response,response_label]],…].
                # There are about 1,110,000 pairs in the training data
                info = line.strip().split("_label_")

                convObj = {}  #add post
                convObj["lines"] = []
                convObj["lines"].append(dict(zip(field, info)))
                conversations.append(convObj)

        return conversations

    def getSamples(self):
        return self.trainSamples,self.testSamples