import os
import json


class NLPCC:
    def __init__(self, dirName):
        self.trainSamples = []
        self.testSamples = []

        NLPCC_FIELDS = ["text", "label"]

        self.trainSamples = self.loadSamples(os.path.join(dirName, "train_data.json"), NLPCC_FIELDS)
        self.testSamples  = self.loadTestSamples(os.path.join(dirName, "test_data.txt"),NLPCC_FIELDS)

    def loadSamples(self, fileName, field):
        conversations = []
        with open(fileName, 'r', encoding='utf-8') as f:
            for line in json.loads(f.read()):
                # The training dataset looks like: [[[post,post_label],
                # [response,response_label]],[[post,post_label],[response,response_label]],…].
                # There are about 1,110,000 pairs in the training data
                post_info = line[0]
                res_info = line[1]

                convObj = {}  #add post
                convObj["lines"] = []
                convObj["lines"].append(dict(zip(field, post_info)))
                conversations.append(convObj)

                convObj = {} #add response info
                convObj["lines"] = []
                convObj["lines"].append(dict(zip(field, res_info)))
                conversations.append(convObj)
        return conversations

    def loadTestSamples(self,fileName,field):
        conversations = []
        with open(fileName, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # The training dataset looks like: [[[post,post_label],
                # [response,response_label]],[[post,post_label],[response,response_label]],…].
                # There are about 1,110,000 pairs in the training data
                info = line.strip().split('\t')
                try:
                    info[1] = int(info[1])
                except:
                    continue
                convObj = {}  #add post
                convObj["lines"] = []
                convObj["lines"].append(dict(zip(field, info)))
                conversations.append(convObj)

        return conversations

    def getSamples(self):
        return self.trainSamples,self.testSamples