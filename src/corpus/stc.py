import os
import json


class STC:
    def __init__(self, dirName):
        self.trainSamples = []
        self.testSamples = []
        self.ecm_test_data = []

        STC_POST_FIELDS = ["post_id", "text","post_label"]
        STC_RES_FIELDS = ["res_id", "text","response_label"]
        ECM_TEST_DATA_FIELDS = ["text", "post_label"]

        self.ecm_test_data = self.load_ecm_test_data(os.path.join(dirName, "ecm_test_data.txt"), ECM_TEST_DATA_FIELDS)
        self.trainSamples, self.testSamples = self.loadSamples(os.path.join(dirName, "repos/repository/stc2-repos-id-post"),
                                                               os.path.join(dirName, "repos/repository/stc2-repos-id-cmnt"),
                                                               STC_POST_FIELDS,
                                                               STC_RES_FIELDS)

    def loadSamples(self, post_fileName,res_fileName, post_field, res_field):
        samples = []
        f_post = open(post_fileName,'r',encoding='utf-8')
        f_res = open(res_fileName,'r',encoding='utf-8')
        for pline,rline in zip(f_post.readlines(),f_res.readlines()):
            # The training dataset looks like: [[[post,post_label],
            # [response,response_label]],[[post,post_label],[response,response_label]],…].
            # There are about 1,110,000 pairs in the training data
            post_info = pline.strip().split('\t')
            res_info = rline.strip().split('\t')
            if len(post_info)<=2:
                post_info.append(-1)
                res_info.append(-1)

            convObj = {}
            convObj["lines"] = []
            convObj["lines"].append(dict(zip(post_field, post_info)))
            convObj["lines"].append(dict(zip(res_field, res_info)))
            samples.append(convObj)
        trainSamples = samples[:len(samples) - 5000]
        testSamples = samples[-5000:]
        f_post.close()
        f_res.close()
        return trainSamples, testSamples

    def load_ecm_test_data(self, fileName, field):
        samples = []
        with open(fileName, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                try:
                    info = line.strip().split('\t')
                    info[1] = int(info[1])

                    convObj = {}
                    convObj['lines'] = []
                    convObj['lines'].append(dict(zip(field, info)))
                    samples.append(convObj)
                except:
                    continue
        return samples

    def getSamples(self):
        return self.trainSamples, self.testSamples

    def get_ecm_test_data(self):
        return self.ecm_test_data