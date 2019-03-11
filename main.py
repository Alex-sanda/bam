

"""
mail : 908337832@qq.com
author : Alex伞大
"""

""""
说明： 情感识别模块
核心算法： GRU + Attention
数据集：/qiyi/nlpcc
参考框架： deepQA
"""
from src import recognizer



if __name__ == "__main__":
    recognizer = recognizer.EmotionRecognizer()
    recognizer.main()