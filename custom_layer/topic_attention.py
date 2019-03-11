from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, merge, Layer
from keras.layers import add,multiply,dot
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.models import model_from_json
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
from keras import backend as K

class Attention_Over_Latent_Topic(Layer):
    def __init__(self, context_emb_size,topic_emb_size,combine_method='add', **kwargs):
        self.context_emb_size = context_emb_size
        self.topic_emb_size = topic_emb_size
        self.combine_method = combine_method
        super(Attention_Over_Latent_Topic, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None


    def build(self, input_shape):
        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(self.context_emb_size, self.topic_emb_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(self.topic_emb_size,),
                                 initializer="zeros",
                                 trainable=True)
        super(Attention_Over_Latent_Topic, self).build(input_shape)

    def call(self, inputs, mask=None):
        """

        :param inputs: 1.context is context embedding of shape(_,seqL,context_emb)
                        2.theta is the feature for latent topic(representation of the internal state of Neural Topic Model)
                         of shape (_,K,topic_emb_size)
                        3.S internal representation for the latent topic (as match) shape(_,K,topic_emb_size)
                        4.V internal representation for the latent topic (as calculation) shape(_,K,topic_emb_size)
        :param mask: Not use for now
        :return:  sequence feature extracted from  latent topic model of shape(_,seqL,topic_emb_size)
        """
        context,theta,S,V = inputs
        context = K.tanh(K.dot(context,self.W) + self.b)
        et_raw = dot([context,S],axes=(2,2))
        if self.combine_method == 'add':
            et = add([et_raw,theta])
        elif self.combine_method == 'mul':
            et = multiply([et_raw,theta])
        else:
            raise ValueError('bad value for combine method !')
        at = K.softmax(et)
        output = dot([at,V],axes=(2,1))
        return [output,at,et_raw]

    def get_config(self):
        config = {'context_emb_size': self.context_emb_size,
                  'topic_emb_size': self.topic_emb_size,
                  'combine_method': self.combine_method
                  }
        base_config = super(Attention_Over_Latent_Topic, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], input_shape[0][1],self.topic_emb_size),(input_shape[0][0],input_shape[0][1],input_shape[1][1]),(input_shape[0][0],input_shape[0][1],input_shape[1][1])]