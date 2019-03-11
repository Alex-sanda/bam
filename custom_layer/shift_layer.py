from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, merge, Layer
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

class Shift(Layer):
    def __init__(self, shift, **kwargs):
        self.shift = shift
        super(Shift, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None


    def build(self, input_shape):
        super(Shift, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        if self.shift>=0:
            x = x[:,self.shift:,:]
            pad_shape = (K.shape(x)[0],self.shift,K.shape(x)[2])
            x = K.concatenate([K.zeros(pad_shape),x],axis=1)
        else:
            x = x[:,:self.shift,:]
            pad_shape = (K.shape(x)[0],-self.shift,K.shape(x)[2])
            x = K.concatenate([x,K.zeros(pad_shape)],axis=1)
        return x

    def get_config(self):
        config = {'shift': self.shift}
        base_config = super(Shift, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1],input_shape[2])