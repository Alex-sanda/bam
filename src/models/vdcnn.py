import keras
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, BatchNormalization, Activation, Add, MaxPooling1D, Dense, Flatten
from keras.engine.topology import get_source_inputs
from keras.optimizers import SGD
from custom_layer.k_maxpooling import KMaxPooling
from gensim.models.keyedvectors import KeyedVectors
from utils import metric
import numpy as np

class VDCNN:
    def __init__(self,args,textData):
        print("-----------------------")
        print("Model : VDCNN ::Very Deep Convolutional Networks for Text Classification")
        print("-----------------------")
        self.args = args
        self.textData = textData

        #model settings
        self.depth = 9
        self.shortcut = True
        self.pool_type = 'max'
        self.input_length = 64

        if self.depth == 9:
            self.num_conv_blocks = (1,1,1,1)
        elif self.depth == 17:
            self.num_conv_blocks = (2,2,2,2)
        elif self.depth == 29:
            self.num_conv_blocks = (5,5,2,2)
        elif self.depth == 49:
            self.num_conv_blocks = (8,8,5,3)
        else:
            raise ValueError('unsupported depth for VDCNN.')

        #build the network in keras code
        self.network = None
        self.callbacks = None
        self.buildModel()

    def buildModel(self):
        def base_block(inputs,filters,kernel_size=3,shortcut=False):
            conv1 = Conv1D(filters=filters,kernel_size=kernel_size,strides=1,padding='same')(inputs)
            bn1 = BatchNormalization()(conv1)
            relu1 = Activation('relu')(bn1)

            conv2 = Conv1D(filters=filters,kernel_size=kernel_size,strides=1,padding='same')(relu1)
            out = BatchNormalization()(conv2)
            if shortcut:
                out = Add()([out,inputs])
            out = Activation('relu')(out)
            return out

        def conv_and_down_sample_block(inputs,filters,kernel_size=3,shortcut=False,pool_type='max',sorted=True,stage=1):
            conv1 = Conv1D(filters=filters,kernel_size=kernel_size,strides=1,padding='same')(inputs)
            bn1 = BatchNormalization()(conv1)
            relu1 = Activation('relu')(bn1)

            conv2 = Conv1D(filters=filters,kernel_size=kernel_size,strides=1,padding='same')(relu1)
            out = BatchNormalization()(conv2)
            if shortcut:
                residual = Conv1D(filters=filters,kernel_size=1,strides=2,name='shortcut_conv1d_{}'.format(stage))(inputs)
                residual = BatchNormalization(name='shortcut_batch_normalization_{}'.format(stage))(residual)
                out = downsample(out,pool_type=pool_type,sorted=sorted,stage=stage)
                out = Add()([out,residual])
                out = Activation('relu')(out)
            else:
                out = Activation('relu')(out)
                out = downsample(out,pool_type=pool_type,sorted=sorted,stage=stage)
            if pool_type is not None:
                out = Conv1D(filters=filters*2,kernel_size=1,strides=1,padding='same',name='1_1_conv_{}'.format(stage))(out)
                out = BatchNormalization()(out)
            return out

        def downsample(inputs,pool_type='max',sorted=True,stage=1):
            if pool_type =='max':
                out = MaxPooling1D(pool_size=3,strides=2,padding='same',name='pool_{}'.format(stage))(inputs)
            elif pool_type == 'k_max':
                k = int(inputs._keras_shape[1]/2)
                out = KMaxPooling(k=k,sorted=sorted,name='pool_{}'.format(stage))
            elif pool_type == 'conv':
                out = Conv1D(filters=inputs._keras_shape[-1],kernel_size=3,strides=2,padding='same',name='pool_{}'.format(stage))(inputs)
                out = BatchNormalization()(out)
            elif pool_type is None:
                out = inputs
            else:
                raise ValueError('unsurported pool type!')
            return out

        #word2vec settings
        def useword2vec(word_index):
            # KeyedVectors.load_word2vec_format("/data/word2vec/word2vec/GoogleNews-vectors-negative300.bin", binary=True)
            word_vectors = KeyedVectors.load_word2vec_format(self.args.word2vec_path)
            embedding_matrix = np.zeros((len(word_index) + 1, self.args.embedding_dims))
            mu, sigma = 0, 0.1  # 均值与标准差
            rarray = np.random.normal(mu, sigma, self.args.embedding_dims)
            for word, i in word_index.items():
                if word in word_vectors.vocab:
                    embedding_vector = word_vectors[word]
                else:
                    embedding_vector = rarray
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector[:self.args.embedding_dims]

            return embedding_matrix

        if self.args.using_word2vec:
            print('Using word2vec to initialize embedding layer ... ')
            embedding_mat = useword2vec(self.textData.word2id)

            embedding_layer = Embedding(embedding_mat.shape[0],
                                        embedding_mat.shape[1],
                                        weights=[embedding_mat],
                                        mask_zero=False,
                                        trainable=False,
                                        input_length=self.input_length)
        else:
            embedding_layer = Embedding(len(self.textData.word2id.keys()) + 1,  # max_features,
                                        self.args.embedding_dims,
                                        mask_zero=False,
                                        trainable=False,
                                        input_length=self.input_length)

        model_input = Input(shape=(self.input_length,))
        sequence_embedding = embedding_layer(model_input)
        out = Conv1D(filters=64,kernel_size=3,strides=1,padding='same',name='temp_conv')(sequence_embedding)

        # VDCNN Structure
        # Convolutional Block 64
        for _ in range(self.num_conv_blocks[0]-1):
            out = base_block(out,filters=64,kernel_size=3,shortcut=self.shortcut)
        out = conv_and_down_sample_block(out,filters=64,kernel_size=3,shortcut=self.shortcut,pool_type=self.pool_type,stage=1)

        # Convolutional Block 128
        for _ in range(self.num_conv_blocks[1]-1):
            out = base_block(out,filters=128,kernel_size=3,shortcut=self.shortcut)
        out = conv_and_down_sample_block(out,filters=128,kernel_size=3,shortcut=self.shortcut,pool_type=self.pool_type,stage=2)

        # Convolutional Block 256
        for _ in range(self.num_conv_blocks[2]-1):
            out = base_block(out,filters=256,kernel_size=3,shortcut=self.shortcut)
        out = conv_and_down_sample_block(out,filters=256,kernel_size=3,shortcut=self.shortcut,pool_type=self.pool_type,stage=3)
        
        # Convolutional Block 512
        for _ in range(self.num_conv_blocks[3]-1):
            out = base_block(out,filters=512,kernel_size=3,shortcut=self.shortcut)
        out = conv_and_down_sample_block(out,filters=512,kernel_size=3,shortcut=self.shortcut,pool_type=self.pool_type,stage=4)




        # k-maxpooling with k=8
        out = KMaxPooling(k=2,sorted=True)(out)
        out = Flatten()(out)

        # Dense Layers
        out = Dense(2048,activation='relu')(out)
        out = Dense(2048,activation='relu')(out)
        if self.args.num_class==2:
            out = Dense(1, activation='sigmoid', name='sigmoid')(out)
            loss = 'binary_crossentropy'
        else:
            out = Dense(self.args.num_class,activation='softmax')(out)
            loss = 'categorical_crossentropy'

        model = Model(inputs=model_input,outputs=out)
        self.network = model

        metrics = ['accuracy', metric.precision, metric.recall, metric.f1]
        model.compile(optimizer=SGD(lr=self.args.learningRate, momentum=0.9), loss=loss, metrics=metrics)


        print(self.network.summary())

if __name__ =='__main__':
    VDCNN = VDCNN()

