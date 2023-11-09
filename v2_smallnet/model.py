import this
from keras.models import Model
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.layers import Dense, Conv2D, Conv1D, Flatten, GlobalAveragePooling1D, GlobalAveragePooling2D, \
	GlobalMaxPooling1D, GlobalMaxPooling2D, BatchNormalization, LayerNormalization, Layer, Add, LSTM
from params import *


class TrainingModel(Model):
    def __init__(self):
        super().__init__()
        
        inputs = keras.layers.Input(shape=(SIZE_H, SIZE_X, SIZE_Y)) # this is for the 2d cnn layer
        initial_output = keras.layers.Conv2D(32, (3, 3), strides = 2, \
                                             padding = 'same', kernel_initializer = 'he_normal')(inputs)
        #act = tf.nn.leaky_relu
        act = tf.nn.relu

        # first resnet block without downsampling
        channels_1 = 32
        strides_1 = [1, 1]
        kernel_1 = (3, 3)
        res_1 = initial_output
        outputs_1 = keras.layers.Conv2D(channels_1, strides=strides_1[0], kernel_size=kernel_1, \
                           padding="same", kernel_initializer='he_normal')(initial_output)
        outputs_1 = act(outputs_1)
        outputs_1 = keras.layers.Conv2D(channels_1, strides=strides_1[1], kernel_size=kernel_1, \
                           padding="same", kernel_initializer='he_normal')(outputs_1)
        outputs_1 = keras.layers.Add()([outputs_1, res_1])
        outputs_1 = act(outputs_1)
        # end of first resnet block without downsampling

        # now the second resnet block, with downsampling
        channels_2 = 64
        strides_2 = [2, 1]
        kernel_2 = (3, 3)
        res_2 = outputs_1
        outputs_2 = keras.layers.Conv2D(channels_2, strides=strides_2[0], kernel_size=kernel_2, \
                           padding="same", kernel_initializer='he_normal')(outputs_1)
        outputs_2 = act(outputs_2)
        outputs_2 = keras.layers.Conv2D(channels_2, strides=strides_2[1], kernel_size=kernel_2, \
                           padding="same", kernel_initializer='he_normal')(outputs_2)
        # down sampling for residual too
        res_2 = keras.layers.Conv2D(channels_2, strides=2, kernel_size=(1, 1), \
                                  kernel_initializer='he_normal', padding="same")(res_2)
        outputs_2 = keras.layers.Add()([outputs_2, res_2])
        outputs_2 = act(outputs_2)
        # end of second resnet block with downsampling

        # now the third resnet block w/o downsampling
        channels_3 = 64
        strides_3 = [1, 1]
        kernel_3 = (3, 3)
        outputs_3 = keras.layers.Conv2D(channels_3, strides=strides_3[0], kernel_size=kernel_3, \
                           padding="same", kernel_initializer='he_normal')(outputs_2)
        outputs_3 = act(outputs_3)
        outputs_3 = keras.layers.Conv2D(channels_3, strides=strides_3[1], kernel_size=kernel_3, \
                           padding="same", kernel_initializer='he_normal')(outputs_3)
        outputs_3 = act(outputs_3)
        # end of second resnet block with downsampling

        # now the fourth resnet block, with downsampling
        channels_4 = 128
        strides_4 = [2, 1]
        kernel_4 = (3, 3)
        res_4 = outputs_3
        outputs_4 = keras.layers.Conv2D(channels_4, strides=strides_4[0], kernel_size=kernel_4, \
                           padding="same", kernel_initializer='he_normal')(outputs_3)
        outputs_4 = act(outputs_4)
        outputs_4 = keras.layers.Conv2D(channels_4, strides=strides_4[1], kernel_size=kernel_4, \
                           padding="same", kernel_initializer='he_normal')(outputs_4)
        # down sampling for residual too
        res_4 = keras.layers.Conv2D(channels_4, strides=2, kernel_size=(1, 1), \
                                  kernel_initializer='he_normal', padding="same")(res_4)
        outputs_4 = keras.layers.Add()([outputs_4, res_4])
        outputs_4 = act(outputs_4)
        # end of fourth resnet block with downsampling

        # now the final block, leading to HIDDEN_2D dimension
        channels_5 = 128
        strides_5 = [1, 1]
        kernel_5 = (3, 3)
        outputs_5 = keras.layers.Conv2D(channels_5, strides=strides_5[0], kernel_size=kernel_5, \
                           padding="same", kernel_initializer='he_normal')(outputs_4)
        outputs_5 = act(outputs_5)
        outputs_5 = keras.layers.Conv2D(channels_5, strides=strides_5[1], kernel_size=kernel_5, \
                           padding="same", kernel_initializer='he_normal')(outputs_5)
        outputs_5 = act(outputs_5)
        # end of final block

        # final pooling and flatten
        final_output = keras.layers.GlobalAveragePooling2D()(outputs_5)
        final_output = keras.layers.Flatten()(final_output)


        self.two_d_encoder = keras.Model(inputs=inputs, outputs=final_output)
        
        # HIDDEN_2D = 128
        # LSTM_DIM = 128
        # lstm_inputs = keras.layers.Input(shape=(SIZE_T, HIDDEN_2D))
        # lstm_out = LSTM(LSTM_DIM)(lstm_inputs)
        # outputs = Dense(OUT_DIM)(lstm_out)
        # self.time_slice_encoder = keras.Model(inputs=lstm_inputs, outputs=lstm_out)

        # fc_inputs = keras.layers.Input(shape = (LSTM_DIM,))
        # fc_out = Dense(8)(fc_inputs)
        # outputs = Dense(OUT_DIM)(fc_out)

        # self.final_dense = keras.Model(inputs = fc_inputs, outputs = outputs)

        HIDDEN_2D = 128
        LSTM_DIM = 128
        lstm_inputs = keras.layers.Input(shape=(SIZE_T, HIDDEN_2D))
        lstm_out = LSTM(LSTM_DIM)(lstm_inputs)
        outputs = Dense(OUT_DIM)(lstm_out)
        self.time_slice_encoder = keras.Model(inputs=lstm_inputs, outputs=outputs)
        
        
    def call(self, inputs):
        """
        input dimension: B*T*H*X*Y*1
        output dimension: B*output_dim
        """

        # out = tf.reshape(inputs, shape=[-1, SIZE_H, SIZE_X, SIZE_Y])
        # out = self.two_d_encoder(out)
        # out = tf.reshape(out, shape=[-1, SIZE_T, HIDDEN_2D])
        # out = self.time_slice_encoder(out)
        # out = self.final_dense(out)

        out = tf.reshape(inputs, shape=[-1, SIZE_H, SIZE_X, SIZE_Y])
        out = self.two_d_encoder(out)
        out = tf.reshape(out, shape=[-1, SIZE_T, HIDDEN_2D])
        out = self.time_slice_encoder(out)

        return out


class TrainingModel_v2(Model):
    def __init__(self):
        super().__init__()
        
        inputs = keras.layers.Input(shape=(SIZE_H, SIZE_X, SIZE_Y)) # this is for the 2d cnn layer
        initial_output = keras.layers.Conv2D(32, (3, 3), strides = 2, \
                                             padding = 'same', kernel_initializer = 'he_normal')(inputs)
        #act = tf.nn.leaky_relu
        act = tf.nn.relu

        # first resnet block without downsampling
        channels_1 = 32
        strides_1 = [1, 1]
        kernel_1 = (3, 3)
        res_1 = initial_output
        outputs_1 = keras.layers.Conv2D(channels_1, strides=strides_1[0], kernel_size=kernel_1, \
                           padding="same", kernel_initializer='he_normal')(initial_output)
        outputs_1 = act(outputs_1)
        outputs_1 = keras.layers.Conv2D(channels_1, strides=strides_1[1], kernel_size=kernel_1, \
                           padding="same", kernel_initializer='he_normal')(outputs_1)
        outputs_1 = keras.layers.Add()([outputs_1, res_1])
        outputs_1 = act(outputs_1)
        # end of first resnet block without downsampling

        # now the second resnet block, with downsampling
        channels_2 = 64
        strides_2 = [2, 1]
        kernel_2 = (3, 3)
        res_2 = outputs_1
        outputs_2 = keras.layers.Conv2D(channels_2, strides=strides_2[0], kernel_size=kernel_2, \
                           padding="same", kernel_initializer='he_normal')(outputs_1)
        outputs_2 = act(outputs_2)
        outputs_2 = keras.layers.Conv2D(channels_2, strides=strides_2[1], kernel_size=kernel_2, \
                           padding="same", kernel_initializer='he_normal')(outputs_2)
        # down sampling for residual too
        res_2 = keras.layers.Conv2D(channels_2, strides=2, kernel_size=(1, 1), \
                                  kernel_initializer='he_normal', padding="same")(res_2)
        outputs_2 = keras.layers.Add()([outputs_2, res_2])
        outputs_2 = act(outputs_2)
        # end of second resnet block with downsampling

        # now the third resnet block w/o downsampling
        channels_3 = 64
        strides_3 = [1, 1]
        kernel_3 = (3, 3)
        outputs_3 = keras.layers.Conv2D(channels_3, strides=strides_3[0], kernel_size=kernel_3, \
                           padding="same", kernel_initializer='he_normal')(outputs_2)
        outputs_3 = act(outputs_3)
        outputs_3 = keras.layers.Conv2D(channels_3, strides=strides_3[1], kernel_size=kernel_3, \
                           padding="same", kernel_initializer='he_normal')(outputs_3)
        outputs_3 = act(outputs_3)
        # end of second resnet block with downsampling

        # now the fourth resnet block, with downsampling
        channels_4 = 128
        strides_4 = [2, 1]
        kernel_4 = (3, 3)
        res_4 = outputs_3
        outputs_4 = keras.layers.Conv2D(channels_4, strides=strides_4[0], kernel_size=kernel_4, \
                           padding="same", kernel_initializer='he_normal')(outputs_3)
        outputs_4 = act(outputs_4)
        outputs_4 = keras.layers.Conv2D(channels_4, strides=strides_4[1], kernel_size=kernel_4, \
                           padding="same", kernel_initializer='he_normal')(outputs_4)
        # down sampling for residual too
        res_4 = keras.layers.Conv2D(channels_4, strides=2, kernel_size=(1, 1), \
                                  kernel_initializer='he_normal', padding="same")(res_4)
        outputs_4 = keras.layers.Add()([outputs_4, res_4])
        outputs_4 = act(outputs_4)
        # end of fourth resnet block with downsampling

        # now the final block, leading to HIDDEN_2D dimension
        channels_5 = 128
        strides_5 = [1, 1]
        kernel_5 = (3, 3)
        outputs_5 = keras.layers.Conv2D(channels_5, strides=strides_5[0], kernel_size=kernel_5, \
                           padding="same", kernel_initializer='he_normal')(outputs_4)
        outputs_5 = act(outputs_5)
        outputs_5 = keras.layers.Conv2D(channels_5, strides=strides_5[1], kernel_size=kernel_5, \
                           padding="same", kernel_initializer='he_normal')(outputs_5)
        outputs_5 = act(outputs_5)
        # end of final block

        # final pooling and flatten
        final_output = keras.layers.GlobalAveragePooling2D()(outputs_5)
        final_output = keras.layers.Flatten()(final_output)


        self.two_d_encoder = keras.Model(inputs=inputs, outputs=final_output)
        
        HIDDEN_2D = 128
        LSTM_DIM = 128
        lstm_inputs = keras.layers.Input(shape=(SIZE_T, HIDDEN_2D))
        lstm_out = LSTM(LSTM_DIM)(lstm_inputs)
        self.time_slice_encoder = keras.Model(inputs=lstm_inputs, outputs=lstm_out)

        fc_inputs = keras.layers.Input(shape = (LSTM_DIM,))
        fc_out = Dense(8)(fc_inputs)
        outputs = Dense(OUT_DIM)(fc_out)

        self.final_dense = keras.Model(inputs = fc_inputs, outputs = outputs)

        
    def call(self, inputs):
        """
        input dimension: B*T*H*X*Y*1
        output dimension: B*output_dim
        """

        out = tf.reshape(inputs, shape=[-1, SIZE_H, SIZE_X, SIZE_Y])
        out = self.two_d_encoder(out)
        out = tf.reshape(out, shape=[-1, SIZE_T, HIDDEN_2D])
        out = self.time_slice_encoder(out)
        out = self.final_dense(out)

        return out
