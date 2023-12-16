from keras import layers as ksl
import tensorflow as tf
import numpy as np
from config import STATE_SIZE, AUTOENCODER_PATH
import random
class AutoEncoder:
    def __init__(self) -> None:
        self.encoders=[]
        self.decoders=[]
        self.AEs=[]
        for i in range(2):
            self.name=str(i)
            self.build_model()
    def build_model(self):
        encoderInput = ksl.Input(STATE_SIZE)

        x = ksl.Conv2D(32,kernel_size=3,padding='same',activation='relu')(encoderInput)
        x = ksl.BatchNormalization()(x)
        x = ksl.AveragePooling2D(2)(x)

        x = ksl.Conv2D(3,kernel_size=3,padding='same',activation='relu')(x)
        x = ksl.BatchNormalization()(x)
        x = ksl.AveragePooling2D(2)(x)

        encoder = tf.keras.Model(encoderInput,x, name='Encoder_'+self.name)

        decoderInput = ksl.Input([20,20,3])
        x = ksl.Conv2DTranspose(16,kernel_size=3,strides=2,padding='same',activation='relu')(decoderInput)
        x = ksl.BatchNormalization()(x)

        x = ksl.Conv2DTranspose(STATE_SIZE[-1],kernel_size=3,strides=2,padding='same',activation='sigmoid')(x)

        decoder = tf.keras.Model(decoderInput,x, name='Decoder_'+self.name)
        AEModel = tf.keras.Sequential([encoder,decoder], name='AE_'+self.name)
        self.encoders.append(encoder)
        self.decoders.append(decoder)
        self.AEs.append(AEModel)
    def compile_model(self):
        encoder=random.sample(self.encoders,1)
        decoder=random.sample(self.decoders,1)
        model = tf.keras.Sequential([encoder[0],decoder[0]])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(0.01), metrics=['mae'])
        return model
    def train_model(self,states):
        model=self.compile_model()
        # states=np.concatenate(states,axis=0)
        state=[]
        for s in states:
            state.append(s[None,:])
        state=np.concatenate(state,axis=0)
        model.fit(state, state, epochs=5, batch_size=32)
        for i in range(2):
            model.layers[i].save(AUTOENCODER_PATH+'/'+model.layers[i].name+'.h5')
