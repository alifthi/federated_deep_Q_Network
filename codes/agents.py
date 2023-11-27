import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

class agent:
    def __init__(self,state_size,action_size,discount_factor=0.9,input_size=[8,8]) -> None:
        self.state_size=state_size
        self.action_size=action_size
        self.discount_factor=discount_factor
        self.buffer=[]
        self.input_size=input_size
        self.main_network=self.build_model()
        self.target_network=self.build_model()
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32,input_size=self.input_size, padding='same', input_shape=self.state_size))
        model.add(MaxPooling2D(2))
        model.add(Activation('relu'))

        model.add(Conv2D(64, strides=2, padding='same'))
        model.add(MaxPooling2D(2))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(MaxPooling2D(2))        
        model.add(Activation('relu'))
        
        model.add(Flatten())


        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam())

        return model
    def load_main_network(slef):
        pass
    def train(self):
        pass
    

class agent1(agent):
    def __init__(self) -> None:
        pass
    def load_main_network(slef):
        pass
    def load_target_network(self):
        pass


class agent2(agent):
    def __init__(self) -> None:
        pass
    def load_main_network(slef):
        pass
    def load_target_network(self):
        pass

    