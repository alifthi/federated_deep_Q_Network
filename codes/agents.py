import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as ksl 
from config import ENV,STATE_SIZE
import gym 

class agent:
    def __init__(self,discount_factor=0.9,input_size=[8,8]) -> None:
        self.state_size=STATE_SIZE
        self.discount_factor=discount_factor
        self.buffer=[]
        self.input_size=input_size
        self.main_network=self.build_model()
        self.target_network=self.build_model()
        self.env=gym.env(ENV)
        self.action_size=self.env.action_space.n
        self.eps=0.9
    def build_model(self):
        model = Sequential()
        model.add(ksl.Conv2D(32,input_size=self.input_size, padding='same', input_shape=self.state_size))
        model.add(ksl.MaxPooling2D(2))
        model.add(ksl.Activation('relu'))

        model.add(ksl.Conv2D(64, strides=2, padding='same'))
        model.add(ksl.MaxPooling2D(2))
        model.add(ksl.Activation('relu'))

        model.add(ksl.Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(ksl.MaxPooling2D(2))        
        model.add(ksl.Activation('relu'))
        
        model.add(ksl.Flatten())

        model.add(ksl.Dense(512, activation='relu'))
        model.add(ksl.Dense(self.action_size, activation='linear'))
        return model
    def update_buffer(self, state, action, reward, n_state, is_done):
        self.buffer.append([state, action, reward, n_state, is_done])
    def sellect_action(self,Q_value):
        # epsilone greedy action sellection
        if np.random.uniform(0,1)>self.eps:
            return np.random.randint(self.action_size)
        return np.argmax(Q_value)
    def load_main_network(slef):
        pass
    def train_local_models(self):
        pass
    def loss(self):
        pass
    

class agent1(agent):
    def __init__(self) -> None:
        super().__init__()
    def load_main_network(slef):
        pass
    def load_target_network(self):
        pass
    def train_local_models(self):
        return 


class agent2(agent):
    def __init__(self) -> None:
        super().__init__()
    def load_main_network(slef):
        pass
    def load_target_network(self):
        pass
    def train_local_models(self):
        return 
