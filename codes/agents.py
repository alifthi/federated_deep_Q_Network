import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as ksl 
from utils import utils
from config import ENV, STATE_SIZE, BATCH_SIZE, TARGET_NETWORK_UPDATE_RATE, DISCOUNT_FACTOR 
import gym 

class agent:
    def __init__(self) -> None:
        self.state_size=STATE_SIZE
        self.batch_size=BATCH_SIZE
        self.discount_factor=DISCOUNT_FACTOR
        self.buffer=[]
        self.env=gym.make(ENV,render_mode='rgb_array')
        self.action_size=self.env.action_space.n
        self.eps=0.9
        self.utils=utils()
        self.target_network_update_rate=TARGET_NETWORK_UPDATE_RATE
        self.total_updates=1
        self.main_network=self.build_model()
        self.target_network=self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(ksl.Conv2D(32,kernel_size=3, padding='same', input_shape=self.state_size))
        model.add(ksl.MaxPooling2D(2))
        model.add(ksl.Activation('relu'))

        model.add(ksl.Conv2D(64,kernel_size=3, strides=2, padding='same'))
        model.add(ksl.MaxPooling2D(2))
        model.add(ksl.Activation('relu'))

        model.add(ksl.Conv2D(64, kernel_size=3, strides=1, padding='same'))
        model.add(ksl.MaxPooling2D(2))        
        model.add(ksl.Activation('relu'))

        model.add(ksl.Flatten())

        model.add(ksl.Dense(512, activation='relu'))
        model.add(ksl.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer='sgd')
        return model
    def update_buffer(self, state, action, reward, n_state, is_done):
        self.buffer.append([state, action, reward, n_state, is_done])
    def sellect_action(self,Q_value):
        # epsilone greedy action sellection
        if np.random.uniform(0,1)>self.eps:
            return np.random.randint(self.action_size)
        return np.argmax(Q_value[0])
    def start_episode(self, time_steps):
        return_=0
        self.update_target_network()
        state,_=self.env.reset()
        state=self.utils.preprocessing(image=state)
        for _ in range(time_steps):
            self.total_updates+=1
            self.env.render()
            if self.total_updates%self.target_network_update_rate==0:
                self.update_target_network()
            Q_values=self.main_network.predict(state[None,:])
            action=self.sellect_action(Q_value=Q_values)
            n_state, reward, done,_,_=self.env.step(action)
            n_state=self.utils.preprocessing(n_state)
            self.update_buffer(action=action, reward=reward, n_state=n_state, state=state,is_done=done)
            state=n_state
            return_+=reward
            if done:
                break
            if len(self.buffer)>self.batch_size:
                self.train_local_models()
    def train_local_models(self):
        batch=np.random.sample(self.buffer,self.batch_size)
        for state, action, reward, n_state, is_done in batch:
            if is_done:
                Q=reward
            else:
                Q=reward+self.discount_factor*np.max(self.target_network.predict(n_state[None,:]))
            Q_values=self.main_network.predict(state[None,:])
            Q_values[0][action]=Q
            self.main_network.fit(state, Q_values, epochs=1)

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.weights)
    def loss(self):
        pass
class agent1(agent):
    def __init__(self) -> None:
        super().__init__()
    def train_local_models(self):
        return 
class agent2(agent):
    def __init__(self) -> None:
        super().__init__()
    def train_local_models(self):
        return 
