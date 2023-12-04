import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import numpy as np
from keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as ksl 
from utils import utils
import gym 
from config import ENV, STATE_SIZE, BATCH_SIZE,\
                    TARGET_NETWORK_UPDATE_RATE,\
                    DISCOUNT_FACTOR, NUM_OF_EPISODES,\
                    AGGREGATE_RATE, NUM_OF_TIMESTEPS,MODE

class agent:
    def __init__(self) -> None:
        self.state_size=STATE_SIZE
        self.batch_size=BATCH_SIZE
        self.discount_factor=DISCOUNT_FACTOR
        self.num_of_timesteps=NUM_OF_TIMESTEPS
        self.buffer=[]
        self.env=gym.make(ENV,render_mode='rgb_array')
        self.action_size=self.env.action_space.n
        self.eps=0.9
        self.utils=utils()
        self.target_network_update_rate=TARGET_NETWORK_UPDATE_RATE
        self.total_updates=1
        self.main_network=self.build_model()
        self.target_network=self.build_model()
        self.prox_factor=0.05
        self.last_aggregation_weights=None
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

        model.add(ksl.Conv2D(32, kernel_size=3, strides=1, padding='same'))
        model.add(ksl.MaxPooling2D(2))        
        model.add(ksl.Activation('relu'))
        
        model.add(ksl.Flatten())

        model.add(ksl.Dense(128, activation='relu'))
        model.add(ksl.Dense(self.action_size, activation='linear'))
        if MODE=='FedAvg':
            model.compile(loss='mae',optimizer='sgd')
        elif MODE=='FedProx':
            pass
        return model
    def update_buffer(self, state, action, reward, n_state, is_done):
        self.buffer.append([state, action, reward, n_state, is_done])
    def sellect_action(self,Q_value):
        # epsilone greedy action sellection
        if np.random.uniform(0,1)>self.eps:
            return np.random.randint(self.action_size)
        return np.argmax(Q_value[0])
    def start_episode(self):
        return_=0
        self.update_target_network()
        state,_=self.env.reset()
        state=self.utils.preprocessing(image=state)
        for _ in range(self.num_of_timesteps):
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
                self.train_main_models()
        return return_
    def train_main_models(self):
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
        
    def loss(self,yTrue,yPred):
        dist_aggregation=[]
        for i,layer in enumerate(self.last_aggregation_weights):
            dist_aggregation.append(layer-self.main_network[i])
        losses=tf.keras.losses.MAE(yTrue,yPred)+tf.norm(dist_aggregation)**2
        return losses
class agent1(agent):
    def __init__(self,cooprator) -> None:
        super().__init__()
        self.cooprator=cooprator
    def train_local_models(self,weights,aggregated_weights,call_for_aggregation):
        return_=0
        for i in range(NUM_OF_EPISODES):
            if i%AGGREGATE_RATE==0:
                weights.put(self.main_network.weights)
                # time.sleep(0.1)
                if not call_for_aggregation.value:
                    print(1)
                    call_for_aggregation.value=True 
                    self.cooprator.fedavg_aggregate(weights,aggregated_weights)
                self.main_network.set_weights(aggregated_weights.get())
                print('aggregation done!')
            r=self.start_episode()
            return_+=r
class agent2(agent):
    def __init__(self,cooprator) -> None:
        super().__init__()
        self.cooprator=cooprator
    def train_local_models(self,weights,aggregated_weights,call_for_aggregation):
        return_=0
        for i in range(NUM_OF_EPISODES):
            if i%AGGREGATE_RATE==0:
                weights.put(self.main_network.weights)
                if not call_for_aggregation.value:
                    print(2)
                    call_for_aggregation.value=True
                    self.cooprator.fedavg_aggregate(weights,aggregated_weights)
                self.main_network.set_weights(aggregated_weights.get())
                print('aggregation done!')
            r=self.start_episode()
            return_+=r