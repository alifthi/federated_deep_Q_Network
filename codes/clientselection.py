import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import numpy as np
from keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as ksl 
from keras.optimzers import Adam
class policygradient:
    def __init__(self,numberOfAgents=3) -> None:
        self.state_size=[numberOfAgents*2]
        self.discount_factor=0.95
        self.action_size=numberOfAgents
        self.total_updates=1
        self.model=self.build_model()
        self.optim=Adam(0.01)
    def build_model(self):
        model = Sequential()
        model.add(ksl.Dense(64, activation='gelu',input_shape=self.state_size))         
        model.add(ksl.Dense(32, activation='gelu'))
        model.add(ksl.Dense(self.action_size, activation='softmax'))
        model.summary()
        model.compile(loss='mae',optimizer=SGD(0.01),metrics=['mae','mse'])
        return model
    def sellect_action_dist(self,polisy):
        dist=tf.nn.softmax(polisy).numpy()[0]
        dist=dist/sum(list(dist))
        return np.random.choice(np.arange(self.action_size),p=dist)
    def train_model(self, states, rewards, actions):
        sum_reward = 0
        discnt_rewards = []
        rewards.reverse()
        for r in rewards:
            sum_reward = r + self.gamma*sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()  
        for state, reward, action in zip(states, discnt_rewards, actions):
            with tf.GradientTape() as tape:
                policy = self.model(np.array([state]), training=True)
                loss = self.loss(policy, action, reward)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
    def loss(self,policy,action,reward):
        return -tf.math.log(sum([policy[a] for a in action]))*reward