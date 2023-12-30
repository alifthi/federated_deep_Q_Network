import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as ksl 
class policygradient:
    def __init__(self,numberOfAgents=3) -> None:
        self.num_of_connection=numberOfAgents
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
        return model
    def sellect_action_by_distribution(self,polisy):
        dist=tf.nn.softmax(polisy).numpy()[0]
        dist=dist/sum(list(dist))
        return [np.random.choice(np.arange(self.action_size),int(self.num_of_connection/2),p=dist),dist]
    def sellect_action(self,states):
        return self.sellect_action_by_distribution(self.model(np.array([states])))
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
        return -(sum([tf.math.log(policy[a]) for a in action])/len(action))*reward