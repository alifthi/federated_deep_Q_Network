import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import numpy as np
from keras.optimizers import SGD,Adam
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as ksl 
import tensorflow_probability as tfp
from keras.losses import CategoricalCrossentropy
class policygradient:
    def __init__(self,numberOfAgents=3) -> None:
        self.num_of_connection=numberOfAgents
        self.state_size=[numberOfAgents*2]
        self.discount_factor=0.95
        self.action_size=numberOfAgents
        self.total_updates=1
        self.model=self.build_model()
        self.optim=Adam(0.01)
        self.centropy=CategoricalCrossentropy()
    def build_model(self):
        inp = ksl.Input(self.state_size)
        x=ksl.Dense(32, activation='gelu')(inp)         
        # x=ksl.Dense(32, activation='gelu')(x)
        outs=[]
        for _ in range(int(self.num_of_connection/2)):
            outs.append(ksl.Dense(self.action_size, activation='linear')(x))
        model=tf.keras.Model(inp,outputs=outs)
        return model
    def sellect_action_by_distribution(self,polisy):
        dist=tf.nn.softmax(polisy/10).numpy()
        dist=dist.reshape([-1])
        dist=dist/dist.sum()
        return [np.random.choice(np.arange(self.action_size),p=dist,replace=False),dist]
    def sellect_action(self,states):
        selected_actions=[]
        distribution=[]
        policy=self.model(np.array([states]))
        for p in policy:
            action,dist=self.sellect_action_by_distribution(p)
            selected_actions.append(action)
            distribution.append(dist)
        return selected_actions,distribution
    def train_model(self, states, rewards, actions):
        sum_reward = 0
        discnt_rewards = []
        rewards.reverse()
        for r in rewards:
            sum_reward = r + self.discount_factor*sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()  
        discnt_rewards=np.array(discnt_rewards)
        discnt_rewards -= np.mean(discnt_rewards)
        discnt_rewards /= np.std(discnt_rewards)
        action=[]
        for a in actions:
            act=tf.keras.utils.to_categorical(a,num_classes=self.action_size)
            action.append(act)
        actions=np.array(action)
        states=np.array(states)
        losses=[]
        with tf.GradientTape() as tape:
            policy = self.model(np.array(states), training=True)
            losses=self.loss(policy, actions, discnt_rewards)
            loss=tf.math.reduce_mean(losses)
        print('Loss: ',loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
    def loss(self,policy,action,reward):
        loss=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits = policy, labels = action)
        loss=tf.math.reduce_mean(loss*reward)
        return loss