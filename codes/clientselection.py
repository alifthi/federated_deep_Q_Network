import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import numpy as np
from keras.optimizers import SGD,Adam
import tensorflow as tf
from tensorflow.keras import layers as ksl 
from keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import plot_model
class policygradient:
    def __init__(self,numberOfAgents=3) -> None:
        self.num_of_connection=numberOfAgents
        self.state_size=[numberOfAgents*1]
        self.discount_factor=0.9
        self.action_size=numberOfAgents
        self.total_updates=1
        self.model=self.build_model()
        plot_model(self.model,'model.png')
        self.Q_network=self.build_model(Q_network=True)
        self.optim=Adam(0.01)
        self.optim_phi=Adam(0.01)
        self.centropy=CategoricalCrossentropy()
    def build_model(self,Q_network=False):
        inp = ksl.Input(self.state_size,name='Input')
        x=ksl.Dense(32, activation='relu',name='FC1')(inp)     
        # x = ksl.Dropout(0.5)(x)    
        x=ksl.Dense(16, activation='relu',name='FC2')(x)
        outs=[]
        for _ in range(int(self.num_of_connection/2)+1):
            if Q_network:
                outs.append(ksl.Dense(self.action_size, activation='linear')(x))
            else:
                outs.append(ksl.Dense(self.action_size, activation='relu',name='Output'+str(_))(x))
        model=tf.keras.Model(inp,outputs=outs)
        return model
    def sellect_action_by_distribution(self,polisy):
        dist=tf.nn.softmax(polisy/30).numpy()
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
    def train_model(self, states, rewards, actions,n_states):
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
        action=np.array(action)
        actions=np.array(actions)
        states=np.array(states)
        n_states=np.array(n_states)
        losses=[]
        with tf.GradientTape() as tape_phi:
            tape_phi.watch(self.Q_network.trainable_variables)
            Q=self.Q_network(states, training=True)
            Q_n_state=self.Q_network(n_states,training=True)
            td=self.clac_td(Q,Q_n_state,rewards,actions)
            phi_loss=tf.math.reduce_mean(-td)
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            policy = self.model(np.array(states), training=True)
            losses=self.loss(policy, action, td)
            loss=tf.math.reduce_mean(losses)
        print('Loss: ',loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        phi_grad=tape_phi.gradient(phi_loss,self.Q_network.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.optim_phi.apply_gradients(zip(phi_grad,self.Q_network.trainable_variables))
    def loss(self,policy,action,td):
        loss=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits = policy, labels = action)
        loss=tf.math.reduce_mean(loss*td)
        return loss
    def clac_td(self,Q_state,Q_n_state,reward,action):
        td=[]
        for act,q_s,q_n_s in zip(action.T,Q_state,Q_n_state):
            for a,q,q_n in zip(act,q_s,q_n_s):
                td.append(reward+ self.discount_factor*max(q_n)-q[a])
        
        td=tf.math.reduce_mean(td)
        return td 