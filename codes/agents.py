import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import numpy as np
from keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as ksl 
from tensorflow.keras.activations import softmax
from utils import utils
import gym 
import random
from config import ENV, STATE_SIZE, BATCH_SIZE,\
                    TARGET_NETWORK_UPDATE_RATE,\
                    DISCOUNT_FACTOR, NUM_OF_EPISODES,\
                     NUM_OF_TIMESTEPS,MODE
class agent:
    def __init__(self) -> None:
        self.state_size=STATE_SIZE
        self.batch_size=BATCH_SIZE
        self.discount_factor=DISCOUNT_FACTOR
        self.num_of_timesteps=NUM_OF_TIMESTEPS
        self.buffer=[]
        self.env=gym.make(ENV,render_mode='rgb_array')
        self.action_size=self.env.action_space.n
        self.eps=0.6
        self.utils=utils()
        self.target_network_update_rate=TARGET_NETWORK_UPDATE_RATE
        self.total_updates=1
        self.main_network=self.build_model()
        # self.target_network=self.build_model()
        self.prox_factor=0.05
        self.last_aggregation_weights=None
      
    def build_model(self):
        model = Sequential()
        model.add(ksl.Conv2D(32,kernel_size=3, padding='same', input_shape=self.state_size))
        model.add(ksl.MaxPooling2D(2))
        model.add(ksl.Activation('relu'))

        model.add(ksl.Conv2D(64,kernel_size=3, padding='same'))
        model.add(ksl.MaxPooling2D(2))
        model.add(ksl.Activation('relu'))

        model.add(ksl.Conv2D(64, kernel_size=3, strides=1, padding='same'))
        model.add(ksl.MaxPooling2D(2))        
        model.add(ksl.Activation('relu'))

        model.add(ksl.Conv2D(32, kernel_size=3, strides=1, padding='same'))
        model.add(ksl.MaxPooling2D(2))        
        model.add(ksl.Activation('relu'))
        
        model.add(ksl.Flatten())

        model.add(ksl.Dense(64, activation='relu'))
        model.add(ksl.Dense(self.action_size, activation='linear'))
        model.summary()
        if MODE=='FedAvg':
            model.compile(loss='mae',optimizer=SGD(0.01),metrics=['mae','mse'])
        elif MODE=='FedProx':
            model.compile(loss=self.FedProx_loss,optimizer=SGD(0.01),metrics=['mae','mse'])
        elif MODE=='FedADMM':
            self.roh=0.1
            self.yk=[]
            for i in model.weights:
                self.yk.append(tf.reshape(i,[1,-1]))
        return model
    def update_buffer(self, state, action, reward, n_state, is_done):
        self.buffer.append([state, action, reward, n_state, is_done])
    def sellect_action(self,Q_value):
        # epsilone greedy action sellection
        if np.random.uniform(0,1)>self.eps:
            return np.random.randint(self.action_size)
        return np.argmax(Q_value[0])
    def sellect_action_dist(self,Q_value):
        dist=tf.nn.softmax(Q_value).numpy()[0]
        dist=dist/sum(list(dist))
        return np.random.choice(np.arange(self.action_size),p=dist)
    def start_episode(self):
        return_=0
        self.update_target_network()
        state,_=self.env.reset()
        state=self.env.render()
        state=self.utils.preprocessing(image=state)
        for i in range(self.num_of_timesteps):
            self.total_updates+=1
            # self.env.render()
            # if self.total_updates%self.target_network_update_rate==0:
            Q_values=self.main_network.predict(state[None,:],verbose=0)
            action=self.sellect_action(Q_value=Q_values)
            n_state, reward, done,_,_=self.env.step(action)
            n_state=self.utils.preprocessing(n_state)
            self.update_buffer(action=action, reward=reward, n_state=n_state, state=state,is_done=done)
            state=n_state
            return_+=reward

            if done or i==32:
                break
            # if len(self.buffer)%self.batch_size==0:
        self.batch_size=np.shape(self.buffer)[0]

        self.train_main_models()
        self.update_target_network()
        return return_
    def train_main_models(self):
        batch=random.sample(self.buffer,self.batch_size)
        states=[]
        val=[]
        i=0
        for state, action, reward, n_state, is_done in batch:
            if action==5 and i%10==0:
                i+=1
                continue
            elif action==5 and not i%10==0:
                i+=1
            if is_done:
                Q=reward
            else:
                Q=reward+self.discount_factor*np.max(self.target_network.predict(n_state[None,:],verbose=0))
            Q_values=self.main_network.predict(state[None,:],verbose=0)
            Q_values[0][action]=Q
            val.append(Q_values)
            states.append(state[None,:])
        states=np.concatenate(states,axis=0)
        val=np.concatenate(val,axis=0)
        self.train(states, val, batch_size=32,epochs=2)
        # self.main_network.fit(states, val, batch_size=32,epochs=2)
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.weights)  
        print('updating Target Network...')
    def FedProx_loss(self,yTrue,yPred):
        model_difference = tf.nest.map_structure(lambda a, b: a - b,
                                        self.main_network.weights,
                                        self.last_aggregation_weights)
        model_difference=tf.linalg.global_norm(model_difference)**2
        losses=tf.math.reduce_mean(tf.keras.losses.MSE(yTrue,yPred))+model_difference
        return losses
    def ADMM_loss(self,yTrue,yPred):
        model_difference = tf.nest.map_structure(lambda a, b: a - b,
                                        self.main_network.weights,
                                        self.last_aggregation_weights)
        l=[]
        for i, layer in enumerate(self.yk):
            l.append(tf.tensordot(layer,tf.transpose(tf.reshape(model_difference[i],[1,-1])),1))
        residual=sum(l)
        model_difference=tf.linalg.global_norm(model_difference)**2
        losses=tf.math.reduce_mean(tf.keras.losses.MSE(yTrue,yPred))+\
                                    model_difference+residual
        return losses      
    def train(self,states,values,batch_size,epochs):
        sgd=SGD(0.01)
        if MODE=='FedADMM':
            epochs=1
        for _ in range(epochs):
            for iteration in range(int(np.shape(states)[0]/batch_size)):    
                batch_states=states[iteration*batch_size:(iteration+1)*batch_size]
                batch_values=values[iteration*batch_size:(iteration+1)*batch_size]
                with tf.GradientTape() as tape:
                    tape.watch(self.main_network.weights)
                    outs=self.main_network(batch_states)
                    if MODE=='FedProx':
                        losses=self.FedProx_loss(yTrue=batch_values,yPred=outs)
                    elif MODE=='FedADMM':
                        losses=self.ADMM_loss(yTrue=batch_values,yPred=outs)
                gradients=tape.gradient(losses,self.main_network.weights)
                sgd.apply_gradients(zip(gradients,self.main_network.weights))
                print(losses)
                    
            

class agent1(agent):
    def train_local_models(self):
        for _ in range(NUM_OF_EPISODES):
            r=self.start_episode()
            self.main_network.save('../model/model1.h5')
            print(f'[INFO] 1.{_}th round ended, Total return {r}!')
class agent2(agent):
    def train_local_models(self):
        for _ in range(NUM_OF_EPISODES):
            r=self.start_episode()
            self.main_network.save('../model/model2.h5')
            print(f'[INFO] 2.{_}th round ended, Total return {r}!')