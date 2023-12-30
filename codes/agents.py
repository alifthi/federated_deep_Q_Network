import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import numpy as np
from keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as ksl 
from utils import utils
import gym 
import random
from matplotlib import pyplot as plt
from config import ENV, STATE_SIZE, BATCH_SIZE,\
                    TARGET_NETWORK_UPDATE_RATE,\
                    DISCOUNT_FACTOR, NUM_OF_EPISODES,\
                     NUM_OF_TIMESTEPS,MODE,ROBUST_METHODE,\
                     FIGURE_PATH,MODEL_PATH
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
        self.target_network=self.build_model()
        self.prox_factor=0.05
        self.last_aggregation_weights=None
        self.losses=[]
        self.rewards=[]
        if MODE=='FedADMM':
            self.roh=0.2
            self.yk=self.build_model().weights
    def build_model(self):
        model = Sequential()
        model.add(ksl.Dense(64, activation='gelu',input_shape=self.state_size))         
        model.add(ksl.Dropout(0.2))
        model.add(ksl.Dense(32, activation='gelu'))
        model.add(ksl.Dense(self.action_size, activation='linear'))
        model.summary()
        if MODE=='FedAvg':
            model.compile(loss='mae',optimizer=SGD(0.01),metrics=['mae','mse'])
        elif MODE=='FedProx':
            model.compile(loss=self.FedProx_loss,optimizer=SGD(0.01),metrics=['mae','mse'])
        else:
            model.compile(loss=self.FedProx_loss,optimizer=SGD(0.01),metrics=['mae','mse'])
        return model
    def update_buffer(self, state, action, reward, n_state, is_done,TD=None):
        if not isinstance(type(TD),type(None)):
            self.buffer.append([state, action, reward, n_state, is_done,TD])
            return
        self.buffer.append([state, action, reward, n_state, is_done,TD])
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
        state,_=self.env.reset()
        for i in range(self.num_of_timesteps):
            self.total_updates+=1
            Q_values=self.main_network.predict(state[None,:],verbose=0)
            action=self.sellect_action(Q_value=Q_values)
            n_state, reward, done,_,_=self.env.step(action)
            if ROBUST_METHODE=='priorized':
                td=reward+self.discount_factor*np.max(self.target_network.predict(n_state[None,:],verbose=0))-Q_values[0,action]
                td=abs(td)
                self.update_buffer(action=action, reward=reward, n_state=n_state, state=state,is_done=done,TD=td)
            else:    
                self.update_buffer(action=action, reward=reward, n_state=n_state, state=state,is_done=done)
            state=n_state
            
            return_+=reward
            if self.total_updates%1000==0: 
                self.update_target_network()
            if len(self.buffer)>self.batch_size:
                self.train_main_models()
            if done:
                break
        return return_
    def train_main_models(self):
        if not ROBUST_METHODE=='priorized':
            batch=random.sample(self.buffer,self.batch_size)
        else:
            p=self.periorizeation()
            batch=np.random.choice(np.arange(len(self.buffer)),self.batch_size,p=p.astype('float'))
            batch=np.array(self.buffer)[batch]
        states=[]
        value=[]
        i=0
        for state, action, reward, n_state, is_done,_ in batch:
            val=self.target_network.predict(n_state[None,:],verbose=0)
            act=np.argmax(val)
            if is_done:
                Q=reward
            else:
                if ROBUST_METHODE=='DDQN':
                    Q=reward+self.discount_factor*self.target_network.predict(n_state[None,:],verbose=0)[act]
                else:
                    Q=reward+self.discount_factor*np.max(self.target_network.predict(n_state[None,:],verbose=0))
            Q_values=self.main_network.predict(state[None,:],verbose=0)
            Q_values[0][action]=Q
            value.append(Q_values)
            states.append(state[None,:])
        states=np.concatenate(states,axis=0)
        value=np.concatenate(value,axis=0)
        self.train(states, value, batch_size=self.batch_size,epochs=1)
    def update_target_network(self):
        if self.eps < 0.9:
            self.eps+=0.1
            print(f'Epsilon: {self.eps}')
        self.target_network.set_weights(self.main_network.weights)  
        print('updating Target Network...')
    def FedProx_loss(self,yTrue,yPred):
        model_difference = tf.nest.map_structure(lambda a, b: a - b,
                                        self.main_network.weights,
                                        self.last_aggregation_weights)
        model_difference=tf.linalg.global_norm(model_difference)**2
        losses=tf.math.reduce_mean(tf.keras.losses.MSE(yTrue,yPred))
        losses=+0.001*model_difference
        return losses
    def ADMM_loss(self,yTrue,yPred):
        model_difference = tf.nest.map_structure(lambda a, b: a - b,
                                        self.main_network.weights,
                                        self.last_aggregation_weights)
        l=[]
        for i, layer in enumerate(self.yk):
            l.append(tf.tensordot(tf.reshape(layer,[1,-1]),
                                  tf.transpose(tf.reshape(model_difference[i],[1,-1])),1))
        residual=sum(l)
        model_difference=tf.linalg.global_norm(model_difference)**2
        mse=tf.keras.losses.MSE(yTrue,yPred)
        mse=tf.math.reduce_mean(mse)
        losses=mse+self.roh*model_difference/2+residual
        return losses[0][0]    
    def train(self,states,values,batch_size,epochs):
        sgd=SGD(0.001)
        for _ in range(epochs):
            print(f'Epoch: {_}')
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
                    elif MODE=='FedAvg':
                        mse=tf.keras.losses.MSE(batch_values,outs)
                        losses=tf.math.reduce_mean(mse)
                        
                gradients=tape.gradient(losses,self.main_network.weights)
                if ROBUST_METHODE=='SAM':
                    epsilon = tf.nest.map_structure(lambda a: 0.1*a/tf.linalg.norm(a),
                                                    gradients)
                    new_weights=tf.nest.map_structure(lambda a,b: a+b,
                                                self.main_network.weights,epsilon)
                    tmp_model=tf.keras.models.clone_model(self.main_network)
                    tmp_model.set_weights(new_weights)
                    with tf.GradientTape() as tape:
                        tape.watch(tmp_model.weights)
                        outs=tmp_model(batch_states)
                        if MODE=='FedProx':
                            losses=self.FedProx_loss(yTrue=batch_values,yPred=outs)
                        elif MODE=='FedADMM':
                            losses=self.ADMM_loss(yTrue=batch_values,yPred=outs)
                        elif MODE=='FedAvg':
                            mse=tf.keras.losses.MSE(batch_values,outs)
                            losses=tf.math.reduce_mean(mse)
                    gradients=tape.gradient(losses,tmp_model.weights)     
                      
                sgd.apply_gradients(zip(gradients,self.main_network.weights))
                self.losses.append(losses.numpy())
                print(f'Mode: {MODE}, loss: {losses.numpy()}')
    def periorizeation(self):
        arr=np.array(self.buffer)
        tds=arr[:,-1]
        sum_of_td_error=sum(tds)
        return tds/sum_of_td_error
    def plot(self,agent):
        plt.figure()
        plt.plot(self.losses)
        plt.xlabel('Episode')
        plt.ylabel('Losses')
        plt.savefig(f'{FIGURE_PATH}/Losses_{MODE}_{ROBUST_METHODE}_{agent}.png')
        plt.figure()
        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.savefig(f'{FIGURE_PATH}/Rewards_{MODE}_{ROBUST_METHODE}_{agent}.png')
class agent1(agent):
    def train_local_models(self):
        for _ in range(NUM_OF_EPISODES):
            self.total_reward=self.start_episode()
            self.main_network.save('../model/model1.h5')
            self.rewards.append(self.total_reward)
            if self.total_reward>= max(self.rewards):
                self.main_network.save(MODEL_PATH+'/best/model1.h5')
            self.plot('1')
            print(f'[INFO] 1.{_}th round ended, Total return {self.total_reward}!')
        return[self.total_reward,sum(self.losses)/len(self.losses)]

class agent2(agent):
    def train_local_models(self):
        for _ in range(NUM_OF_EPISODES):
            self.total_reward=self.start_episode()
            self.rewards.append(self.total_reward)
            self.main_network.save('../model/model2.h5')
            if self.total_reward>= max(self.rewards):
                self.main_network.save(MODEL_PATH+'/best/model2.h5')
            self.plot('2')
            print(f'[INFO] 2.{_}th round ended, Total return {self.total_reward}!')
        return[self.total_reward,sum(self.losses)/len(self.losses)]
class agent3(agent):
    def train_local_models(self):
        for _ in range(NUM_OF_EPISODES):
            self.total_reward=self.start_episode()
            self.rewards.append(self.total_reward)
            self.main_network.save('../model/model3.h5')
            if self.total_reward>= max(self.rewards):
                self.main_network.save(MODEL_PATH+'/best/model3.h5')
            self.plot('3')
            print(f'[INFO] 3.{_}th round ended, Total return {self.total_reward}!')
        return[self.total_reward,sum(self.losses)/len(self.losses)]

class agent4(agent):
    def train_local_models(self):
        for _ in range(NUM_OF_EPISODES):
            self.total_reward=self.start_episode()
            self.rewards.append(self.total_reward)
            self.main_network.save('../model/model4.h5')
            if self.total_reward>= max(self.rewards):
                self.main_network.save(MODEL_PATH+'/best/model4.h5')
            self.plot('4')
        print(f'[INFO] 4.{_}th round ended, Total return {self.total_reward}!')
        return[self.total_reward,sum(self.losses)/len(self.losses)]

class agent5(agent):
    def train_local_models(self):
        for _ in range(NUM_OF_EPISODES):
            self.total_reward=self.start_episode()
            self.rewards.append(self.total_reward)
            self.main_network.save('../model/model5.h5')
            if self.total_reward>= max(self.rewards):
                self.main_network.save(MODEL_PATH+'/best/model5.h5')
            self.plot('5')
            print(f'[INFO] 5.{_}th round ended, Total return {self.total_reward}!')
        return[self.total_reward,sum(self.losses)/len(self.losses)]
