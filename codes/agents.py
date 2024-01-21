import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import numpy as np
from vpp_env import vpp_env
from keras.optimizers import SGD,Adam
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as ksl 
from utils import utils
import gym 
import random
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from keras.losses import CategoricalCrossentropy
from config import ENV, STATE_SIZE, BATCH_SIZE,\
                    TARGET_NETWORK_UPDATE_RATE,\
                    DISCOUNT_FACTOR, NUM_OF_EPISODES,\
                     NUM_OF_TIMESTEPS,MODE,ROBUST_METHODE,\
                     FIGURE_PATH,MODEL_PATH, ATTACK
class agent:
    def __init__(self,is_attacker=False) -> None:
        self.state_size=STATE_SIZE
        self.batch_size=BATCH_SIZE
        self.discount_factor=DISCOUNT_FACTOR
        self.num_of_timesteps=NUM_OF_TIMESTEPS
        self.buffer=[]
        if not ENV=='VPP':
            self.env=gym.make(ENV,render_mode='rgb_array')
            self.action_size=self.env.action_space.n
        else:
            self.env=vpp_env()
            self.action_size=self.env.action_space
            self.num_EV=self.env.num_EV_charger
            self.state_size=[2+self.num_EV]
        self.eps=0.6
        self.utils=utils()
        self.target_network_update_rate=TARGET_NETWORK_UPDATE_RATE
        self.total_updates=1
        self.main_network=self.build_model()
        self.target_network=self.build_model()
        self.prox_factor=0.05
        self.last_aggregation_weights=self.main_network.weights
        self.optim=Adam(0.01)
        self.centropy=CategoricalCrossentropy()
        self.losses=[]
        self.rewards=[]
        self.is_attacker=is_attacker
        if ATTACK=='model_targeted_poisoning'and self.is_attacker:
            self.attacker_target=self.build_model()
            self.attacker_optimizer=SGD(0.01)
        if MODE=='FedADMM':
            self.roh=0.2
            self.yk=self.build_model().weights
    def build_model(self):
        inp=ksl.Input(self.state_size)
        x=ksl.Dense(32, activation='gelu',name='fc1')(inp)
        x=ksl.Dense(32, activation='gelu',name='fc2')(x)
        x=ksl.Dense(32, activation='gelu',name='fc3')(x)
        if not ENV=='VPP':
            outs=ksl.Dense(self.action_size, activation='linear',name='Output')(x)
        else:
            outs=[]
            for i in range(self.num_EV):
                outs.append(ksl.Dense(self.action_size, activation='linear',name='Output'+str(i))(x))
        model=tf.keras.Model(inp,outs)
        model.summary()
        if MODE=='FedAvg':
            model.compile(loss='mae',optimizer=SGD(0.01),metrics=['mae','mse'])
        elif MODE=='FedProx':
            model.compile(loss=self.FedProx_loss,optimizer=SGD(0.01),metrics=['mae','mse'])
        else:
            model.compile(loss=self.FedProx_loss,optimizer=SGD(0.01),metrics=['mae','mse'])
        plot_model(model,to_file=FIGURE_PATH+'/model.png',show_shapes=True)
        return model
    def update_buffer(self, state, action, reward, n_state, is_done,TD=None):
        if not isinstance(type(TD),type(None)):
            self.buffer.append([state, action, reward, n_state, is_done])
            return
        self.buffer.append([state, action, reward, n_state, is_done,TD])
    def sellect_action(self,Q_value):
        # epsilone greedy action sellection
        if np.random.uniform(0,1)>self.eps:
            return np.random.randint(self.action_size)
        return np.argmax(Q_value[0])
    def sellect_action_dist(self,prob):

        prob=tf.nn.softmax(prob).numpy()
        prob=prob/sum(list(prob))
        return np.random.choice(np.arange(self.action_size),p=prob)
    def policy_gradient_method(self):
        Return=0
        state,_=self.env.reset()
        self.buffer=[]
        for i in range(self.num_of_timesteps):
            self.total_updates+=1
            prob=self.main_network.predict(state[None,:],verbose=0)[0]
            if self.is_attacker and ATTACK=='label_flipping':
                action=0
            else:
                action=self.sellect_action_dist(prob)
                
            n_state, reward, done,_,_=self.env.step(action)
            self.update_buffer(action=action, reward=reward,
                               n_state=n_state, state=state,
                               is_done=done)
            state=n_state
            Return+=reward
            if done or Return==500:
                break
        self.discount_rewards()
        self.train_policy_gradient()
        return Return
    def model_targeted_loss(self,states,action,reward):
        while True:
            states=tf.cast(states,tf.float32)
            action=tf.cast(action,tf.float32)
            reward=tf.cast(reward,tf.float32)
            with tf.GradientTape() as tape:
                tape.watch([states,action,reward])
                pred_target=self.main_network(states)
                attacker_prediction=self.attacker_target(states)
                main_loss=self.policy_loss(policy=pred_target,
                                           actions=action,
                                           reward=reward)
                attacker_loss=self.policy_loss(policy=attacker_prediction,
                                           actions=action,
                                           reward=reward)
                loss=tf.math.square(main_loss-attacker_loss)
            grad=tape.gradient(loss,[states,action,reward])
            states+=1*grad[0]
            action+=1*grad[1]
            reward+=1*grad[2]
            print(loss)
            if loss>1:
                break
        return list(states),list(action),list(reward)
    def train_policy_gradient(self):
        rewards=[]
        actions=[]
        states=[]
        for state, action, reward, _, _ in self.buffer:
            act=np.zeros(self.action_size)
            act[action]=1
            actions.append(act)
            rewards.append(reward)
            states.append(state)
        if self.is_attacker and ATTACK=='model_targeted_poisoning':
            attacked_states,attacked_action,attacked_reward=self.model_targeted_loss(states,actions,rewards)
            actions = attacked_action
            rewards = attacked_reward
            states=attacked_states
        
        actions=np.array(actions)
        rewards=np.array(rewards)
        state=np.array(states)
        
        with tf.GradientTape() as tape:
            tape.watch(self.main_network.trainable_variables)
            policy = self.main_network(state, training=True)
            loss=self.policy_loss(policy,actions,rewards)
            if MODE=='FedProx':
                loss+=self.FedProx_loss(losses=loss)
            elif MODE=='FedADMM':
                loss+=self.ADMM_loss(losses=loss)
            
        self.losses.append(loss)

        print(loss)
        grads = tape.gradient(loss, self.main_network.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.main_network.trainable_variables))
        
    def policy_loss(self,policy,actions,reward):
        loss=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits = policy, labels = actions)
        loss=tf.math.reduce_mean(loss*reward)
        return loss               
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
    def discount_rewards(self):
        reward_to_go = 0.0
        for i in reversed(range(len(self.buffer))):
            reward_to_go = reward_to_go * self.discount_factor + self.buffer[i][2]
            self.buffer[i][2] = reward_to_go
        tmp=np.array(self.buffer)
        tmp=tmp[:,2]
        tmp -= np.mean(tmp)
        tmp /= np.std(tmp)
        for i,r in enumerate(tmp):
            self.buffer[i][2]=r
    def vpp_env(self):
        pass

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
    def FedProx_loss(self,yTrue=None,yPred=None,losses=None):
        model_difference = tf.nest.map_structure(lambda a, b: a - b,
                                        self.main_network.weights,
                                        self.last_aggregation_weights)
        model_difference=tf.linalg.global_norm(model_difference)**2
        if isinstance(losses,type(None)):
            losses=tf.math.reduce_mean(tf.keras.losses.MSE(yTrue,yPred))
        losses=+0.001*model_difference
        return losses
    def ADMM_loss(self,yTrue=None,yPred=None,losses=None):
        model_difference = tf.nest.map_structure(lambda a, b: a - b,
                                        self.main_network.weights,
                                        self.last_aggregation_weights)
        l=[]
        for i, layer in enumerate(self.yk):
            l.append(tf.tensordot(tf.reshape(layer,[1,-1]),
                                  tf.transpose(tf.reshape(model_difference[i],[1,-1])),1))
        residual=sum(l)
        model_difference=tf.linalg.global_norm(model_difference)**2
        if isinstance(losses,type(None)):
            losses=tf.keras.losses.MSE(yTrue,yPred)
            losses=tf.math.reduce_mean(losses)
        losses=losses+self.roh*model_difference/2+residual
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
            # self.total_rewards=self.start_episode()
            self.total_rewards=self.policy_gradient_method()
            self.main_network.save('../model/model1.h5')
            self.rewards.append(self.total_rewards)
            if self.total_rewards>= max(self.rewards):
                self.main_network.save(MODEL_PATH+'/best/model1.h5')
            self.plot('1')
            print(f'[INFO] 1.{_}th round ended, Total return {self.total_rewards}!')
        return[self.total_rewards]
class agent2(agent):
    def train_local_models(self):
        for _ in range(NUM_OF_EPISODES):
            # self.total_rewards=self.start_episode()
            self.total_rewards=self.policy_gradient_method()
            self.rewards.append(self.total_rewards)
            self.main_network.save('../model/model2.h5')
            if self.total_rewards>= max(self.rewards):
                self.main_network.save(MODEL_PATH+'/best/model2.h5')
            self.plot('2')
            print(f'[INFO] 2.{_}th round ended, Total return {self.total_rewards}!')
        return[self.total_rewards]
        return[self.total_rewards,sum(self.losses)/len(self.losses)]
class agent3(agent):
    def train_local_models(self):
        for _ in range(NUM_OF_EPISODES):
            # self.total_rewards=self.start_episode()
            self.total_rewards=self.policy_gradient_method()
            self.rewards.append(self.total_rewards)
            self.main_network.save('../model/model3.h5')
            if self.total_rewards>= max(self.rewards):
                self.main_network.save(MODEL_PATH+'/best/model3.h5')
            self.plot('3')
            print(f'[INFO] 3.{_}th round ended, Total return {self.total_rewards}!')
        return[self.total_rewards]

class agent4(agent):
    def train_local_models(self):
        for _ in range(NUM_OF_EPISODES):
            # self.total_rewards=self.start_episode()
            self.total_rewards=self.policy_gradient_method()
            self.rewards.append(self.total_rewards)
            self.main_network.save('../model/model4.h5')
            if self.total_rewards>= max(self.rewards):
                self.main_network.save(MODEL_PATH+'/best/model4.h5')
            self.plot('4')
        print(f'[INFO] 4.{_}th round ended, Total return {self.total_rewards}!')
        return[self.total_rewards]

class agent5(agent):
    def train_local_models(self):
        for _ in range(NUM_OF_EPISODES):
            # self.total_rewards=self.start_episode()
            self.total_rewards=self.policy_gradient_method()
            self.rewards.append(self.total_rewards)
            self.main_network.save('../model/model5.h5')
            if self.total_rewards>= max(self.rewards):
                self.main_network.save(MODEL_PATH+'/best/model5.h5')
            self.plot('5')
            print(f'[INFO] 5.{_}th round ended, Total return {self.total_rewards}!')
        return[self.total_rewards]
