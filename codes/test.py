import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import tensorflow as tf
from utils import utils
import numpy as np
import cv2 as cv
import gym 
from agents import agent
from config import ENV, MODEL_PATH

env=gym.make(ENV,render_mode='rgb_array')
utils=utils()
model=tf.keras.models.load_model(MODEL_PATH+'/best/model1.h5',
                                 custom_objects={'FedProx_loss':'nothing'})
agent=agent()

state,_=env.reset()
env.render()
n_action=12
total_rewards=0
while True:
  
  sstate=env.render()
  
  cv.imshow('Pacman',cv.resize(sstate,[256,256]))
  Q_values=model.predict(state[None,:],verbose=0)
  action=np.argmax(Q_values)
  print(action)
  action=agent.sellect_action(Q_values)
  n_state, reward, done,_,_=env.step(action)
  total_rewards+=reward
  state=n_state
  n_action=action
  if cv.waitKey(25) & 0xFF == ord('q'):
    break
  if done:
      break
cv.destroyAllWindows()
print(total_rewards)