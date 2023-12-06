import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import tensorflow as tf
from utils import utils
import numpy as np
import cv2 as cv
import gym 
from matplotlib import pyplot as plt
import time
from agents import agent
from config import ENV, MODEL_PATH

env=gym.make(ENV,render_mode='rgb_array')
utils=utils()
model=tf.keras.models.load_model(MODEL_PATH+'/model.h5',
                                 custom_objects={'loss':'mae'})
agent=agent()
i=0


for i in range(10):
  state,_=env.reset()
  env.render()
  state=utils.preprocessing(state)
  while True:
    i+=1
    cv.imshow('Pacman',cv.resize(state,[256,256]))
    Q_values=model.predict(state[None,:],verbose=0)
    action=np.argmax(Q_values)
    # dist=tf.nn.softmax(Q_values).numpy()[0]
    # dist=dist/sum(list(dist))
    print(action)
    action=agent.sellect_action(Q_values)
    n_state, reward, done,_,_=env.step(action)
    state=env.render()
    state=utils.preprocessing(state)
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
    if done:
        cv.destroyAllWindows()
        break
