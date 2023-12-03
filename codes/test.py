import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import tensorflow as tf
from utils import utils
import numpy as np
import cv2 as cv
import gym 
from matplotlib import pyplot as plt
import time
from config import ENV, MODEL_PATH

env=gym.make(ENV)
utils=utils()
model=tf.keras.models.load_model(MODEL_PATH+'/model.h5')



state,_=env.reset()
state=utils.preprocessing(state)
while True:
    cv.imshow('Pacman',state)
    Q_values=model.predict(state[None,:],verbose=0)
    action=np.argmax(Q_values)
    n_state, reward, done,_,_=env.step(action)
    state=utils.preprocessing(n_state)
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
    if done:
        cv.destroyAllWindows()
        break
    