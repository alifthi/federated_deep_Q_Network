ENV="MsPacman-v4"
MODE='FedAvg'
ROBUST_METHODE= 0# 'SAM'
NUMBER_OF_AGENTS=2
STATE_SIZE=(80, 80, 3)
DISCOUNT_FACTOR=0.9
BATCH_SIZE=32
NUM_OF_EPISODES=1
AGGREGATE_RATE=500
TARGET_NETWORK_UPDATE_RATE=1000
NUM_OF_TIMESTEPS=15000
MODEL_PATH='../model'
AUTOENCODER_PATH='../AEmodels'