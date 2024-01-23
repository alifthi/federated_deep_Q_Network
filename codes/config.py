ENV='VPP' # 'CartPole-v1'
MODEL_SELECTION='policy_gradient_method'
MODE='FedAvg' # FedProx FedADMM FedAvg
ROBUST_METHODE= '_' # SAM priorized DDQN
AGREEGATION= 'weightedAveraging'# 'weightedAveraging'
ATTACK='ـ' # 'label_flipping','model_targeted_poisoning'
POLICY_UPDATE_RATE=5
NUMBER_OF_AGENTS=5
STATE_SIZE=[4]
DISCOUNT_FACTOR=0.99
BATCH_SIZE=8
NUM_OF_EPISODES=1
AGGREGATE_RATE=500
TARGET_NETWORK_UPDATE_RATE=1000
NUM_OF_TIMESTEPS=15000
MODEL_PATH='../model'
AUTOENCODER_PATH='../AEmodels'
FIGURE_PATH='../figurs'