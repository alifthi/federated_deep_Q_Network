from agents import agent1,agent2
from cooprator import cooprator
from multiprocessing import Process, Queue, Value
import tensorflow as tf
from config import MODEL_PATH,MODE
co=cooprator()
agent1=agent1()
agent2=agent2()
# model1=tf.keras.models.load_model(MODEL_PATH+'/model.h5',{'FedProx_loss':agent1.FedProx_loss})
# model1.compile(loss=agent1.FedProx_loss,optimizer=tf.keras.optimizers.SGD(0.1),metrics=['mae','mse'])
# model2=tf.keras.models.load_model(MODEL_PATH+'/model.h5',{'FedProx_loss':agent2.FedProx_loss})
# model2.compile(loss=agent2.FedProx_loss,optimizer=tf.keras.optimizers.SGD(0.1),metrics=['mae','mse'])
# agent1.main_network=model1
# agent2.main_network=model2
# agent1.target_network=model1
# agent2.target_network=model2
for _ in range(50):
    print('>>>>>>>> ',_)
    weights=[agent1.main_network.weights,agent2.main_network.weights]
    y=[agent1.yk,agent2.yk]
    if not MODE=='FedADMM':
        co.fedavg_aggregate(weights)
    else:
        co.fedADMM_aggregate(weights,y)
        agent1.yk=co.yk_1
        agent2.yk=co.yk_2
    agent1.last_aggregation_weights=co.last_weights
    agent2.last_aggregation_weights=co.last_weights
    agent1.main_network.set_weights(co.last_weights)
    agent2.main_network.set_weights(co.last_weights)
    agent1.train_local_models()
    agent2.train_local_models()
    co.save_model(agent1.main_network)
    