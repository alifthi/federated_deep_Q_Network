from agents import agent1,agent2
from cooprator import cooprator
import numpy as np
from config import MODE, ROBUST_METHODE
if ROBUST_METHODE=='AE':
    from Autoencoder import AutoEncoder
    aemodel=AutoEncoder()    
co=cooprator()
agent1=agent1()
agent2=agent2()
for i in range(500):
    print('Iteration....',i)
    weights=[agent1.main_network.weights,agent2.main_network.weights]
    if not MODE=='FedADMM':
        if i ==0:
            co.fedavg_aggregate(weights,[1,1])
        else:
            co.fedavg_aggregate(weights,[agent1.total_reward,agent2.total_reward])
    else:
        y=[agent1.yk,agent2.yk]
        co.fedADMM_aggregate(weights,y)
        agent1.yk=co.yk_1
        agent2.yk=co.yk_2
    agent1.last_aggregation_weights=co.last_weights
    agent2.last_aggregation_weights=co.last_weights
    agent1.main_network.set_weights(co.last_weights)
    agent2.main_network.set_weights(co.last_weights)
    agent1.train_local_models()
    if ROBUST_METHODE=='AE':
        states=np.array(agent1.buffer)
        aemodel.train_model(states=states[:,0])
    agent2.train_local_models()
    if ROBUST_METHODE=='AE':
        states=np.array(agent2.buffer)
        aemodel.train_model(states=states[:,0])
    co.save_model(agent1.main_network)
    