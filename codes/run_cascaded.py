from agents import *
from cooprator import cooprator
import numpy as np
from config import MODE, ROBUST_METHODE, AGREEGATION
if ROBUST_METHODE=='AE':
    from Autoencoder import AutoEncoder
    aemodel=AutoEncoder()    
co=cooprator()
agents={'agent1':agent1(),
        'agent2':agent2(),
        'agent3':agent3(),
        'agent4':agent4(),
        'agent5':agent5()}
for i in range(500):
    print('Iteration....',i)
    states={}
    for key in agents.keys():
        tmp_st=[0,0]
        for _ in range(20):
            st=agents[key].train_local_models()
            tmp_st[0]+=st[0]
            # tmp_st[1]+=st[1]
        states.update({key:tmp_st})
    weights=[agents[ag].main_network.weights for ag in agents.keys()]
    if AGREEGATION=='weightedAveraging':
        aggregation=co.weightedAveraging(weights,states=states)
        for i,ag in enumerate(agents.keys()):
            agents[ag].main_network.set_weights(aggregation[i])
    else:
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
    if ROBUST_METHODE=='AE':
        states=np.array(agent1.buffer)
        aemodel.train_model(states=states[:,0])
        states=np.array(agent2.buffer)
        aemodel.train_model(states=states[:,0])

    