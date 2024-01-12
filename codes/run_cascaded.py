from agents import *
from cooprator import cooprator
import numpy as np
from config import MODE, ROBUST_METHODE, AGREEGATION
if ROBUST_METHODE=='AE':
    from Autoencoder import AutoEncoder
    aemodel=AutoEncoder()    
co=cooprator()
agents={'agent1':agent1(is_attacker=False),
        'agent2':agent2(),
        'agent3':agent3(),
        'agent4':agent4(),
        'agent5':agent5()}
for i in range(500):
    print('Iteration....',i)
    states={}
    for key in agents.keys():
        tmp_st=[0]
        solved_counter=0
        for i in range(50):
            st=agents[key].train_local_models()
            if st[0]==500:
                solved_counter+=1
            else:
                solved_counter=0
            if solved_counter==4:
                agents[key].rewards=agents[key].rewards+[500]*(50-i)
                agent[key].plot(key[-1])
                tmp_st[0]+=500*(50-i)
                break
            tmp_st[0]+=st[0]
        states.update({key:tmp_st})
    weights=[agents[ag].main_network.weights for ag in agents.keys()]
    if AGREEGATION=='weightedAveraging':
        if not MODE=='FedADMM':
            aggregation=co.weightedAveraging(weights,states=states)
            for i,ag in enumerate(agents.keys()):
                agents[ag].last_aggregation_weights=aggregation[i]
                agents[ag].main_network.set_weights(aggregation[i])
        else:
            y_k=[]
            for agent in agents.values():
                y_k.append(agent.yk)
            co.fedADMM_aggregate(weights,y_k,states=states)
            for name,agent in agents.items():
                agent.yk=co.yk[int(name[-1])-1]
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

    