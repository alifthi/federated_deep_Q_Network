from agents import agent1,agent2
from cooprator import cooprator
from multiprocessing import Process, Queue, Value

co=cooprator()
agent1=agent1(co)
agent2=agent2(co)

for _ in range(10):
    agent1.train_local_models()
    agent2.train_local_models()
    weights=[agent1.main_network.weights,agent2.main_network.weights]
    co.fedavg_aggregate(weights)
    agent1.main_network.set_weights(co.last_weights)
    agent2.main_network.set_weights(co.last_weights)
    co.save_model(agent1.main_network)