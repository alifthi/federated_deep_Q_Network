from agents import agent1,agent2
from cooprator import cooprator
from multiprocessing import Process, Queue, Value

co=cooprator()
agent1=agent1(co)
agent2=agent2(co)

aggregated_weights=Queue()
weights=Queue()
call_for_aggregation=Value('b',False)
agent_process1=Process(target=agent1.train_local_models,
                       args=(weights,aggregated_weights,call_for_aggregation))
agent_process2=Process(target=agent2.train_local_models,
                       args=(weights,aggregated_weights,call_for_aggregation))
agent_process1.start()
agent_process2.start()