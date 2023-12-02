from config import NUMBER_OF_AGENTS
class cooprator:
    def __init__(self,model) -> None:
        self.agents_weights=[]
        self.is_ready=False
        self.model=model
        self.number_of_agents=NUMBER_OF_AGENTS
    def aggregate(self,weights,network):
        model_weights=[]
        for _ in range(self.number_of_agents):
            self.agents_weights.append(weights.get())
        for i,_ in range(len(self.agents_weights[0])):
            layer_weights=[]
            for ag in range(self.number_of_agents):
                layer_weights.append(self.agents_weights[ag][i])
            model_weights.append(sum(layer_weights)/self.number_of_agents)
        for _ in range(self.number_of_agents):
            network.put(model_weights) 