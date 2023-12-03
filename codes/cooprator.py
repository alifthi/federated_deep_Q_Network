from config import NUMBER_OF_AGENTS,MODEL_PATH
class cooprator:
    def __init__(self) -> None:
        self.is_ready=False
        self.number_of_agents=NUMBER_OF_AGENTS
    def fedavg_aggregate(self,agents_weights):
        model_weights=[]
        for i in range(len(agents_weights[0])):
            layer_weights=[]
            for ag in range(self.number_of_agents):
                layer_weights.append(agents_weights[ag][i])
            model_weights.append(sum(layer_weights)/self.number_of_agents)
        self.last_weights=model_weights
    @staticmethod
    def save_model(model):
        model.save(MODEL_PATH+'/model.h5')