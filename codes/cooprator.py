from config import NUMBER_OF_AGENTS, MODEL_PATH
import tensorflow as tf
class cooprator:
    def __init__(self) -> None:
        self.is_ready=False
        self.number_of_agents=NUMBER_OF_AGENTS
        self.roh=0.9
    def fedavg_aggregate(self,agents_weights):
        model_weights=[]
        for i in range(len(agents_weights[0])):
            layer_weights=[]
            for ag in range(self.number_of_agents):
                layer_weights.append(agents_weights[ag][i])
            model_weights.append(sum(layer_weights)/self.number_of_agents)
        self.last_weights=model_weights
    def fedADMM_aggregate(self,agents_weights,y_k):
        model_weights=[]
        for i in range(len(agents_weights[0])):
            layer_weights=[]
            for ag in range(self.number_of_agents):
                layer_weights.append(agents_weights[ag][i])
                layer_weights.append(y_k[ag][i]/self.roh)
            model_weights.append(sum(layer_weights)/self.number_of_agents)
        self.last_weights=model_weights
        model1_weights=[]
        model2_weights=[]
        for ag in range(self.number_of_agents):
            layer_weights=[]
            for i in range(len(agents_weights[0])):
                layer_weights=y_k[ag][i]+self.roh*(agents_weights[ag][i]-self.last_weights[i])
                if ag==0:
                    model1_weights.append(layer_weights/self.number_of_agents)
                elif ag==1:
                    model2_weights.append(layer_weights/self.number_of_agents)
        self.yk_1=model1_weights
        self.yk_2=model2_weights

    @staticmethod
    def save_model(model):
        model.save(MODEL_PATH+'/model.h5')