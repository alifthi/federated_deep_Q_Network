import pandas as pd
class env_vpp:
    def __init__(self) -> None:
        self.load_datasets('../Dataset/scenario_datasets/')
    def load_datasets(self,path):
        self.sp_data=pd.read_csv(path+'PV_load_2020_profile.csv')
        self.wt_data=pd.read_csv(path+'WT_load_2020_profile.csv')
        self.price=pd.read_csv(path+'market_prices_2020_profile.csv')
        self.houseload_data=pd.read_csv(path+'households_load_profile.csv')
        self.t=0 # second
        self.time_delay=15 # second
    def step_reward_function(self,total_load,soc,state_in_table):
        rewards=[0,0]
        load_value=self.price.iloc[state_in_table,1]*total_load
        if load_value<-1:
            rewards[0]=load_value+1
        elif load_value>=-1 and load_value<0:
            rewards[0]=15*load_value+15
        elif load_value>=0 and load_value<1:
            rewards[0]=-15*load_value+15
        else:
            rewards[0]=-load_value+1
        
        if soc<90:
            rewards[1]=5*soc-300
        else:
            rewards[1]=-10*soc+1050
        return sum(rewards)
    def trajectory_ending_reward(self,soc,greed_energy,renewable_energy,cost):
        rewards=[0,0,0,0]
        if soc <75:
            rewards[0]=-9+4*soc/25
        else:
            rewards[0]=9-2*soc/25
            
        if greed_energy<800:
            rewards[1]=-25*greed_energy+20000
        else:
            rewards[1]=-greed_energy+800
            
        if renewable_energy<3000:
            rewards[2]=-5*renewable_energy/3+5000
        else:
            rewards[2]=-3*renewable_energy/2+4500
            
        if cost<450:
            rewards[3]=-40*cost+18000
        else:
            rewards[3]=-10*cost+4500
            
        return sum(rewards)
    def step(self,action):
        pass