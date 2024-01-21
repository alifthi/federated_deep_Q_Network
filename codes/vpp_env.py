import pandas as pd
import numpy as np
class vpp_env:
    def __init__(self) -> None:
        self.load_datasets('../Dataset/scenario_datasets/')
        self.num_EV_charger=4
        self.action_space=3
        self.t=0 # second
        self.time_delay=15 # second
        self.state_in_table=0
        self.naminal_power=3.7
        self.EV_cap=100
        self.time_to_fullcharge=self.EV_cap/self.naminal_power #hours
        self.evs_per_week=10
        self.num_EV=10
        self.soc=np.random.normal(0.5,0.1,size=self.num_EV)
        self.time_to_park=np.random.normal(24,1,size=self.num_EV)
        self.parked_time=np.zeros_like(self.soc)
        self.charging_EVs=np.random.choice(range(self.num_EV),self.num_EV_charger)
    def load_datasets(self,path):
        self.sp_data=pd.read_csv(path+'PV_load_2020_profile.csv')
        self.wt_data=pd.read_csv(path+'WT_load_2020_profile.csv')
        self.price=pd.read_csv(path+'market_prices_2020_profile.csv')
        self.houseload_data=pd.read_csv(path+'households_load_profile.csv')
    def step(self,action):
        for i in range(self.num_EV):
            if self.parked_time[i]>self.time_to_park[i] or self.soc[i]==1:
                self.charging_EVs[self.charging_EVs==i]=np.random.sample(range(self.num_EV))
                self.parked_time[i]=0
                
        self.t+=self.time_delay
        self.parked_time[self.charging_EVs]=self.parked_time[self.charging_EVs]+self.time_delay/3600
        house_demand=self.houseload_data.iloc[self.state_in_table,1]
        renewable_energy=self.sp_data.iloc[self.state_in_table,1]+self.wt_data.iloc[self.state_in_table,1]
        exist_energy=renewable_energy-house_demand                
        self.soc[self.charging_EVs]=self.soc[self.charging_EVs]\
            +(action.T*self.time_delay/(self.time_to_fullcharge*60*60))
        states=[]
        states.append(action.T@(np.ones_like(action)*self.naminal_power))
        states.append(exist_energy+states[0])
        states.append(self.soc*self.EV_cap)
        
        if not self.t==0 and self.t%900==0:
            self.t=0
            self.state_in_table+=1
            
        
    def step_reward_function(self,total_load,soc):
        rewards=[0,0]
        load_value=self.price.iloc[self.state_in_table,1]*total_load
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
            
        