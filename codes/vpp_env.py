import pandas as pd
import numpy as np
class vpp_env:
    def __init__(self) -> None:
        self.load_datasets('../Dataset/scenario_datasets/')
        self.num_EV_charger=4
        self.action_space=3
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
        self.states=None
    def load_datasets(self,path):
        self.sp_data=pd.read_csv(path+'PV_load_2020_profile.csv')
        self.wt_data=pd.read_csv(path+'WT_load_2020_profile.csv')
        self.price=pd.read_csv(path+'market_prices_2020_profile.csv')
        self.houseload_data=pd.read_csv(path+'households_load_profile.csv')
    def step(self,action):
        episode_done=False
        rewards=[0,0,0]
        for i in range(self.num_EV):
            if self.parked_time[i]>self.time_to_park[i] or self.soc[i]==1:
                self.soc_of_departed_EVs.append(self.soc[i])
                rewards[1]+=self.reward_for_soc(self.soc[i]*100)
                self.charging_EVs[self.charging_EVs==i]=np.random.choice(range(self.num_EV),1)
                self.parked_time[i]=0
                
        self.t+=self.time_delay
        self.parked_time[self.charging_EVs]=self.parked_time[self.charging_EVs]+self.time_delay/3600
        house_demand=self.houseload_data.iloc[self.state_in_table,1]
        renewable_energy=self.sp_data.iloc[self.state_in_table,1]\
            +self.wt_data.iloc[self.state_in_table,1]
        exist_energy=renewable_energy-house_demand          
        
        self.soc[self.charging_EVs]=self.soc[self.charging_EVs]\
            +(action.T*self.time_delay/(self.time_to_fullcharge*60*60))
        self.states=np.zeros(3)
        self.states[0]=action.T@(np.ones_like(action)*self.naminal_power)
        power=exist_energy+self.states[0]
        self.states[1]=power
        self.states[2]=self.soc[self.charging_EVs].T@np.ones_like(self.charging_EVs)*self.EV_cap
        rewards[1]=self.reward_for_load_value(power)
        self.total_power+=power
        if power<0:
            self.purchased_energy+=power
        else:
            self.excess_energy+=power
        if not self.t==0 and self.t%900==0:
            try:
                soc=sum(self.soc_of_departed_EVs)/len(self.soc_of_departed_EVs)
            except:
                soc=0
            rewards[2]=self.trajectory_ending_reward(soc*100,
                                                     abs(self.purchased_energy),
                                                     self.excess_energy,
                                                     abs(self.total_power))

            episode_done=True
        return self.states,sum(rewards),episode_done    
    def reset(self):
        self.t=0
        self.state_in_table+=1
        self.purchased_energy=0
        self.excess_energy=0
        self.total_power=0
        self.soc_of_departed_EVs=[]
        if isinstance(self.states,type(None)):
            return np.zeros(3)
        else:
            return self.states
    def reward_for_load_value(self,total_load):
        load_value=self.price.iloc[self.state_in_table,1]*total_load
        if load_value<-1:
            reward=load_value+1
        elif load_value>=-1 and load_value<0:
            reward=15*load_value+15
        elif load_value>=0 and load_value<1:
            reward=-15*load_value+15
        else:
            reward=-load_value+1
        return reward
    def reward_for_soc(self,soc):
        if soc<90:
            reward=5*soc-300
        else:
            reward=1050-10*soc
        return reward
    def trajectory_ending_reward(self,soc,greed_energy,renewable_energy,cost):
        rewards=[0,0,0,0]
        if soc <75:
            rewards[0]+=-9+4*soc/25
        else:
            rewards[0]+=9-2*soc/25
            
        if greed_energy<800:
            rewards[1]=-25*greed_energy+20000
        else:
            rewards[1]=-greed_energy+800
            
        if renewable_energy<3000:
            rewards[2]=-5*renewable_energy/3+5000
        else:
            rewards[2]=-3*renewable_energy/2+4500
        print('Exceed power ',renewable_energy,'purched power ',greed_energy,'Cost ',cost)
        if cost<450:
            rewards[3]=-40*cost+18000
        else:
            rewards[3]=-10*cost+4500
            
        return sum(rewards)
            
        