import pandas as pd
import numpy as np
from config import IS_CONTINOUS 
class vpp_env:
    def __init__(self) -> None:
        self.load_datasets('../Dataset/scenario_datasets/')
        self.num_EV_charger=4
        self.action_space=3
        self.time_delay=30 # second
        self.state_in_table=0
        self.naminal_power=3.7
        self.EV_cap=100
        self.time_to_fullcharge=self.EV_cap/self.naminal_power #hours
        self.evs_per_week=10
        self.num_EV=10
        self.soc=np.random.normal(0.5,0.1,size=self.num_EV)
        self.time_to_park=np.random.normal(24,1,size=self.num_EV)
        self.parked_time=np.zeros_like(self.soc)
        self.charging_EVs=np.random.choice(range(self.num_EV),self.num_EV_charger,replace=False)
        self.states=None
        self.num_car_per_week=0
    def load_datasets(self,path):
        self.sp_data=pd.read_csv(path+'PV_load_2020_profile.csv')
        self.wt_data=pd.read_csv(path+'WT_load_2020_profile.csv')
        self.price=pd.read_csv(path+'market_prices_2020_profile.csv')
        self.houseload_data=pd.read_csv(path+'households_load_profile.csv')
    def step(self,action):
        if IS_CONTINOUS:
            action=(action-0.5)*2
            action=action*11.3
        else:
            action=action*3.3
        for i in range(self.num_EV_charger):
            if action[i]>=-1.1 and action[i]<=1.1:
                action[i]=0
        episode_done=False
        rewards=[0,0,0]
        for i,ev in enumerate(self.charging_EVs):
            rewards[0]+=self.reward_for_soc(self.soc[ev]*100) 
            if self.soc[ev]<=0.1 and (not action[i]>0):# ==1):
                return None,None,None
            elif 0.1<=self.soc[ev] and self.soc[ev] <=0.2 and action[i]<0: # ==-1:
                return None,None,None
        for i in range(self.num_EV):
            if not i in self.charging_EVs and self.soc[i]>0.1:
                self.soc[i]=self.soc[i]-0.1*self.time_delay/(self.time_to_fullcharge*60*60) 
            if self.parked_time[i]>self.time_to_park[i] or (self.soc[i]>=1 and i in self.charging_EVs):
                self.soc_of_departed_EVs.append(self.soc[i])
                choice=self.charging_EVs[0]
                while choice in self.charging_EVs:
                    choice=np.random.choice(range(self.num_EV),1)
                self.charging_EVs[self.charging_EVs==i]=choice
                self.parked_time[i]=0        
        self.t+=self.time_delay
        self.parked_time[self.charging_EVs]=self.parked_time[self.charging_EVs]+self.time_delay/3600
        house_demand=self.houseload_data.iloc[self.state_in_table,1]
        renewable_energy=40*self.sp_data.iloc[self.state_in_table,1]+8*self.wt_data.iloc[self.state_in_table,1]
        exist_energy=renewable_energy-4*house_demand   
        self.soc[self.charging_EVs]=self.soc[self.charging_EVs]\
            +(action*self.time_delay/(10*self.time_to_fullcharge*60*60))
        self.states=np.zeros(2+self.num_EV_charger)
        self.states[0]=action.T@np.ones_like(action)
        power=exist_energy-self.states[0]
        self.states[1]=power
        self.states[2:]=self.soc[self.charging_EVs]
        
        rewards[1]=self.reward_for_load_value(power)
        self.total_power+=power
        if power<0:
            self.purchased_energy+=abs(abs(power)-renewable_energy)
        elif power>0:
            self.excess_energy+=abs(power-renewable_energy)
        if not self.t==0 and self.t%900==0:
            print(self.soc)
            try:
                soc=sum(self.soc_of_departed_EVs)/len(self.soc_of_departed_EVs)
            except:
                soc=0
            rewards[2]=self.trajectory_ending_reward(soc*100,
                                                     abs(self.purchased_energy),
                                                     self.excess_energy,
                                                     abs(self.total_power))

            episode_done=True
            if self.state_in_table %672==0:
                self.time_to_park=np.concatenate([self.time_to_park,np.random.normal(24,1,size=self.num_car_per_week)])
                self.soc=np.concatenate([self.soc,np.random.normal(0.5,0.1,size=self.num_car_per_week)])
                self.parked_time=np.concatenate([self.parked_time,np.zeros(self.num_car_per_week)])
                self.num_EV+=self.num_car_per_week
            print(rewards,'>>>>>>>>>>>>',self.charging_EVs,'>>>>>>>',self.soc[self.charging_EVs],'>>>>>>>',action)
        print(sum(rewards))
        return self.states,sum(rewards),episode_done    
    def reset(self):
        self.t=0
        self.state_in_table+=1
        self.purchased_energy=0
        self.excess_energy=0
        self.total_power=0
        self.soc_of_departed_EVs=[]
        if isinstance(self.states,type(None)):
            return np.array([0]*(2+self.num_EV_charger))
        else:
            return self.states
    def reward_for_load_value(self,total_load):
        load_value=self.price.iloc[self.state_in_table,1]*total_load
        if load_value<-1:
            reward=load_value+1
        elif load_value>=-1 and load_value<0:
            reward=30*load_value+30
        elif load_value>=0 and load_value<1:
            reward=-30*load_value+30
        else:
            reward=-load_value+1
        return reward
    def reward_for_soc(self,soc):
        if soc<90:
            reward=20*soc-600
        else:
            reward=-40*(soc-90)
        return reward
    def trajectory_ending_reward(self,soc,greed_energy,renewable_energy,cost):
        rewards=[0,0,0,0]
        if soc <75:
            rewards[0]+=-9+4*soc/5
        else:
            rewards[0]+=9-2*soc/5
            
        if greed_energy<800:
            rewards[1]=-10*greed_energy+20000
        else:
            rewards[1]=-greed_energy+800
            
        if renewable_energy<250:
            rewards[2]=-5*renewable_energy/3+5000
        else:
            rewards[2]=-3*renewable_energy/2+4500
        print('Exceed power ',renewable_energy,'purched power ',greed_energy,'Cost ',cost)
        if cost<200:
            rewards[3]=-40*cost+18000
        else:
            rewards[3]=-10*cost+4500
            
        return sum(rewards)
            
        