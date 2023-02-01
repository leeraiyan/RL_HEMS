#!/usr/bin/env python
# coding: utf-8




import gym
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3 import DDPG





import numpy as np
import torch
SEEDLIST = [10129,10353,22373,54284,35519,40046,75647,66957,85409,92451]
DATASEED = 10

import numpy as np
import torch
def set_seeds(seed):
    torch.manual_seed(seed)  # Sets seed for PyTorch RNG
    torch.cuda.manual_seed_all(seed)  # Sets seeds of GPU RNG
    np.random.seed(seed=seed)  # Set seed for NumPy RNG
    random.seed(seed)  # Set seed for random RNG


# Define Classes for Loads




class CriticalLoad():
  def __init__(self, loadName, powerRating):
    self.powerRating = powerRating
    self.loadName = loadName
    self.isOn = 1

  def takeOneTimestep(self):
    return self.powerRating

  def checkStatus(self):
    print(self.loadName)
    print("Load Power Rating:", self.powerRating)
    print("Load Status:", self.isOn)





class AdjustableLoad():
  def __init__(self, loadName, minPowerRating, maxPowerRating, alpha):
    self.minPowerRating = minPowerRating
    self.maxPowerRating = maxPowerRating
    self.powerRating = minPowerRating
    self.alpha = alpha
    self.loadName = loadName

  def setPower(self, powerToBeSet):
    if powerToBeSet > self.maxPowerRating:
      self.powerRating = self.maxPowerRating
    elif powerToBeSet < self.minPowerRating:
      self.powerRating = self.minPowerRating
    else:
      self.powerRating = powerToBeSet

  def takeOneTimestep(self):
    return self.powerRating

  def checkStatus(self):
    print(self.loadName)
    print("Min Power Rating:", self.minPowerRating)
    print("Max Power Rating:", self.maxPowerRating)
    print("Current Power Set:", self.powerRating)
    print("Alpha", self.alpha)





class ShiftableInterruptible():
  def __init__(self, loadName, powerRating, startTime, endTime, requiredHours):
    self.loadName = loadName
    self.powerRating = powerRating
    self.startTime = startTime
    self.endTime = endTime
    self.requiredHours = requiredHours
    self.requiredHoursRemaining = requiredHours
    self.isOn = 0

  def initiateLoad(self):
    if self.requiredHoursRemaining > 0:
      self.isOn = 1 
    else:
      print("Already completed todays load usage")

  def takeOneTimestep(self):
    if self.isOn == 1:
      self.requiredHoursRemaining -= 1
      print(self.loadName, "is running this hour, required hours remaining today:", self.requiredHoursRemaining)
      self.isOn = 0
      return self.powerRating
    else:
      return 0

  def resetDay(self):
    self.requiredHoursRemaining = self.requiredHours

  def checkStatus(self):
    print(self.loadName)
    print("Power Rating:", self.powerRating)
    print("Required Hours per Day:", self.requiredHours)
    print("Required Hours Remaining Today:", self.requiredHoursRemaining)
    print("Time Allotted for Load:", self.startTime, "to", self.endTime)





class ShiftableUninterruptible():
  def __init__(self, loadName, powerRating, startTime, endTime, requiredHours):
    self.loadName = loadName
    self.powerRating = powerRating
    self.startTime = startTime
    self.endTime = endTime
    self.requiredHours = requiredHours
    self.requiredHoursRemaining = requiredHours
    self.isOn = 0

  def initiateLoad(self):
    if self.isOn:
      print(self.loadName, "already on")
    elif self.requiredHoursRemaining > 0:
      self.isOn = 1 
    else:
      print("Already completed todays load usage")
     

  def takeOneTimestep(self):
    if self.isOn == 1:
      self.requiredHoursRemaining -= 1
      print(self.loadName, "is running this hour, required hours remaining today:", self.requiredHoursRemaining)
      if self.requiredHoursRemaining == 0:
        self.isOn = 0
      return self.powerRating
    else:
      return 0
  

  def resetDay(self):
    self.requiredHoursRemaining = self.requiredHours

  def checkStatus(self):
    print(self.loadName)
    print("Power Rating:", self.powerRating)
    print("Required Hours per Day:", self.requiredHours)
    print("Required Hours Remaining Today:", self.requiredHoursRemaining)
    print("Time Allotted for Load:", self.startTime, "to", self.endTime)


# Setting Seeds




import numpy as np
import torch
SEEDLIST = [10129,10353,22373,54284,35519,40046,75647,66957,85409,92451]
DATASEED = 10

def set_seeds(seed):
    torch.manual_seed(seed)  # Sets seed for PyTorch RNG
    torch.cuda.manual_seed_all(seed)  # Sets seeds of GPU RNG
    np.random.seed(seed=seed)  # Set seed for NumPy RNG
    random.seed(seed)  # Set seed for random RNG


# Define Loads




import tomli

with open("home.toml", "rb") as f:
    toml_dict = tomli.load(f)





for key in toml_dict:
  print(key, toml_dict[key])





fridge = CriticalLoad(toml_dict["CR"][0]["id"], toml_dict["CR"][0]["P"])
alarm = CriticalLoad(toml_dict["CR"][1]["id"], toml_dict["CR"][1]["P"])
 
heater = AdjustableLoad(toml_dict["AD"][0]["id"],toml_dict["AD"][0]["Pmin"], toml_dict["AD"][0]["Pmax"], toml_dict["AD"][0]["α"])
aircon1 = AdjustableLoad(toml_dict["AD"][1]["id"],toml_dict["AD"][1]["Pmin"], toml_dict["AD"][1]["Pmax"], toml_dict["AD"][1]["α"])
aircon2 = AdjustableLoad(toml_dict["AD"][2]["id"],toml_dict["AD"][2]["Pmin"], toml_dict["AD"][2]["Pmax"], toml_dict["AD"][2]["α"])

washingMachine = ShiftableUninterruptible(toml_dict["SU"][0]["id"], toml_dict["SU"][0]["P"], toml_dict["SU"][0]["ts"], toml_dict["SU"][0]["tf"], toml_dict["SU"][0]["L"])
dishWasher = ShiftableUninterruptible(toml_dict["SU"][1]["id"], toml_dict["SU"][1]["P"], toml_dict["SU"][1]["ts"], toml_dict["SU"][1]["tf"], toml_dict["SU"][1]["L"])

electricVehicle = ShiftableInterruptible(toml_dict["SI"][0]["id"], toml_dict["SI"][0]["P"], toml_dict["SI"][0]["ts"], toml_dict["SI"][0]["tf"], toml_dict["SI"][0]["L"])

loadList = [fridge, alarm, heater, aircon1, aircon2, washingMachine, dishWasher, electricVehicle]





for load in loadList:
  load.checkStatus()
  print("\n")


# Define Scenarios




import numpy as np
data2019 = np.load('2019.npy')
data2020 = np.load('2020.npy')
data2021 = np.load('2021.npy')
data2019and2020 =np.concatenate((data2019, data2020), axis=2)
allData = np.concatenate((data2019and2020, data2021), axis=2)





print("2019", data2019.shape)
print("2020", data2020.shape)
print("2021", data2021.shape)
print("2019 and 2020", data2019and2020.shape)





import random
random.seed(10)
#Extract Validation Set, 10 percent
validation = random.sample(range(365+366), 70)
validationSet = []
trainingSet = []
trainingFullSet = []
for i in range(365+366):
  trainingFullSet.append(data2019and2020[:,0,i])
  if i in validation:
    validationSet.append(data2019and2020[:,0,i])
  else:
    trainingSet.append(data2019and2020[:,0,i])

validationData = np.asarray(validationSet)
trainingData = np.asarray(trainingSet)
trainingFullData = np.asarray(trainingFullSet)
print(validationData.shape)
print(trainingData.shape)

testingSet = data2021[:,0,:]
testingData = np.transpose(testingSet)
print(testingData.shape)

allDataTemp = allData[:,0,:]
allDataProcessed = np.transpose(allDataTemp)
print(allDataProcessed.shape)





#solar data
solarTestingSet = data2021[:,1,:]
solarTestingData = np.transpose(solarTestingSet)
print(solarTestingData.shape)

solarTrainingSet = data2019and2020[:,1,:]
solarTrainingData = np.transpose(solarTrainingSet)
print(solarTrainingData.shape)





print(np.min(validationData), np.max(validationData))
print(np.min(trainingData), np.max(trainingData))
print(np.min(testingData), np.max(testingData))
print(len(trainingData))





trainingData[1]


# Pipeline Adjustable Load




import gym
from gym import spaces

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  # metadata = {'render.modes': ['human']}

  def __init__(self, load, set):
    super(CustomEnv, self).__init__()
    self.load = load
    self.solar = None
    self.price = [0.1061027 , 0.10518237, 0.10348876, 0.1025972 , 0.10004596,
       0.10728957, 0.14616401, 0.14855754, 0.13336482, 0.14566902,
       0.13109758, 0.13705286, 0.20942042, 0.13099825, 0.12543957,
       0.12171211, 0.15030475, 0.15189125, 0.13866458, 0.13601246,
       0.13050256, 0.11916902, 0.11551519, 0.11015341]
    self.set = set
    if set == "training":
      self.price = trainingData[0]
    elif set == "validation":
      self.price =  validationData[0] 
    elif set == "testing":
      self.price = testingData[0]
    else:
      print("No set specified, will use dummy price data")
    self.hour = 0
    self.episodeCount = 0

    self.scenario = None

    #we define the action space: loadDispatch
    self.action_space = spaces.Box(low = np.array([self.load.minPowerRating]), high = np.array([self.load.maxPowerRating]))
    # we define the state space: price
    self.observation_space = spaces.Box(low=np.array([0]), high=np.array([999]))



  def step(self, action):
    powerSet = action[0]
    reward = -1 * (self.price[self.hour] * powerSet + self.load.alpha*(self.load.maxPowerRating - powerSet)**2)
    #done flagged only when its the last hour of the day
    done = False
    if self.hour >= 23:
      done = True
      self.episodeCount += 1
      
    #info is always set to nothing
    info = {}
    #update the hour so the next data is corret, but for the last piece of data we dont wanna overflow
    self.hour += 1

    if done:
      self.hour -= 1
    observation = [999]


    return np.array(observation), reward, done, info
    
  def reset(self):
    if self.set == "training":
      exampleX = random.randint(0,len(trainingData)-1)
      self.price = trainingData[exampleX]
    elif self.set == "validation":
      self.price =  validationData[self.episodeCount%len(validationData)] 
    elif self.set == "testing":
      self.price = testingData[self.episodeCount%len(testingData)]
    else:
      print("No set specified, will use dummy price data")

    #initial solar, wind, totalLoads, price, SOC
    observation = [self.price[0]]
    self.hour = 0
    return np.array(observation)  # reward, done, info can't be included





from stable_baselines3.common.env_checker import check_env

env = CustomEnv(heater, "training")
check_env(env)


# Heater




import gym
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
# SET SEED HERE
set_seeds(22373)
env = CustomEnv(heater, "training")
env = Monitor(env, "heaterEnv")
model = PPO("MlpPolicy", env, verbose=0)

validationCostArray = []
for episode in range(1500):
    #train first on one training episode
    print("training heater" , "episode", episode)
    env.reset()
    model.learn(total_timesteps=24)
    validationTotalCost = 0
    #then check cost on validation set
    for validationScenario in range(70):
        envValidation = CustomEnv(heater, "validation")
        obs = envValidation.reset()
        episodeReward = 0
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = envValidation.step(action)
            # print(obs, rewards, dones, info)
            episodeReward += rewards
            if dones == True:
                break
        validationTotalCost += episodeReward
    validationCostArray.append(validationTotalCost/70)
    print("Average Cost on Validation Set:", str(validationTotalCost/70))
    np.savetxt('PPO_Heater_22373_validationCost.txt', np.asarray(validationCostArray), delimiter=',')
    model.save("PPO_Heater_22373")


# Aircon1




import gym
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3 import DDPG

#SET SEED HERE
set_seeds(10129)
# set_seeds(10353)
# set_seeds(22373)
env = CustomEnv(aircon1, "training")
env = Monitor(env, "aircon1Env")
model = PPO("MlpPolicy", env, verbose=0)

validationCostArray = []
for episode in range(1500):
    #train first on one training episode
    print("training aircon1" , "episode", episode)
    env.reset()
    model.learn(total_timesteps=24)
    validationTotalCost = 0
    #then check cost on validation set
    for validationScenario in range(70):
        envValidation = CustomEnv(aircon1, "validation")
        obs = envValidation.reset()
        episodeReward = 0
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = envValidation.step(action)
            # print(obs, rewards, dones, info)
            episodeReward += rewards
            if dones == True:
                break
        validationTotalCost += episodeReward
    validationCostArray.append(validationTotalCost/70)
    print("Average Cost on Validation Set:", str(validationTotalCost/70))
    np.savetxt('PPO_Aircon1_10129_validationCost.txt', np.asarray(validationCostArray), delimiter=',')
    model.save("PPO_Aircon1_10129")


# Aircon2




import gym
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3 import DDPG

#SET SEED HERE
set_seeds(10129)
# set_seeds(10353)
# set_seeds(22373)
env = CustomEnv(aircon2, "training")
env = Monitor(env, "aircon2Env")
model = PPO("MlpPolicy", env, verbose=0)

validationCostArray = []
for episode in range(1500):
    #train first on one training episode
    print("training aircon2" , "episode", episode)
    env.reset()
    model.learn(total_timesteps=24)
    validationTotalCost = 0
    #then check cost on validation set
    for validationScenario in range(70):
        envValidation = CustomEnv(aircon2, "validation")
        obs = envValidation.reset()
        episodeReward = 0
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = envValidation.step(action)
            # print(obs, rewards, dones, info)
            episodeReward += rewards
            if dones == True:
                break
        validationTotalCost += episodeReward
    validationCostArray.append(validationTotalCost/70)
    print("Average Cost on Validation Set:", str(validationTotalCost/70))
    np.savetxt('PPO_Aircon2_10129_validationCost.txt', np.asarray(validationCostArray), delimiter=',')
    model.save("PPO_Aircon2_10129")

