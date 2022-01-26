# -*- coding: utf-8 -*-
"""Submittable DQN.ipynb

# Define Classes for Loads
"""

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

"""# Setting Seeds

"""

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

"""# Start Defining Loads"""

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



# fridge = CriticalLoad('REF', 0.5)
# alarm = CriticalLoad('AS', 0.2)

# heater = AdjustableLoad('H', 0.0, 3.5, 0.1)
# aircon1 = AdjustableLoad('AC1', 0.0, 2, 0.2)
# aircon2 = AdjustableLoad('AC2', 0.0, 3, 0.1)
# lLoad = AdjustableLoad('L', 0.0, 0.6, 0.4)

# washingMachine = ShiftableUninterruptible('WM', 1.5, 18, 23, 2)
# dishWasher = ShiftableUninterruptible('DW', 0.95, 21, 24, 2)

# electricVehicle = ShiftableInterruptible('EV', 5, 9, 17, 3)

# loadList = [fridge, alarm, heater, aircon1, aircon2, lLoad, washingMachine, dishWasher, electricVehicle]

for load in loadList:
  load.checkStatus()
  print("\n")

"""# Start Loading in Scenarios"""

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

from sklearn.preprocessing import KBinsDiscretizer
est = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
est.fit(allDataProcessed)
validationDataDiscrete = est.transform(validationData)
trainingDataDiscrete = est.transform(trainingData)
testingDataDiscrete = est.transform(testingData)
trainingFullDataDiscrete = est.transform(trainingFullData)

print(validationDataDiscrete.shape)
print(trainingDataDiscrete.shape)
print(testingDataDiscrete.shape)
print(np.min(validationDataDiscrete), np.max(validationDataDiscrete))
print(np.min(trainingDataDiscrete), np.max(trainingDataDiscrete))
print(np.min(testingDataDiscrete), np.max(testingDataDiscrete))

"""# Pipeline for Adjustable Loads
the cost for adjustable loads consist of eletrcity cost: price * power
and also discomfort cost: alpha * (maxPower - currentPower)
"""

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
print(T.__version__)

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x) 

        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size = 1_000_000,
                eps_end = 0.01, eps_dec = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        #new addition
        self.reserveFlag = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                            fc1_dims=100, fc2_dims=50)

        self.target_Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                            fc1_dims=100, fc2_dims=50)
                            
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype= np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype= np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype= np.bool)
        self.freq_table = np.zeros((100, 11))
    
    def store_transition(self, state, action, reward, state_, done):
        if self.mem_cntr == 50_000:
            self.reserveFlag = 1
        if self.reserveFlag == 1:
            index = self.mem_cntr % (self.mem_size - 50_000)
            index += 50_000
            self.state_memory[index] = state
            self.new_state_memory[index] = state_
            self.reward_memory[index] = reward
            self.action_memory[index] = action
            self.terminal_memory[index] = done
        else:    
            index = self.mem_cntr % self.mem_size
            self.state_memory[index] = state
            self.new_state_memory[index] = state_
            self.reward_memory[index] = reward
            self.action_memory[index] = action
            self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.Tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self, sync):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.target_Q_eval.forward(new_state_batch)
        
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * q_next[batch_index, action_batch]
        #calculate loss
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()


        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        if sync:
            self.target_Q_eval.load_state_dict(self.Q_eval.state_dict())

import random
#setting up world
DISCOUNT_FACTOR = 0.99 #this is an undiscounted MDP
ACTIONS = [0,1,2,3,4,5,6,7,8,9,10]
PRICEDISCRETIZATION = 100


def run_AdjustableQLEARNING(agent, adjustableLoad, data):
    hour = 0
    observation = data[0]
    totalRewards = 0.0
    auditList = []
    while (hour < 24):
        action = agent.choose_action(observation)
        observation_ = data[hour]
        powerSet = action*((adjustableLoad.maxPowerRating-adjustableLoad.minPowerRating) /10) +adjustableLoad.minPowerRating
        reward = -(data[hour]*(powerSet) + adjustableLoad.alpha*(adjustableLoad.maxPowerRating - powerSet)**2)
        done = 1
        info = {}

        totalRewards += reward
        agent.store_transition(observation, action, reward, observation_, done)

        auditList.append(action)


        observation = observation_
        hour += 1
        auditList.append(action)
    agent.learn(sync = False)
    return totalRewards, auditList


def test_AdjustableQLEARNING(agent, adjustableLoad, data):
    hour = 0
    observation = data[0]
    totalRewards = 0.0
    auditList = []
    while (hour < 24):
        action = agent.choose_action(observation)
        observation_ = data[hour]
        powerSet = action*((adjustableLoad.maxPowerRating-adjustableLoad.minPowerRating) /10) +adjustableLoad.minPowerRating
        reward = -(data[hour]*(powerSet) + adjustableLoad.alpha*(adjustableLoad.maxPowerRating - powerSet)**2)
        done = 0
        info = {}

        totalRewards += reward

        auditList.append(powerSet)
        observation = observation_
        hour += 1
    return totalRewards, auditList

def runAdjustableLoad(run, seed, load):
  set_seeds(seed)
  agent = Agent(gamma=1, epsilon=1, batch_size=50, n_actions=11,
                      eps_end=0.01, input_dims=[1], lr=0.0005)
  #Training
  convergenceCurve = []
  trainingCurve = []
  q_tableSum = []
  for run in range(1):
    for episode in range(1500):
      print("run", run, "training", load.loadName, "episode", episode)
      agent.epsilon = 0.1
      exampleX = random.randint(0,len(trainingDataDiscrete)-1)
      qlearning_episode_rewards, auditList = run_AdjustableQLEARNING(agent, load, trainingData[exampleX])
      if episode%20 == 0:
        agent.learn(sync = True)
      validationReward = 0
      for validationScenario in range(70):

        agent.epsilon = 0
        validation_average_qlearning_episode_rewards, auditList = test_AdjustableQLEARNING(agent, load, validationData[validationScenario])
        validationReward += validation_average_qlearning_episode_rewards
      convergenceCurve.append(validationReward/70)
      trainingCurve.append(qlearning_episode_rewards)

  TestingResults = []
  TestingAuditList = []
  print("now testing", load.loadName)
  for testEpisode in range(len(testingDataDiscrete)):
    agent.epsilon = 0
    test_qlearning_episode_rewards, auditList  = test_AdjustableQLEARNING(agent, load, testingData[testEpisode])
    TestingResults.append(test_qlearning_episode_rewards)
    TestingAuditList.append(auditList)

  generalisationResults = []
  generalisationAuditList = []
  print("now testing with training set", load.loadName)
  for testEpisode in range(len(trainingFullData)):
    agent.epsilon = 0
    test_qlearning_episode_rewards, auditList = test_AdjustableQLEARNING(agent, load, trainingFullData[testEpisode])
    generalisationResults.append(test_qlearning_episode_rewards)
    generalisationAuditList.append(auditList)
  return convergenceCurve, TestingResults, generalisationResults, TestingAuditList, generalisationAuditList


class ShiftableAgent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size = 1_000_000,
                eps_end = 0, eps_dec = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        #new addition
        self.reserveFlag = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                            fc1_dims=100, fc2_dims=50)

        self.target_Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                            fc1_dims=100, fc2_dims=50)
                            
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype= np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype= np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype= np.bool)
    
    def store_transition(self, state, action, reward, state_, done):
        if self.mem_cntr == 50_000:
            self.reserveFlag = 1
        if self.reserveFlag == 1:
            index = self.mem_cntr % (self.mem_size - 50_000)
            index += 50_000
            self.state_memory[index] = state
            self.new_state_memory[index] = state_
            self.reward_memory[index] = reward
            self.action_memory[index] = action
            self.terminal_memory[index] = done
        else:    
            index = self.mem_cntr % self.mem_size
            self.state_memory[index] = state
            self.new_state_memory[index] = state_
            self.reward_memory[index] = reward
            self.action_memory[index] = action
            self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation, shiftableLoad):
        if np.random.random() > self.epsilon:
            state = T.Tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        if shiftableLoad.requiredHoursRemaining <= 0:
            return 0
        elif shiftableLoad.isOn:
            return 1
        return action

    def learn(self, sync):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]

        q_next = self.target_Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0


        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        #calculate loss
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()


        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        for target_param, local_param in zip(self.target_Q_eval.parameters(), self.Q_eval.parameters()):
            target_param.data.copy_((1e-3)*local_param.data + (1.0-(1e-3))*target_param.data)
        

import random
DISCOUNT_FACTOR = 1 #this is an undiscounted MDP
ACTIONSbinary = [0,1]


def checkLegalShiftableUn(hour, shiftableLoad, interimAction):
  #now we check if the interim action is legal
  if shiftableLoad.requiredHoursRemaining <= 0:
    return 0

  else:
    if shiftableLoad.isOn == 1:
      return 1

    else:
      if interimAction == 1:
        return interimAction
      #if not, we need to look ahead and check if there is still room for the agent to do its duty by the given time
      else:
        if hour+shiftableLoad.requiredHoursRemaining < shiftableLoad.endTime:
          return interimAction
        #there is no choice, you need to turn it on now to finish by the end time
        else:
          return 1    

def run_ShiftableUnQLEARNING(agent, shiftableLoad, data):
    hour = 0
    totalRewards = 0.0
    window = shiftableLoad.endTime - shiftableLoad.startTime
    shiftableLoad.requiredHoursRemaining = shiftableLoad.requiredHours
    observation = [data[0], shiftableLoad.requiredHoursRemaining, window, hour]
    auditList = []
    legalAuditList = []
    while (hour < 24):
        reward = 0
        action = 0
        interimAction = 0
        if hour >= shiftableLoad.startTime and hour < shiftableLoad.endTime:
          interimAction = agent.choose_action(observation, shiftableLoad)
          action = checkLegalShiftableUn(hour, shiftableLoad, interimAction)
          if interimAction == 1:
            shiftableLoad.requiredHoursRemaining -= 1
          window -= 1
          observation_ =  [data[hour], shiftableLoad.requiredHoursRemaining, window, hour]
          info = {}
          reward = -(data[hour]*(interimAction)*shiftableLoad.powerRating)
          totalRewards += reward
          shiftableLoad.isOn = interimAction
          done = 1
        
          agent.store_transition(observation, interimAction, reward, observation_, done)
          agent.learn(sync = False)
          observation = observation_
        auditList.append(interimAction)
        legalAuditList.append(action)
        hour += 1
    if(auditList.count(1) != 2):
      totalRewards = -(data[shiftableLoad.endTime -2] + data[shiftableLoad.endTime -1]) * shiftableLoad.powerRating
      auditList = legalAuditList
    agent.learn(sync = True)
    return totalRewards, auditList


def test_ShiftableUnQLEARNING(agent, shiftableLoad, data):
    hour = 0
    totalRewards = 0.0
    window = shiftableLoad.endTime - shiftableLoad.startTime
    shiftableLoad.requiredHoursRemaining = shiftableLoad.requiredHours
    observation = [data[0], shiftableLoad.requiredHoursRemaining, window, hour]
    auditList = []
    legalAuditList = []
    while (hour < 24):
        reward = 0
        action = 0
        interimAction = 0
        if hour >= shiftableLoad.startTime and hour < shiftableLoad.endTime:
          interimAction = agent.choose_action(observation, shiftableLoad)
          action = checkLegalShiftableUn(hour, shiftableLoad, interimAction)
          if interimAction == 1:
            shiftableLoad.requiredHoursRemaining -= 1
          window -= 1
          observation_ =  [data[hour], shiftableLoad.requiredHoursRemaining, window, hour]
          info = {}
          reward = -(data[hour]*(interimAction)*shiftableLoad.powerRating)
          totalRewards += reward
          shiftableLoad.isOn = interimAction
          done = 0
        
          observation = observation_
        auditList.append(interimAction)
        legalAuditList.append(action)
        hour += 1
    if(auditList.count(1) != shiftableLoad.requiredHours):
      totalRewards = -(data[shiftableLoad.endTime -2] + data[shiftableLoad.endTime -1]) * shiftableLoad.powerRating
      auditList = legalAuditList

    return totalRewards, auditList



def runShiftableUnLoad(seed, load):
  set_seeds(seed)
  averageConvergenceCurve = np.zeros((1500))
  for run in range(1):
    agent = ShiftableAgent(gamma=1, epsilon=1, batch_size=10, n_actions=2,
                        eps_end=0.1, input_dims=[4], lr=0.00003)
    #Training
    convergenceCurve = []
    trainingCurve = []
    qvaluesSum = []
    for episode in range(1500):
      print("run", run, "training", load.loadName, "episode", episode)
      tempEpsilon = agent.epsilon
      exampleX = random.randint(0,len(trainingDataDiscrete)-1)
      qlearning_episode_rewards, auditList = run_ShiftableUnQLEARNING(agent, load, trainingData[exampleX])

      validationReward = 0
      for validationScenario in range(70):
        agent.epsilon = 0
        validation_average_qlearning_episode_rewards, auditList = test_ShiftableUnQLEARNING(agent, load , validationData[validationScenario])
        
        validationReward += validation_average_qlearning_episode_rewards
      convergenceCurve.append(validationReward/70)
      trainingCurve.append(qlearning_episode_rewards)
      agent.epsilon = tempEpsilon
      print("run", run, "episode", episode, "done", "Episode Reward",qlearning_episode_rewards ,"Validation reward", validationReward/70)

  TestingResults = []
  TestingAuditList = []
  print("now testing with test set", load.loadName)
  for testEpisode in range(len(testingDataDiscrete)):
    agent.epsilon = 0
    test_qlearning_episode_rewards, auditList = test_ShiftableUnQLEARNING(agent, load, testingData[testEpisode])
    print(auditList)
    TestingResults.append(test_qlearning_episode_rewards)
    TestingAuditList.append(auditList)

  generalisationResults = []
  generalisationAuditList = []
  print("now testing with training set", load.loadName)
  for testEpisode in range(len(trainingFullData)):
    agent.epsilon = 0
    test_qlearning_episode_rewards, auditList = test_ShiftableUnQLEARNING(agent, load, trainingFullData[testEpisode])
    generalisationResults.append(test_qlearning_episode_rewards)
    generalisationAuditList.append(auditList)
  return convergenceCurve, TestingResults, generalisationResults, TestingAuditList, generalisationAuditList




class ShiftableInterruptibleAgent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size = 1_000_000,
                eps_end = 0.001, eps_dec = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        #new addition
        self.reserveFlag = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                            fc1_dims=100, fc2_dims=50)

        self.target_Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                            fc1_dims=100, fc2_dims=50)
                            
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype= np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype= np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype= np.bool)
    
    def store_transition(self, state, action, reward, state_, done):
        if self.mem_cntr == 50_000:
            self.reserveFlag = 1
        if self.reserveFlag == 1:
            index = self.mem_cntr % (self.mem_size - 50_000)
            index += 50_000
            self.state_memory[index] = state
            self.new_state_memory[index] = state_
            self.reward_memory[index] = reward
            self.action_memory[index] = action
            self.terminal_memory[index] = done
        else:    
            index = self.mem_cntr % self.mem_size
            self.state_memory[index] = state
            self.new_state_memory[index] = state_
            self.reward_memory[index] = reward
            self.action_memory[index] = action
            self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation, shiftableInterruptibleLoad, window):
        if np.random.random() > self.epsilon:
            state = T.Tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        if shiftableInterruptibleLoad.requiredHoursRemaining <= 0:
            return 0

        else:
          if action == 1:
            return action
          #if not, we need to look ahead and check if there is still room for the agent to do its duty by the given time
          else:
            if shiftableInterruptibleLoad.requiredHoursRemaining <  window:
              return action
            #there is no choice, you need to turn it on now to finish by the end time
            else:
              return 1
        return action

    def learn(self, sync):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]

        q_next = self.target_Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0


        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        #calculate loss
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()


        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        for target_param, local_param in zip(self.target_Q_eval.parameters(), self.Q_eval.parameters()):
            target_param.data.copy_((1e-3)*local_param.data + (1.0-(1e-3))*target_param.data)
        


def checkLegalShiftableInt(hour, shiftableInterruptibleLoad, interimAction):
        #now we check if the interim action is legal
        if shiftableInterruptibleLoad.requiredHoursRemaining <= 0:
          return 0
        else:
          if interimAction == 1:
            return interimAction
          #if not, we need to look ahead and check if there is still room for the agent to do its duty by the given time
          else:
            if hour+shiftableInterruptibleLoad.requiredHoursRemaining < shiftableInterruptibleLoad.endTime:
              return interimAction
            #there is no choice, you need to turn it on now to finish by the end time
            else:
              return 1   

def run_ShiftableIntQLEARNING(agent, shiftableInterruptibleLoad, data):
    hour = 0
    totalRewards = 0.0
    window = shiftableInterruptibleLoad.endTime - shiftableInterruptibleLoad.startTime
    shiftableInterruptibleLoad.requiredHoursRemaining = shiftableInterruptibleLoad.requiredHours
    observation = [data[0], shiftableInterruptibleLoad.requiredHoursRemaining, window, hour]
    auditList = []
    legalAuditList = []
    while (hour < 24):
        reward = 0
        action = 0
        interimAction = 0
        if hour >= shiftableInterruptibleLoad.startTime and hour < shiftableInterruptibleLoad.endTime:
          interimAction = agent.choose_action(observation, shiftableInterruptibleLoad, window)
          action = checkLegalShiftableInt(hour, shiftableInterruptibleLoad, interimAction)
          if interimAction == 1:
            shiftableInterruptibleLoad.requiredHoursRemaining -= 1
          window -= 1
          observation_ =  [data[hour], shiftableInterruptibleLoad.requiredHoursRemaining, window, hour]
          info = {}
          reward = -(data[hour]*(interimAction)*shiftableInterruptibleLoad.powerRating)
          totalRewards += reward
          shiftableInterruptibleLoad.isOn = interimAction
          done = 1
        
          agent.store_transition(observation, interimAction, reward, observation_, done)
          agent.learn(sync = False)
          observation = observation_
        auditList.append(interimAction)
        legalAuditList.append(action)
        hour += 1
    if(auditList.count(1) != shiftableInterruptibleLoad.requiredHours):
      totalRewards = -(data[shiftableInterruptibleLoad.endTime -4] + data[shiftableInterruptibleLoad.endTime -3] + data[shiftableInterruptibleLoad.endTime -2] + data[shiftableInterruptibleLoad.endTime -1]) * shiftableInterruptibleLoad.powerRating
      auditList = legalAuditList
    agent.learn(sync = True)
    return totalRewards, auditList


def test_ShiftableIntQLEARNING(agent, shiftableInterruptibleLoad, data):
    hour = 0
    totalRewards = 0.0
    window = shiftableInterruptibleLoad.endTime - shiftableInterruptibleLoad.startTime
    shiftableInterruptibleLoad.requiredHoursRemaining = shiftableInterruptibleLoad.requiredHours
    observation = [data[0], shiftableInterruptibleLoad.requiredHoursRemaining, window, hour]
    auditList = []
    legalAuditList = []
    while (hour < 24):
        reward = 0
        action = 0
        interimAction = 0
        if hour >= shiftableInterruptibleLoad.startTime and hour < shiftableInterruptibleLoad.endTime:
          interimAction = agent.choose_action(observation, shiftableInterruptibleLoad, window)
          action = checkLegalShiftableInt(hour, shiftableInterruptibleLoad, interimAction)
          if interimAction == 1:
            shiftableInterruptibleLoad.requiredHoursRemaining -= 1
          window -= 1
          observation_ =  [data[hour], shiftableInterruptibleLoad.requiredHoursRemaining, window, hour]
          info = {}
          reward = -(data[hour]*(interimAction)*shiftableInterruptibleLoad.powerRating)
          totalRewards += reward
          shiftableInterruptibleLoad.isOn = interimAction
          done = 1
        
          observation = observation_
        auditList.append(interimAction)
        legalAuditList.append(action)
        hour += 1
    if(auditList.count(1) != shiftableInterruptibleLoad.requiredHours):
      totalRewards = -(data[shiftableInterruptibleLoad.endTime -4] + data[shiftableInterruptibleLoad.endTime -3] + data[shiftableInterruptibleLoad.endTime -2] + data[shiftableInterruptibleLoad.endTime -1]) * shiftableInterruptibleLoad.powerRating
      auditList = legalAuditList
    return totalRewards, auditList


def runShiftableIntLoad(run, seed, load):
  set_seeds(seed)
  for run in range(1):
    
    agent = ShiftableInterruptibleAgent(gamma=1, epsilon=1, batch_size=10, n_actions=2,
                        eps_end=0.01, input_dims=[4], lr=0.00001)
    #Training
    convergenceCurve = []
    trainingCurve = []
    qvaluesSum = []
    for run in range(1):
      for episode in range(1500):
        print("run", run, "training", load.loadName, "episode", episode)
        agent.epsilon = 0.1
        qlearning_episode_rewards, auditList = run_ShiftableIntQLEARNING(agent, electricVehicle, trainingData[episode%len(trainingData)])
        validationReward = 0
        for validationScenario in range(10):
          agent.epsilon = 0
          validation_average_qlearning_episode_rewards, auditList = test_ShiftableIntQLEARNING(agent, electricVehicle, validationData[validationScenario])
          validationReward += validation_average_qlearning_episode_rewards
        convergenceCurve.append(validationReward/10)
        trainingCurve.append(qlearning_episode_rewards)
  TestingResults = []
  TestingAuditList = []
  print("now testing with training set", load.loadName)
  for testEpisode in range(len(testingDataDiscrete)):
    agent.epsilon = 0
    test_qlearning_episode_rewards, auditList = test_ShiftableIntQLEARNING(agent, electricVehicle, testingData[testEpisode])
    TestingResults.append(test_qlearning_episode_rewards)
    TestingAuditList.append(auditList)

  generalisationResults = []
  generalisationAuditList = []
  print("now testing with training set", load.loadName)
  for testEpisode in range(len(trainingFullData)):
    agent.epsilon = 0
    test_qlearning_episode_rewards, auditList = test_ShiftableIntQLEARNING(agent, electricVehicle, trainingFullData[testEpisode])
    generalisationResults.append(test_qlearning_episode_rewards)
    generalisationAuditList.append(auditList)
  return convergenceCurve, TestingResults, generalisationResults, TestingAuditList, generalisationAuditList



def calculateCost(actionListH, actionListAC1, actionListAC2, actionListWM, actionListDW, actionListEV,
                  heater, aircon1, aircon2, washingMachine, dishWasher, electricVehicle, fridge, alarm,
                  price, solarGeneration):
  
  cost = []
  for i in range(len(price)):
    print("now calculating cost for day", i, "of day", len(price))
    totalElectricCost = 0
    totalDiscomfortCost = 0
    for j in range(24):
      powerUseThisHour = 0
      powerUseThisHour += fridge.powerRating
      powerUseThisHour += alarm.powerRating
      powerUseThisHour += actionListH[i][j]
      powerUseThisHour += actionListAC1[i][j]
      powerUseThisHour += actionListAC2[i][j]
      # powerUseThisHour += actionListlLoad[i][j]
      powerUseThisHour += actionListWM[i][j]*washingMachine.powerRating
      powerUseThisHour += actionListDW[i][j]*dishWasher.powerRating
      powerUseThisHour += actionListEV[i][j]*electricVehicle.powerRating
      powerUseThisHour -= solarGeneration[i,j]

      if powerUseThisHour >= 0:
        totalElectricCost += price[i,j]*abs(powerUseThisHour)
      else:
        totalElectricCost -= 0.5*price[i,j]*abs(powerUseThisHour)

      totalDiscomfortCost += heater.alpha*(heater.maxPowerRating - actionListH[i][j])**2
      totalDiscomfortCost += aircon1.alpha*(aircon1.maxPowerRating - actionListAC1[i][j])**2 
      totalDiscomfortCost += aircon2.alpha*(aircon2.maxPowerRating - actionListAC2[i][j])**2


    cost.append(totalElectricCost + totalDiscomfortCost)  
  return cost

import time
trainingCurveEVavg = np.zeros((1500))
trainingCurveWMavg = np.zeros((1500))
trainingCurveDWavg = np.zeros((1500))
trainingCurveHavg = np.zeros((1500))
trainingCurveAC1avg = np.zeros((1500))
trainingCurveAC2avg = np.zeros((1500))
testingCostavg = np.zeros(len(testingData))
generalisationCostavg = np.zeros(len(trainingFullData))

for seed in range(10):

  trainingCurveEV, testCurveEV, generalisationCurveEV, actionListTestEV, actionListTrainEV = runShiftableIntLoad(seed, SEEDLIST[seed], electricVehicle)
  trainingCurveEVavg += trainingCurveEV

  trainingCurveH, testCurveH, generalisationCurveH, actionListTestH, actionListTrainH = runAdjustableLoad(seed, SEEDLIST[seed], heater)
  trainingCurveHavg += trainingCurveH
  trainingCurveAC1, testCurveAC1, generalisationCurveAC1, actionListTestAC1, actionListTrainAC1 = runAdjustableLoad(seed, SEEDLIST[seed], aircon1)
  trainingCurveAC1avg += trainingCurveAC1
  trainingCurveAC2, testCurveAC2, generalisationCurveAC2, actionListTestAC2, actionListTrainAC2 = runAdjustableLoad(seed, SEEDLIST[seed], aircon2)
  trainingCurveAC2avg += trainingCurveAC2




  trainingCurveWM, testCurveWM, generalisationCurveWM, actionListTestWM, actionListTrainWM = runShiftableUnLoad(seed, SEEDLIST[seed], washingMachine)
  trainingCurveWMavg += trainingCurveWM
  trainingCurveDW, testCurveDW, generalisationCurveDW, actionListTestDW, actionListTrainDW = runShiftableUnLoad(seed, SEEDLIST[seed], dishWasher)
  trainingCurveDWavg += trainingCurveDW





  testingCost = calculateCost(actionListTestH, actionListTestAC1, actionListTestAC2, actionListTestWM, actionListTestDW, actionListTestEV,
                    heater, aircon1, aircon2, washingMachine, dishWasher, electricVehicle, fridge, alarm,
                    testingData, solarTestingData)
  testingCostavg += testingCost


  generalisationCost = calculateCost(actionListTrainH, actionListTrainAC1, actionListTrainAC2, actionListTrainWM, actionListTrainDW, actionListTrainEV,
                    heater, aircon1, aircon2, washingMachine, dishWasher, electricVehicle, fridge, alarm,
                    trainingFullData, solarTrainingData)
  generalisationCostavg += generalisationCost

trainingCurveEVavg /= 10
trainingCurveWMavg /= 10
trainingCurveDWavg /= 10
trainingCurveHavg /= 10
trainingCurveAC1avg /= 10
trainingCurveAC2avg /= 10
testingCostavg /= 10
generalisationCostavg /= 10

np.save("trainingCurveEVavg.npy", trainingCurveEVavg)
np.save("trainingCurveWMavg.npy", trainingCurveWMavg)
np.save("trainingCurveDWavg.npy", trainingCurveDWavg)
np.save("trainingCurveHavg.npy", trainingCurveHavg)
np.save("trainingCurveAC1avg.npy", trainingCurveAC1avg)
np.save("trainingCurveAC2avg.npy", trainingCurveAC2avg)
np.save("testingCostavg.npy", testingCostavg)
np.save("generalisationCostavg.npy", generalisationCostavg)