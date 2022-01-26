# -*- coding: utf-8 -*-
"""Submittable Q Learning.ipynb


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

"""# Setting Seeds"""

import numpy as np
import torch
SEEDLIST = [10129,10353,22373,54284,35519,40046,75647,66957,85409,92451]
DATASEED = 10

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

import random
#setting up world
DISCOUNT_FACTOR = 1 #this is an undiscounted MDP
ACTIONS = [0,1,2,3,4,5,6,7,8,9,10]
PRICEDISCRETIZATION = 20


class AdjustableAgent:
    def __init__(self, adjustableLoad):
        self.epsilon = 0.1
        self.learningRate = 0.2
        self.q_table = np.zeros((PRICEDISCRETIZATION, len(ACTIONS)))
        self.freq_table = np.zeros((PRICEDISCRETIZATION, len(ACTIONS)))
        self.state = 0
        self.totalRewards = 0.0
        self.minPowerRating = adjustableLoad.minPowerRating
        self.maxPowerRating = adjustableLoad.maxPowerRating
        self.powerRating = adjustableLoad.minPowerRating
        self.alpha = adjustableLoad.alpha
        self.loadName = adjustableLoad.loadName
    
        
    def choose_action(self, state):
        i = int(state)
        x = random.uniform(0, 1)
        if x <= self.epsilon:
            return np.random.choice(ACTIONS)
        else:
            arr = self.q_table[i, :]
            best_action = np.where(arr == np.amax(arr)) #there might be ties            
            real_action = np.random.choice(np.asarray(best_action).flatten())
            return real_action

def run_QLEARNING(agent, data, dataNonDiscrete):
    hour = 0
    
    agent.totalRewards = 0.0
    auditList = []
    while (hour < 24):
        agent.state = data[hour]
        action = agent.choose_action(agent.state)
        powerSet = action*((agent.maxPowerRating-agent.minPowerRating) /10) +agent.minPowerRating
        reward = -(dataNonDiscrete[hour]*(powerSet) + agent.alpha*(agent.maxPowerRating - powerSet)**2)
        next_state = data[hour]
        agent.totalRewards += reward
        
        #update rule
        target = np.max(agent.q_table[next_state, :])

        agent.q_table[agent.state, action] = (1-agent.learningRate)*(agent.q_table[agent.state, action]) \
                                                                + (agent.learningRate)*(reward + DISCOUNT_FACTOR*0)

        hour += 1
        auditList.append(action)
    return agent.q_table, agent.totalRewards, auditList

def validate_QLEARNING(agent, data, dataNonDiscrete):
    hour = 0
    
    agent.totalRewards = 0.0
    auditList = []
    while (hour < 24):
        agent.state = data[hour]
        action = agent.choose_action(agent.state)
        powerSet = action*((agent.maxPowerRating-agent.minPowerRating) /10) +agent.minPowerRating
        reward = -(dataNonDiscrete[hour]*(powerSet) + agent.alpha*(agent.maxPowerRating - powerSet)**2)
        agent.totalRewards += reward
        
        hour += 1
        auditList.append(action)
    return agent.q_table, agent.totalRewards, auditList

def test_QLEARNING(agent, data, dataNonDiscrete):
    hour = 0
    
    agent.totalRewards = 0.0
    auditList = []
    while (hour < 24):
        agent.state = data[hour]
        action = 0
        arr = agent.q_table[agent.state, :]
        best_action = np.where(arr == np.amax(arr)) #there might be ties            
        real_action = min(np.random.choice(np.asarray(best_action).flatten())+2,10) #fix index, tiebreaking
        action = real_action
        powerSet = action*((agent.maxPowerRating-agent.minPowerRating) /10) +agent.minPowerRating
        reward = -(dataNonDiscrete[hour]*(powerSet) + agent.alpha*(agent.maxPowerRating - powerSet)**2)
        agent.totalRewards += reward
        
        hour += 1
        auditList.append(action)
    return agent.q_table, agent.totalRewards, auditList



def random_NOLEARNING(agent, data, dataNonDiscrete):
    hour = 0
    agent.state = [data[0], 0]
    agent.totalRewards = 0.0
    while (hour < 24):
        action = np.random.choice(ACTIONS)
        powerSet = action*(agent.maxPowerRating/10)
        reward = -(dataNonDiscrete[hour]*(powerSet) + agent.alpha*(agent.maxPowerRating - powerSet)**2)
        next_state = [data[hour],action]
        agent.totalRewards += reward
        
        agent.state = next_state
        hour += 1
    return agent.q_table, agent.totalRewards 



def runAdjustableLoad(run, seed, load):
  set_seeds(seed)
  Loadagent = AdjustableAgent(load)
  Loadagent.epsilon = 0.1
  #Training
  convergenceCurve = []
  trainingCurve = []
  q_tableSum = []
  for run in range(1):
    for episode in range(1500):
      print("run", run, "training", load.loadName, "episode", episode)
      exampleX = random.randint(0,len(trainingDataDiscrete)-1)
      qlearning_episode_qtable, qlearning_episode_rewards, auditList = run_QLEARNING(Loadagent, trainingDataDiscrete[exampleX].astype(int), trainingData[exampleX])

      validationReward = 0
      for validationScenario in range(70):
        validationAgent = AdjustableAgent(load)
        validationAgent.epsilon = 0
        validationAgent.q_table = qlearning_episode_qtable
        validation_qlearning_episode_qtable, validation_average_qlearning_episode_rewards, auditList = validate_QLEARNING(validationAgent, validationDataDiscrete[validationScenario].astype(int), validationData[validationScenario])
        validationReward += validation_average_qlearning_episode_rewards
      convergenceCurve.append(validationReward/70)
      trainingCurve.append(qlearning_episode_rewards)
      q_tableSum.append(np.sum(qlearning_episode_qtable))
  TestingResults = []
  TestingAuditList = []
  print("now testing", load.loadName)
  for testEpisode in range(len(testingDataDiscrete)):
    testAgent = AdjustableAgent(load)
    testAgent.epsilon = 0
    testAgent.q_table = qlearning_episode_qtable
    test_qlearning_episode_qtable, test_qlearning_episode_rewards, auditList = test_QLEARNING(testAgent, testingDataDiscrete[testEpisode].astype(int), testingData[testEpisode])
    TestingResults.append(test_qlearning_episode_rewards)
    TestingAuditList.append(auditList)

  generalisationResults = []
  generalisationAuditList = []
  print("now testing with training set", load.loadName)
  for testEpisode in range(len(trainingFullData)):
    testAgent = AdjustableAgent(load)
    testAgent.epsilon = 0
    testAgent.q_table = qlearning_episode_qtable
    test_qlearning_episode_qtable, test_qlearning_episode_rewards, auditList = test_QLEARNING(testAgent, trainingFullDataDiscrete[testEpisode].astype(int), trainingFullData[testEpisode])
    
    generalisationResults.append(test_qlearning_episode_rewards)
    generalisationAuditList.append(auditList)
  return convergenceCurve, TestingResults, generalisationResults, TestingAuditList, generalisationAuditList


import random
DISCOUNT_FACTOR = 1 #this is an undiscounted MDP
ACTIONSbinary = [0,1]
PRICEDISCRETIZATION = 20

class ShiftableAgent:
    def __init__(self, shiftableLoad):
        self.epsilon = 0.1
        self.learningRate = 0.1
        self.requiredHoursRemaining = shiftableLoad.requiredHours
        self.endTime = shiftableLoad.endTime
        self.startTime = shiftableLoad.startTime
        self.window = shiftableLoad.endTime-shiftableLoad.startTime
        self.q_table = np.zeros((PRICEDISCRETIZATION, self.requiredHoursRemaining+1, self.window+1, 24, len(ACTIONSbinary)))
        self.state = [0, 0, 0, 0]
        self.totalRewards = 0.0
        self.powerRating = shiftableLoad.powerRating
        self.startTime = shiftableLoad.startTime
        self.requiredHours = shiftableLoad.requiredHours
        self.requiredHoursRemaining = shiftableLoad.requiredHours
        self.isOn = 0
    
        
    def choose_action(self, state):
        i = int(state[0])
        j = int(state[1])
        k = int(state[2])
        l = int(state[3])
        x = random.uniform(0, 1)
        interimAction = 0
        if x <= self.epsilon:
            interimAction = np.random.choice(ACTIONSbinary)
        else:
            arr = self.q_table[i, j, k, l, :]
            best_action = np.where(arr == np.amax(arr)) #there might be ties            
            interimAction = np.random.choice(np.asarray(best_action).flatten())


        if self.requiredHoursRemaining <= 0:
          return 0

        else:
          if self.isOn == 1:
            return 1

          else:
            if interimAction == 1:
              return interimAction
            #if not, we need to look ahead and check if there is still room for the agent to do its duty by the given time
            else:
              if self.requiredHoursRemaining < k:
                return interimAction
              #there is no choice, you need to turn it on now to finish by the end time
              else:
                return 1    

class randomShiftableAgent:
    def __init__(self, shiftableLoad):
        self.epsilon = 0.1
        self.learningRate = 0.1
        self.requiredHoursRemaining = shiftableLoad.requiredHours
        self.endTime = shiftableLoad.endTime
        self.startTime = shiftableLoad.startTime
        self.window = shiftableLoad.endTime-shiftableLoad.startTime
        self.q_table = np.zeros((PRICEDISCRETIZATION, self.requiredHoursRemaining+1, self.window+1, 24, len(ACTIONSbinary)))
        self.state = [0, 0, 0, 0]
        self.totalRewards = 0.0
        self.powerRating = shiftableLoad.powerRating
        self.startTime = shiftableLoad.startTime
        self.requiredHours = shiftableLoad.requiredHours
        self.requiredHoursRemaining = shiftableLoad.requiredHours
        self.isOn = 0
    
        
    def choose_action(self, state):
        i = int(state[0])
        j = int(state[1])
        k = int(state[2])
        l = int(state[3])
        x = random.uniform(0, 1)
        interimAction = 0
        interimAction = np.random.choice(ACTIONSbinary)

        if self.requiredHoursRemaining <= 0:
          return 0


        else:

          if self.isOn == 1:
            return 1

          else:
            if interimAction == 1:
              return interimAction
            else:
              if self.requiredHoursRemaining < k:
                return interimAction
              #there is no choice, you need to turn it on now to finish by the end time
              else:
                return 1    


def run_ShiftableUnQLEARNING(agent, data, dataNonDiscrete):
    hour = 0
    agent.totalRewards = 0.0
    agent.requiredHoursRemaining = agent.requiredHours
    agent.window = agent.endTime - agent.startTime
    agent.state = [data[0], agent.requiredHoursRemaining, agent.window, hour]
    auditList = []
    while (hour < 24):
        reward = 0
        action = 0
        if hour >= agent.startTime and hour < agent.endTime:
          agent.state = [data[hour], agent.requiredHoursRemaining, agent.window, hour]
          action = agent.choose_action(agent.state)
          if action == 1:
            agent.requiredHoursRemaining -= 1
          agent.isOn = action
          agent.window -= 1
          reward = -(dataNonDiscrete[hour]*(action)*agent.powerRating)
          next_hour = 0 if hour==23 else hour+1
          next_state = [data[next_hour], agent.requiredHoursRemaining, agent.window, next_hour]
          agent.totalRewards += reward
          agent.isOn = action
        
          #update rule
          target = np.max(agent.q_table[next_state[0], next_state[1], next_state[2], next_state[3], :])
          agent.q_table[agent.state[0],agent.state[1], agent.state[2], agent.state[3], action] = (1-agent.learningRate)*(agent.q_table[agent.state[0],agent.state[1],agent.state[2], agent.state[3], action]) \
                                                                  + (agent.learningRate)*(reward + DISCOUNT_FACTOR*target)
          agent.state = next_state
        auditList.append(action)
        hour += 1
    return agent.q_table, agent.totalRewards, auditList


def validate_ShiftableUnQLEARNING(agent, data, dataNonDiscrete):
    hour = 0
    agent.totalRewards = 0.0
    agent.requiredHoursRemaining = agent.requiredHours
    agent.window = agent.endTime - agent.startTime
    agent.state = [data[0], agent.requiredHoursRemaining, agent.window, hour]
    auditList = []
    while (hour < 24):
        reward = 0
        action = 0
        if hour >= agent.startTime and hour < agent.endTime:
          agent.state = [data[hour], agent.requiredHoursRemaining, agent.window, hour]
          action = agent.choose_action(agent.state)
          if action == 1:
            agent.requiredHoursRemaining -= 1
          agent.isOn = action
          agent.window -= 1

          reward = -(dataNonDiscrete[hour]*(action)*agent.powerRating)
          agent.totalRewards += reward
          agent.isOn = action
        
        auditList.append(action)
        hour += 1
    return agent.q_table, agent.totalRewards, auditList


def test_ShiftableUnQLEARNING(agent, data, dataNonDiscrete):
    hour = 0
    agent.totalRewards = 0.0
    agent.requiredHoursRemaining = agent.requiredHours
    agent.window = agent.endTime - agent.startTime
    agent.state = [data[0], agent.requiredHoursRemaining, agent.window, hour]
    auditList = []
    while (hour < 24):
        reward = 0
        action = 0
        if hour >= agent.startTime and hour < agent.endTime:
          agent.state = [data[hour], agent.requiredHoursRemaining, agent.window, hour]
          action = agent.choose_action(agent.state)
          if action == 1:
            agent.requiredHoursRemaining -= 1
          agent.isOn = action
          agent.window -= 1
          reward = -(dataNonDiscrete[hour]*(action)*agent.powerRating)
          agent.totalRewards += reward
          agent.isOn = action
        

        auditList.append(action)
        hour += 1
    return agent.q_table, agent.totalRewards, auditList

def runShiftableUnLoad(run, seed, load):
  set_seeds(seed)
  loadQagent = ShiftableAgent(load)

  #Training
  convergenceCurve = []
  trainingCurve = []
  qvaluesSum = []
  for run in range(1):
    for episode in range(1500):
      print("run", run, "training", load.loadName, "episode", episode)
      loadQagent.epsilon = 0.1
      exampleX = random.randint(0,len(trainingDataDiscrete)-1)
      qlearning_episode_qtable, qlearning_episode_rewards, auditList = run_ShiftableUnQLEARNING(loadQagent, trainingDataDiscrete[exampleX].astype(int), trainingData[exampleX])

      validationReward = 0
      for validationScenario in range(70):
        validationAgent = ShiftableAgent(load)
        validationAgent.q_table = np.copy(qlearning_episode_qtable)
        validationAgent.epsilon = 0
        validation_qlearning_episode_qtable, validation_average_qlearning_episode_rewards, auditList = validate_ShiftableUnQLEARNING(validationAgent, validationDataDiscrete[validationScenario].astype(int), validationData[validationScenario])
        validationReward += validation_average_qlearning_episode_rewards
      convergenceCurve.append(validationReward/70)
      trainingCurve.append(qlearning_episode_rewards)
      qvaluesSum.append(np.sum(qlearning_episode_qtable))


  TestingResults = []
  TestingAuditList = []
  print("now testing", load.loadName)
  for testEpisode in range(len(testingDataDiscrete)):
    testAgent = ShiftableAgent(load)
    testAgent.q_table = np.copy(qlearning_episode_qtable)
    testAgent.epsilon = 0
    test_qlearning_episode_qtable, test_qlearning_episode_rewards, auditList = test_ShiftableUnQLEARNING(testAgent, testingDataDiscrete[testEpisode].astype(int), testingData[testEpisode])
    TestingResults.append(test_qlearning_episode_rewards)
    TestingAuditList.append(auditList)
  generalisationResults = []
  generalisationAuditList = []
  print("now testing with training set", load.loadName)
  for testEpisode in range(len(trainingFullData)):
    testAgent = ShiftableAgent(load)
    testAgent.epsilon = 0
    testAgent.q_table = qlearning_episode_qtable
    test_qlearning_episode_qtable, test_qlearning_episode_rewards, auditList = test_ShiftableUnQLEARNING(testAgent, trainingFullDataDiscrete[testEpisode].astype(int), trainingFullData[testEpisode])
    
    generalisationResults.append(test_qlearning_episode_rewards)
    generalisationAuditList.append(auditList)
  return convergenceCurve, TestingResults, generalisationResults, TestingAuditList, generalisationAuditList

import random
DISCOUNT_FACTOR = 1 #this is an undiscounted MDP
ACTIONSbinary = [0,1]
PRICEDISCRETIZATION = 20

class ShiftableInterruptibleAgent:
    def __init__(self, shiftableInterruptibleLoad):
        self.epsilon = 0.1
        self.learningRate = 0.1
        self.requiredHoursRemaining = shiftableInterruptibleLoad.requiredHours
        self.endTime = shiftableInterruptibleLoad.endTime
        self.startTime = shiftableInterruptibleLoad.startTime
        self.window = shiftableInterruptibleLoad.endTime-shiftableInterruptibleLoad.startTime
        self.q_table = np.zeros((PRICEDISCRETIZATION, self.requiredHoursRemaining+1, self.window+1, 24, len(ACTIONSbinary)))
        self.state = [0, 0, 0, 0]
        self.totalRewards = 0.0
        self.powerRating = shiftableInterruptibleLoad.powerRating
        self.startTime = shiftableInterruptibleLoad.startTime
        self.requiredHours = shiftableInterruptibleLoad.requiredHours
        self.requiredHoursRemaining = shiftableInterruptibleLoad.requiredHours
        self.isOn = 0
    
        
    def choose_action(self, state):
        i = int(state[0])
        j = int(state[1])
        k = int(state[2])
        l = int(state[3])
        x = random.uniform(0, 1)
        interimAction = 0
        if x <= self.epsilon:
            interimAction = np.random.choice(ACTIONSbinary)
        else:
            arr = self.q_table[i, j, k, l, :]
            best_action = np.where(arr == np.amax(arr)) #there might be ties            
            interimAction = np.random.choice(np.asarray(best_action).flatten())


        if self.requiredHoursRemaining <= 0:
          return 0

        else:
          if interimAction == 1:
            return interimAction
          #if not, we need to look ahead and check if there is still room for the agent to do its duty by the given time
          else:
            if self.requiredHoursRemaining < k:
              return interimAction
            #there is no choice, you need to turn it on now to finish by the end time
            else:
              return 1  
              
class randomShiftableInterruptibleAgent:
    def __init__(self, shiftableInterruptibleLoad):
        self.epsilon = 0.1
        self.learningRate = 0.1
        self.requiredHoursRemaining = shiftableInterruptibleLoad.requiredHours
        self.endTime = shiftableInterruptibleLoad.endTime
        self.startTime = shiftableInterruptibleLoad.startTime
        self.window = shiftableInterruptibleLoad.endTime-shiftableInterruptibleLoad.startTime
        self.q_table = np.zeros((PRICEDISCRETIZATION, self.requiredHoursRemaining+1, self.window+1, 24, len(ACTIONSbinary)))
        self.state = [0, 0, 0, 0]
        self.totalRewards = 0.0
        self.powerRating = shiftableInterruptibleLoad.powerRating
        self.startTime = shiftableInterruptibleLoad.startTime
        self.requiredHours = shiftableInterruptibleLoad.requiredHours
        self.requiredHoursRemaining = shiftableInterruptibleLoad.requiredHours
        self.isOn = 0
    
        
    def choose_action(self, state):
        i = int(state[0])
        j = int(state[1])
        k = int(state[2])
        l = int(state[3])
        x = random.uniform(0, 1)
        interimAction = 0
        interimAction = np.random.choice(ACTIONSbinary)


        if self.requiredHoursRemaining <= 0:
          return 0

        else:
          if interimAction == 1:
            return interimAction
          else:
            if self.requiredHoursRemaining < k:
              return interimAction
            #there is no choice, you need to turn it on now to finish by the end time
            else:
              return 1                 

def run_ShiftableIntQLEARNING(agent, data, dataNonDiscrete):
    hour = 0
    agent.totalRewards = 0.0
    agent.requiredHoursRemaining = agent.requiredHours
    agent.window = agent.endTime - agent.startTime
    agent.state = [data[0], agent.requiredHoursRemaining, agent.window, hour]
    auditList = []
    while (hour < 24):
        reward = 0
        action = 0
        if hour >= agent.startTime and hour < agent.endTime:
          agent.state = [data[hour], agent.requiredHoursRemaining, agent.window, hour]
          action = agent.choose_action(agent.state)
          if action == 1:
            agent.requiredHoursRemaining -= 1
          agent.isOn = action
          agent.window -= 1
          reward = -(dataNonDiscrete[hour]*(action)*agent.powerRating)
          next_state = [data[hour+1], agent.requiredHoursRemaining, agent.window, hour+1]
          agent.totalRewards += reward
        
          #update rule
          target = np.max(agent.q_table[next_state[0], next_state[1], next_state[2], next_state[3], :])
          agent.q_table[agent.state[0],agent.state[1], agent.state[2], agent.state[3], action] = (1-agent.learningRate)*(agent.q_table[agent.state[0],agent.state[1],agent.state[2], agent.state[3], action]) \
                                                                  + (agent.learningRate)*(reward + DISCOUNT_FACTOR*0)
          agent.state = next_state
        auditList.append(action)
        hour += 1
    return agent.q_table, agent.totalRewards, auditList


def validate_ShiftableIntQLEARNING(agent, data, dataNonDiscrete):
    hour = 0
    agent.totalRewards = 0.0
    agent.requiredHoursRemaining = agent.requiredHours
    agent.window = agent.endTime - agent.startTime
    agent.state = [data[0], agent.requiredHoursRemaining, agent.window, hour]
    auditList = []
    while (hour < 24):
        reward = 0
        action = 0
        if hour >= agent.startTime and hour < agent.endTime:
          agent.state = [data[hour], agent.requiredHoursRemaining, agent.window, hour]
          action = agent.choose_action(agent.state)
          if action == 1:
            agent.requiredHoursRemaining -= 1
          agent.isOn = action
          agent.window -= 1
          reward = -(dataNonDiscrete[hour]*(action)*agent.powerRating)
          next_state = [data[hour+1], agent.requiredHoursRemaining, agent.window, hour+1]
          agent.totalRewards += reward

          agent.state = next_state
        auditList.append(action)
        hour += 1
    return agent.q_table, agent.totalRewards, auditList

def test_ShiftableIntQLEARNING(agent, data, dataNonDiscrete):
    hour = 0
    agent.totalRewards = 0.0
    agent.requiredHoursRemaining = agent.requiredHours
    agent.window = agent.endTime - agent.startTime
    agent.state = [data[0], agent.requiredHoursRemaining, agent.window, hour]
    auditList = []
    while (hour < 24):
        reward = 0
        action = 0
        if hour >= agent.startTime and hour < agent.endTime:
          agent.state = [data[hour], agent.requiredHoursRemaining, agent.window, hour]
          action = agent.choose_action(agent.state)
          if action == 1:
            agent.requiredHoursRemaining -= 1
          agent.isOn = action
          agent.window -= 1
          reward = -(dataNonDiscrete[hour]*(action)*agent.powerRating)
          next_state = [data[hour+1], agent.requiredHoursRemaining, agent.window, hour+1]
          agent.totalRewards += reward
        
          agent.state = next_state
        auditList.append(action)
        hour += 1
    return agent.q_table, agent.totalRewards, auditList


def runShiftableIntLoad(run, seed, load):
  set_seeds(seed)
  loadQagent = ShiftableInterruptibleAgent(load)
  #Training
  convergenceCurve = []
  trainingCurve = []
  qvaluesSum = []
  for run in range(1):
    for episode in range(1500):
      print("run", run, "training", load.loadName, "episode", episode)
      qlearning_episode_qtable, qlearning_episode_rewards, auditList = run_ShiftableIntQLEARNING(loadQagent, trainingDataDiscrete[episode%len(trainingData)].astype(int), trainingData[episode%len(trainingData)])
      validationReward = 0
      for validationScenario in range(10):
        validationAgent = ShiftableInterruptibleAgent(load)
        validationAgent.q_table = np.copy(qlearning_episode_qtable)
        validationAgent.epsilon = 0
        validation_qlearning_episode_qtable, validation_average_qlearning_episode_rewards, auditList = validate_ShiftableIntQLEARNING(validationAgent, validationDataDiscrete[validationScenario].astype(int), validationData[validationScenario])
        validationReward += validation_average_qlearning_episode_rewards
      convergenceCurve.append(validationReward/10)
      trainingCurve.append(qlearning_episode_rewards)
      qvaluesSum.append(np.sum(qlearning_episode_qtable))

  TestingResults = []
  TestingAuditList = []
  print("now testing", load.loadName)
  for testEpisode in range(len(testingDataDiscrete)):
    testAgent = ShiftableInterruptibleAgent(load)
    testAgent.q_table = qlearning_episode_qtable
    testAgent.epsilon = 0
    test_qlearning_episode_qtable, test_qlearning_episode_rewards, auditList = test_ShiftableIntQLEARNING(testAgent, testingDataDiscrete[testEpisode].astype(int), testingData[testEpisode])
    TestingResults.append(test_qlearning_episode_rewards)
    TestingAuditList.append(auditList)

  generalisationResults = []
  generalisationAuditList = []
  print("now testing with training set", load.loadName)
  for testEpisode in range(len(trainingFullData)):
    testAgent = ShiftableAgent(load)
    testAgent.epsilon = 0
    testAgent.q_table = qlearning_episode_qtable
    test_qlearning_episode_qtable, test_qlearning_episode_rewards, auditList = test_ShiftableIntQLEARNING(testAgent, trainingFullDataDiscrete[testEpisode].astype(int), trainingFullData[testEpisode])
    
    generalisationResults.append(test_qlearning_episode_rewards)
    generalisationAuditList.append(auditList)
  return convergenceCurve, TestingResults, generalisationResults, TestingAuditList, generalisationAuditList


def calculateCost(actionListH, actionListAC1, actionListAC2, actionListWM, actionListDW, actionListEV,
                  heater, aircon1, aircon2, washingMachine, dishWasher, electricVehicle, fridge, alarm,
                  price, solarGeneration):
  
  cost = []
  for i in range(len(price)):
    totalElectricCost = 0
    totalDiscomfortCost = 0
    for j in range(24):
      powerUseThisHour = 0
      powerUseThisHour += fridge.powerRating
      powerUseThisHour += alarm.powerRating
        # powerSet = action*((agent.maxPowerRating-agent.minPowerRating) /10) +agent.minPowerRating
      powerUseThisHour += actionListH[i][j]*((heater.maxPowerRating-heater.minPowerRating) /10) + heater.minPowerRating
      powerUseThisHour += actionListAC1[i][j]*((aircon1.maxPowerRating-aircon1.minPowerRating) /10) + aircon1.minPowerRating
      powerUseThisHour += actionListAC2[i][j]*((aircon2.maxPowerRating-aircon2.minPowerRating) /10) + aircon2.minPowerRating
      # powerUseThisHour += actionListlLoad[i][j]*(lLoad.maxPowerRating/10)
      powerUseThisHour += actionListWM[i][j]*washingMachine.powerRating
      powerUseThisHour += actionListDW[i][j]*dishWasher.powerRating
      powerUseThisHour += actionListEV[i][j]*electricVehicle.powerRating
      powerUseThisHour -= solarGeneration[i,j]

      if powerUseThisHour >= 0:
        totalElectricCost += price[i,j]*abs(powerUseThisHour)
      else:
        totalElectricCost -= 0.5*price[i,j]*abs(powerUseThisHour)
      
      totalDiscomfortCost += heater.alpha*(heater.maxPowerRating - (actionListH[i][j]*((heater.maxPowerRating - heater.minPowerRating)/10) + heater.minPowerRating))**2
      totalDiscomfortCost += aircon1.alpha*(aircon1.maxPowerRating - (actionListAC1[i][j]*((aircon1.maxPowerRating- aircon1.minPowerRating)/10) + aircon1.minPowerRating))**2 
      totalDiscomfortCost += aircon2.alpha*(aircon2.maxPowerRating - (actionListAC2[i][j]*((aircon2.maxPowerRating - aircon2.minPowerRating)/10) + aircon2.minPowerRating))**2




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

  trainingCurveWM, testCurveWM, generalisationCurveWM, actionListTestWM, actionListTrainWM = runShiftableUnLoad(seed, SEEDLIST[seed], washingMachine)
  trainingCurveWMavg += trainingCurveWM
  trainingCurveDW, testCurveDW, generalisationCurveDW, actionListTestDW, actionListTrainDW = runShiftableUnLoad(seed, SEEDLIST[seed], dishWasher)
  trainingCurveDWavg += trainingCurveDW


  trainingCurveH, testCurveH, generalisationCurveH, actionListTestH, actionListTrainH = runAdjustableLoad(seed, SEEDLIST[seed], heater)
  trainingCurveHavg += trainingCurveH
  trainingCurveAC1, testCurveAC1, generalisationCurveAC1, actionListTestAC1, actionListTrainAC1 = runAdjustableLoad(seed, SEEDLIST[seed], aircon1)
  trainingCurveAC1avg += trainingCurveAC1
  trainingCurveAC2, testCurveAC2, generalisationCurveAC2, actionListTestAC2, actionListTrainAC2 = runAdjustableLoad(seed, SEEDLIST[seed], aircon2)
  trainingCurveAC2avg += trainingCurveAC2

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