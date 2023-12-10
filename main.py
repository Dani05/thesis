import os
import random
import numpy
import numpy as np
import traci
import matplotlib.pyplot as plt

def init_sumo(senario_name):
    SUMO_HOME = os.environ["SUMO_HOME"]
    MOST_ROOT = "input_files"
    sumoBinary = SUMO_HOME + "/bin/sumo"
    sumoCmd = [sumoBinary, "-c", "%s/sumoconfig.sumocfg" % MOST_ROOT]
    sumoCmd += ["--net-file", "%s/uthalozat.net.xml" % MOST_ROOT]
    sumoCmd += ["-a", "%s/det.xml" % MOST_ROOT]
    sumoCmd += ["--random"]
    sumoCmd += ["--random-depart-offset", "900"]
    sumoCmd += ["--no-warnings"]
    if senario_name == "reggel":
        sumoCmd += ["-r", "%s/flow.rou.xml" % MOST_ROOT]
    return sumoCmd

def epoch(epsilon=.96):
    global bestQtable_reward
    global bestQtable
    actionIndex = -1
    os = [0, 0, 0]
    ns = [0, 0, 0]
    steps = 0

    while traci.simulation.getMinExpectedNumber() > 0:
    
        for veh_id in traci.vehicle.getIDList():
            position = traci.vehicle.getPosition(veh_id)
            
        edges = traci.edge.getIDList()
        
        for i in edges:
            res = traci.edge.getFuelConsumption(i)

        traci.simulation.getArrivedIDList()
        traci.simulationStep()

        if steps % 120 == 0:
            timeLossDetector1 = traci.multientryexit.getLastIntervalMeanTimeLoss(detectors[0])
            timeLossDetector2 = traci.multientryexit.getLastIntervalMeanTimeLoss(detectors[1])
            timeLossDetector3 = traci.multientryexit.getLastIntervalMeanTimeLoss(detectors[2])

            reward = -timeLossDetector1 if timeLossDetector1 > 0 else 0
            reward += -timeLossDetector2 if timeLossDetector2 > 0 else 0
            reward += -timeLossDetector3 if timeLossDetector3 > 0 else 0

            rewardList.append(reward)
            if bestQtable_reward == None or bestQtable_reward < reward:
                bestQtable = Qtable
                bestQtable_reward = reward

            if actionIndex != -1: learn(os, ns, actionIndex, reward)

        if steps % 120 == 0:

            os = quantize_state(
                traci.multientryexit.getLastStepVehicleNumber(detectors[0]),
                traci.multientryexit.getLastStepVehicleNumber(detectors[1]),
                traci.multientryexit.getLastStepVehicleNumber(detectors[2])
            )

            if epsilon > random.random():
                actionIndex = random.randint(0, len(bus_allowed) - 1)
            else:
                actionIndex = np.argmax(Qtable[os[0]][os[1]][os[2]])

            allowed = {'all'} if bus_allowed[actionIndex] == 1 else {'bus'}
            bus_allowed[actionIndex] = 0 if bus_allowed[actionIndex] == 1 else 1
            traci.lane.setAllowed(bus_allowed_id[actionIndex], allowed)

            ns = quantize_state(
                traci.multientryexit.getLastStepVehicleNumber(detectors[0]),
                traci.multientryexit.getLastStepVehicleNumber(detectors[1]),
                traci.multientryexit.getLastStepVehicleNumber(detectors[2])
            )
            epsilon *= .999

        steps += 1

    return epsilon


def epoch_no_action():
    global bestQtable_reward
    global bestQtable
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        steps = 0
        if steps % 120 == 0:
            timeLossDetector1 = traci.multientryexit.getLastIntervalMeanTimeLoss(detectors[0])
            timeLossDetector2 = traci.multientryexit.getLastIntervalMeanTimeLoss(detectors[1])
            timeLossDetector3 = traci.multientryexit.getLastIntervalMeanTimeLoss(detectors[2])

            reward = -timeLossDetector1 if timeLossDetector1 > 0 else 0
            reward += -timeLossDetector2 if timeLossDetector2 > 0 else 0
            reward += -timeLossDetector3 if timeLossDetector3 > 0 else 0

            rewardList.append(reward)
            if bestQtable_reward == None or bestQtable_reward < reward:
                bestQtable = Qtable
                bestQtable_reward = reward

        steps += 1


def epoch_best_action():
    global bestQtable_reward
    global bestQtable
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        steps = 0
        if steps % 120 == 0:
            timeLossDetector1 = traci.multientryexit.getLastIntervalMeanTimeLoss(detectors[0])
            timeLossDetector2 = traci.multientryexit.getLastIntervalMeanTimeLoss(detectors[1])
            timeLossDetector3 = traci.multientryexit.getLastIntervalMeanTimeLoss(detectors[2])

            reward = -timeLossDetector1 if timeLossDetector1 > 0 else 0
            reward += -timeLossDetector2 if timeLossDetector2 > 0 else 0
            reward += -timeLossDetector3 if timeLossDetector3 > 0 else 0

            rewardList.append(reward)
            if bestQtable_reward == None or bestQtable_reward < reward:
                bestQtable = Qtable
                bestQtable_reward = reward
            if steps % 120 == 0:
                os = quantize_state(
                    traci.multientryexit.getLastStepVehicleNumber(detectors[0]),
                    traci.multientryexit.getLastStepVehicleNumber(detectors[1]),
                    traci.multientryexit.getLastStepVehicleNumber(detectors[2])
                )

                actionIndex = np.argmax(Qtable[os[0]][os[1]][os[2]])

                allowed = {'all'} if bus_allowed[actionIndex] == 1 else {'bus'}
                bus_allowed[actionIndex] = 0 if bus_allowed[actionIndex] == 1 else 1
                traci.lane.setAllowed(bus_allowed_id[actionIndex], allowed)

            steps += 1


sumoCmd = init_sumo("normal")
sumoCmd_reggel = init_sumo("reggel")


def init():
    traci.start(sumoCmd)
    detectors = traci.multientryexit.getIDList()

    Qtable = np.zeros(
        [len(detectorTraffic_det0) + 1, len(detectorTraffic_det1) + 1, len(detectorTraffic_det2) + 1, len(bus_allowed)])
    traci.close()
    return detectors, Qtable


def learn(os, ns, action, reward):
    maxQns = np.array(Qtable[ns[0]][ns[1]][ns[2]]).max()
    Qtable[os[0]][os[1]][os[2]][action] += alfa * (reward + gamma * maxQns - Qtable[os[0]][os[1]][os[2]][action])
    rewardList.append(reward)


# returns indexes of the state
def quantize_state(trafficFlow_det0, trafficFlow_det1, trafficFlow_det2):
    state = [len(detectorTraffic_det0), len(detectorTraffic_det1), len(detectorTraffic_det2)]

    for i in range(len(detectorTraffic_det0) - 1):
        if trafficFlow_det0 < detectorTraffic_det0[i]:
            state[0] = i

    for j in range(len(detectorTraffic_det1) - 1):
        if trafficFlow_det1 < detectorTraffic_det1[j]:
            state[1] = j

    for k in range(len(detectorTraffic_det2) - 1):
        if trafficFlow_det2 < detectorTraffic_det1[k]:
            state[2] = k

    return state


bus_allowed_id = ['E26_0', '-E25_0', 'E33_0']
bus_allowed = [0, 0, 0]  # 0 = only buses, 1 = all vehicles ACTION
detectorTraffic_det0 = [10, 20, 40, 80]
detectorTraffic_det1 = [5, 10, 25, 40]
detectorTraffic_det2 = [5, 10, 25, 40]

epsilon = 1
alfa = 0.9
gamma = 0.6
detectors, Qtable = init()
bestQtable = Qtable
bestQtable_reward = None
rewardList = []

for simulationIndex in range(10):
    Qtable = np.zeros(
        [len(detectorTraffic_det0) + 1, len(detectorTraffic_det1) + 1, len(detectorTraffic_det2) + 1, len(bus_allowed)])
    bestQtable = Qtable
    bestQtable_reward = None

    # tanulas nelkul
    rewardList = []
    for i in range(210):
        current_sumoCmd = sumoCmd
        if i % 7 == 0:
            current_sumoCmd = sumoCmd_reggel
        traci.start(current_sumoCmd)
        epoch_no_action()
        
        with open('rewards_before/rewardList%d_%d.npy' %(simulationIndex,  i), 'wb') as f:
            np.save(f, numpy.array(rewardList))

        traci.close()

    #tanulas
    epsilon = 1
    alfa = 0.9
    gamma = 0.6
    rewardList = []
    for i in range(210):
        current_sumoCmd = sumoCmd
        if i % 7 == 0:
            current_sumoCmd = sumoCmd_reggel

        traci.start(current_sumoCmd)
        epsilon = epoch(epsilon)

        with open('rewards/rewardList%d_%d.npy' %(simulationIndex,  i), 'wb') as f:
            np.save(f, numpy.array(rewardList))

        with open('Qtables/besQtable%d_%d' %(simulationIndex,  i), 'wb') as f:
            np.save(f, bestQtable)

        with open('steps/stepsInEpoch%d_%d' %(simulationIndex,  i), "w+") as f:
            f.write(str(traci.simulation.getTime()))
        traci.close()

    #tanulas utan
    rewardList = []
    np.save('rewards_after/besQtable%d' %simulationIndex, bestQtable)
    for i in range(210):
        current_sumoCmd = sumoCmd
        if i % 7 == 0:
            current_sumoCmd = sumoCmd_reggel
        traci.start(current_sumoCmd)
        epoch_best_action()
        with open('rewards_after/rewardList%d_%d.npy' %(simulationIndex,  i), 'wb') as f:
            np.save(f, numpy.array(rewardList))
        traci.close()
