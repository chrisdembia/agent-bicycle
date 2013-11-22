from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners.valuebased import NFQ, ActionValueNetwork
from pybrain.rl.experiments import EpisodicExperiment

from environment import Environment
from tasks import BalanceTask
from training import NFQTraining

task = BalanceTask()

task.performAction(1)


print task.getObservation()

task.performAction(2)

print task.getObservation()

task.performAction(3)

print task.getObservation()