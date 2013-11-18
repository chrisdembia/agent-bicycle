from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners.valuebased import NFQ, ActionValueNetwork
from pybrain.rl.experiments import EpisodicExperiment

from environment import Environment
from tasks import BalanceTask
from training import NFQTraining

task = BalanceTask()
action_value_function = ActionValueNetwork(task.outdim, task.nactions,
        name='BalanceNFQActionValueNetwork')
learner = NFQ()
agent = LearningAgent(action_value_function, learner)
performance_agent = LearningAgent(action_value_function, None)
experiment = EpisodicExperiment(task, agent)

tr = NFQTraining('balance_nfq', experiment, performance_agent)

tr.train(7000)

