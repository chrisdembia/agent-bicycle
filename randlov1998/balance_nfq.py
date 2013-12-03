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
learner.gamma = 0.9999
learner.explorer.epsilon = 0.9
task.discount = learner.gamma
agent = LearningAgent(action_value_function, learner)
performance_agent = LearningAgent(action_value_function, None)
experiment = EpisodicExperiment(task, agent)

tr = NFQTraining('balance_nfq', experiment, performance_agent)

tr.train(7000, performance_interval=1, n_performance_episodes=1, plotsave_interval=10, plot_action_history=True)

