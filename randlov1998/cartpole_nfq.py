from pybrain.rl.environments.cartpole import CartPoleEnvironment, DiscreteBalanceTask
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners.valuebased import NFQ, ActionValueNetwork
from pybrain.rl.experiments import EpisodicExperiment

from training import NFQTraining

task = DiscreteBalanceTask(CartPoleEnvironment(), 100)
action_value_function = ActionValueNetwork(4, 3,
        name='CartPoleNFQActionValueNetwork')
learner = NFQ()
#learner.gamma = 0.99
learner.explorer.epsilon = 0.4
task.discount = learner.gamma
agent = LearningAgent(action_value_function, learner)
performance_agent = LearningAgent(action_value_function, None)
experiment = EpisodicExperiment(task, agent)

tr = NFQTraining('cartpole_nfq', experiment, performance_agent)

tr.train(7000, performance_interval=1, n_performance_episodes=5)

