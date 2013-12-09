from pybrain.rl.agents.linearfa import LinearFA_Agent
from pybrain.rl.learners.valuebased.linearfa import LSPI
from pybrain.rl.experiments import EpisodicExperiment

from environment import Environment
from tasks import LSPIGotoTask
from training import LinearFATraining

x_g = 10
y_g = 30

task = LSPIGotoTask(butt_disturbance_amplitude = 0.0000, randomInitState = False, five_actions = True,  rewardType = 1, x_goal = x_g, y_goal = y_g)
learner = LSPI(task.nactions, task.outdim, randomInit = False)

# TODO this LSPI does not have eligibility traces.
#learner._lambda = 0.95

# lagoudakis uses 0.8 discount factor
learner.rewardDiscount = 0.8
task.discount = learner.rewardDiscount

agent = LinearFA_Agent(learner)
# The state has a huge number of dimensions, and the logging causes me to run
# out of memory. We needn't log, since learning is done online.
agent.logging = False
agent.epsilonGreedy = True
#learner.exploring = True
performance_agent = LinearFA_Agent(learner)
performance_agent.logging = False
performance_agent.greedy = True
performance_agent.learning = False
experiment = EpisodicExperiment(task, agent)

# TODO PyBrain says that the learning rate needs to decay, but I don't see that
# described in Randlov's paper.
# A higher number here means the learning rate decays slower.
learner.learningRateDecay = 300
# NOTE increasing this number above from the default of 100 is what got the
# learning to actually happen, and fixed the bug/issue where the performance
# agent's performance stopped improving.

tr = LinearFATraining('goto_lspi', experiment,
        performance_agent, verbose=True)

tr.train(200000, performance_interval=10, n_performance_episodes=5)