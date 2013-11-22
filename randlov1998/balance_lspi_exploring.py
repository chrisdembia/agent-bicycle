from pybrain.rl.agents.linearfa import LinearFA_Agent
from pybrain.rl.learners.valuebased.linearfa import LSPI
from pybrain.rl.experiments import EpisodicExperiment

from environment import Environment
from tasks import LSPIBalanceTask
from training import LinearFATraining

task = LSPIBalanceTask(only_steer=True)
learner = LSPI(task.nactions, task.outdim)
theta = np.loadtxt('/home/fitze/Dropbox/stanford/21quarter/229cs/proj/data/balance_lspi_experimental_112011H17M18S/theta_800.dat')
learner._theta = theta
# TODO this LSPI does not have eligibility traces.
#learner._lambda = 0.95
task.discount = learner.rewardDiscount
agent = LinearFA_Agent(learner)
# The state has a huge number of dimensions, and the logging causes me to run
# out of memory. We needn't log, since learning is done online.
agent.logging = False
#learner.exploring = True
performance_agent = LinearFA_Agent(learner)
performance_agent.logging = False
performance_agent.greedy = True
performance_agent.learning = False
experiment = EpisodicExperiment(task, agent)

# TODO PyBrain says that the learning rate needs to decay, but I don't see that
# described in Randlov's paper.
# A higher number here means the learning rate decays slower.
learner.learningRateDecay = 100000
# NOTE increasing this number above from the default of 100 is what got the
# learning to actually happen, and fixed the bug/issue where the performance
# agent's performance stopped improving.

tr = LinearFATraining('balance_lspi_experimental', experiment,
        performance_agent, verbose=True)

tr.train(55000, performance_interval=10, n_performance_episodes=1)
