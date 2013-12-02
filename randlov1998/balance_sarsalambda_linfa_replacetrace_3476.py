from pybrain.rl.agents.linearfa import LinearFA_Agent
from pybrain.rl.experiments import EpisodicExperiment

from environment import Environment
from tasks import LinearFATileCoding3476BalanceTask
from training import LinearFATraining_setAlpha
from learners import SARSALambda_LinFA_setAlpha

# learning rate applied to heading states 
reduced_rate = 0.25
# number of states for the balancing task only, this is used to selectively
# different learning rates for balancing states and heading states
num_states_1 = 3456

# this task includes 20 discretized heading states
# the reward for this task is identical to the balance task
task = LinearFATileCoding3476BalanceTask()

# creating a modified SARSALambda learner, which applies a reduced rate to the
# heading states
learner = SARSALambda_LinFA_setAlpha(reduced_rate, num_states_1, task.nactions, task.outdim)
learner._lambda = 0.95

task.discount = learner.rewardDiscount
agent = LinearFA_Agent(learner)

# The state has a huge number of dimensions, and the logging causes me to run
# out of memory. We needn't log, since learning is done online.
agent.logging = False
agent.epsilonGreedy = True
performance_agent = LinearFA_Agent(learner)
performance_agent.logging = False
performance_agent.greedy = True
performance_agent.learning = False
experiment = EpisodicExperiment(task, agent)

# TODO PyBrain says that the learning rate needs to decay, but I don't see that
# described in Randlov's paper.
# A higher number here means the learning rate decays slower.
learner.learningRateDecay = 5000
# NOTE increasing this number above from the default of 100 is what got the
# learning to actually happen, and fixed the bug/issue where the performance
# agent's performance stopped improving.

tr = LinearFATraining_setAlpha('goto_sarsalambda_linfa_replacetrace',
        experiment, performance_agent, verbose=True)

tr.train(200000, performance_interval=10, n_performance_episodes=5)