from pybrain.rl.agents.linearfa import LinearFA_Agent
from pybrain.rl.experiments import EpisodicExperiment

from environment import Environment
from tasks import LinearFATileCoding3456BalanceTaskRewardPower5
from training import LinearFATraining
from learners import SARSALambda_LinFA_ReplacingTraces

task = LinearFATileCoding3456BalanceTaskRewardPower5()
learner = SARSALambda_LinFA_ReplacingTraces(task.nactions, task.outdim)
learner._lambda = 0.95
task.discount = learner.rewardDiscount
agent = LinearFA_Agent(learner)
agent.epsilonGreedy = True
agent.init_exploration = 0.5
# The state has a huge number of dimensions, and the logging causes me to run
# out of memory. We needn't log, since learning is done online.
agent.logging = False
performance_agent = LinearFA_Agent(learner)
performance_agent.logging = False
performance_agent.greedy = True
performance_agent.learning = False
experiment = EpisodicExperiment(task, agent)

# TODO PyBrain says that the learning rate needs to decay, but I don't see that
# described in Randlov's paper.
# A higher number here means the learning rate decays slower.
learner.learningRateDecay = 2000
# NOTE increasing this number above from the default of 100 is what got the
# learning to actually happen, and fixed the bug/issue where the performance
# agent's performance stopped improving.

tr = LinearFATraining('balance_sarsalambda_linfa_replacetrace_anneal_RewardPower4_take1', experiment,
        performance_agent, verbose=True)

tr.train(7000, performance_interval=1, n_performance_episodes=1,
        serialization_interval=50)
