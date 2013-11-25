from pybrain.rl.agents.linearfa import LinearFA_Agent
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.utilities import one_to_n

from environment import Environment
from tasks import LinearFATileCoding3456BalanceTask
from training import LinearFATraining
from learners import SARSALambda_LinFA_ReplacingTraces

task = LinearFATileCoding3456BalanceTask()
learner = SARSALambda_LinFA_ReplacingTraces(task.nactions, task.outdim)
learner._lambda = 0.95
task.discount = learner.rewardDiscount
agent = LinearFA_Agent(learner)
# The state has a huge number of dimensions, and the logging causes me to run
# out of memory. We needn't log, since learning is done online.
agent.logging = False

# TODO PyBrain says that the learning rate needs to decay, but I don't see that
# described in Randlov's paper.
# A higher number here means the learning rate decays slower.
learner.learningRateDecay = 100000
# NOTE increasing this number above from the default of 100 is what got the
# learning to actually happen, and fixed the bug/issue where the performance
# agent's performance stopped improving.

for i in np.arange(2000, 3800, 50):
    theta = np.loadtxt('/home/fitze/Documents/agent-bicycle/data/balance_sarsalambda_linfa_replacetrace_anneal_112217H56M04S/theta_%i.dat' % i)
    learner._theta = theta
    Q = learner._qValues(one_to_n(task.getBin(0, 0, 0, 0, 0), task.outdim))
    pl.plot(Q, label='%s' % i)
#pl.legend()
pl.show()
print learner._greedyAction(one_to_n(task.getBin(0, 0, 0, 0, 0), task.outdim))
