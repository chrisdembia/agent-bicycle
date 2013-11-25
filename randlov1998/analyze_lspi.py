
from pybrain.rl.agents.linearfa import LinearFA_Agent
from pybrain.rl.learners.valuebased.linearfa import LSPI
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.utilities import one_to_n

from environment import Environment
from tasks import LSPIBalanceTask
from training import LinearFATraining

task = LSPIBalanceTask()
learner = LSPI(task.nactions, task.outdim)
# TODO this LSPI does not have eligibility traces.
#learner._lambda = 0.95
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

for idx in np.arange(0, 2500, 100):
theta = np.loadtxt('/home/fitze/Dropbox/stanford/21quarter/229cs/proj/data/balance_lspi_experimental_112011H17M18S/theta_%i.dat' % idx)
learner._theta = theta
Q = learner._qValues(one_to_n(task.getPhi(0, 0, 0, 0, 0), task.outdim))
pl.plot(Q)
print Q
pl.show()

print learner._greedyAction(one_to_n(task.getPhi(0, 0, 0, 0, 0), task.outdim))
