from pybrain.rl.agents.linearfa import LinearFA_Agent
from pybrain.rl.learners.valuebased.linearfa import LSPI
from pybrain.rl.experiments import EpisodicExperiment

from environment import Environment
from tasks import LSPIGotoTask
from training import LinearFATraining

task = LSPIGotoTask(butt_disturbance_amplitude = 0.0000)
learner = LSPI(task.nactions, task.outdim,
        learningRateDecay=1000, randomInit=False)
task.env.x_goal = 3
task.env.y_goal = 15
task.env.r_goal = 2
# TODO this LSPI does not have eligibility traces.
learner.rewardDiscount = 0.8
learner.exploring = True
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


tr = LinearFATraining('goto_lspi', experiment,
        performance_agent, verbose=True)

tr.train(2000, performance_interval=10, n_performance_episodes=1)
