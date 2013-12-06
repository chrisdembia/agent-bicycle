from numpy import loadtxt

from pybrain.rl.agents.linearfa import LinearFA_Agent

from environment import Environment
from tasks import LinearFATileCoding3456BalanceTask
from training import LinearFATraining
from learners import SARSALambda_LinFA_ReplacingTraces

from game import Game

task = LinearFATileCoding3456BalanceTask()
learner = SARSALambda_LinFA_ReplacingTraces(task.nactions, task.outdim)
#theta = loadtxt('/home/fitze/Documents/agent-bicycle/data/balance_sarsalambda_linfa_replacetrace_anneal_112217H56M04S/theta_3800.dat')
theta = loadtxt('theta_linearfatilecoding3456balancetask_balance_works.dat')
learner._theta = theta
performance_agent = LinearFA_Agent(learner)
performance_agent.logging = False
performance_agent.greedy = True
performance_agent.learning = False

Game(performance_agent, task).run()
