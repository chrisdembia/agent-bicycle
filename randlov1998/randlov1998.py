"""Implementation of Randlov, 1998.

"""

import numpy as np
from scipy import asarray
import pybrain.rl.environments

from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.rl.learners.valuebased import NFQ, ActionValueNetwork

from environment import Environment

# The agent's actions are T and d.

# TODO where do we set up the generalization?
# TODO must pass omegadd to the learner.
# TODO the tiling might be achieved by implementing Task.getObservation.
# TODO states and actions are converted to int's within sarsa.py, ...what does
# this mean?
# TODO might need to use NFQ instead of Q or Sarsa.
# TODO NFQ might used a fix value of alpha as 0.5.
# TODO set epsilon for epsilon-greedy learning using learner.explorer.epsilon.
# TODO pybrain has limited examples of doing RL using continuous states and
# value-based learners (generalizing). Then we can use ActionValueNetwork, but
# it's not clear to me yet how this 'discretizes'/generalizes the state space.

class BalanceTask(pybrain.rl.environments.EpisodicTask):
    """The rider is to simply balance the bicycle while moving with the
    prescribed speed.

    This class is heavily guided by
    pybrain.rl.environments.cartpole.balancetask.BalanceTask.

    """
    # See Randlov's code. Paper and thesis say 12 degrees, but his code uses
    # pi/15. These are actually equivalent.
    #max_tilt = 12.0 * np.pi / 180.0
    max_tilt = np.pi / 15.0
    max_time = 1000.0 # seconds.

    def __init__(self):
        super(BalanceTask, self).__init__(Environment())
        # Keep track of time in case we want to end episodes based on number of
        # time steps.
        self.t = 0

        # TODO Sensor limits to normalize the sensor readings.
        # TODO Actor limits.
        #T_limits = (-2, 2) # Newtons.
        #d_limits = (-0.02, 0.02) # meters.
        ## None for sensor limits; does not normalize sensor values.
        ## outdim should be set to the length of the sensors vector.
        #self.setScaling([None] * self.env.outdim, [T_limits, d_limits])

    def reset(self):
        super(BalanceTask, self).reset()
        self.t = 0

    def performAction(self, action):
        """Incoming action is an int between 0 and 8. The action we provide to
        the environment consists of a torque T in {-2 N, 0, 2 N}, and a
        displacement d in {-.02 m, 0, 0.02 m}.

        """
        self.t += 1
        # Map the action integer to a torque and displacement.
        assert round(action) == action
        # -1 for action in {0, 1, 2}, 0 for action in {3, 4, 5}, 1 for action
        # in {6, 7, 8}
        torque_selector = np.floor(action / 3) - 1.0
        T = 2.0 * torque_selector
        ## Random number in [-1, 1]:
        #p = 2.0 * (np.random.rand() - 0.5)
        ## Max noise is 2 cm:
        #s = 0.02
        # -1 for action in {0, 3, 6}, 0 for action in {1, 4, 7}, 1 for action
        # in {2, 5, 8}
        disp_selector = action % 3 - 1.0
        d = 0.02 * disp_selector # TODO add in noise + s * p
        super(BalanceTask, self).performAction([T, d])

    def getObservation(self):
        (theta, thetad, omega, omegad, omegadd,
                xf, yf, xb, yb, psi) = self.env.getSensors()
        # TODO not calling superclass to do normalization, etc.
        return self.env.getSensors()[0:5]

    def isFinished(self):
        # Criterion for ending an episode.
        # "When the agent can balance for 1000 seconds, the task is considered
        # learned."
        if np.abs(self.env.getTilt()) > self.max_tilt:
            return True
        elapsed_time = self.env.time_step * self.t
        if elapsed_time > self.max_time:
            return True
        return False

    def getReward(self):
        # -1 reward for falling over; no reward otherwise.
        if np.abs(self.env.getTilt()) > self.max_tilt:
            return -1.0
        return 0.0

    @property
    def indim(self):
        return 5

    @property
    def outdim(self):
        return 1


#class DiscretizedActionTask(Task):
#    (theta, thetad, omega, omegad, omegadd, xf, yf, xb, yb, psi) = self.sensors


task = BalanceTask()
action_value_function = ActionValueNetwork(5, 9,
        name='RandlovActionValueNetwork')
learner = NFQ() # QLambda(alpha=0.5, gamma=0.99, lambda=0.95)
# TODO would prefer to use SARSALambda but it doesn't exist in pybrain (yet).
agent = LearningAgent(action_value_function, learner)
# TODO net.agent = agent
experiment = EpisodicExperiment(task, agent)
# See Randlov, 1998, fig 2 caption.
for i in range(7000):
    r = experiment.doEpisodes(1)
    print i, learner.explorer.epsilon
    agent.learn()
    
    # print agent.history.getSample(i)
