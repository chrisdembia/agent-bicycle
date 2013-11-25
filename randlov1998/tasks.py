import numpy as np

import pybrain.rl.environments
from pybrain.utilities import one_to_n

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

    nactions = 9

    def __init__(self, butt_disturbance_amplitude=0.02, only_steer=False,
            max_time=1000.0):
        """
        Parameters
        ----------
        butt_disturbance_amplitude : float; optional
            In meters.
        """
        super(BalanceTask, self).__init__(Environment())
        # Keep track of time in case we want to end episodes based on number of
        # time steps.
        self._butt_disturbance_amplitude = butt_disturbance_amplitude
        self.only_steer = only_steer
        self.max_time = max_time
        self.t = 0

        # TODO Sensor limits to normalize the sensor readings.
        # TODO Actor limits.
        #T_limits = (-2, 2) # Newtons.
        #d_limits = (-0.02, 0.02) # meters.
        ## None for sensor limits; does not normalize sensor values.
        ## outdim should be set to the length of the sensors vector.
        #self.setScaling([None] * self.env.outdim, [T_limits, d_limits])

    @property
    def indim(self):
        return 1

    @property
    def outdim(self):
        return 5

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
        assert round(action[0]) == action[0]

        if self.only_steer:
            T = 2 * (action[0] / 4.0 - 1.0)
            d = 0.
        else:
            # -1 for action in {0, 1, 2}, 0 for action in {3, 4, 5}, 1 for
            # action in {6, 7, 8}
            torque_selector = np.floor(action[0] / 3.0) - 1.0
            T = 2 * torque_selector
            # Random number in [-1, 1]:
            p = 2.0 * np.random.rand() - 1.0
            # -1 for action in {0, 3, 6}, 0 for action in {1, 4, 7}, 1 for
            # action in {2, 5, 8}
            disp_selector = action[0] % 3 - 1.0
            d = 0.02 * disp_selector + self._butt_disturbance_amplitude * p
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
            print 'hit max time.', self.t, elapsed_time
            return True
        return False

    def getReward(self):
        # -1 reward for falling over; no reward otherwise.
        if np.abs(self.env.getTilt()) > self.max_tilt:
            return -1.0
        # TODO return -np.abs(self.env.getSensors()[0])
        return 0.0


class LinearFATileCoding3456BalanceTask(BalanceTask):
    """An attempt to exactly implement Randlov's function approximation. He
    discretized (tiled) the input space into 3456 tiles.

    """
    # From Randlov, 1998:
    theta_bounds = np.array(
            [-0.5 * np.pi, -1.0, -0.2, 0, 0.2, 1.0, 0.5 * np.pi])
    thetad_bounds = np.array(
            [-np.inf, -2.0, 0, 2.0, np.inf])
    omega_bounds = np.array(
            [-BalanceTask.max_tilt, -0.15, -0.06, 0, 0.06, 0.15,
                BalanceTask.max_tilt])
    omegad_bounds = np.array(
            [-np.inf, -0.5, -0.25, 0, 0.25, 0.5, np.inf])
    omegadd_bounds = np.array(
            [-np.inf, -2.0, 0, 2.0, np.inf])
    # http://stackoverflow.com/questions/3257619/numpy-interconversion-between-multidimensional-and-linear-indexing
    nbins_across_dims = [
            len(theta_bounds) - 1,
            len(thetad_bounds) - 1,
            len(omega_bounds) - 1,
            len(omegad_bounds) - 1,
            len(omegadd_bounds) - 1]
    # This array, when dotted with the 5-dim state vector, gives a 'linear'
    # index between 0 and 3455.
    magic_array = np.cumprod([1] + nbins_across_dims)[:-1]

    def __init__(self, *args, **kwargs):
        super(LinearFATileCoding3456BalanceTask, self).__init__(*args, **kwargs)
        # Count the number of times that each state is visited.
        self.bin_count = np.zeros(self.outdim)

    @property
    def outdim(self):
        # Used when constructing LinearFALearner's.
        return 3456

    def getBin(self, theta, thetad, omega, omegad, omegadd):
        bin_indices = [
                np.digitize([theta], self.theta_bounds)[0] - 1,
                np.digitize([thetad], self.thetad_bounds)[0] - 1,
                np.digitize([omega], self.omega_bounds)[0] - 1,
                np.digitize([omegad], self.omegad_bounds)[0] - 1,
                np.digitize([omegadd], self.omegadd_bounds)[0] - 1,
                ]
        linear_index = np.dot(self.magic_array, bin_indices)
        if linear_index > self.outdim:
            # DEBUGGING PRINTS
            print self.isFinished()
            print self.env.getTilt()
            print np.abs(self.env.getTilt())
            print self.max_tilt
            print np.abs(self.env.getTilt()) > self.max_tilt
            print self.env.getSensors()[0:5]
            print self.magic_array
            print bin_index_for_each_dim
            print linear_index
        return linear_index

    def getBinIndices(self, linear_index):
        """Given a linear index (integer between 0 and outdim), returns the bin
        indices for each of the state dimensions.

        """
        return linear_index / self.magic_array % self.nbins_across_dims

    def getObservation(self):
        (theta, thetad, omega, omegad, omegadd,
                xf, yf, xb, yb, psi) = self.env.getSensors()
        # TODO not calling superclass to do normalization, etc.
        state = one_to_n(self.getBin(theta, thetad, omega, omegad, omegadd),
                self.outdim)
        self.bin_count += state
        return state


class LSPIBalanceTask(BalanceTask):
    """Lagoudakis, 2002; simplified for just balancing. Also, we're still using
    all 9 possible actions.

    """
    @property
    def outdim(self):
        # Used when constructing LinearFALearner's.
        return 14

    def getPhi(self, theta, thetad, omega, omegad, omegadd):
        return np.array([
            1, omega, omegad, omega**2, omegad**2, omega * omegad,
            theta, thetad, theta**2, thetad**2, theta * thetad,
            omega * theta, omega * theta**2, omega**2 * theta,
            ])

    def getObservation(self):
        (theta, thetad, omega, omegad, omegadd,
                xf, yf, xb, yb, psi) = self.env.getSensors()
        return self.getPhi(theta, thetad, omega, omegad, omegadd)


class LinearFATileCoding3456GoToTask(BalanceTask):
    """An attempt to exactly implement Randlov's function approximation. He
    discretized (tiled) the input space into 3456 tiles.

    """
    # From Randlov, 1998:
    theta_bounds = np.array(
            [-0.5 * np.pi, -1.0, -0.2, 0, 0.2, 1.0, 0.5 * np.pi])
    thetad_bounds = np.array(
            [-np.inf, -2.0, 0, 2.0, np.inf])
    omega_bounds = np.array(
            [-BalanceTask.max_tilt, -0.15, -0.06, 0, 0.06, 0.15,
                BalanceTask.max_tilt])
    omegad_bounds = np.array(
            [-np.inf, -0.5, -0.25, 0, 0.25, 0.5, np.inf])
    omegadd_bounds = np.array(
            [-np.inf, -2.0, 0, 2.0, np.inf])
    # http://stackoverflow.com/questions/3257619/numpy-interconversion-between-multidimensional-and-linear-indexing
    nbins_across_dims = [
            len(theta_bounds) - 1,
            len(thetad_bounds) - 1,
            len(omega_bounds) - 1,
            len(omegad_bounds) - 1,
            len(omegadd_bounds) - 1]
    # This array, when dotted with the 5-dim state vector, gives a 'linear'
    # index between 0 and 3455.
    magic_array = np.cumprod([1] + nbins_across_dims)[:-1]

    @property
    def outdim(self):
        # Used when constructing LinearFALearner's.
        return 3456

    def getBin(self, theta, thetad, omega, omegad, omegadd):
        bin_indices = [
                np.digitize([theta], self.theta_bounds)[0] - 1,
                np.digitize([thetad], self.thetad_bounds)[0] - 1,
                np.digitize([omega], self.omega_bounds)[0] - 1,
                np.digitize([omegad], self.omegad_bounds)[0] - 1,
                np.digitize([omegadd], self.omegadd_bounds)[0] - 1,
                ]
        linear_index = np.dot(self.magic_array, bin_indices)
        if linear_index > self.outdim:
            # DEBUGGING PRINTS
            print self.isFinished()
            print self.env.getTilt()
            print np.abs(self.env.getTilt())
            print self.max_tilt
            print np.abs(self.env.getTilt()) > self.max_tilt
            print self.env.getSensors()[0:5]
            print self.magic_array
            print bin_index_for_each_dim
            print linear_index
        return linear_index

    def getObservation(self):
        (theta, thetad, omega, omegad, omegadd,
                xf, yf, xb, yb, psi) = self.env.getSensors()
        # TODO not calling superclass to do normalization, etc.
        return one_to_n(self.getBin(theta, thetad, omega, omegad, omegadd),
                self.outdim)

    def getReward(self):
        # -1 reward for falling over; no reward otherwise.
        x = 15.
        y = 20.
        if np.abs(self.env.getTilt()) > self.max_tilt:
            return -1.0
        else:
            return 1e-5 * ((self.env.getXF() - x)**2 + (self.env.getYF() - y)**2)
