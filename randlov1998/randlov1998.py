"""Implementation of Randlov, 1998.

"""

from scipy import asarray
from matplotlib.mlab import rk4
import pybrain.rl.environments.environment
import pybrain.rl.environments

from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Reinforce

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

# TODO consider moving some calculations, like psi, from the environment to the
# task. psi seems particularly task-dependent.
class Environment(pybrain.rl.environments.environment.Environment):
        # TODO RL-state is [theta, thetad, omega, omegad, omegadd]^T

    # For superclass.
    indim = 2
    outdim = 10

    # Environment parameters.
    time_step = 0.001

    # Acceleration on Earth's surface due to gravity (m/s^2):
    g = 9.8
    
    # See the paper for a description of these quantities:
    # Distances (in meters):
    c = 0.66
    dCM = 0.30
    h = 0.94
    L = 1.11
    r = 0.34
    # Masses (in kilograms):
    Mc = 15.0
    Md = 1.7
    Mp = 60.0
    # Velocity of a bicycle (in meters per second), equal to 10 km/h:
    v = 10.0 * 1000.0 / (3600.0 * 24.0)
    
    # Derived constants.
    M = Mc + Mp # See Randlov's code.
    Idc = Md * r**2
    Idv = 1.5 * Md * r**2
    Idl = 0.5 * Md * r**2
    Itot = 13.0 / 3.0 * Mc * h**2 + Mp * (h + dCM)**2
    sigmad = self.v / self.r

    def __init__(self):
        super(Environment, self).__init__()
        self.reset()
        self.actions = [0.0, 0.0]
        # TODO self.delay

    def getTilt(self):
        return self.sensors[0]

    def getSensors(self):
        return asarray(self.sensors)

    def performAction(self, actions):
        self.actions = actions
        self.step()

    def step(self):
        # Unpack the state and actions.
        # -----------------------------
        # Want to ignore the previous value of omegadd; it could only cause a
        # bug if we assign to it.
        (theta, thetad, omega, omegad, _,
                xf, yf, xb, yb, psi) = self.sensors
        (T, d) = self.actions

        # Process the actions.
        # --------------------
        # TODO Add noise to the inputs, as Randlov did.
        # d_noised += 0.04 * (0.5 - np.random.rand())
        # Control should be trivial otherwise.

        # Intermediate time-dependent quantities.
        # ---------------------------------------
        # Avoid divide-by-zero, just as Randlov did.
        if theta == 0:
            rf = 1e9
            rb = 1e9
            rCM = 1e9
        else:
            rf = self.L / np.abs(np.sin(theta))
            rb = self.L / np.abs(np.tan(theta))
            rCM = np.sqrt((self.L - self.c)**2 + self.L**2 / np.tan(theta)**2)

        phi = omega + np.arctan(d / self.h)

        # Equations of motion.
        # --------------------
        # Second derivative of angular acceleration:
        omegadd = 1 / self.Itot * (self.M * self.h * self.g * np.sin(phi)
                - np.cos(phi) * (self.Idc * self.sigmad * thetad
                    + np.sign(theta) * self.v**2 * (
                        self.Md * self.r / rf + self.Md * self.r / rb
                        + self.M * self.h / rCM)))
        thetadd = (T - self.Idv * self.sigmad * omegad) / self.Idl

        # Integrate equations of motion using Euler's method.
        # ---------------------------------------------------
        # yt+1 = yt + yd * dt.
        omegad += omegadd * dt
        omega += omegad * dt
        thetad += thetadd * dt
        theta += theta * dt

        # Handlebars can't be turned more than 80 degrees.
        theta = np.clip(theta, -1.3963, 1.3963)

        # Wheel ('tyre') contact positions.
        # ---------------------------------

        # Front wheel contact position.
        front_temp = self.v * self.time_step / (2 * rf)
        # See Randlov's code.
        if front_temp > 1:
            front_temp = np.sign(psi + theta) * 0.5 * np.pi
        else:
            front_temp = np.sign(psi + theta) * np.arcsin(front_temp)
        xf += self.v * self.time_step * -np.sin(psi + theta + front_temp)
        yf += self.v * self.time_step * np.cos(psi + theta + front_temp)

        # Rear wheel.
        back_temp = self.v * self.time_step / (2 * rb)
        # See Randlov's code.
        if back_temp > 1:
            back_temp = np.sign(psi) * 0.5 * np.pi
        else:
            back_temp = np.sign(psi) * np.arcsin(back_temp)
        xb += self.v * self.time_step * -np.sin(psi + back_temp)
        yb += self.v * self.time_step * np.cos(psi + back_temp)

        # Preventing numerical drift.
        # ---------------------------
        # Copying what Randlov did.
        current_wheelbase = np.sqrt((xf - xb)**2 + (yf - yb)**2)
        if np.abs(current_wheelbase - self.L) > 0.01:
            relative_error = self.L / current_wheelbase - 1.0
            xb += (xb - xf) * relative_error
            yb += (yb - yf) * relative_error

        # Update heading, psi.
        # --------------------
        delta_y = yf - yb
        if (xf == xb) and delta_y < 0.0:
            psi = np.pi
        else:
            if deltay_y < 0.0:
                psi = np.arctan((xb - xf) / delta_y)
            else:
                psi = (np.sign(xb - xf) * 0.5 * np.pi
                        - np.arctan(delta_y / (xb - xf)))

        self.sensors = (theta, thetad, omega, omegad, omegadd,
                xf, yf, xb, yb, psi)

    def reset(self):
        omega = 0
        omegad = 0
        omegadd = 0
        theta = 0
        thetad = 0
        xf = 0
        yf = self.L
        xb = 0
        yb = 0
        psi = np.arctan((xb - xf) / (yf - yb))
        self.sensors = (omega, omegad, omegadd, theta, thetad,
                xf, yf, xb, yb, psi)


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
        super(BalanceTask, self).__init__(self, Environment())
        # Keep track of time in case we want to end episodes based on number of
        # time steps.
        self.t = 0

        # TODO Sensor limits to normalize the sensor readings.
        # TODO Actor limits.
        T_limits = (-2, 2) # Newtons.
        d_limits = (-0.02, 0.02) # meters.
        # None for sensor limits; does not normalize sensor values.
        # outdim should be set to the length of the sensors vector.
        self.setScaling([None] * self.env.outdim, [T_limits, d_limits])

    def reset(self):
        super(BalanceTask, self).reset(self)
        self.t = 0

    def performAction(self, action):
        """Incoming action is an int between 0 and 8. The action we provide to
        the environment consists of a torque T in {-2 N, 0, 2 N}, and a
        displacement d in {-.02 m, 0, 0.02 m}.

        """
        self.t += 1
        # Map the action integer to a torque and displacement.
        assert type(action) == int
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
        super(BalanceTask, self).performAction(self, [T, d])

    def getObservation(self):
        (theta, thetad, omega, omegad, omegadd,
                xf, yf, xb, yb, psi) = self.env.sensors
        # TODO not calling superclass to do normalization, etc.
        return asarray([theta, thetad, omega, omegad, omegadd])

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


class DiscretizedTask(Task):
    (theta, thetad, omega, omegad, omegadd, xf, yf, xb, yb, psi) = self.sensors


task = BalanceTask()
action_value_function = buildNetwork(TODO)
action_value_function = ActionValueNetwork(5, 9,
        name='RandlovActionValueNetwork')
learner = QLambda(alpha=0.5, gamma=0.99, lambda=0.95)
# TODO would prefer to use SARSALambda but it doesn't exist in pybrain (yet).
agent = LearningAgent(action_value_function, learner)
# TODO net.agent = agent
experiment = EpisodicExperiment(task, agent)
# See Randlov, 1998, fig 2 caption.
for _ in range(7000):
    experiment.doEpisodes(1)
    agent.learn()
