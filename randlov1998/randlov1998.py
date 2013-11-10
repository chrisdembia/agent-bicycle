"""Implementation of Randlov, 1998.

"""

from scipy import asarray
from matplotlib.mlab import rk4
import pybrain.rl.environments.environment
import pybrain.rl.environments
# The agent's actions are T and d.

class BalanceTask(pybrain.rl.environments.EpisodicTask):
    def __init__(self, maxsteps):
        super(BalanceTask, self).__init__(self, Environment())
        self.maxsteps = maxsteps

    def isFinished(self):
        # TODO
        pass

    def getReward(self):
        # TODO
        psi = np.arctan((xb - xf) / (yf - yb))
        xfd = -self.v * np.sin(psi + theta + np.sign(psi + theta) * np.arcsin(



class Environment(pybrain.rl.environments.environment.Environment):
        # TODO RL-state is [theta, thetad, omega, omegad, omegadd]^T

    # For superclass.
    indim = 2
    outdim = 6

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

    # TODO randomInitalization?

    def __init__(self):
        super(Environment, self).__init__()
        self.reset()
        self.actions = [0.0, 0.0]
        # TODO self.delay
        
    def getSensors(self):
        return asarray(self.sensors)
 
    def performAction(self, actions):
        self.actions = actions
        self.step()

    def step(self):
        # Unpack the state and actions.
        # -----------------------------
        (omega, omegad, theta, thetad, xf, yf, xb, yb) = self.sensors
        (T, d) = self.actions

        # TODO Add noise to the inputs, as Randlov did.
        # d_noised += 0.04 * (0.5 - np.random.rand())
    
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

        # Second derivative of angular acceleration:
        omegadd = 1 / self.Itot * (self.M * self.h * self.g * np.sin(phi)
                - np.cos(phi) * (self.Idc * self.sigmad * thetad
                    + np.sign(theta) * self.v**2 * (
                        self.Md * self.r / rf + self.Md * self.r / rb
                        + self.M * self.h / rCM)))
        thetadd = (T - self.Idv * self.sigmad * omegad) / self.Idl

        # Integrate using Euler's method.
        # yt+1 = yt + yd * dt.
        omegad += omegadd * dt
        omega += omegad * dt
        thetad += thetadd * dt
        theta += theta * dt

        # Handlebars can't be turned more than 80 degrees.
        theta = np.clip(theta, -1.3963, 1.3963)

        # Front wheel contact position.
        front_temp = self.v * self.time_step / (2 * rf)
        if front_temp > 1:
            front_temp = np.sign(psi + theta) * 0.5 * np.pi
        else:
            front_temp = np.sign(psi + theta) * np.arcsin(front_temp)
        xf += self.v * self.time_step * -np.sin(psi + theta + front_temp)


        self.sensors = (omega, omegad, theta, thetad, xf, yf, xb, yb)

    def reset(self):
        omega = 0
        omegad = 0
        theta = 0
        thetad = 0
        xf = 0
        yf = self.L
        xb = 0
        yb = 0
        self.sensors = (omega, omegad, theta, thetad, xf, yf, xb, yb)




env = Environment()
env.actions = [0, 0]
env._derivs([0, 0, 0, 0], 0)
