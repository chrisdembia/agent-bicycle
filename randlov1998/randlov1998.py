"""Implementation of Randlov, 1998.

"""

from scipy import asarray
from matplotlib.mlab import rk4
import pybrain.rl.environments.environment.Environment

# The agent's actions are T and d.

class Environment(pybrain.rl.environments.Environment):
        # TODO RL-state is [theta, thetad, omega, omegad, omegadd]^T

    # Abstract environment parameters.
    time_step = 0.01

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
    M = Mc + Md + Mp # Never defined explicity.
    Idc = md * r**2
    Idv = 1.5 * Md * r**2
    Idl = 0.5 * Md * r**2
    Itot = 13.0 / 3.0 * Mc * h**2 + Mp * (h + dCM)**2

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
        self.sensors = rk4(self._derivs, self.sensors, [0, self.time_step])
        # TODO what do we assign to sensors?

    def reset(self):
        self.sensors = (0.0, 0.0, 0.0, 0.0)

    def _derivs(self, x, t):
    
        # Unpack the state.
        # -----------------
        # ODE-state is [theta, thetad, omega, omegad]^T
        (theta, thetad, omega, omegad) = x

        # Get the control actions.
        (T, d) = self.actions
    
        rf = self.L / np.abs(sin(theta))
        rb = self.L / np.abs(tan(theta))
        rcm = np.sqrt((self.L - self.c)**2 + self.L**2 / np.tan(theta)**2)
        phi = omega + np.atan(self.d / self.h)

        # Second derivative of angular acceleration:
        omegadd = 1 / self.Itot * (self.M * self.h * self.g * np.sin(phi)
                - np.cos(phi) * (self.Idc * sigmad * thetad
                    + np.sign(theta) * v**2 * (
                        Md * self.r / rf + self.Md * self.r / rb + self.M * self.h / rCM)))
        thetadd = (T - self.Idv * simgad * omegad) / self.Idl
