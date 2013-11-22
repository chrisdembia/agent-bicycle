import numpy as np
from numpy import clip, argwhere

from pybrain.rl.learners.valuebased.linearfa import SARSALambda_LinFA
from pybrain.utilities import one_to_n

class SARSALambda_LinFA_ReplacingTraces(SARSALambda_LinFA):
    def _updateEtraces(self, state, action, responsibility=1.):
        self._etraces *= self.rewardDiscount * self._lambda * responsibility
        # TODO I think this assumes that state is an identity vector (like,
        # from one_to_n).
        self._etraces[action] = clip(self._etraces[action] + state, -np.inf, 1.)
        # Set the trace for all other actions in this state to 0:
        action_bit = one_to_n(action, self.num_actions)
        self._etraces[argwhere(action_bit != 1), argwhere(state == 1)] = 0.
