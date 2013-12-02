import numpy as np
from numpy import clip, argwhere

from pybrain.rl.learners.valuebased.linearfa import SARSALambda_LinFA
from pybrain.utilities import one_to_n
from scipy import dot

class SARSALambda_LinFA_ReplacingTraces(SARSALambda_LinFA):
    def _updateEtraces(self, state, action, responsibility=1.):
        self._etraces *= self.rewardDiscount * self._lambda * responsibility
        # TODO I think this assumes that state is an identity vector (like,
        # from one_to_n).
        self._etraces[action] = clip(self._etraces[action] + state, -np.inf, 1.)
        # Set the trace for all other actions in this state to 0:
        action_bit = one_to_n(action, self.num_actions)
        
        # changed this line to allow for multiple 
        # (state == 1) occurences
        for argstate in argwhere(state == 1) :
        	self._etraces[argwhere(action_bit != 1), argstate] = 0.   

class SARSALambda_LinFA_setAlpha(SARSALambda_LinFA_ReplacingTraces):
	
	reduced_rate = 0.25 # the learning rate to be applied to the heading states

	def __init__(self, reduced_rate, num_states_1, *args, **kwargs):
		""" additional arguments to pass in: reduced_rate and num_states_1
		reduced_rate is the learning rate that will be applied to 
		the states[num_states_1:] 
		"""
		SARSALambda_LinFA_ReplacingTraces.__init__(self, *args, **kwargs)
		self.reduced_rate = reduced_rate
		self.num_states_1 = num_states_1
	

	def _updateWeights(self, state, action, reward, next_state, next_action):
		num_states_1 = self.num_states_1
		reduced_rate = self.reduced_rate

		td_error1 =  self.rewardDiscount*dot(self._theta[next_action, 0:num_states_1], next_state[0:num_states_1]) - dot(self._theta[action, 0:num_states_1], state[0:num_states_1])
		td_error2 =  self.rewardDiscount*dot(self._theta[next_action, num_states_1:], next_state[num_states_1:]) - dot(self._theta[action, num_states_1:], state[num_states_1:])

		self._updateEtraces(state, action)
		self._theta += ( self.learningRate * (reward + td_error1) + ( reduced_rate * td_error2 )  ) * self._etraces
	
	def newEpisode(self):
		SARSALambda_LinFA_ReplacingTraces.newEpisode(self)
		self.reduced_rate *= ((self.learningRateDecay + self._callcount)/(self.learningRateDecay + self._callcount + 1.))