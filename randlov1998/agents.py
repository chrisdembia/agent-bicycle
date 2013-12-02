
from pybrain.rl.agents.learning import LearningAgent
from random import random, randint

class LinFA_QAgent(LearningAgent):
    """ Customization of the Agent class for linear function approximation learners. """

    epsilon = 0.1
    logging = False
    
    def __init__(self, learner):
        LearningAgent.__init__(self, None, learner)
        self.learner = learner
        self.previousobs = None
        
    def getAction(self):
        if random() < self.epsilon:
            a = randint(0, self.learner.num_actions-1)
        else:
            a = self.learner._greedyAction(self.lastobs)  
        self.lastaction = a
        return [a]
    
    def giveReward(self, r):
        LearningAgent.giveReward(self, r)
        if self.previousobs is not None:
            #print  self.previousobs, a, self.lastreward, self.lastobs
            self.learner._updateWeights(self.previousobs, self.previousaction, self.previousreward, self.lastobs)
        self.previousobs = self.lastobs
        self.previousaction = self.lastaction
        self.previousreward = self.lastreward
        
    def newEpisode(self):
        """ Indicate the beginning of a new episode in the training cycle. """
        if self.logging:
            self.history.newSequence()
        if self.learning and not self.learner.batchMode:
            self.learner.newEpisode()
        else:
            self.learner.newEpisode()

    def reset(self):
        LearningAgent.reset(self)
        self._temperature = self.init_temperature
        self._expl_proportion = self.init_exploration
        self.learner.reset()    
        self._oaro = None
        self.newEpisode()
