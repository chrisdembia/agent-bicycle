from pybrain.utilities import abstractMethod

class Training:
    """This is the top-level class for a reinforcement learning problem. It
    manages the entire process of training and learning. It manages the running
    of episodes and the learning from the episodes. Also, it manages output of
    the learning (like report cards).

    This class is abstract; it must be implemented in a way that is relevant to
    the particular algorithm it is used with. For example, NFQ is off-line, so
    learn() must be called manually after episodes are carried out. Usually,
    then, you'd run just one episode, then learn. Online methods take care of
    the learning as part of running the episodes, So a rehearsal for an online
    method could run 100 episodes, let's say. 

    Training consists of a number of rehearsals. Rehearsals are blocks of
    learning; what happens in them and how long they are is up to the subclass.

    """
    def __init__(self, experiment):
        """
        Parameters
        ----------
        experiment : pybrain.rl.experiments.Experiment

        """
        self.exp = experiment

    def train(max_n_rehearsals, do_plot=True, serialization_interval=10):
        """Training consists of a loop of (1) rehearsing, (2) plotting the
        reward and bicycle wheel trajectory, and (3) writing the learner to a
        file.

        Parameters
        ----------
        max_n_rehearsals : int
            Maximum number of rehearsals to run. The only way to stop the
            training is to reach the number 
        do_plot : bool; optional
            Interactively plot (1) rewards vs rehearsal number, and (2) 
        serialization_interval : int
            How many rehearsals should pass before writing out the learner to a
            file?

        """
        for irehearsal in range(max_n_rehearsals):
            self.rehearse()
            self.perform()
            if irehearsal % serialization_interval == 0:
                self.serialize_learner()

    def perform(self):
        """
        Returns
        -------
        success_metric : float
            Some number that represents the success of the learning.

        """
        #experiment.agent = testagent
        #r = np.sum(experiment.doEpisodes(1))
        # Plot the wheel trajectories.
        #testagent.reset()
        #experiment.agent = agent


    def rehearse(self):
        """Run episodes, and ensure learning occurs. The implementation of this
        depends on the learning algorithm used.
        
        """
        abstractMethod()

    def 


class NFQTraining(Training):
    def rehearse(self):
        r = experiment.doEpisodes(1)

class Escapade:
    """An escapade is a learning endeavor. Metaphorically, it represents an
    evening that you'll spend with your child at the park trying to learn to
    ride a bicycle. Such an event consists of multiple episodes. In terms of
    PyBrain, an escapade is a specific way of running episodes on an
    experience. The class manages running the episodes, learning after the
    episodes, and most importantly, data plotting and output.
    
    """
    def __init__(self, experiment, do_plot=True):

    def run(n_iterations):
        for i in range(n_iterations):
            r = experiment

task = BalanceTask()
action_value_function = ActionValueNetwork(5, 9,
        name='RandlovActionValueNetwork')
learner = NFQ() # QLambda(alpha=0.5, gamma=0.99, lambda=0.95)
# TODO would prefer to use SARSALambda but it doesn't exist in pybrain (yet).
agent = LearningAgent(action_value_function, learner)
testagent = LearningAgent(action_value_function, None)
#learner.explorer.epsilon = 0.1
# TODO net.agent = agent
experiment = EpisodicExperiment(task, agent)
# See Randlov, 1998, fig 2 caption.

plt.ion()
plt.figure(figsize=(8, 4))

performance = []

for i in range(7000):
    r = experiment.doEpisodes(1)
    R = np.sum(r)
    agent.learn()

    #experiment.agent = testagent
    #r = np.sum(experiment.doEpisodes(1))
    # Plot the wheel trajectories.
    pl.subplot(122)
    if i > 0:
        for L in L1:
            L.set_color([0.8, 0.8, 0.8])
        for L in L2:
            L.set_color([0.8, 0.8, 0.8])
    L1 = plt.plot(task.env.get_xfhist(), task.env.get_yfhist(), 'r')
    L2 = plt.plot(task.env.get_xbhist(), task.env.get_ybhist(), 'b')
    plt.axis('equal')
    plt.draw()
    #testagent.reset()
    #experiment.agent = agent

    print i, learner.explorer.epsilon, len(r[0])

    performance.append(R)
    pl.subplot(121)
    plt.cla()
    plt.plot(performance, 'o--')
    plt.draw()
    plt.pause(0.001)

    # TODO plot wheel trajectory (two different colors for rear/front).
    
    # print agent.history.getSample(i)
    if i % 10 == 0:
        NetworkWriter.writeToFile(action_value_function.network, 'randlov1998_actionvaluenetwork_%i.xml' % i)

