from datetime import datetime
import getpass
import os

from numpy import mean, array_str, empty
from matplotlib import pyplot as plt

from pybrain.tools.customxml.networkwriter import NetworkWriter
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
    def __init__(self, name, experiment, performance_agent, description=None,
            verbose=True):
        """
        Parameters
        ----------
        name : str
            Name for this training. This is used to create a folder that'll
            store output.
        experiment : pybrain.rl.experiments.EpisodicExperiment
            We expect to be able to call doEpisodes() on this experiment.
        performance_agent : pybrain.rl.agents
            An instance of the same type of agent given in the experiment,
            except this agent does not learn OR explore. This agent is used in
            performances to evaluate the success of the learned policy. For
            non-LinearFA learners, this can usually be a LearningAgent with the
            same module as given to the agent in the experiment. For LinearFA
            learners, you need to set the `greedy`, `epsilonGreedy`, and
            `learning` class attributes to False.
        description : str; optional
            Written to a README alongside all file outputs.
        verbose : bool; optional
            Write information to the command window.

        """
        self.exp = experiment
        # TODO with pointers etc, this might not end up getting the desired
        # "switch agent" functionality.
        self.learning_agent = self.exp.agent
        self.performance_agent = performance_agent
        self.verbose = verbose
        self.name = name

        # For plotting, to grey out previous trajectories.
        self.front_lines = []
        self.back_lines = []

        # Create data output directory.
        reldir = self.name + '_' + datetime.now().strftime('%m%d%HH%MM%SS')
        if 'AGENT_BICYCLE_DATA_PATH' in os.environ:
            self.outdir = os.path.join(
                    self.outdirs.environ['AGENT_BICYCLE_DATA_PATH'], reldir)
        else:
            self.outdir = reldir
        if os.path.exists(self.outdir):
            raise Exception('Training output path %s exists.' % self.outdir)
        else:
            os.makedirs(self.outdir)

        # Write a brief README in this directory.
        freadme = open(os.path.join(self.outdir, 'README.txt'), 'w')
        freadme.write('This training was executed by %s.\n' % getpass.getuser())
        if description:
            freadme.write(description)
        freadme.close()

    def train(self, n_max_rehearsals, do_plot=True, performance_interval=10,
            n_performance_episodes=5, serialization_interval=10):
        """Training consists of a loop of (1) rehearsing, (2) plotting the
        reward and bicycle wheel trajectory, and (3) writing output to a file
        (including the learner; e.g., weights).

        Parameters
        ----------
        n_max_rehearsals : int
            Maximum number of rehearsals to run. The only way to stop the
            training is to reach this number of rehearsals.
        do_plot : bool; optional
            Interactively plot (1) success metric vs rehearsal number, and (2)
            wheel trajectories. The plot is updated according to the
            performance interval. Note that the plot of wheel trajectories is
            only for the last episode executed in the preceding performance. On
            the other hand, the success metric can depend on all of the
            episodes executed in the preceding performance.
        performance_interval : int
            How many rehearsals should pass before performing with an agent
            that doesn't learn? This controls when the plot is updated.
        n_performance_episodes : int
            Number of episodes to execute in each performance.
        serialization_interval : int
            How many rehearsals should pass before writing data out to files?

        """
        # TODO unless we start writing out the success metric to a file, the
        # only reason to 'perform' is if do_plot is True.
        if do_plot:
            self.exp.task.env.saveWheelContactTrajectories(True)
            plt.ion()
            plt.figure(figsize=(8, 4))

        success_metric_history = []
        for irehearsal in range(n_max_rehearsals):
            # The trailing comma prevents `print` from print a newline.
            if self.verbose:
                print('Rehearsal %i.' % irehearsal),

            # Rehearse.
            self.rehearse(irehearsal)

            # Perform.
            if irehearsal % performance_interval == 0:
                success_metric_history.append(self.perform(n_performance_episodes))
                if do_plot:
                    # Success history.
                    plt.subplot(121)
                    plt.cla()
                    plt.plot(success_metric_history, '.--')
                    # Wheel trajectories.
                    plt.subplot(122)
                    self.update_wheel_trajectories()
                    # Necessary (?) plotting mechanics.
                    plt.draw()
                    plt.pause(0.001)

            # Write learning to file.
            if irehearsal % serialization_interval == 0:
                self.serialize(irehearsal)

    def update_wheel_trajectories(self):
        # Grey out previous lines.
        for line in self.front_lines:
            line.set_color([0.8, 0.8, 0.8])
        for line in self.back_lines:
            line.set_color([0.8, 0.8, 0.8])
        env = self.exp.task.env
        self.front_lines = plt.plot(env.get_xfhist(), env.get_yfhist(), 'r')
        self.back_lines = plt.plot(env.get_xbhist(), env.get_ybhist(), 'b')
        plt.axis('equal')

    def perform(self, n_perform_episodes):
        """Run `n_perform_episodes` on an agent that does NOT do any
        exploring/learning. This is how to really test out the learned policy.

        Returns
        -------
        success_metric : float
            Some number that represents the success of the learning. In this
            implementation, it's the average of the sum of rewards achieved
            over all the performance episodes.

        """
        # TODO have this return the actual discounted return achieved:
        # sum_t gamma^{t} r_{t+1}
        # Hold onto the original learning agent.
        # TODO sometimes the performance agent gets stuck.
        learning_agent = self.exp.agent
        self.exp.agent = self.performance_agent
        # The old/original success metric:
        #r = mean([sum(x) for x in self.exp.doEpisodes(n_perform_episodes)])
        R = empty(n_perform_episodes)
        for iep in range(n_perform_episodes):
            r = self.exp.doEpisodes(1)
            R[iep] = self.exp.task.getTotalReward()
        success = mean(R)
        if self.verbose:
            print 'PERFORMANCE: mean(R): %.2f, last nsteps: %.1f' % (
                    success, len(r[0]))

        self.performance_agent.reset()
        self.exp.agent = learning_agent
        return success

    def rehearse(self):
        """Run episodes, and ensure learning occurs. The implementation of this
        depends on the learning algorithm used.

        """
        abstractMethod()

    def serialize(self):
        """Serialize whatever you want; probably the learner (e.g., neural
        network) though.

        """
        abstractMethod()

    def fullFilePath(self, fname):
        """Takes a filename, and returns a full file path to a file with the
        name specified, but in the data output directory for this training.
        Useful when serializing.

        """
        return os.path.join(self.outdir, fname)


class NFQTraining(Training):
    def rehearse(self, irehearsal):
        r = self.exp.doEpisodes(1)
        self.exp.agent.learn()
        print 'epsilon: %.4f; nsteps: %i' % (
                self.exp.agent.learner.explorer.epsilon, len(r[0]))
    def serialize(self, irehearsal):
        NetworkWriter.writeToFile(self.exp.agent.module.network,
                self.fullFilePath('network_%i.xml' % irehearsal))

class LinearFATraining(Training):
    def __init__(self, *args, **kwargs):
        Training.__init__(self, *args, **kwargs)
        self.thetafile = open(self.fullFilePath('theta.dat'), 'w')
    def __del__(self):
        self.thetafile.close()
    def rehearse(self, irehearsal):
        r = self.exp.doEpisodes(1)
        # Discounted reward/return (I think):
        cumreward = self.exp.task.getTotalReward()
        if self.verbose:
            print 'cumreward: %.4f; nsteps: %i; learningRate: %.4f' % (
                    cumreward, len(r[0]), self.exp.agent.learner.learningRate)
    def serialize(self, irehearsal):
        #flattheta = self.exp.agent.learner._theta.flatten()
        #flatthetastr = array_str(flattheta).replace('[', '').replace(']', '')
        #self.thetafile.write('%i %s' % (irehearsal, flatthetastr))
        # TODO
        pass

