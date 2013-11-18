"""Functions for plotting results, etc.

"""

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import pylab as pl
from scipy import r_
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.utilities import one_to_n


def plot_nfq_action_value_history(network_name_prefix, count, state=[0, 0, 0, 0, 0], n_actions=9):
    """Example::

    >>> plot_nfq_action_value_history('randlov_actionvaluenetwork_',
            np.arange(0, 30, 10))

    This will plot the data from the files:
        randlov_actionvaluenetwork_0.xml
        randlov_actionvaluenetwork_10.xml
        randlov_actionvaluenetwork_20.xml
        randlov_actionvaluenetwork_30.xml

    """
    # TODO any file naming.
    n_times = len(count)

    actionvalues = np.empty((n_times, n_actions))
    for i in range(n_times):
        fname = network_name_prefix + '%i.xml' % count[i]
        actionvalues[i, :] = nfq_action_value(fname)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    actions = np.arange(n_actions)
    X, Y = np.meshgrid(actions, count)
    #ax.plot_surface(X, Y, actionvalues)
    ax.plot_wireframe(X, Y, actionvalues)
    plt.show()

def plot_nfq_action_value(network_name, state=[0, 0, 0, 0, 0]):
    """Plots Q(a) for the given state. Must provide a network serialization
    (.xml). Assumes there are 9 action values.

    Example::

    >>> plot_nfq_action_value('randlov_actionvaluenetwork.xml', [0, 0, 0, 0, 0])

    """
    pl.ion()
    n_actions = 9
    actionvalues = nfq_action_value(network_name, state)
    actions = np.arange(len(actionvalues))
    bar_width = 0.35
    pl.bar(actions, actionvalues, bar_width)
    pl.xticks(actions + 0.5 * bar_width, actions)
    pl.show()


def nfq_action_value(network_fname, state=[0, 0, 0, 0, 0]):
    # TODO generalize away from 9 action values. Ask the network how many
    # discrete action values there are.
    n_actions = 9
    network = NetworkReader.readFrom(network_fname)
    actionvalues = np.empty(n_actions)
    for i_action in range(n_actions):
        network_input = r_[state, one_to_n(i_action, n_actions)]
        actionvalues[i_action] = network.activate(network_input)
    return actionvalues
