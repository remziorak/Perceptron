import numpy as np
from py_expression_eval import Parser
from  Perceptron import Perceptron
import matplotlib.pyplot as plt
import os

def create_dataset(filename, expression, start, end, n_data=2000):
    """Evaluate given expression and create a dataset from this function
    with noise."""
    data = []
    x = np.linspace(start, end, n_data )
    parser = Parser()
    expr = parser.parse(expression)
    variable = expr.variables()

    for i in x:
        data.append([i, parser.evaluate(expression, {variable[0]: i})])

    data = np.array(data)
    row,col = data.shape
    noise =np.random.RandomState(721).normal(0, 1, n_data)
    data[:,1] = data[:,1] + noise

    f = open(filename, '+w')
    for i in range(row):
        f.write('{:.3f},{:.3f}\n'.format(data[i,0],data[i,1]))
    f.close()



def compare_l_rate(filename, data):
    """
    Calculate loss for different learning rates, same batch size
    sort calculated values by loss and write to file. Also plot
    the Learning rate vs Loss graph.
    """
    l_rate_performance = []
    for l_rate in np.arange(0.001, 0.1, 0.001):
        _ = Perceptron(l_rate=l_rate)
        _.fit(data)
        l_rate_performance.append([l_rate, _.test_set_errors[-1], len(_.test_set_errors)])

    sorted_by_loss = np.array(l_rate_performance)[np.array(l_rate_performance)[:, 1].argsort()]
    f = open(filename, '+w')
    f.write("Learning Rate    Final Loss    Iteration Count")
    for i in range(len(sorted_by_loss)):
        f.write("\n{:<17.3f}{:<14.2E}{:<19.1f}"
                .format(sorted_by_loss[i, 0], sorted_by_loss[i, 1], sorted_by_loss[i, 2]))

    plt.figure()
    plt.plot(np.array(l_rate_performance)[:,0],np.array(l_rate_performance)[:,-2])
    plt.scatter(np.array(l_rate_performance)[:, 0], np.array(l_rate_performance)[:, -2], s=5, c='r')
    plt.title('Learning Rate Performance')
    plt.ylim(0, np.mean(np.array(l_rate_performance)[3:15, -2])*10)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.draw()

def save_figures(path):
    """Save all figures plotted with matplotlib to path directory"""

    # create folder for png files
    if not os.path.isdir(path):
        os.makedirs(path)

    # plt.get_fignums returns a list of existing figure numbers.
    # then we save all existing figures
    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig(os.path.join(path, "figure_{}.png".format(i)), format='png')

    # close all figure to clear figure numbers
    plt.close("all")
    print("Figures for the dataset saved in {}".format(path))
