import pandas as pd
from Perceptron import Perceptron
from Functions import *


file1 = 'data_regression1.txt'  # filename
data = pd.read_csv(file1, sep=',', header=None)  # read values from file

# Calculate loss for each different learning rates, same batch size
# Function writes results to given file
compare_l_rate('dataset1_l_rate.txt', data )

rgs1 = Perceptron(l_rate=0.0218) # create an instance of Percepton
rgs1.fit(data)  # fit data
rgs1.visualize()  # plot the graphs

#############################################################################################
create_dataset('dataset2.txt', ' x^3   + 150 ', -3, 10)  # create a new dataset with noise
file2 = 'dataset2.txt'  # filename
data = pd.read_csv(file2, sep=',', header=None)  # read values from file

# Calculate loss for different learning rates, same batch size
# Function writes results to given file
compare_l_rate('dataset2_l_rate.txt', data)

rgs2 = Perceptron(l_rate=0.002)  # create an instance of Percepton
rgs2.fit(data)  # fit data
rgs2.visualize()  # plot the graphs

#############################################################################################
create_dataset('dataset3.txt', ' 6 * x + 10 ', -3, 10)  # create a new dataset with noise
file3 = 'dataset3.txt'  # filename
data = pd.read_csv(file3, sep=',', header=None)  # read values from file

# Calculate loss for different learning rates,
# Function writes results to given file
compare_l_rate('dataset3_l_rate.txt', data)

rgs3 = Perceptron(l_rate=0.009)  # create an instance of Percepton
rgs3.fit(data)  # fit data
rgs3.visualize()  # plot the graphs

# save all figures to the output folder
save_figures('outputs')