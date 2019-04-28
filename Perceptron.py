import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, l_rate=0.01,  max_iter=2000, tol=0.1,):
        self.max_iter = max_iter
        self.tol = tol
        self.l_rate = l_rate

    def fit(self, dataset):
        self.dataset = dataset
        self.train_dataset, self.test_dataset = self.split_data(test_size=0.5)
        self.X_train = self.train_dataset[:, : -1]
        self.y_train = self.train_dataset[:, -1]
        self.X_test = self.test_dataset[:, : -1]
        self.y_test = self.test_dataset[:, -1]

        # initialize each w_i to small number
        self.w =  np.zeros(1 + self.X_train.shape[1]) +0.00001

        self.train_set_errors = []
        self.test_set_errors = []


        for i in range(len(self.train_dataset)):
            x_i, target = self.train_dataset[i,0], self.train_dataset[i,1]
            update = self.l_rate * (target - self.predict(x_i))
            self.w[1:] += update * x_i
            self.w[0] += update

            self.train_set_errors.append(np.linalg.norm(self.y_train - self.predict(self.X_train)))
            self.test_set_errors.append(np.linalg.norm(self.y_test - self.predict(self.X_test)))

            if(self.train_set_errors[i] <= self.tol):
                break

    def predict(self, X):
        """calculate X * weight[1] + bias """
        return np.dot(X, self.w[1:]) + self.w[0]

    def split_data(self, test_size=0.5):
        """Function shuffle the data points and return splitted sets """
        shuffled_data = np.random.RandomState(seed=721).permutation(self.dataset)
        train_set = shuffled_data[: int(len(self.dataset) * (1 - test_size)), :]
        test_set = shuffled_data[int(len(self.dataset) * (1 - test_size)):, :]
        return train_set, test_set

    def visualize(self):
        """Plot graphs after training perceptron"""

        plt.figure()
        plt.scatter(self.train_dataset[:, 0], self.train_dataset[:, 1], s=3, )
        plt.xlabel('Feature')
        plt.ylabel('Output')
        plt.title('Dataset')
        plt.draw()

        plt.figure()
        plt.scatter(self.train_dataset[:, 0], self.train_dataset[:, 1], s=3, )
        plt.plot(np.arange(np.min(self.train_dataset[:, 0]), np.max(self.train_dataset[:, 0]), 1),
                 self.w[1] * np.arange(np.min(self.train_dataset[:, 0]), np.max(self.train_dataset[:, 0]), 1) + self.w[0], 'r')
        plt.title("Fitted Function\n (learning rate = {})".format(self.l_rate))
        plt.xlabel('Feature')
        plt.ylabel('Output')
        plt.draw()

        plt.figure()
        plt.plot(self.train_set_errors, label='Training Set')
        plt.scatter([*range(1, len(self.train_set_errors)+1)], self.train_set_errors,  s=5)
        plt.plot([*range(1, len(self.test_set_errors)+1)], self.test_set_errors, label='Test Set')
        plt.scatter([*range(1,len(self.test_set_errors)+1)], self.test_set_errors,  s=5)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.draw()