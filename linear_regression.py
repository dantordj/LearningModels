import numpy as np
import math
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, x, y):
        # Data for the model
        self.x = x
        self.y = y
        self.n = len(x)

        # Parameters of the model
        self.w = np.zeros(3)


    def fit(self):
        """ Fit the model : Estimate w, which contains the biais w[0] """
        x_ = np.append(np.ones((self.n,1)), self.x, axis=1)
        w_hat = np.linalg.pinv(x_.T.dot(x_)).dot(x_.T).dot(self.y)
        self.w = w_hat


    def predict(self, x):
        """ Return y = wx + b """
        x_ = np.append(1, x)
        return self.w.dot(x_)

    def plot_boundary(self, N=100):
        """ Plot the boundary of the model """
        xmin = min(self.x[:,0]) - 2
        xmax = max(self.x[:,0]) + 2
        ymin = min(self.x[:,1]) - 2
        ymax = max(self.x[:,1]) + 2
        w = self.w
        x_1 = np.linspace(xmin,xmax,N)
        x_2 = (0.5 - w[0]- w[1]* x_1) / w[2]
        plt.plot(x_1,x_2)
        plt.ylim(top=ymax, bottom=ymin)
