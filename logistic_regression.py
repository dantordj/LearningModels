import numpy as np
import math
import matplotlib.pyplot as plt

class LogisiticRegression():

    def __init__(self, x, y):
        # Data for the model
        self.x = x
        self.y = y
        self.n = len(x)
        # Parameters of the model, computed in the fit function
        self.w = np.zeros(3)



    def fit(self, e=10 ** (-3)):
        """ Fit the model: compute w with the IRLS algorithm
        args :: e float : precision for the optimization algorithm """

        x_ = np.append(np.ones((self.n,1)), self.x, axis=1)
        # initate the error at e + 1 to enter in the loop
        error = e + 1
        while error > e:
            diag = np.array([self.eta(x_i,self.w) for x_i in x_])
            D = np.diag(diag)
            diff_w = np.linalg.inv(x_.T.dot(D).dot(x_)).dot(x_.T).dot((self.y - diag))
            self.w, prev_w = self.w + diff_w, self.w
            error = np.linalg.norm(self.w - prev_w)


    def eta(self,x,w):
        return 1 / (1 + math.exp(-1 * w.T.dot(x)))


    def predict(self, x):
        """ Returns p(y=1|x) """
        x_ = np.append(1, x)
        return self.eta(x_, self.w)


    def plot_boundary(self, N=100):
        min_x = min(self.x[:,0])
        max_x = max(self.x[:,0])
        x1 = np.linspace(min_x, max_x, N)
        x2 = [- 1 * (self.w[0] + x * self.w[1]) / self.w[2] for x in x1]
        plt.plot(x1,x2)
