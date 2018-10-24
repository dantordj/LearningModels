import math
import numpy as np
import matplotlib.pyplot as plt

class QDA():

    def __init__(self, x, y):
        # Data for the model
        self.x = x
        self.y = y
        self.n = len(x)

        # Parameters of the model
        self.mu_0 = None
        self.mu_1 = None
        self.sigma_0 = None
        self.sigma_1 = None
        self.pi = None



    def estimate_mu(self, j):
        """ Estimate the expectation
        ::args:: j int : 0 or 1"""
        x_j =  np.array([self.x[i] for i in range(self.n) if self.y[i] == j])
        if j == 0:
            self.mu_0 = np.sum(x_j, axis=0) / len(x_j)
        elif j == 1:
            self.mu_1 = np.sum(x_j, axis=0) / len(x_j)

    def estimate_sigma(self, j):
        """ Estimate sigma
        ::args:: j int : 0 or 1"""
        if j == 0:
            mu_j = self.mu_0
        elif j == 1:
            mu_j = self.mu_1
        if mu_j is None:
            raise Exception("Estimate mu_j before sigma_j")
        x_j =  np.array([self.x[i] for i in range(self.n) if self.y[i] == j]) - mu_j
        if j == 0:
            self.sigma_0 = x_j.T.dot(x_j) / len(x_j)
        elif j == 1:
            self.sigma_1 = x_j.T.dot(x_j) / len(x_j)


    def estimate_pi(self):
        """ Estimate pi, parameter of the Bernouilli """
        self.pi = np.sum(self.y) / self.n

    def fit(self):
        """ Fit the model: computes all the parameters """
        self.estimate_pi()
        self.estimate_mu(0)
        self.estimate_mu(1)

        self.estimate_sigma(0)
        self.estimate_sigma(1)

    def predict(self,x):
        """ Return p(y=1|x) """
        pi = self.pi
        mu_0 = self.mu_0
        mu_1 = self.mu_1
        sigma_0 = self.sigma_0
        sigma_1 = self.sigma_1
        inv_sigma_0 = np.linalg.inv(sigma_0)
        inv_sigma_1 = np.linalg.inv(sigma_1)
        det_sigma_0 = np.linalg.det(sigma_0)
        det_sigma_1 = np.linalg.det(sigma_1)


        X1 = -1 / 2 * (x - mu_1).T.dot(inv_sigma_1).dot(x - mu_1)
        X0 = -1 / 2 * (x - mu_0).T.dot(inv_sigma_0).dot(x - mu_0)
        a = pi * math.exp(X1) / math.sqrt(det_sigma_1)
        b = (1 - pi) * math.exp(X0) / math.sqrt(det_sigma_0)
        return a / (a + b)




    def plot_boundary(self, N=100):
        """ Plot the boundary of the model """
        pi = self.pi
        mu_0, mu_1 = self.mu_0, self.mu_1
        sigma_0, sigma_1 = self.sigma_0, self.sigma_1

        inv_sigma_0 = np.linalg.inv(sigma_0)
        inv_sigma_1 = np.linalg.inv(sigma_1)
        det_sigma_0 = np.linalg.det(sigma_0)
        det_sigma_1 = np.linalg.det(sigma_1)

        # Equation of the boundary : tZAZ + 2tZB + c = 0
        A = (inv_sigma_0 - inv_sigma_1)
        B = (inv_sigma_1.dot(mu_1) - inv_sigma_0.dot(mu_0))
        c = mu_0.T.dot(inv_sigma_0).dot(mu_0) - mu_1.T.dot(inv_sigma_1).dot(mu_1)
        c += (2 * math.log(pi / (1 - pi)) - math.log(det_sigma_0) + math.log(det_sigma_1))

        xmin = min(self.x[:,0]) - 2
        xmax = max(self.x[:,0]) + 2
        ymin = min(self.x[:,1]) - 2
        ymax = max(self.x[:,1]) + 2
        x1 = np.linspace(xmin, xmax, N)
        x2 = np.linspace(ymin, ymax, N)

        X, Y = np.meshgrid(x1,x2)

        F2 = (A[0,0]) * (X ** 2) + (A[1,1]) * (Y ** 2)
        F2 +=  (A[1,0] + A[0,1]) * X * Y
        F1 = 2 * (B[0] * X + B[1] * Y)

        F = F1 + F2 + c

        plt.contour(X,Y,F,[0])
