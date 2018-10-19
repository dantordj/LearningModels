import numpy as np
import matplotlib.pyplot as plt
class LDA:

    def __init__(self, x, y):
        # Data for thhe model
        self.x = x
        self.y = y
        self.n = len(x)

        # Parameters of the model, computed in the fit function
        self.mu_0 = None
        self.mu_1 = None
        self.pi = None
        self.sigma = None

    def estimate_mu(self, j):
        """Estimation of the expectation mu_j"""
        x_j =  np.array([self.x[i] for i in range(self.n) if self.y[i] == j])
        mu_j = np.sum(x_j, axis=0) / len(x_j)
        if j == 0:
            self.mu_0 = mu_j
        elif j == 1:
            self.mu_1 = mu_j


    def estimate_sigma(self):
        """ Estimation of sgima """
        mu_0 = self.mu_0
        mu_1 = self.mu_1
        if mu_0 is None or mu_1 is None:
            raise Exception("Estimate mu_0 and mu_1 before sigma")
        x_0 =  np.array([self.x[i] for i in range(self.n) if self.y[i] == 0]) - mu_0
        x_1 =  np.array([self.x[i] for i in range(self.n) if self.y[i] == 1]) - mu_1
        self.sigma = (x_0.T.dot(x_0) + x_1.T.dot(x_1)) / self.n

    def estimate_pi(self):
        """ Estimate pi, parameter of the Bernouilli """
        self.pi = np.sum(self.y) / self.n


    def fit(self):
        """ Fit the model: estimation of pi, mu_0, mu_1 and sigma """
        self.estimate_pi()
        self.estimate_mu(0)
        self.estimate_mu(1)
        self.estimate_sigma()

    def predict(self, x):
        """ Returns the p(y=1|x) """
        mu_0 = self.mu_0
        mu_1 = self.mu_1
        sigma = self.sigma
        inv_sigma = np.linalg.inv(sigma)
        pi = self.pi
        mu = mu_0.T.dot(inv_sigma).dot(mu_0) - mu_1.T.dot(inv_sigma).dot(mu_1)
        a_tilde = 0.5 * mu + np.log(pi) - np.log(1 - pi)
        b_tilde = 0.5 * (mu_1.T.dot(inv_sigma) - mu_0.T.dot(inv_sigma))
        res = np.exp(a_tilde + b_tilde.dot(x.T) + x.dot(b_tilde))
        res /= (1 + np.exp(a_tilde + b_tilde.dot(x.T) + x.dot(b_tilde)))
        return res

    def plot_boundary(self, N=100):

        mu_0 = self.mu_0
        mu_1 = self.mu_1
        pi = self.pi
        inv_sigma = np.linalg.inv(self.sigma)
        mu = mu_0.T.dot(inv_sigma).dot(mu_0) - mu_1.T.dot(inv_sigma).dot(mu_1)
        a_tilde = 0.5 * mu + np.log(pi) - np.log(1 - pi)
        b_tilde = 0.5 * (mu_1.T.dot(inv_sigma) - mu_0.T.dot(inv_sigma))
        xmin = min(self.x[:,0])
        xmax = max(self.x[:,0])
        x_1 = np.linspace(xmin,xmax,N)
        x_2 = (-1  * a_tilde - 2 * b_tilde[0] * x_1) / (2 * b_tilde[1])
        plt.plot(x_1, x_2)
