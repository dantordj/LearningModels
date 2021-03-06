import pandas as pd
import numpy as np
from lda import LDA
from linear_regression import LinearRegression
from logistic_regression import LogisiticRegression
from qda import QDA
import matplotlib.pyplot as plt


def read_file(filename):
    """ Load the data
    return:: x array, y array """
    print("Loading data " + filename)
    df = pd.read_csv(filename, delimiter='\t',  header=None)
    return df.values[:,:2], df.values[:,2]



def accuracy(y_true, y_pred):
    """ Return the accuracy """
    y_pred = [int(a >= 0.5) for a in y_pred]
    return np.sum((y_true == y_pred)) * 1.0 / len(y_true)



def main(dataset, compute_errors=True, plot_boundaries=True, save=False):
    """ Fit the four models on the training sets, depending on the parameters
    compute the accuracy et plot the boundary
    args :: dataset : array(str) """


    filename = "data/" + dataset + ".train"
    x_train, y_train = read_file(filename)
    filename = "data/" + dataset + ".test"
    x_test, y_test = read_file(filename)

    models = [LDA(x_train,y_train), LinearRegression(x_train,y_train),
              LogisiticRegression(x_train,y_train), QDA(x_train,y_train)]

    model_names = ["LDA", "LinearRegression", "LogisiticRegression", "QDA"]
    for i, model in enumerate(models):
        model_name = model_names[i]
        model.fit()
        if compute_errors:
            y_pred_train = [model.predict(x) for x in x_train]
            e = accuracy(y_train, y_pred_train)
            print("Accuracy with " + model_name)
            print("Training: ", e)
            y_pred_test = [model.predict(x) for x in x_test]
            e = accuracy(y_test, y_pred_test)
            print("Testing: ", e)
        if plot_boundaries:
            model.plot_boundary()
            plt.scatter(model.x[:,0],model.x[:,1], c=model.y, s=1)
            title = "Model: " + model_name + ", " + dataset + " (Train)"
            plt.title(title)
            if save:
                plt.savefig("figs/" + model_name+"_"+dataset[-1]+"Train.png")
            plt.show()
            model.plot_boundary()
            plt.scatter(x_test[:,0],x_test[:,1], c=y_test, s=1)
            title = "Model: " + model_name + ", " + dataset + " (Test)"
            plt.title(title)
            if save:
                plt.savefig("figs/" + model_name+"_"+dataset[-1]+"Test.png")
            plt.show()





if __name__ == "__main__":
    datasets = ["classificationA", "classificationB", "classificationC"]
    for dataset in datasets:
        main(dataset)
