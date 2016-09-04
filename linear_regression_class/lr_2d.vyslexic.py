import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv("data_2d.csv").as_matrix().astype(np.float32)
    X = data[:,0:2]
    # adding the bias term
    bias_vec = np.ones((np.size(X,0), 1))
    X = np.concatenate((X, bias_vec), axis=1)
    Y = data[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], Y)
    #plt.show()

    X_Transpose = np.transpose(X)
    mat_1=np.linalg.inv(np.matmul(X_Transpose, X))
    mat_2=np.matmul(X_Transpose, Y)
    W = np.matmul(mat_1, mat_2)

    # now let's compute the residual error
    Y_hat = np.dot(X, W)

    d1 = Y-Y_hat
    d2 = Y-Y.mean()
    r = 1.0 - d1.dot(d1)/d2.dot(d2)

    print("The r-squared is", r)


if __name__ == '__main__':
    main()
