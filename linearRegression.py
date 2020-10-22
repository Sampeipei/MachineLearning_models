import numpy as np
import matplotlib.pyplot as plt

class linearRegression:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.dNum = X.shape[0]
        self.xDim = X.shape[1]
    
    def train(self):
        # Concat ones vector
        Z = np.concatenate([self.X, np.ones([self.dNum, 1])], axis=1)
        # Prepare for calculation
        ZZ = np.matmul(Z.T, Z) / self.dNum
        ZY = np.matmul(Z.T, self.Y) / self.dNum
        # Optimize param
        v_opt = np.matmul(ZZ.T, ZY)

        self.w = v_opt[:-1]
        self.b = v_opt[-1]
    
    def trainRegularized(self, lamb=0.1):
        # Concat ones vector to input
        Z = np.concatenate([self.X, np.ones([self.dNum, 1])], axis=1)
        # Prepare for optim calculation
        ZZ = np.matmul(Z.T, Z) / self.dNum + lamb * np.eye(self.xDim)
        ZY = np.matmul(Z.T, self.Y) / self.dNum
        # Optimize param
        v_opt = np.matmul(ZZ.T, ZY)

        self.w = v_opt[:-1]
        self.b = v_opt[-1]
    
    def predict(self, X):
        return np.matmul(X, self.w) + self.b
    
    def RMSE(self, X, Y):
        return np.sqrt(np.mean(np.square(self.predict(X) - Y)))
    
    def R2(self, X, Y):
        return 1 - np.sum(np.square(self.predict(X) - Y)) / np.sum(np.square(Y - np.mean(Y, axis=0)))
    
    def plotResult(self, X=[], Y=[], xLabel="", yLabel=""):
        if X.shape[1] != 1:
            return
        
        fig = plt.figure(figsize=(8, 5), dpi=100)
        
        # Calc plot of line
        Xlin = np.array([[0], [np.max(X)]])
        Ylin = self.predict(Xlin)

        # Plot data and linear model
        plt.plot(X, Y, '.', label="Data")
        plt.plot(Xlin, Ylin, 'r', label="Linear model")
        plt.legend()

        # Set axis range and label
        plt.ylim([0, np.max(Y)])
        plt.xlim([0, np.max(X)])
        plt.xlabel(xLabel, fontsize=14)
        plt.ylabel(yLabel, fontsize=14)
        
        plt.show()