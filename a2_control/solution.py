import sys
import math
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class LQR(MathematicalProgram):
    """
    Parameters
    K integer
    A in R^{n x n}
    B in R^{n x n}
    Q in R^{n x n} symmetric
    R in R^{n x n} symmetric
    yf in R^n

    Variables
    y[k] in R^n for k=1,...,K
    u[k] in R^n for k=0,...,K-1

    Optimization Problem:
    LQR with terminal state constraint

    min 1/2 * sum_{k=1}^{K}   y[k].T Q y[k] + 1/2 * sum_{k=0}^{K-1}      u [k].T R u [k]
    s.t.
    y[1] - Bu[0]  = 0
    y[k+1] - Ay[k] - Bu[k] = 0  ; k = 1,...,K-1
    y[K] - yf = 0

    Hint: Use the optimization variable:
    x = [ u[0], y[1], u[1],y[2] , ... , u[K-1], y[K] ]

    Use the following features:
    1 - a single feature of types OT.f
    2 - the features of types OT.eq that you need
    """

    def __init__(self, K, A, B, Q, R, yf):
        """
        Arguments
        -----
        T: integer
        A: np.array 2-D
        B: np.array 2-D
        Q: np.array 2-D
        R: np.array 2-D
        yf: np.array 1-D
        """
        # in case you want to initialize some class members or so...
        self.K = K
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.yf = yf
        self.n = len(yf)
        n = len(yf)
        self.H_dyn = np.zeros((K*n, 2*K*n))
        
        
        for t in range(K):
            if t > 0:
                self.H_dyn[n*t : n*(t + 1), n * (2*t - 1) : n * 2*t] = - A
            
            self.H_dyn[n*t : n*(t + 1), n * 2*t : n * (2*t + 1)] = - B
            self.H_dyn[n*t : n*(t + 1), n * (2*t + 1) : n * (2*t + 2)] = np.eye(n)

    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """
        xReshape = x.reshape((2 * self.K, self.n))
        u = xReshape[ : : 2]
        y = xReshape[1 : : 2]
        
        sos = 0
        Jsos = np.zeros((2*self.n*self.K))
    
        for t in range(self.K):
            sos += .5 * u[t].T @ self.R @ u[t] + .5 * y[t].T @ self.Q @ y[t]

            Jsos[self.n*2*t : self.n*(2*t + 1)] = self.R @ u[t]
            Jsos[self.n*(2*t + 1) : self.n*(2*t + 2)] = self.Q @ y[t]
            
        Jsos = np.reshape(Jsos, (1, -1))
        
        
        h_dyn = self.H_dyn @ x
        Jh_dyn = self.H_dyn
        
        h_des = y[self.K - 1,:] - self.yf
        Jh_des = np.zeros((self.n, 2*self.K*self.n))
        Jh_des[:, -self.n:] = np.eye(self.n)
      
        phi = np.hstack(([sos], h_dyn, h_des))        
        J = np.concatenate([Jsos, Jh_dyn, Jh_des], axis = 0)
        
        return  phi, J

    
    
    def getFHessian(self, x):
        """
        """
        H = np.zeros((2*self.K*self.n, 2*self.K*self.n))

        for t in range(self.K):
            H[self.n*2*t : self.n*(2*t + 1) , self.n*2*t : self.n*(2*t + 1)] = self.R
            H[self.n*(2*t+1) : self.n*(2*t+2) , self.n*(2*t+1) : self.n*(2*t+2)] = self.Q

        return H

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        return 2 * self.n * self.K

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.zeros(self.getDimension())

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.f] + [OT.eq] * ((self.K + 1) * self.n)
