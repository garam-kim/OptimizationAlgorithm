# import sys
# import math
# import numpy as np

# sys.path.append("..")
# from optimization_algorithms.interface.mathematical_program import MathematicalProgram
# from optimization_algorithms.interface.objective_type import OT


# class LQR(MathematicalProgram):
#     """
#     Parameters
#     K integer
#     A in R^{n x n}
#     B in R^{n x n}
#     Q in R^{n x n} symmetric
#     R in R^{n x n} symmetric
#     yf in R^n

#     Variables
#     y[k] in R^n for k=1,...,K
#     u[k] in R^n for k=0,...,K-1

#     Optimization Problem:
#     LQR with terminal state constraint

#     min 1/2 * sum_{k=1}^{K}   y[k].T Q y[k] + 1/2 * sum_{k=1}^{K-1}      u [k].T R u [k]
#     s.t.
#     y[1] - Bu[0]  = 0
#     y[k+1] - Ay[k] - Bu[k] = 0  ; k = 1,...,K-1
#     y[K] - yf = 0

#     Hint: Use the optimization variable:
#     x = [ u[0], y[1], u[1],y[2] , ... , u[K-1], y[K] ]

#     Use the following features:
#     1 - a single feature of types OT.f
#     2 - the features of types OT.eq that you need
#     """

#     def __init__(self, K, A, B, Q, R, yf):
#         """
#         Arguments
#         -----
#         T: integer
#         A: np.array 2-D
#         B: np.array 2-D
#         Q: np.array 2-D
#         R: np.array 2-D
#         yf: np.array 1-D
#         """
#         # in case you want to initialize some class members or so...

#     def evaluate(self, x):
#         """
#         See also:
#         ----
#         MathematicalProgram.evaluate
#         """

#         # add the main code here! E.g. define methods to compute value y and Jacobian J
#         # y = ...
#         # J = ...

#         # y is a 1-D np.array of dimension m
#         # J is a 2-D np.array of dimensions (m,n)
#         # where m is the number of features and n is dimension of x
#         # return  y  , J

#     def getFHessian(self, x):
#         """
#         """
#         # return

#     def getDimension(self):
#         """
#         See Also
#         ------
#         MathematicalProgram.getDimension
#         """
#         # return

#     def getInitializationSample(self):
#         """
#         See Also
#         ------
#         MathematicalProgram.getInitializationSample
#         """
#         return np.zeros(self.getDimension())

#     def getFeatureTypes(self):
#         """
#         returns
#         -----
#         output: list of feature Types
#         See Also
#         ------
#         MathematicalProgram.getFeatureTypes
#         """
#         # return

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

    min 1/2 * sum_{k=1}^{K}   y[k].T Q y[k] + 1/2 * sum_{k=1}^{K-1}      u [k].T R u [k]
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

        self.n = A.shape[1]

    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """

        # add the main code here! E.g. define methods to compute value y and Jacobian J
        # y = ...
        # J = ...

        # y is a 1-D np.array of dimension d1
        # J is a 2-D np.array of dimensions (d1,d2)
        # where d1 is the number of features and d2 is dimension of x
        # return  y  , J
        
        
        

        phi = 0
        phif = np.zeros((self.K + 1, self.n))

        J = np.zeros((self.K*2, self.n ))
        #[constraint number, all , index of input in x, all]
        Jf = np.zeros((self.K + 1, self.n , self.K *2, self.n))
        
        xc = x.reshape((self.K*2,self.n))
        u = xc[::2]
        y = xc[1::2]

        
        for i in range(len(u)):
            vec = u[i]
            phi += 0.5*vec.T @ self.R @ vec
            J[2*i] = self.R@vec
            
            vec = y[i]
            phi += 0.5*vec.T @ self.Q @vec
            J[2*i + 1] = self.Q @vec
                

        phif[0] = y[0] - self.B@u[0]
        Jf[0,:,0] = -self.B
        Jf[0,:,1] = np.eye(self.n,self.n)

        for k in range(1, self.K):
            phif[k] = y[k] - self.A@y[k-1] - self.B@u[k]
            Jf[k,:,k*2 +1] = np.eye(self.n,self.n)
            Jf[k,:,k*2 -1] = -self.A
            Jf[k,:,k*2] = -self.B
        
        phif[-1] = y[-1] - self.yf
        Jf[-1,:,-1] = np.eye(self.n, self.n)

        phif = phif.reshape(-1)
        phi = np.concatenate(([phi],phif))
        
        Jf = Jf.reshape((-1, self.K *2, self.n))
        Jf = Jf.reshape(-1, self.getDimension())
        J = J.reshape(1,-1)
        J = np.concatenate((J, Jf),axis=0)

        return phi ,  J


    def getFHessian(self, x):
        """
        """
        # return
        H = np.zeros((self.K*2,self.n, self.K*2,self.n))
        xc = x.reshape((self.K*2,self.n))
        u = xc[::2]
        y = xc[1::2]

        for i in range(self.K):
            H[2*i , : , 2*i] = self.R
            H[2*i+1,:, 2*i+1] = self.Q

        H= H.reshape((self.K*2,self.n,-1))
        H = H.reshape((self.getDimension(),self.getDimension()))
        return H

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        return 2*self.n * self.K

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
        return [OT.f] + [OT.eq]*((self.K+1) *self.n)
