import math
import sys
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class AntennaPlacement(MathematicalProgram):
    """
    """

    def __init__(self, P, w):
        """
        Arguments
        ----
        P: list of 1-D np.arrays
        w: 1-D np.array
        """
        # in case you want to initialize some class members or so...
        self.P = P
        self.w = w
        self.dim = 2
        
        
    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """

        # add the main code here! E.g. define methods to compute value y and Jacobian J
        y = 0
        J = np.zeros(self.dim)
        
        for j in range(len(self.w)):
            expTerm = np.exp(- np.linalg.norm(x - self.P[j]) ** 2)
            y += - self.w[j] * expTerm
            J += 2 * self.w[j] * expTerm * (x - self.P[j])

        return np.array([y]) , np.array([J])


    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        return self.dim

    def getFHessian(self, x):
        """
        See Also
        ------
        MathematicalProgram.getFHessian
        """
        # add code to compute the Hessian matrix
       
        H = np.zeros((self.dim, self.dim))
        
        for i in range(len(self.w)):
            expTerm = np.exp(- np.linalg.norm(x - self.P[i]) ** 2)
            H[0][0] += (2 - 4 * (x[0] - self.P[i][0]) ** 2) * self.w[i] * expTerm
            H[1][1] += (2 - 4 * (x[1] - self.P[i][1]) ** 2) * self.w[i] * expTerm
            H[1][0] += -4 * self.w[i]  *expTerm * (x[1] - self.P[i][1]) * (x[0] - self.P[i][0])
        H[0][1] = H[1][0]
        
        return H



    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        
        x0 = 0
        for j in range(len(self.P)):
            x0 += self.P[j]
            
        return x0/len(self.P)

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types

        """
        return [OT.f]
