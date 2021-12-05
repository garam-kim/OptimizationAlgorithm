import sys
import math
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class RobotTool(MathematicalProgram):
    """
    """

    def __init__(self, q0, pr, l):
        """
        Arguments
        ----
        q0: 1-D np.array
        pr: 1-D np.array
        l: float
        """
        # in case you want to initialize some class members or so...
        self.q0 = q0
        self.pr = pr
        self.l = l

    def evaluate(self, x):
        p1 = np.cos(x[0])+0.5*np.cos(x[0]+x[1])+1/3*np.cos(x[0]+x[1]+x[2])
        p2 = np.sin(x[0])+0.5*np.sin(x[0]+x[1])+1/3*np.sin(x[0]+x[1]+x[2])
        p = np.array([p1, p2])

        phi = np.zeros((2,))
        phi[0] = np.linalg.norm(p - self.pr)
        phi[1] = np.sqrt(self.l)*np.linalg.norm(x-self.q0)
        y = phi

        p1_J_1 = -np.sin(x[0]) - 0.5 * np.sin(x[0] + x[1]) - 1 / 3 * np.sin(x[0] + x[1] + x[2])
        p1_J_2 = -0.5 * np.sin(x[0] + x[1]) - 1 / 3 * np.sin(x[0] + x[1] + x[2])
        p1_J_3 = - 1 / 3 * np.sin(x[0] + x[1] + x[2])
        p2_J_1 = np.cos(x[0]) + 0.5 * np.cos(x[0] + x[1]) + 1 / 3 * np.cos(x[0] + x[1] + x[2])
        p2_J_2 = 0.5 * np.cos(x[0] + x[1]) + 1 / 3 * np.cos(x[0] + x[1] + x[2])
        p2_J_3 = 1 / 3 * np.cos(x[0] + x[1] + x[2])

        J = np.zeros((2,3))
        J[0, 0] = np.matmul((p - self.pr), np.array([p1_J_1, p2_J_1])) / (np.linalg.norm(p - self.pr)+1e-10)
        J[0, 1] = np.matmul((p - self.pr), np.array([p1_J_2, p2_J_2])) / (np.linalg.norm(p - self.pr)+1e-10)
        J[0, 2] = np.matmul((p - self.pr), np.array([p1_J_3, p2_J_3])) / (np.linalg.norm(p - self.pr)+1e-10)
        J[1, 0] = np.sqrt(self.l) * (x[0] - self.q0[0]) / (np.linalg.norm(self.q0 - x)+1e-10)
        J[1, 1] = np.sqrt(self.l) * (x[1] - self.q0[1]) / (np.linalg.norm(self.q0 - x)+1e-10)
        J[1, 2] = np.sqrt(self.l) * (x[2] - self.q0[2]) / (np.linalg.norm(self.q0 - x)+1e-10)

        # y is a 1-D np.array of dimension m
        # J is a 2-D np.array of dimensions (m,n)
        # where m is the number of features and n is dimension of x
        return  y , J

        """
        See also:
        ----
        MathematicalProgram.evaluate
        """

        # add the main code here! E.g. define methods to compute value y and Jacobian J

#         q1, q2, q3 = x[0], x[1], x[2]
        
#         p1 = np.cos(q1) + .5 * np.cos(q1 + q2) + 1/3 * np.cos(q1 + q2 + q3)
#         p2 = np.sin(q1) + .5 * np.sin(q1 + q2) + 1/3 * np.sin(q1 + q2 + q3)
#         # pq = np.array([p1, p2])

#         m = len(np.concatenate([self.q0, self.pr]))
#         n = len(self.q0)
#         y = np.zeros(m)
#         y[0] = p1-self.pr[0]
#         y[1] = p2-self.pr[1]
#         y[2] = np.sqrt(self.l)*(q1-self.q0[0])
#         y[3] = np.sqrt(self.l)*(q2-self.q0[1])
#         y[4] = np.sqrt(self.l)*(a3]-self.q0[2])
        
#         J = np.zeros([m, n])
#         J[0][0] = -p2
#         J[1][0] = p1
#         J[2][0] = np.sqrt(self.l)
        
#         J[0][1] = -0.5*np.sin(x[0]+x[1])-np.sin(x[0]+x[1]+x[2])/3
#         J[1][1] = 0.5*np.cos(x[0]+x[1])+np.cos(x[0]+x[1]+x[2])/3
#         J[3][1] = np.sqrt(self.l)
        
#         J[0][2] = -np.sin(x[0]+x[1]+x[2])/3
#         J[1][2] = np.cos(x[0]+x[1]+x[2])/3
#         J[4][2] = np.sqrt(self.l)
#         return  y  , J
    


    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        return 3

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return self.q0

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.sos] * 5
