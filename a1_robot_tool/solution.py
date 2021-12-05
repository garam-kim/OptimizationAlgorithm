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
        self.q0 = q0
        self.pr = pr
        self.l = l
        # in case you want to initialize some class members or so...

    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """

        # add the main code here! E.g. define methods to compute value y and Jacobian J
        q1, q2, q3 = x[0], x[1], x[2]
        m = len(np.concatenate([self.q0, self.pr]))
        n = len(x)
                
        p1 = np.cos(q1) + .5 * np.cos(q1 + q2) + np.cos(q1 + q2 + q3) / 3
        p2 = np.sin(q1) + .5 * np.sin(q1 + q2) + np.sin(q1 + q2 + q3) / 3
        pq = [p1, p2]
        y = np.zeros(m)

        y[ :2] = pq - self.pr
        y[2: ] = np.sqrt(self.l) * (x - self.q0) 
    
        
        J = np.zeros([m,n])
        j11 = - p2
        j21 = p1
        
        j12 = -.5 * np.sin(q1 + q2) - np.sin(q1 + q2 + q3)/3
        j22 = .5 * np.cos(q1 + q2) + np.cos(q1 + q2 + q3) /3
        
        j13 = -np.sin(q1 + q2 + q3) /3
        j23 = np.cos(q1 + q2 + q3) / 3
        
        J[:2, : ] = np.array([[j11, j12, j13], [j21, j22, j23] ])
        J[2: , : ] = np.sqrt(self.l) * np.eye(n)
        
        return  y  , J

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
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
