import numpy as np
import sys


sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT



class SolverInteriorPoint(NLPSolver):

    def __init__(self):
        """
        See also:
        ----
        NLPSolver.__init__
        """


    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """

        # write your code here

        # Initialization
        x = self.problem.getInitializationSample()
        n = self.problem.getDimension()


        def Compute(x, mu):

            ot = self.problem.getFeatureTypes()
            index_f = [i for i, x in enumerate(ot) if x == OT.f] 
            assert( len(index_f) <= 1 )
            index_r = [i for i, x in enumerate(ot) if x == OT.sos] 
            index_i = [i for i, x in enumerate(ot) if x == OT.ineq]



            phi, J = self.problem.evaluate(x)

            c = 0
            H = np.zeros([n, n])
            grad = np.zeros(n)

            if len(index_f) > 0:
                c += phi[index_f][0]
                grad += J[index_f][0]
                H += self.problem.getFHessian(x) 
           


            if len(index_r) > 0:
                c += phi[index_r].T @ phi[index_r]
                grad += 2 * J[index_r].T @ phi[index_r]
                H += 2 * J[index_r].T @ J[index_r] 



            if len(index_i) > 0:
                for i in index_i:
                    if phi[i] > 0:
                        c = np.inf
                        g = np.zeros(n)
                        break
                    c -= mu * np.log(-phi[i])
                    grad -= mu * J[i] / phi[i]
                    H += mu * (1/phi[i]**2) * np.outer(J[i],J[i])

            # Check H is positive definite
            if np.any(np.linalg.eigvals(H) < 0):
                H += (abs(min(np.linalg.eigvals(H)))+.02) * np.eye(n)
            return c, grad, H

    
    
        mu = 1
        muRho = .5
        alpha = 1
        rhoPlus = 1.3
        rhoMinus = .5
        rhols = .1
        theta = 1e-4
        iternum = 1
        inner_iter = 1
        

        while iternum <= 1000:
            xt = x.copy()
            mu *= muRho
            
            while inner_iter <= 1000:
                c, grad, H = Compute(x, mu)
                delta = np.linalg.solve(H , -grad)
                if np.linalg.norm(delta) > 0:
                    delta = delta/np.linalg.norm(delta,2)
                
                count = 0
                c1, grad1, H1 = Compute(x + alpha *delta, mu)

                while c1 > c:
                    alpha = rhoMinus * alpha
                    c1, grad1, H1 = Compute(x + alpha *delta, mu)
                    if count == 100:
                        print('Cost dose not improve!')
                        raise NotImplementedError
                    count += 1
                
                x += alpha * delta            
                alpha = np.min([rhoPlus * alpha, np.inf])   
                
                if np.linalg.norm(alpha * delta) < theta:
                    break
                inner_iter += 1
                
            if np.linalg.norm(xt-x) < 1e-6:
                print(x)
                break
                
            iternum += 1
        return x
