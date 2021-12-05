import numpy as np
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT



def Compute(problem, x):
    
    ot = problem.getFeatureTypes()
    index_f = [i for i, x in enumerate(ot) if x == OT.f] 
    assert( len(index_f) <= 1 )
    index_r = [i for i, x in enumerate(ot) if x == OT.sos] 
        

    phi, J = problem.evaluate(x)
    
    c = 0
    n = problem.getDimension()
    H = np.zeros([n, n])
    grad = np.zeros(n)
    
    if len(index_f) > 0:
        c += phi[index_f][0]
        grad += J[index_f][0]
        H += problem.getFHessian(x) 
        
        # Check H is positive definite
        if np.any(np.linalg.eigvals(H) < 0):
            H += (abs(min(np.linalg.eigvals(H)))+.02) * np.eye(n)
                        
                        
    if len(index_r) > 0:
        c += phi[index_r].T @ phi[index_r]
        grad += 2 * J[index_r].T @ phi[index_r]
        H += 2 * J[index_r].T @ J[index_r] 


    return c, grad, H



class SolverUnconstrained(NLPSolver):

    def __init__(self):
        """
        See also:
        ----
        NLPSolver.__init__
        """
    
        # in case you want to initialize some class members or so...
        

   

    # def LineSearch(x,c, delta, alpha = 1):
    #     while 1:
    #         c_new = Compute(x+alpha*delta)[0]
    #         if c_new > c:
    #             break
    #         else:
    #             alpha = self.rho*alpha
    #     return alpha
    
    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """

        # write your code here

        # Initialization
        x = self.problem.getInitializationSample()
        x = x.astype(float)
       
            
        
        alpha = 1
        rhoPlus = 1.3
        rhoMinus = .5
        rhols = .1
                        
        theta = 1e-3
        iteration_num = 1
        c, grad, H = Compute(self.problem, x)    
        delta = np.linalg.solve(H, -grad)
  
        while np.max(np.abs(alpha * delta)) >= theta or iteration_num >= 500:
            
                
            c, grad, H = Compute(self.problem, x)
            
            if np.any(np.linalg.eigvals(H)<=0):
                print('Hessian is positive definite!')
                raise NotImplementedError
              
            delta = np.linalg.solve(H, -grad)
            delta = delta/np.linalg.norm(delta,2)
            
            
            
            count = 0
            c1, grad1, H1 = Compute(self.problem, x + alpha *delta)
            
            while c1 > c:
                alpha = rhoMinus * alpha
                c1, grad1, H1 = Compute(self.problem, x + alpha *delta)
                if count == 100:
                    print('Cost dose not improve!')
                    raise NotImplementedError
                count += 1
            
            x += alpha * delta
            alpha = np.min([rhoPlus * alpha, np.inf])    
            iteration_num += 1
        print('\n The number of iterations is ', iteration_num)
        print('The Optimum is ', x)
        print('The Objective valueat optimum is ', c1)
        return x
    
       