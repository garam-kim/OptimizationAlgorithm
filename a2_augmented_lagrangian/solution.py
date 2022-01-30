import numpy as np
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverAugmentedLagrangian(NLPSolver):

    def __init__(self):
        """
        See also:
        ----
        NLPSolver.__init__
        """
  
        
    def Compute(self, x, mu, nu):
        
        ot = self.problem.getFeatureTypes()
        n = self.problem.getDimension()

        self.index_f = [i for i, x in enumerate(ot) if x == OT.f] 
        assert( len(self.index_f) <= 1 )
        self.index_r = [i for i, x in enumerate(ot) if x == OT.sos] 
        self.index_i = [i for i, x in enumerate(ot) if x == OT.ineq]
        self.index_e = [i for i, x in enumerate(ot) if x == OT.eq]
        
        self.kappaVec  = np.zeros(len(ot))   # Lagrange multiplier for equalities
        self.lamVec  = np.zeros(len(ot))

        phi, J = self.problem.evaluate(x)

        c = 0
        H = np.zeros([n, n])
        grad = np.zeros(n)

        if len(self.index_f) > 0:
            c += phi[self.index_f][0]
            grad += J[self.index_f][0]
            H += self.problem.getFHessian(x) 


        if len(self.index_r) > 0:
            c += phi[self.index_r].T @ phi[self.index_r]
            grad += 2 * J[self.index_r].T @ phi[self.index_r]
            H += 2 * J[self.index_r].T @ J[self.index_r] 



        if len(self.index_i) > 0:
            for ind in self.index_i:
                lam = self.lamVec[ind]
                c +=  (phi[ind] >= 0 or lam > 0) * mu * phi[ind]**2 + lam * phi[ind]
                grad +=  (2 * (phi[ind] >= 0 or lam > 0) * mu * phi[ind] + lam) * J[ind]
                H += 2 * mu * (phi[ind] >= 0 or lam > 0) * np.outer(J[ind], J[ind])


        if len(self.index_e) > 0:
            for ind in self.index_e:
                kappa = self.kappaVec[ind]
                c += nu * phi[ind]**2 + kappa * phi[ind]
                grad +=  2 * nu * phi[ind] * J[ind] + kappa * J[ind]
                H += 2 * nu * np.outer(J[ind], J[ind])

        if np.any(np.linalg.eigvals(H) <= 0):
            H += (abs(min(np.linalg.eigvals(H)))+.02) * np.eye(n)
        return c, grad, H
    
   
    
    
    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """
        # Initialization
        x = self.problem.getInitializationSample()
        n = self.problem.getDimension()
    
        theta = 1e-3
        epsilon = 1e-3
        
        # parameter update
        mu = 1
        rhomu = 1.2
        nu = 1
        rhonu = 1.2
        
        
        rhoPlus = 1.3
        rhoMinus = .5
        rhols = .1

        alpha = 1
        
        sigmin = 0.5
        sigmax = 1.2
       
      
        inner_iter = 1        
        iternum = 1
        
        
      
        
        while iternum <= 1e4:
            xt = x.copy()
            
            while inner_iter <= 1e4:
                c, grad, H = self.Compute(x, mu, nu)
                delta = np.linalg.solve(H , -grad)
                if np.linalg.norm(delta) > 0:
                    delta = delta/np.linalg.norm(delta,2)
                
                count = 0
                c1, grad1, H1 = self.Compute(x + alpha * delta, mu, nu)

                while c1 > c:
                    alpha = rhoMinus * alpha
                    c1, grad1, H1 = self.Compute(x + alpha * delta, mu, nu)
                    if count == 100:
                        print('Cost dose not improve!')
                        raise NotImplementedError
                    count += 1
                
                x += alpha * delta            
                alpha = np.min([rhoPlus * alpha, np.inf])   
                
                if np.linalg.norm(alpha * delta) < theta:
                    break
                inner_iter += 1
            
            
        
            phi, J = self.problem.evaluate(x)
            for ind in self.index_i:
                self.lamVec[ind] = np.max(self.lamVec[ind] +  2 * mu * phi[ind], 0 )
            
            for ind in self.index_e:
                self.kappaVec[ind] = self.kappaVec[ind] +  2 * nu * phi[ind]    

                
            # increase mu, nu
            mu *= rhomu
            nu *= rhonu


            testOne = np.linalg.norm(xt-x) < theta
            testTwo = np.all(phi[self.index_i] < epsilon)
            testThree = np.all(abs(phi[self.index_e]) < epsilon)
            if testOne * testTwo * testThree == 1 :
          
                print(x)
                break
                
            iternum += 1
        return x
