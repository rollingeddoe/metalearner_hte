import numpy as np
import random
from scipy import stats

class synthetic_data():
    
    def __init__(self,
                 N = None,
                 d= None,
                 simulation = None
                ):
        """Initialize a synthetic data generator according to the paper
        N: the # of sample to generate
        d: the dimension, d ∈ N, of the feature space; 
        simulation: the simulation we are reproducing refer to the paper
        """
        self.nsample = N
        self.dimension = d
        self.propensity = None
        self.simulation = simulation
            
    
    def get_treatment(self,X=None,ex = None,e = None):
        """ get propensity and treatment/control label
            X: must be a np.array/matrix with shape(n_samples,features)
            ex: the propensity function, ex = constant if e(x) is constant, ex = confounding if ex depend on x;
            e: the propensity score if ex is constant;
            output: 
            self.propensity with true propensity
            treatment: treatment label, np.array of shape(nsample,)
        """
        if ex  == 'constant':
            treatment= np.array(random.choices([0,1],[1-e,e],k=X.shape[0]))
            self.propensity = np.array([e]*X.shape[0])         
        else:
            self.propensity = (1+stats.beta.pdf(X[:,1],2,4))/4
            treatment = [1 if t > 0.5 else 0 for t in self.propensity]
        return treatment
                     
    # we define tw0 distributions referring to the paper
 
    def get_linear_outcome(self, g,  X, beta, c=0):
        """get outcome for linear outcome functions
            g: global linear or piecewise linear or 0/1 linear
            X: np.array/matrix(without treatment label)
            treatment: treatment from get_treatment
            beta: coefficient vector for X with dimension(n_feature,)
            c: constant
            
            output:
            y: outcome
        """
        if g == 'global':
            # global linear
            y= np.dot(X,beta)+c
            
        elif g ==' piecewise':
            # piecewise linear
            y = []
            for s in range(X.shape[0]):
                if X[s,19] < -0.4:
                    beta[5:]=0
                elif X[s,19]<=0.4:
                    beta[5:10] = 0
                else:
                    beta[10:15] = 0
                y.append(np.dot(X[s,:],beta)+c)
        else:
            y = []
            for s in range(X.shape[0]):
                if X[s,0] > 0.5:
                    c*=1
                else:
                    c*=0            
                y.append(np.dot(X[s,:],beta)+c)
        
        return np.array(y)
                    
                    
    def get_nonlinear_outcome(self,X):
        
        """ get outcome for complex-nonlinear outcome functions
            X: np.array/matrix(without treatment label)
            distribution of the nonlinear function is: 
            $$\mu_0(x) = \frac{1}{2} \eta(x_1) \eta(x_2)$$
            ,where 
            $$\eta(x) = \frac{2}{1+e^{-12(\frac{x-1}{2})}}$$
            output:
            y: outcome
        
        """
        
        return np.exp(-12*(X[:,0]-0.5))*np.exp(-12*(X[:,1]-0.5))
    
    def get_full_data(self):
        """ get full synthetic data for all 6 simulations
         output: 
         full_dataset: np.array with shape(nsample, nfeatures+1), outcome y at first column X[:,0]
        """
        # covariate X is generated with normal distribution
        X = np.random.normal(size= (self.nsample,self.dimension))
        
        # SI 1.1 The unbalanced case with a simple CATE
        if self.simulation == 1:
            # beta is generated with uniform distribution from (-5,5)
            beta = np.random.uniform(low=-5,high=5,size=(self.dimension,))
            y_c = self.get_linear_outcome(self, g = '0/1',  X = X, beta = beta, c=5)
            y_t = []
            for s in range(X.shape[0]):
                if X[s,1] > 0.1:
                    temp_c = 8
                else:
                    temp_c = 0            
                y_t.append(y_c[s] + temp_c)
            
            # get treatment
            treatment  = self.get_treatment(self,X,ex = 'constant',e=0.01)
            
            # get outcome
            y = np.zeros_like(y_t, dtype=float)
            y[treatment == 0] = y_c[treatment == 0]
            y[treatment == 1] = y_t[treatment == 1]
            
            # get full data
            full_data = np.hstack((y.reshape((-1,1)), X))
            
            # get the true ite simulated
            true_ite = y_t - y_c
            
            
        # SI 1.2
        elif self.simulation == 2:
            pass
        
        else:
            pass
        
        
        return true_ite, full_data
