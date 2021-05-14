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
        if ex  == "constant":
            treatment= np.array(random.choices([0,1],[1-e,e],k=X.shape[0]))
            self.propensity = np.array([e]*X.shape[0])         
        else:
            self.propensity = (1+stats.beta.pdf(X[:,1],2,4))/4
            treatment = [1 if t > 0.5 else 0 for t in self.propensity]
        return treatment,self.propensity
                     
    # we define tw0 distributions referring to the paper
 
    def get_linear_outcome(self, g, X, beta, c=0):
        """get outcome for linear outcome functions
            g: global linear or piecewise linear or 0/1 linear
            X: np.array/matrix(without treatment label)
            treatment: treatment from get_treatment
            beta: coefficient vector for X with dimension(n_feature,)
            c: constant
            
            output:
            y: outcome
        """
        if g == "global":
            # global linear
            y= np.dot(X,beta)+c
            
        elif g =="piecewise":
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
         true_ite: the true ite generated
         X: covariate
         y: outcome
         treatment:
         propensity:
        """
        # covariate X is generated with normal distribution, 
        # in paper the author generate X with a correlation matrix created with vine method,
        # here we just use the simplest setting
        
        X = np.random.normal(size= (self.nsample,self.dimension))
        
        # SI 1 The unbalanced case with a simple CATE
        if self.simulation == 1:
            # beta is generated with uniform distribution from (-5,5)
            beta = np.random.uniform(low=-5,high=5,size=(self.dimension,))
            y_c = self.get_linear_outcome(g ="one",X = X, beta = beta, c=5)
            y_t = []
            for s in range(X.shape[0]):
                if X[s,1] > 0.1:
                    temp_c = 8
                else:
                    temp_c = 0            
                y_t.append(y_c[s] + temp_c)
            y_t = np.array(y_t)
            
            # get treatment
            treatment,propensity  = self.get_treatment(X,ex ="constant",e=0.01)
            
            # get outcome
            y = np.zeros_like(y_t, dtype=float)
            y[treatment == 0] = y_c[treatment == 0]
            y[treatment == 1] = y_t[treatment == 1]
            
#             # get full data
#             full_data = np.hstack((y.reshape((-1,1)), X))
            
            # get the true ite simulated
            true_ite = y_t - y_c
            
            
        # SI 2 The Balanced case without confounding - complex linear
        elif self.simulation == 2:
            
            # beta is generated with uniform distribution from (1,30)
            beta1 = np.random.uniform(low=1,high=30,size=(self.dimension,))
            beta0 = np.random.uniform(low=1,high=30,size=(self.dimension,))
            y_c = self.get_linear_outcome(g ="global",X = X, beta = beta0, c=0)
            y_t = self.get_linear_outcome(g ="global",X = X, beta = beta1, c=0)
            
            # get treatment
            treatment,propensity    = self.get_treatment(X,ex ="constant",e=0.5)
            
            # get outcome
            y = np.zeros_like(y_t, dtype=float)
            y[treatment == 0] = y_c[treatment == 0]
            y[treatment == 1] = y_t[treatment == 1]
            
            # get the true ite simulated
            true_ite = y_t - y_c
            
            
        # SI 3 The Balanced case without confounding - complex nonlinear
        elif self.simulation == 3:
            y_t = self.get_nonlinear_outcome(X)
            y_c = -self.get_nonlinear_outcome(X)
            
            # get treatment
            treatment,propensity = self.get_treatment(X,ex ="constant",e=0.5)
            
            # get outcome
            y = np.zeros_like(y_t, dtype=float)
            y[treatment == 0] = y_c[treatment == 0]
            y[treatment == 1] = y_t[treatment == 1]
            
            # get the true ite simulated
            true_ite = y_t - y_c
          
        
        # SI 4 No treatment effect with global linear
        elif self.simulation == 4:
            # beta is generated with uniform distribution from (1,30)
            beta = np.random.uniform(low=1,high=30,size=(self.dimension,))
            y_c = self.get_linear_outcome(g ="global",X = X, beta = beta, c=0)
            y_t = y_c
            
            # get treatment
            treatment,propensity = self.get_treatment(X,ex ="constant",e=0.5)
            
            # get outcome
            y = np.zeros_like(y_t, dtype=float)
            y[treatment == 0] = y_c[treatment == 0]
            y[treatment == 1] = y_t[treatment == 1]

            
            # get the true ite simulated
            true_ite = y_t - y_c
        
        
        
        # SI 5 No treatment effect with piecewise linear
        elif self.simulation == 5:
            # beta is generated with uniform distribution from (-15,15)
            beta = np.random.uniform(low=-15,high=15,size=(self.dimension,))
            y_c = self.get_linear_outcome(g ="piecewise",X = X, beta = beta, c=0)
            y_t = y_c
            
            # get treatment
            treatment,propensity = self.get_treatment(X,ex ="constant",e=0.5)
            
            # get outcome
            y = np.zeros_like(y_t, dtype=float)
            y[treatment == 0] = y_c[treatment == 0]
            y[treatment == 1] = y_t[treatment == 1]

            
            # get the true ite simulated
            true_ite = y_t - y_c    
            
        # SI 6 Confounding with beta distribution, no treatment effect, 
        # in this case we have to choose X ~ Unif[0,1]
        else:
            
            X = np.random.uniform(low=0,high=1,size= (self.nsample,self.dimension))
            # beta is [2,0,0,...]
            beta = np.zeros(self.dimension)
            beta[0] = 2
            y_c = self.get_linear_outcome(g ="global",X = X, beta = beta, c=-1)
            y_t = y_c
            
            # get treatment
            treatment,propensity = self.get_treatment(X,ex ="confounding",e=None)
            
            # get outcome
            y = np.zeros_like(y_t, dtype=float)
            y[treatment == 0] = y_c[treatment == 0]
            y[treatment == 1] = y_t[treatment == 1]

            
            # get the true ite simulated
            true_ite = y_t - y_c    
        
        
        #return true_ite, full_data
        return  true_ite,  X, y,treatment,propensity



        
if __name__ == '__main__':
    print('synthetic data generator')
