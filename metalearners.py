import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from copy import deepcopy


def rmse(y_hat,y):
    return np.sqrt(np.mean((y_hat-y)**2))


class Slearner():
    
    def __init__(self, baselearner= None, alpha=0.05,is_regressor = False):
        
        """Initialize an S-learner.
        Args:
            baselearner: a model to estimate the treatment effect
            alpha: a significance level for estimating confidence interval
            is_regressor: base model is a regressor or a classifier
        """
        self.model = baselearner
        self.alpha = alpha
        self.is_regressor = is_regressor
      
    
    def fit(self, X, treatment, y):
        
        """fit a baselearner.
        Args:
            X: (np.matrix, np.array) of feature
            y: 1d np.array for outcome observed
            treatment:(np.array or pd.Series) indicating treatment/control groups, 0 for control, 1 for treatment
        """
        X_new = np.hstack((np.array(treatment).reshape((-1,1)), X))
        self.model.fit(X_new, y)
    
    def get_ite(self, X, treatment, y):
        
        """impute effect under the other circumstance
        Args:
            X: (np.matrix, np.array) of feature
            y: 1d np.array for outcome observed
            treatment:(np.array or pd.Series) indicating treatment/control groups, 0 for control, 1 for treatment
        """
        
        # set the treatment column to zero (the control group)
        X_new = np.hstack((np.zeros((X.shape[0], 1)), X))
        if self.is_regressor:
            yhat_cs = self.model.predict(X_new)
        else:
            yhat_cs = self.model.predict_proba(X_new)[:,1]
        
        # set the treatment column to one (the treatment group)
        X_new[:, 0] = 1
        if self.is_regressor:
            yhat_ts = self.model.predict(X_new)
        else:
            yhat_ts = self.model.predict_proba(X_new)[:,1]
            
        # get the prediction for evaluation
        yhat = np.zeros_like(y, dtype=float)
        yhat[treatment == 0] = yhat_cs[treatment == 0]
        yhat[treatment == 1] = yhat_ts[treatment == 1]
        
        # get the treatment effect
        ite = yhat_ts - yhat_cs
        
        return ite, yhat_ts, yhat_cs
    
    def boostrap_interval():
        # todo
        return 
        
class Tlearner():
    
    def __init__(self, baseclearner= None,basetlearner= None, alpha=0.05, is_regressor = False):
        """Initialize T-learner.
        Args:
            baseclearner: base model to fit on control group for control effect
            basetlearner: base model to fit on tratment group for treatment effect
            alpha: a significance level for estimating confidence interval
            is_regressor: base model is a regressor or a classifier
        """
        self.cmodel = baseclearner
        self.tmodel = basetlearner
        self.alpha = alpha
        self.is_regressor = is_regressor
       
    
    def fit(self, X, treatment, y):
        
        """fit baselearners.
        Args:
            X: (np.matrix, np.array) of feature
            y: 1d np.array for outcome observed
            treatment:(np.array or pd.Series) indicating treatment/control groups, 0 for control, 1 for treatment
        """
        X_c, y_c = X[treatment==0],y[treatment==0]
        X_t, y_t = X[treatment==1],y[treatment==1]

        self.cmodel.fit(X = X_c, y = y_c)
        self.tmodel.fit(X = X_t, y = y_t)
        
    def get_ite(self, X, treatment, y):
        
        """get ite treatmentith imputation
        Args:
            X: (np.matrix, np.array) of feature
            y: 1d np.array for outcome observed
            treatment:(np.array or pd.Series) indicating treatment/control groups, 0 for control, 1 for treatment
        """
        if self.is_regressor:
            yhat_cs = self.cmodel.predict(X)
            yhat_ts = self.tmodel.predict(X)
        else:
            yhat_cs = self.cmodel.predict_proba(X)[:,1]
            yhat_ts = self.tmodel.predict_proba(X)[:,1]
            
        # get the prediction for evaluation
        yhat = np.zeros_like(y, dtype=float)
        yhat[treatment == 0] = yhat_cs[treatment == 0]
        yhat[treatment == 1] = yhat_ts[treatment == 1]
        
        # get the treatment effect
        ite = yhat_ts - yhat_cs
        
        return ite, yhat_ts, yhat_cs
        
    def boostrap_interval():
        return 


class Xlearner():
    
    def __init__(self,
                 learner=None,
                 control_outcome_learner=None,
                 treatment_outcome_learner=None,
                 control_effect_learner=None,
                 treatment_effect_learner=None,
                 is_regressor = False,
                 propensity_model = None,
                 alpha=.05):
        
        """Initialize a X-learner.
        Args:
            learner : a model to estimate outcomes and treatment effects in both the control and treatment
                groups
            control_outcome_learner : a model to estimate outcomes in the control group
            treatment_outcome_learner : a model to estimate outcomes in the treatment group
            control_effect_learner : a model to estimate treatment effects in the control group
            treatment_effect_learner : a model to estimate treatment effects in the treatment group
            propensity_model: a model to learn propensity score
            alpha (float, optional): the confidence level alpha of the ATE estimate
        """
        
    
        if control_outcome_learner is None:
            self.model_mu_c = deepcopy(learner)
        else:
            self.model_mu_c = control_outcome_learner

        if treatment_outcome_learner is None:
            self.model_mu_t = deepcopy(learner)
        else:
            self.model_mu_t = treatment_outcome_learner

        if control_effect_learner is None:
            self.model_tau_c = deepcopy(learner)
        else:
            self.model_tau_c = control_effect_learner

        if treatment_effect_learner is None:
            self.model_tau_t = deepcopy(learner)
        else:
            self.model_tau_t = treatment_effect_learner

        self.alpha = alpha
        self.propensity = None
        self.propensity_model = propensity_model
        self.is_regressor = is_regressor
    
    def fit(self, X, treatment, y, p=None):
        """Fit the inference model.
        Args:
            X (np.matrix or np.array): a feature matrix
            treatment (np.array): a treatment vector
            y (np.array): an outcome vector
            p (np.ndarray or dict, optional): an array of propensity scores of float (0,1) in the
                single-treatment case; or, a dictionary of treatment groups that map to propensity vectors of
                float (0,1)
        """
        # Train outcome models
        self.model_mu_c.fit(X[treatment == 0], y[treatment == 0])
        self.model_mu_t.fit(X[treatment == 1], y[treatment == 1])

        # Calculate variances and treatment effects
        if self.is_regressor:
            var_c = (y[treatment == 0] - self.model_mu_c.predict(X[treatment == 0])).var()
            var_t = (y[treatment == 1] - self.model_mu_t.predict(X[treatment == 1])).var()
            d_c = self.model_mu_t.predict(X[treatment == 0]) - y[treatment == 0]
            d_t = y[treatment == 1] - self.model_mu_c.predict(X[treatment == 1])
        else:
            var_c = (y[treatment == 0] - self.model_mu_c.predict_proba(X[treatment == 0])[:,1]).var()
            var_t = (y[treatment == 1] - self.model_mu_t.predict_proba(X[treatment == 1])[:,1]).var()
            d_c = self.model_mu_t.predict_proba(X[treatment == 0])[:,1] - y[treatment == 0]
            d_t = y[treatment == 1] - self.model_mu_c.predict_proba(X[treatment == 1])[:,1]
        
        self.vars_c= var_c
        self.vars_t = var_t

        # Train treatment models
        self.model_tau_c.fit(X[treatment == 0], d_c)
        self.model_tau_t.fit(X[treatment == 1], d_t)
    

    def get_propensity(self,X,treatment):
        self.propensity_model.fit(X,treatment)
        self.propensity = self.propensity_model.predict_proba(X)[:,1]
        return self.propensity

    def get_ite(self, X, treatment, y):
        
        """get ite 
        Args:
            X: (np.matrix, np.array) of feature
            y: 1d np.array for outcome observed
            treatment:(np.array or pd.Series) indicating treatment/control groups, 0 for control, 1 for treatment
        """
        
        p_1 = self.get_propensity(X,treatment)
        p_0 = 1-p_1
        dhat_cs = self.model_tau_c.predict(X)
        dhat_ts = self.model_tau_t.predict(X)
        # get weighted ite
        ite = p_1 * dhat_cs + p_0 * dhat_ts
        # get performance of imputation
        yhat = np.zeros_like(y, dtype=float)
        yhat[treatment == 0] = self.model_mu_c.predict(X[treatment == 0])
        yhat[treatment == 1] = self.model_mu_t.predict(X[treatment == 1])

        return ite, dhat_ts, dhat_cs
        
    def boostrap_interval():
        pass

if __name__ == '__main__':
    print('Metalearner')
