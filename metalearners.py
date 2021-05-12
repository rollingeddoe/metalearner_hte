
class Slearner():
    
    def __init__(self, baselearner= None, alpha=0.05,is_regressor = False):
        
        """Initialize an S-learner.
        Args:
            baselearner: a model to estimate the treatment effect
            alpha: a significance level for estimating confidence interval
            is_regressor: base model is a regressor or a classifier
        """
        self.model = learner
        self.alpha = alpha
        self.is_regressor = is_regressor
      
    
    def fit(self, X, treatment, y):
        
        """fit a baselearner.
        Args:
            X: (np.matrix, np.array) of feature
            y: 1d np.array for outcome observed
            treatment:(np.array or pd.Series) indicating treatment/control groups, 0 for control, 1 for treatment
        """
        X_new = np.hstack((treatment.reshape((-1,1)), X))
        self.model.fit(X_new, y)
    
    def predict(self, X, treatment, y):
        
        """impute effect under the other circumstance
        Args:
            X: (np.matrix, np.array) of feature
            y: 1d np.array for outcome observed
            treatment:(np.array or pd.Series) indicating treatment/control groups, 0 for control, 1 for treatment
        """
        
        # set the treatment column to zero (the control group)
        X_new = np.hstack((np.zeros((X.shape[0], 1)), X))
        if self.is_regressor:
            yhat_cs = model.predict(X_new)
        else:
            yhat_cs = model.predict_proba(X_new)
        
        # set the treatment column to one (the treatment group)
        X_new[:, 0] = 1
        if self.is_regressor:
            yhat_ts = model.predict(X_new)
        else:
            yhat_ts = model.predict_proba(X_new)
            
        # get the prediction for evaluation
        yhat = np.zeros_like(y, dtype=float)
        yhat[treatment == 0] = yhat_cs[treatment == 0]
        yhat[treatment == 1] = yhat_ts[treatment == 1]
        
        # get the treatment effect
        ite = yhat_ts - yhat_cs
        
        return ite, yhat_ts, yhat_cs, rmse(y_hat,y)
    
    def estimate_cate():
        
    def boostrap_interval():




class Tlearner():
    
    def __init__(self, baseclearner= None,basetlearner= None, alpha=0.05, is_regressor = False):
        
        """Initialize T-learner.
        Args:
            baseclearner: base model to fit on control group for control effect
            basetlearner: base model to fit on tratment group for control effect
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
        
        self.cmodel.fit(X_c, y_c)
        self.tmodel.fit(X_t, y_t)
        
    def predict(self, X, treatment, y):
        
        """get ite with imputation
        Args:
            X: (np.matrix, np.array) of feature
            y: 1d np.array for outcome observed
            treatment:(np.array or pd.Series) indicating treatment/control groups, 0 for control, 1 for treatment
        """
        if self.is_regressor:
            yhat_cs = cmodel.predict(X)
            yhat_ts = tmodel.predict(X)
        else:
            yhat_cs = cmodel.predict_proba(X)
            yhat_ts = tmodel.predict_proba(X)
            
        # get the prediction for evaluation
        yhat = np.zeros_like(y, dtype=float)
        yhat[treatment == 0] = yhat_cs[treatment == 0]
        yhat[treatment == 1] = yhat_ts[treatment == 1]
        
        # get the treatment effect
        ite = yhat_ts - yhat_cs
        
        return ite, yhat_ts, yhat_cs, rmse(y_hat,y)
    
    def estimate_cate():
        
    def boostrap_interval():




if __name__ == '__main__':
    print('Metalearner')