import pandas as pd
import numpy as np
import random
pd.set_option('display.max_columns', 100)


class resample_from_GOTV:
    '''
    the data must be preprocessed, with d['W'] = the treatment indicator and d['Y'] = the outcome 
    '''
    def __init__(self, data):
        self.data = data
        self.d_w1 = self.data[self.data['W'] == 1]
        self.d_w0 = self.data[self.data['W'] == 0]
        
    def get_treat_control_equalsize(self, n_sample = 10000):
        # get # treatment = # control 
        
        return self.d_w0.iloc[:n_sample//2], self.d_w1.iloc[:n_sample//2]
    
    def get_treat_control_diffsize(self, n_sample = 10000, ratio = 0.05):
        # get # treatment << # control 
        n_w1 = int(n_sample*ratio)
        n_w0 = n_sample - n_w1
        
        return self.d_w0.iloc[:n_w0], self.d_w1.iloc[:n_w1]
        
        


class resample_from_synthetic_data:
    def __init__(self, n_sample):
        self.n_sample = n_sample
        
    def get_linear_dist_label(self,x, w):
        # a linear func to generate label
        # when 0 < x < 0.8,  return 0, else return 1. 20% of having label = 1 
        generated_num = x 
        if w == 0:
            if generated_num > 0.8:
                return 1
            return 0
        else:
            if generated_num > 0.5:
                return 1
            return 0
        
    def get_complex_dist_label(self,x):
        # a complex func  abs(x*x*x - x) is a quadratic func when x ∈ (0,1)
        # < 20 % change of returning 0
        generated_num = np.abs(x*x*x - x) 
        
        if generated_num > 0.36:
            return 0
        return 1
    
    def get_data_with_diff_distribution(self,ratio = 0.2):
        total_size = self.n_sample
        n_w1 = int(total_size * ratio)
        n_w0 = total_size - n_w1
        df = pd.DataFrame()
        W = []
        Y0 = []
        Y1 = []
        X = []
        Y = []
        for i in range(total_size):
            
            ran_num = random.uniform(0,1)
            X.append(ran_num)
            Y0.append(self.get_linear_dist_label(ran_num,0))
            Y1.append(self.get_complex_dist_label(ran_num))
            if i < n_w0:
                W.append(0)
                Y.append(self.get_linear_dist_label(ran_num,0))
            else:
                W.append(1)
                Y.append(self.get_complex_dist_label(ran_num))
        
        df['W'] = W
        df['Y0'] = Y0
        df['Y1'] = Y1
        df['X'] = X
        df['Y'] = Y
        
        return df
        
    def get_data_with_same_distribution(self,ratio = 0.2):
        total_size = self.n_sample
        n_w1 = int(total_size * ratio)
        n_w0 = total_size - n_w1
        df = pd.DataFrame()
        W = []
        Y0 = []
        Y1 = []
        X = []
        Y = []
        for i in range(total_size):
            
            ran_num = random.uniform(0,1)
            X.append(ran_num)
            Y0.append(self.get_linear_dist_label(ran_num,0))
            Y1.append(self.get_linear_dist_label(ran_num,1))
            if i < n_w0:
                W.append(0)
                Y.append(self.get_linear_dist_label(ran_num,0))
            else:
                W.append(1)
                Y.append(self.get_linear_dist_label(ran_num,1))
        
        df['W'] = W
        df['Y0'] = Y0
        df['Y1'] = Y1
        df['X'] = X
        df['Y'] = Y
        
        return df     
    
    
    def get_data_with_zero_treatment_effect(self,ratio = 0.2):
        total_size = self.n_sample
        n_w1 = int(total_size * ratio)
        n_w0 = total_size - n_w1
        df = pd.DataFrame()
        W = []
        Y0 = []
        Y1 = []
        X = []
        Y = []
        for i in range(total_size):
            while True:
                ran_num = random.uniform(0,1)
                if self.get_linear_dist_label(ran_num,0) != self.get_linear_dist_label(ran_num,1):
                    continue
                else:
                    break
            X.append(ran_num)
            Y0.append(self.get_linear_dist_label(ran_num,0))
            Y1.append(self.get_linear_dist_label(ran_num,1))
            if i < n_w0:
                W.append(0)
                Y.append(self.get_linear_dist_label(ran_num,0))
            else:
                W.append(1)
                Y.append(self.get_linear_dist_label(ran_num,1))
        
        df['W'] = W
        df['Y0'] = Y0
        df['Y1'] = Y1
        df['X'] = X
        df['Y'] = Y
        
        return df     
            
        
        