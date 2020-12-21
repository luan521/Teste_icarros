import math as m
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

class CustomScaler(): 
    
    # init or what information we need to declare a CustomScaler object
    # and what is calculated/declared as we do
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        
        # scaler is nothing but a Standard Scaler object
        self.scaler = StandardScaler(copy,with_mean,with_std)
        # with some columns 'twist'
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    
    # the fit method, which, again based on StandardScale
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    # the transform method which does the actual scaling

    def transform(self, X, y=None, copy=None):
        
        # record the initial order of the columns
        init_col_order = X.columns
        
        # scale all features that you chose when creating the instance of the class
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        
        # declare a variable containing all information that was not scaled
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        
        # return a data frame which contains all scaled features and all 'not scaled' features
        # use the original order (that you recorded in the beginning)
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
    
class ReplaceOutliers():
    
    def __init__(self,columns,lim):        
        self.columns = columns
        self.cols_drop = None
        self.lim = lim
        
    def fit(self, X):
        self.cols_drop = {}
        for col in self.columns:            
            series = (X.groupby([col])[col].count()/X.shape[0]).sort_values(ascending=False)
            i = 1
            while series[:i].sum() < self.lim:
                i += 1
            self.cols_drop[col] = series.index[i:]
            
    def transform(self, X):
        Y = X.copy()
        inicio = True
        for col in self.cols_drop:
            Y[col] = Y[col].replace(self.cols_drop[col],'other')
                    
        return Y
    
class Model():
    
    def __init__(self,cols_drop, cols_drop_out, lim, cols_standard):        
        self.columns_drop = cols_drop
        self.cols_replace_out = ReplaceOutliers(cols_drop_out,lim)
        self.cols_standard = CustomScaler(cols_standard)
        self.reg = LinearRegression()
        self.cols_inputs = None
        
    def fit(self,X):
        Y = X.copy()
        Y = X.drop(self.columns_drop, axis = 1)
        self.cols_replace_out.fit(Y)
        Y = self.cols_replace_out.transform(Y)
        self.cols_standard.fit(Y)
        Y = self.cols_standard.transform(Y)
        Y = pd.get_dummies(Y,drop_first=True)
        self.cols_inputs = [c for c in Y.columns if c != 'price']
        
        inputs = np.array(Y.drop(['price'],axis=1))
        targets = Y['price'].apply(m.log)
        self.reg.fit(inputs,targets)
    
    def yhat_targets(self,X):
        Y = X.copy()
        Y = X.drop(self.columns_drop, axis = 1)
        Y = self.cols_replace_out.transform(Y)
        self.cols_standard.fit(Y)
        Y = self.cols_standard.transform(Y)
        Y = pd.get_dummies(Y,drop_first=True)
        
        for c in self.cols_inputs:
            if c not in Y.columns:
                Y[c] = [0 for i in range(Y.shape[0])]
        for c in Y.columns:
            if c not in self.cols_inputs and c != 'price':
                Y = Y.drop([c],axis=1)
        
        inputs = np.array(Y.drop(['price'],axis=1))
        targets = Y['price'].apply(m.log)
        y_hat = self.reg.predict(inputs)
        
        return y_hat, targets