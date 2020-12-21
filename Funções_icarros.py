import itertools
import numpy as np
import pandas as pd
import math as m

def entropia(X):
    
    possibilidades = X.unique()
    
    h = 0
    for x in possibilidades:
        prob_x = (X ==x).sum()/len(X)
        if prob_x > 0:
            h -= prob_x*m.log(prob_x)/m.log(2)
    return h

def inf_mutua(df):
    
    col_0 = df.columns[0]
    col_1 = df.columns[1]
    possibilidades = itertools.product(*[df[col_0].unique(),df[col_1].unique()])
    len_possibilidades = len(df[col_0].unique())*len(df[col_1].unique())
    
    inf = 0
    for xy in possibilidades:
        
        acertos_x = np.array(df[col_0] == xy[0])
        acertos_y = np.array(df[col_1] == xy[1])
        
        prob_x = acertos_x.sum()/df.shape[0]
        prob_y = acertos_y.sum()/df.shape[0]
        prob_xy = (acertos_x*acertos_y).sum()/df.shape[0]
        
        if prob_xy > 0:
            inf += prob_xy*m.log(prob_xy/(prob_x*prob_y))/m.log(2)
    
    return inf
