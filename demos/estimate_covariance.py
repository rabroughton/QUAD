# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:04:46 2019

@author: Rachel
"""
from __future__ import division  # Use "real" division

# Add another filepath to the Python path
import sys
sys.path.insert(1, r'C:\Users\Rachel\g2conda\GSASII\bindist')
sys.path.insert(2, r'C:\Users\Rachel\g2conda\GSASII')

# Import modules
import numpy as np
from scipy.stats import norm

import statsmodels.api as sm 
lowess = sm.nonparametric.lowess           # Lowess smoothing function
from bspline import Bspline                # Bspline function
from splinelab import augknt  
import matplotlib.pyplot as plt

def z2par(z, lower, upper, grad=False):
    if (grad):
        d = (upper-lower)*norm.pdf(z)
        return d
    else:
        par = lower + (upper-lower)*norm.cdf(z)
        # Fudge the parameter value if we've hit either boundary
        par[np.array([par[j]==upper[j] for j in range(len(par))])] -= 1e-10
        par[np.array([par[j]==lower[j] for j in range(len(par))])] += 1e-10
        return par
    
def estimatecovariance(paramList,start,init_z,Calc,upper,lower): 
    
    Calc.UpdateParameters(dict(zip(paramList, start)))
    f0 = Calc.Calculate()
    
    #Finit difference calculation
    def finitediff(ii,init_z,paramList,f0,delta=1e-3): 
        q_star = np.copy(init_z)
        q_star[ii] = np.copy(init_z[ii])*(1+delta)
        #print(q_star)
        params = z2par(q_star,lower,upper)
        Calc.UpdateParameters(dict(zip(paramList, params)))
        f1 = Calc.Calculate()
        diff = (f1-f0)/(delta*init_z[ii])
        return diff
     
    Index = np.where((Calc._tth>Calc._lowerLimit) & (Calc._tth<Calc._upperLimit) == True)
    y = np.array(Calc._Histograms[list(Calc._Histograms.keys())[0]]['Data'][1][Index], copy=True)
    x = np.array(Calc._tthsample, copy=True)
    # Calculate sensitivity matrix
    sensitivity = np.zeros([np.shape(y)[0],np.shape(init_z)[0]])
    
    for jj in range (0,np.shape(init_z)[0]):
        diff = finitediff(jj,init_z,paramList,f0)
        sensitivity[:,jj] = diff
       
    #Calculate fisher information matrix    
    fisher = np.dot(sensitivity.transpose(),sensitivity)
    
    #Calcualte estimate for s^2  
    L=20
    unique_knots = np.percentile(a=x, q=np.linspace(0, 100, num=(L-2)))
    knots = augknt(unique_knots, 3)
    objB = Bspline(knots, order=3)
    B = objB.collmat(x)
    del unique_knots, knots, objB
    gamma = np.ones(L)
    BG = np.matmul(B, gamma)
    
    R = y-BG-f0 #residuals
    n=np.shape(y)[0]
    p = np.shape(init_z)[0]
    s2 = (1./(n-p))*np.dot(R,R.transpose())
    
    #Calculate covariance matrix
    covariance = s2*np.linalg.inv(fisher)
    
    plt.plot(x,y,x,f0)
    
    evals = np.linalg.eig(covariance)
    print("Covariance eignvalues:\n{}".format(evals[0]))
    return dict(cov=covariance,s2=s2,evals=evals)