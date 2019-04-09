# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 13:41:34 2019
###############################################################################
Quantitative Uncertainty Analysis for Diffraction (QUAD)
###############################################################################
Description: This is a research tool that allows analysis of X-ray and neutron
diffraction data to infer the structure of materials with quantifiable 
uncertainty. QUAD uses Bayesian statistics and Markov chain sampling 
algorithms, together with components from the open source GSAS-II package, 
to create posterior probability distributions on all material structure 
parameters modeled by researchers. 
###############################################################################
Copyright (c) 2018, North Carolina State University
All rights reserved.
Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this 
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.
3. The names “North Carolina State University”, “NCSU” and any trade‐name, 
personal name, trademark, trade device, service mark, symbol, image, icon, or 
any abbreviation, contraction or simulation thereof owned by North Carolina 
State University must not be used to endorse or promote products derived from 
this software without prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
Authors: Susheela Singh, Christopher M. Fancher, Alyson Wilson, Brian Reich, 
Zhen Han, Ralph C. Smith, and Jacob L. Jones
License File: license_QUAD.txt
## Funding:  NSF: DMR-1445926 (RADAR Project ID 2014-2831)
Consortium for Nonproliferation Enabling Capabilities 
[Department of Energy, National Nuclear Security Administration]:  
DE-NA0002576 (RADAR Project ID 2014-0501)
Acknowledgement: This product includes software produced by UChicago Argonne, LLC 
under Contract No. DE-AC02-06CH11357 with the Department of Energy.
Version: 1.0.0
Maintainer: Jacob L. Jones
Email: JacobJones@ncsu.edu 
###############################################################################
"""

from __future__ import division  # Use "real" division

# Add another filepath to the Python path
import sys
sys.path.insert(0, r'C:\Users\Rachel\Documents\QUAD_Python3_7\calculatorfunctions')
sys.path.insert(1, r'C:\Users\Rachel\g2conda\GSASII')
sys.path.insert(2, r'C:\Users\Rachel\g2conda\GSASII\bindist')

# Import specific functions
from timeit import default_timer as timer  # Timing function
from scipy.stats import norm               # Normal distribution
from scipy.stats import multivariate_normal as mvnorm  # Multivariate normal distribution
import statsmodels.api as sm               # Library for lowess smoother
lowess = sm.nonparametric.lowess           # Lowess smoothing function
from bspline import Bspline                # Bspline function
from splinelab import augknt               # Bspline helper function

# Import entire modules
import numpy as np
import matplotlib.pyplot as plt

# Source mean function process from GSAS_Calculator_Opt.py
import GSAS_Calculator_v3_1_0 as gsas

## Helper functions
# Transform between bounded parameter space and continuous space
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

# Log likelihood of the parameters based on the prior distributions
def prior_loglike(par, m0, sd0):
    return -0.5*(sd0**(-2))*np.inner(par-m0, par-m0)

# Log of the posterior distribution
def logf(y, x, BG, Calc, paramList, z, lower, upper, scale, tau_y, m0, sd0):
    # Update the calculator to reflect the current parameter estimates
    params = z2par(z=z, lower=lower, upper=upper)
    Calc.UpdateParameters(dict(zip(paramList, params)))

    # Calculate the potential energy
    R = y-BG-Calc.Calculate()                         # Calculate residuals
    S = np.inner(R/np.sqrt(scale), R/np.sqrt(scale))  # Calculate weighted SSE
    l = 0.5*tau_y*S - prior_loglike(par=z, m0=m0, sd0=sd0)
    return (-1)*l

def calculate_bsplinebasis(x,L):
    # Calculate a B-spline basis for the range of x
    unique_knots = np.percentile(a=x, q=np.linspace(0, 100, num=(L-2)))
    knots = augknt(unique_knots, 3)
    objB = Bspline(knots, order=3)
    B = objB.collmat(x)
    return B

def diffraction_file_data(x,y,Calc):
    
    # Assign the intensity vector (y) from the GPX file, if necessary
    if y is None:
        Index = np.where((Calc._tth>Calc._lowerLimit) & (Calc._tth<Calc._upperLimit) == True)
        y = np.array(Calc._Histograms[list(Calc._Histograms.keys())[0]]['Data'][1][Index], copy=True)

    # Assign the grid of angles (x) from the GPX file, if no values are provided. If values ARE provided, overwrite the _tthsample parameter
    if x is None:
        x = np.array(Calc._tthsample, copy=True)
    else:
        Calc._tthsample = np.array(x, copy=True) # Should update Calc internally for remainder of script
    return x,y

def smooth_ydata(x,y):
    # Smooth the observed Ys on the Xs, patch for negative or 0 values
    y_sm = lowess(endog=y, exog=x, frac=6.0/len(x), return_sorted=False)
    y_sm = np.array([max(0, sm) for sm in y_sm])
    return y_sm

def initialize_cov(initCov, q):
    # Initialize covariance for proposal distribution
    if initCov is None:
        varS1 = np.diag(0.05*np.ones(q))
    elif initCov.shape == (q, q):
        varS1 = initCov
    else:
        raise ValueError("Specification for initCov is not valid. Please provide a (%d x %d) matrix." % (q, q))
    return varS1

def initialize_output(iters,q,n_keep,L,update):
    # Initialize output objects
    all_Z = np.zeros((iters, q))
    keep_params = np.zeros((n_keep, q))
    keep_gamma = np.zeros((n_keep, L))
    keep_b = np.zeros(n_keep)
    keep_tau_y = np.zeros(n_keep)
    keep_tau_b = np.zeros(n_keep)
    accept_rate_S1 = np.zeros(n_keep//update)
    accept_rate_S2 = np.zeros(n_keep//update)
    return all_Z,keep_params,keep_gamma,keep_b,keep_tau_y,keep_tau_b,accept_rate_S1,accept_rate_S2

def update_background(B,var_scale,tau_y,tau_b,L,Calc,y):
    ## Update basis function loadings and then background values
    BtB = np.matmul(np.transpose(B)/var_scale, B)
    VV = np.linalg.inv(tau_y*BtB + tau_b*np.identity(L))
    err = (y-Calc.Calculate())/var_scale
    MM = np.matmul(VV, tau_y*np.sum(np.transpose(B)*err, axis=1))
    gamma = np.random.multivariate_normal(mean=MM, cov=VV)
    BG = np.matmul(B, gamma)
    return gamma,BG

def stage2_acceptprob(can1_post,can2_post,cur_post,can_z1,can_z2,z,varS1):
    # Calculate the acceptance probability
    inner_n = 1 - np.min([1, np.exp(can1_post - can2_post)])
    inner_d = 1 - np.min([1, np.exp(can1_post - cur_post)])
    # Adjust factors for inner_n and inner_d to avoid approaching the boundaries 0, 1
    inner_n = inner_n + 1e-10 if inner_n==0 else inner_n
    inner_n = inner_n - 1e-10 if inner_n==1 else inner_n
    inner_d = inner_d + 1e-10 if inner_d==0 else inner_d
    inner_d = inner_d - 1e-10 if inner_d==1 else inner_d
    numer = can2_post + mvnorm.logpdf(x=can_z1, mean=can_z2, cov=varS1) + np.log(inner_n)
    denom = cur_post + mvnorm.logpdf(x=can_z1, mean=z, cov=varS1) + np.log(inner_d)
    R2 = numer - denom
    return R2

def adapt_covariance(i,adapt,s_p,all_Z,epsilon,q):
    ## Adapt the proposal distribution covariance matrix
    if (0 < i) & (i % adapt is 0):
        varS1 = s_p*np.cov(all_Z[range(i+1)].transpose()) + s_p*epsilon*np.diag(np.ones(q))
    else:
        varS1=varS1
    return varS1

def update_taub(d_g,gamma,c_g,L):
    ## Update tau_b, the background model precision
    rate = d_g + 0.5*np.inner(gamma, gamma)
    tau_b = np.random.gamma(shape=(c_g + 0.5*L), scale=1/rate)
    return tau_b

def update_tauy(y,BG,Calc,var_scale,d_y,c_y,n):
    ## Update tau_y, the model precision
    err = (y-BG-Calc.Calculate())/np.sqrt(var_scale)
    rate = d_y + 0.5*np.inner(err, err)
    tau_y = np.random.gamma(shape=(c_y + 0.5*n), scale=1/rate)
    return tau_y

def print_update(curr_keep,update,n_keep,accept_S1,attempt_S1,accept_S2,attempt_S2,accept_rate_S1,accept_rate_S2):
    # Print an update if necessary
    print("Collected %d of %d samples" % (curr_keep, n_keep))
    print('  %03.2f acceptance rate for Stage 1 (%d attempts)' % (accept_S1/attempt_S1, attempt_S1))
    if attempt_S2 > 0:
        accept_rate_S2 = accept_S2/attempt_S2
    else:
        accept_rate_S2 = 0
    print('  %03.2f acceptance rate for Stage 2 (%d attempts)' % (accept_rate_S2, attempt_S2))
    accept_rate_S1[curr_keep//update] = accept_S1/attempt_S1
    accept_rate_S2[curr_keep//update] = accept_rate_S2
    return accept_rate_S1,accept_rate_S2
 
def traceplots(plot,q,keep_params,curr_keep,paramList,n_keep,update):    
    # Produce trace plots
    if plot is True:
        plt.figure(1, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.4)
        for index in range(q):
            plt.subplot(2, np.ceil(q/2.0), index+1)
            plt.plot(keep_params[range(curr_keep), index], 'k')
            plt.xlabel("Iteration")
            plt.ylabel(paramList[index])
        if ((n_keep-curr_keep) < update):
            plt.savefig('DRAM_Trace.png')
            plt.pause(0.1)
    return;

## MCMC function
def nlDRAM(GPXfile, paramList, variables, init_z, lower, upper, initCov=None, y=None, x=None, L=20, shrinkage=0.2, s_p=(2.4**2), epsilon=1e-4, m0=0, sd0=1, c_y=0.1, d_y=0.1, c_g=0.1, d_g=0.1, c_b=0.1, d_b=0.1, adapt=20, thin=1, iters=5000, burn=2000, update=500, plot=True, fix=False):
    # Args:
    #   GPXfile - string, filepath for the GPX file underlying the current data
    #   paramList - (q x 1) list of GSASII parameter names in the same order as
    #               the upper and lower limits being provided
    #   init_z - (q x 1) vector of initial values in the z-space
    #   lower - (q x 1) vector of lower bounds for the parameter values
    #   upper - (q x 1) vector of upper bounds for the parameter values
    #   initCov - (q x q) matrix to be used as the covariance matrix for the
    #             proposal distribution, default value is None.  If there is no
    #             matrix specified, the function will use a diagonal matrix with
    #             0.05 on the diagonal
    #   y - (n x 1) vector of intensities, default value is None. If no values
    #       are specified, the function uses the values from the provided GPX
    #       file
    #   x - (n x 1) vector of angles (2*theta), default value is None. If no
    #       values are specified, the function uses the values from the provided
    #       GPX file
    #   L - scalar, number of B-spline basis functions to model the background
    #       intensity, default is 20
    #   shrinkage - scalar, governs covariance change between proposal stages,
    #               default is 0.2
    #   s_p - scalar, scaling parameter for the adaptive covariance, default is
    #         set to (2.4**2)/d as in Gelman (1995), where d is the dimension of
    #         the parameter space
    #   epsilon - scalar, ridge constant to prevent singularity of the adaptive
    #             covariance, default is 0.0001
    #   m0, sd0 - scalars, govern the prior distribution on the latent Zs,
    #             default is a standard normal distribution
    #   c_y, d_y - scalars, govern the prior Gamma distribution for the error
    #              variance, default value is 0.1 for both
    #   c_g, d_g - scalars, govern the prior Gamma distribution for the error
    #              in the prior distribution for the basis function loadings,
    #   c_b, d_b - scalars, govern the prior Gamma distribution for scale of
    #              the proportional contribution to the error variance, default
    #              value is 0.1 for both
    #   adapt - scalar, controls the adaptation period, default is 20
    #   thin - scalar, degree of thinning, default is 1
    #   iters - scalar, number of total iterations to run, default is 5000
    #   burn - scalar, number of samples to consider as burn-in, default is 2000
    #   update - scalar, period between updates printed to the console, default
    #            is 500
    #   plot - boolean, indicator for whether or not to create trace plots as
    #          the sampler progresses, default is True
    #
    # Returns: 5-tuple containing the posterior samples for the parameters and
    #          the model timing, tuple entries are
    #            1 - (nSamples x q) matrix of posterior samples for the mean
    #                process parameters of interest
    #            2 - (nSamples x 1) vector of posterior samples for the constant
    #                factor on the smoothed observations in the proportional
    #                variance
    #            3 - (nSamples x 1) vector of posterior samples for the overall
    #                variance / temperature
    #            4 - (nSamples x L) matrix of posterior samples for the basis
    #                function loadings modeling the background intensity
    #            5 - scalar, number of minutes the sampler took to complete

    # Initialize the calculator based on the provided GPX file
    Calc = gsas.Calculator(GPXfile=GPXfile)
    Calc._varyList = variables
    
    # Set the scaling parameter
    s_p = ((2.4**2)/len(paramList))
    
    # Assign the intensity vector (y) and 2-theta angles (x) from the GPX file if no values are provided
    x,y = diffraction_file_data(x=x,y=y,Calc=Calc)
    
    # Calculate a B-spline basis for the range of x
    B = calculate_bsplinebasis(x=x,L=L)

    # Save dimensions
    n = len(y)       # Number of observations
    q = len(init_z)  # Number of parameters of interest

    # Smooth the observed Ys on the Xs with lowess
    y_sm = smooth_ydata(x=x,y=y)

    # Get indices of parameters to refine, even if they are "fixed" by bounds
    useInd = [np.asscalar(np.where(np.array(Calc._varyList)==par)[0]) for par in paramList]
    if (any(np.array(Calc._varyList)[useInd] != paramList)):
        raise ValueError("Parameter list specification is not valid.")

    # Make sure initial z values are given for every parameter in paramList
    if len(paramList)!=len(init_z):
        raise ValueError("Initial value specification for Z is not valid.")

    # Initialize parameter values
    z = np.array(init_z, copy=True)                     # Latent process
    params = z2par(z=init_z, lower=lower, upper=upper)  # Parameters of interest
    tau_y = 1                                           # Error variance for Y
    b = 1                                               # Contribution of y_sm
    gamma = np.ones(L)                                  # Loadings
    tau_b = 1                                           # Variance for loadings
    BG = np.matmul(B, gamma)                            # Background intensity
    Calc.UpdateParameters(dict(zip(paramList, params))) # Calculator
    var_scale = b*y_sm + 1                            # Scale for y_sm/tau_y

    # Initialize covariance for proposal distribution
    varS1 = initialize_cov(initCov=initCov, q=q)

    # Set up counters for the parameters of interest
    attempt_S1 = attempt_S2 = accept_S1 = accept_S2 = 0  # Attempts / acceptances counters

    # Calculate the number of thinned samples to keep
    n_keep = np.floor_divide(iters-burn-1, thin) + 1
    curr_keep = 0

    # Initialize output objects
    all_Z,keep_params,keep_gamma,keep_b,keep_tau_y,keep_tau_b,accept_rate_S1,accept_rate_S2 = initialize_output(iters=iters,q=q,n_keep=n_keep,L=L,update=update)
    
    tick = timer()
    for i in range(iters):
        
        ## Update basis function loadings and then background values
        gamma,BG = update_background(B,var_scale,tau_y,tau_b,L,Calc,y)

        ## Update mean process parameters using 2-stage DRAM
        attempt_S1 += 1
        # Stage 1:
        can_z1 = np.random.multivariate_normal(mean=z, cov=varS1)
        can1_post = logf(y=y, x=x, BG=BG, Calc=Calc, paramList=paramList, z=can_z1, lower=lower, upper=upper, scale=var_scale, tau_y=tau_y, m0=m0, sd0=sd0)
        cur_post = logf(y=y, x=x, BG=BG, Calc=Calc, paramList=paramList, z=z, lower=lower, upper=upper, scale=var_scale, tau_y=tau_y, m0=m0, sd0=sd0)
        R1 = can1_post - cur_post
        if (np.log(np.random.uniform()) < R1) & (np.sum(np.abs(can_z1) > 3)==0):
            accept_S1 += 1
            z = np.array(can_z1, copy=True)                # Update latent
            params = z2par(z=z, lower=lower, upper=upper)  # Update params
            Calc.UpdateParameters(dict(zip(paramList, params)))
        else:
            # Stage 2:
            attempt_S2 += 1
            # Propose the candidate
            can_z2 = np.random.multivariate_normal(mean=z, cov=shrinkage*varS1)
            # Accept or reject the candidate
            if np.sum(np.abs(can_z2) > 3)==0: # Ensures significant distance away from the bounds
                can2_post = logf(y=y, x=x, BG=BG, Calc=Calc, paramList=paramList, z=can_z2, lower=lower, upper=upper, scale=var_scale, tau_y=tau_y, m0=m0, sd0=sd0)
                # Calculate the acceptance probability
                R2 = stage2_acceptprob(can1_post=can1_post,can2_post=can2_post,cur_post=cur_post,can_z1=can_z1,can_z2=can_z2,z=z,varS1=varS1)
                if np.log(np.random.uniform()) < R2:
                    accept_S2 += 1
                    z = np.array(can_z2, copy=True)                # Update latent
                    params = z2par(z=z, lower=lower, upper=upper)  # Update params
                    Calc.UpdateParameters(dict(zip(paramList, params)))
                del can_z2, can2_post, R2
            else:
                del can_z2
        del can_z1, can1_post, cur_post, R1
        all_Z[i] = z

        ## Adapt the proposal distribution covariance matrix
        varS1 = adapt_covariance(i=i,adapt=adapt,s_p=s_p,all_Z=all_Z,epsilon=epsilon,q=q)
            
        ## Update tau_b
        tau_b = update_taub(d_g=d_g,gamma=gamma,c_g=c_g,L=L)

        ## Update tau_y
        tau_y = update_tauy(y=y,BG=BG,Calc=Calc,var_scale=var_scale,d_y=d_y,c_y=c_y,n=n)

        ## Keep track of everything
        if i >= burn:
            # Store posterior draws if appropriate
            if (i-burn) % thin is 0:
                keep_params[curr_keep] = params
                keep_gamma[curr_keep] = gamma
                keep_b[curr_keep] = b
                keep_tau_y[curr_keep] = tau_y
                keep_tau_b[curr_keep] = tau_b
                curr_keep += 1
            
            if curr_keep % update is 0:
                # Print an update if necessary
                accept_rate_S1,accpet_rate_S2 = print_update(curr_keep=curr_keep,update=update,n_keep=n_keep,accept_S1=accept_S1,attempt_S1=attempt_S1,accept_S2=accept_S2,attempt_S2=attempt_S2,accept_rate_S1=accept_rate_S1,accept_rate_S2=accept_rate_S2)
                # Produce trace plots
                traceplots(plot,q,keep_params,curr_keep,paramList,n_keep,update)
                
    tock = timer()

    # Gather output into a tuple
    #output = (keep_params, varS1, keep_b, 1.0/keep_tau_y, keep_gamma, (tock-tick)/60)
    output = (keep_params, curr_keep, varS1, keep_b, 1.0/keep_tau_y, keep_gamma, (tock-tick)/60,accept_rate_S1,accept_rate_S2)
    return output

