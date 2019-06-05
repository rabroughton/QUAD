# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:44:16 2019

@author: Rachel
"""
from __future__ import division  # Use "real" division

# Add another filepath to the Python path
import sys
import os

#sys.path.insert(1, r'C:\Users\Rachel\g2conda\GSASII\bindist')
#sys.path.insert(2, r'C:\Users\Rachel\g2conda\GSASII')

sys.path.append('/Users/Rachel/g2conda/GSASII/bindist')
sys.path.append('/Users/Rachel/g2conda/GSASII')

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from QUAD import gsas_tools as gsas  # GSAS calculator 
# Source the MCMC function
from QUAD import dram
import pickle

# Identify GPX file location and set up the parameter list
gpxFile = '11BM_Si_640C_Refinement1_2thetaLims.gpx'

# Initialize the calculator
Calc = gsas.Calculator(GPXfile=gpxFile)
#Prints which parameters are currently being refined in the GPX file
print('Parameters from gpx file')
print(Calc._varyList)

# Delete parameters you do not want to refine from the paramList
paramList = [key for key in Calc._varyList if (key!=':0:DisplaceX' and key!='0::AUiso:0' and 'Back;' not in key)]

# Add parameters you do want to refine to the paramList
#paramList.append('parameter name')

# Set the desired parameter list to equal the list of parameters to vary (i.e. refine)
variables = np.copy(paramList)

# Pull the starting values for each parameter out of the gpx file
start = [Calc._parmDict[param] for param in paramList] 

# Check that the parameter list is correct
print('Final parameters:')
print (paramList)

# Set up the lower and upper bounds on the parameters. Best to define individually. 
lower = np.multiply((1-0.04),start)
upper = np.multiply((1+0.05),start)
#lower = np.array([1000.0, 0.0, 0.0, 0.0, -50.0, 0, -10, -0.1, 500])
#upper = np.array([2000.0, 20.0, 1.0, 0.5, -19.0, 100, 0, 0.1, 1500])

# Set the chain length parameters
samples = 100
burn = 10
update = 50

# Set initial z-space values
init_z = norm.ppf((start-lower)/(upper-lower))

# Set initial covariance for the proposal distribution of z, choose from the following options

## diagonal matrix
#init_cov = np.diag(0.0002*np.ones(len(init_z)))

## estimated covariance marix from sensitivity matrix
eco = dram.estimatecovariance(paramList=paramList,start=start,init_z=init_z,x=None,y=None,Calc=Calc,
                              upper=upper,lower=lower,L=20,delta=1e-3) 
init_cov = eco["cov"]
#init_cov = 0.95*eco["cov"] + 0.05*np.identity(len(init_z)) #adjust to prevent singularity

# Set remaining input values for DRAM
shrinkage = 0.2
adapt = 100 #adaption interval

################################ RUN DRAM #####################################
print('DRAM with iters={} and burn={}'.format(samples, burn))
(params, curr_keep, varS1, sig2, 
 gamma, mins, accept_rate1, accept_rate2) = dram.nlDRAM(GPXfile=gpxFile, 
                                                        paramList=paramList, 
                                                        variables=variables, 
                                                        init_z=init_z, 
                                                        lower=lower, 
                                                        upper=upper, 
                                                        initCov=init_cov, 
                                                        shrinkage=shrinkage, 
                                                        adapt=adapt, 
                                                        iters=samples, 
                                                        burn=burn, 
                                                        update=update, 
                                                        plot=True)
############################ PROCESS RESULTS #####################################
#SetupOutput Folder
filename = os.path.split(gpxFile)
foldername = filename[1].split('.')
path = './results/' + foldername[0] + 'test' # + other identifying information if desired

os.mkdir(path)

# Calculate mean parameter estimates from the posterior and compare to GSAS-II fit
post_param_mean = np.mean(params, axis=0)

# Print out true versus estimated values for the mean process parameters
print ('Mean parameter estimates:')
for q in range(len(init_z)):
    print (paramList[q] + ' Rietveld: %03.4f, QUAD: %03.4f' % (start[q], post_param_mean[q]))

# Print out run time
print ("Model Time: %03.2f (DRAM)" % (mins))

# Plot parameter posterior distributions
plt.figure(1, figsize=(25, 12))
plt.subplots_adjust(wspace=0.9)
for index in range(0,len(init_z)):
    plt.subplot(2, np.ceil((q+1)/2.0), index+1)
    sns.distplot(params[:, index])
    sns.set_context("talk")
    plt.xlabel(paramList[index])
    plt.ylabel('Probability')

plt.savefig(path + '/Posterior Densities')

# Calculate 95% credible interval 
#np.sort(params,axis=0)

# Save results to output folder
np.savetxt(path + '/parameter samples', params)
np.savetxt(path + '/final proposal covariance', varS1)
np.savetxt(path + '/initial proposal covariance', init_cov)
np.savetxt(path + '/gamma', gamma)
np.savetxt(path + '/sig2', sig2)
np.savetxt(path + '/run time', np.array([mins]))
np.savetxt(path + '/posterior parameter mean', post_param_mean)
np.savetxt(path + '/parameter starting values', start)
np.savetxt(path + '/lower bounds', lower)
np.savetxt(path + '/upper bounds', upper)
np.savetxt(path + '/Stage 1 acceptance', accept_rate1)
np.savetxt(path + '/Stage 2 acceptance', accept_rate2)

# Save DRAM fitting input values
f= open(path + "\DRAM_inputs.txt","w+")
f.write("shrinkage: %05.2f\r\n adaption interval: %10.0f\r\n number of samples: %10.0f\r\n burn-in: %10.0f\r\n" % (shrinkage, adapt, samples, burn))
f.close()

with open(path + '/parameter_list', 'wb') as fp:
    pickle.dump(paramList, fp)