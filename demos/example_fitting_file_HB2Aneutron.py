from __future__ import division
# Import modules
import numpy as np
import sys
import os

# Add GSAS-II filepath to the Python path
sys.path.append('/Users/Rachel/gsas2full/GSASII/bindist')
sys.path.append('/Users/Rachel/gsas2full/GSASII')

# Source the MCMC and GSAS calculator functions
from QUAD import dram  # noqa: E402
from QUAD import gsas_tools as gsas  # noqa: E402
# =============================================================================
# INITIALIZE INPUTS
# =============================================================================

# Identify GPX file location
gpxFile = 'NIST_Si_HB2A_Ge115.gpx'

# Initialize the calculator
Calc = gsas.Calculator(GPXfile=gpxFile)

# Set up the parameter list
# Prints which parameters are currently being refined in the GPX file
print('Parameters from gpx file')
print(Calc._varyList)

# Delete parameters from the paramList to not be refined
paramList = [key for key in Calc._varyList if (key != ':0:DisplaceX' and
                                               key != '0::AUiso:0' and
                                               'Back;' not in key)]

# Add parameters you do want to refine to the paramList
# paramList.append('parameter_name')

# Set the parameter list equal to the list of parameters to vary (i.e. refine)
variables = np.copy(paramList)

# Pull the starting values for each parameter out of the gpx file
start = [Calc._parmDict[param] for param in paramList]

# Check that the parameter list is correct
print('Final parameters:')
print(paramList)

# Check the parameter starting values from GSAS-II
print('Starting Values:')
print(start)

# Define lower and upper bounds on the parameters. Best to define individually.
lower = np.array([0.0, 0.0, 1.53, 0.0, 200.0, -400.0, 125.0, -0.10, 1000.0])
upper = np.array([1200.0, 1.5, 1.55, 0.5, 300.0, -250.0, 225.0, 0.10, 2000.0])

# Set initial covariance for the proposal distribution of z.
# Choose from the following options:

# (1)Diagonal matrix
# init_cov = np.diag(0.002*np.ones(len(start)))

# (2)Estimated covariance marix from sensitivity matrix
eco = dram.estimatecovariance(paramList=paramList, start=start,
                              x=None, y=None, Calc=Calc, upper=upper,
                              lower=lower, L=20, delta=1e-3)
init_cov = eco["cov"]
# Adjust to prevent singularity
# init_cov = 0.95*eco["cov"] + 0.05*np.identity(len(start))

# Set the chain length parameters
samples = 100
burn = 0
update = 50

# Set remaining input values for DRAM
shrinkage = 0.2
adapt = 100     # adaption interval

# SetupOutput Folder
filename = os.path.split(gpxFile)
foldername = filename[1].split('.')
path = './results/' + foldername[0] + '_test1'    # + other naming information
os.mkdir(path)

# =============================================================================
# RUN DRAM
# =============================================================================

print('DRAM with iters={} and burn={}'.format(samples, burn))
results = dram.sample(GPXfile=gpxFile,
                      paramList=paramList,
                      variables=variables,
                      start=start,
                      lower=lower,
                      upper=upper,
                      path=path,
                      initCov=init_cov,
                      shrinkage=shrinkage,
                      adapt=adapt,
                      iters=samples,
                      burn=burn,
                      update=update,
                      plot=True)

# =============================================================================
# PROCESS RESULTS
# =============================================================================

# Print run information to console: compare initial Rietveld values and QUAD
# mean estimates for process parameters, print total run time, plot and save
# final histograms of posterior samples.
dram.run_summary(results=results, start=start, paramList=paramList, path=path)

# Save DRAM output results and input values of prior bounds, starting values,
# and initial covariance as individual files.
dram.save_results(results=results, start=start, lower=lower, upper=upper,
                  paramList=paramList, init_cov=init_cov, path=path) 

# Save DRAM fitting input values
f = open(path + "/DRAM_inputs.txt", "w+")
f.write("shrinkage: %05.2f\r\n adaption interval: %10.0f\r\n number of samples:"
        "%10.0f\r\n burn-in: %10.0f\r\n" % (shrinkage, adapt, samples, burn))
f.close()