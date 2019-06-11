from __future__ import division
# Import modules
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# Add GSAS-II filepath to the Python path
sys.path.append('/Users/Rachel/g2conda/GSASII/bindist')
sys.path.append('/Users/Rachel/g2conda/GSASII')

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
# paramList.append('parameter name')

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

# Set initial z-space values
init_z = norm.ppf((start-lower)/(upper-lower))

# Set initial covariance for the proposal distribution of z.
# Choose from the following options:

# (1)Diagonal matrix
# init_cov = np.diag(0.002*np.ones(len(init_z)))

# (2)Estimated covariance marix from sensitivity matrix
eco = dram.estimatecovariance(paramList=paramList, start=start, init_z=init_z,
                              x=None, y=None, Calc=Calc, upper=upper,
                              lower=lower, L=20, delta=1e-3)
init_cov = eco["cov"]
# Adjust to prevent singularity
# init_cov = 0.95*eco["cov"] + 0.05*np.identity(len(init_z))

# Set the chain length parameters
samples = 100
burn = 0
update = 50

# Set remaining input values for DRAM
shrinkage = 0.2
adapt = 100     # adaption interval

# =============================================================================
# RUN DRAM
# =============================================================================

print('DRAM with iters={} and burn={}'.format(samples, burn))
results = dram.sample(GPXfile=gpxFile,
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

# =============================================================================
# PROCESS RESULTS
# =============================================================================

# SetupOutput Folder
filename = os.path.split(gpxFile)
foldername = filename[1].split('.')
path = './results/' + foldername[0] + '_test2'    # + other naming information
os.mkdir(path)

# Calculate mean parameter estimates from the posterior and compare to
# GSAS-II fit
mins = results["run_time"]
params = results["param_samples"]
post_param_mean = np.mean(params, axis=0)

# Print true versus estimated values for the mean process parameters
print('Mean parameter estimates:')
for q in range(len(init_z)):
    print(paramList[q] + ' Rietveld: %03.4f, QUAD: %03.4f' % (start[q],
          post_param_mean[q]))

# Print run time
print("Model Time: %03.2f (DRAM)" % (mins))

# Plot parameter posterior distributions
plt.figure(1, figsize=(25, 12))
plt.subplots_adjust(wspace=0.9)
for index in range(0, len(init_z)):
    plt.subplot(2, np.ceil((q+1)/2.0), index+1)
    sns.distplot(params[:, index])
    sns.set_context("talk")
    plt.xlabel(paramList[index])
    plt.ylabel('Probability')
plt.savefig(path + '/PosteriorDensities')

# Save results to output folder
np.savetxt(path + '/parameter samples', params)
np.savetxt(path + '/final proposal covariance', results["final_covariance"])
np.savetxt(path + '/initial proposal covariance', init_cov)
np.savetxt(path + '/gamma', results["gamma_samples"])
np.savetxt(path + '/sig2', results["model_variance"])
np.savetxt(path + '/run time', np.array([mins]))
np.savetxt(path + '/posterior parameter mean', post_param_mean)
np.savetxt(path + '/parameter starting values', start)
np.savetxt(path + '/lower bounds', lower)
np.savetxt(path + '/upper bounds', upper)
np.savetxt(path + '/Stage 1 acceptance', results["stage1_accept"])
np.savetxt(path + '/Stage 2 acceptance', results["stage2_accept"])

# Save DRAM fitting input values
f = open(path + "/DRAM_inputs.txt", "w+")
f.write("shrinkage: %05.2f\r\n adaption interval: %10.0f\r\n number ofsamples:"
        "%10.0f\r\n burn-in: %10.0f\r\n" % (shrinkage, adapt, samples, burn))
f.close()

# Save list of refined parameter names to text file
with open(path + '/parameter list.txt', 'w') as outfile:
    for item in paramList:
        outfile.write("%s\n" % item)
