# -*- coding: utf-8 -*-
'''
Quantitative Uncertainty Analysis for Diffraction (QUAD)
========================================================
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
'''
# Import specific functions
from __future__ import division  # Use "real" division
from .utilities import gsas_install_display
# Import modules
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer  # Timing function
# Normal distribution
from scipy.stats import norm
# Multivariate normal distribution
from scipy.stats import multivariate_normal as mvnorm
import statsmodels.api as sm  # Library for lowess smoother
lowess = sm.nonparametric.lowess  # Lowess smoothing function

try:  # Bspline function
    from .bspline import Bspline
except ModuleNotFoundError:
    from bspline import Bspline

try:  # Bspline helper function
    from .splinelab import augknt
except ModuleNotFoundError:
    from bspline.splinelab import augknt

try:
    from .gsas_tools import Calculator as gsas_calculator
except ModuleNotFoundError:
    gsas_install_display(module='gsas_tools')


def estimatecovariance(paramList, start, init_z, Calc, upper, lower,
                       x=None, y=None, L=20, delta=1e-3):
    '''
    Estimate the covariance of the initial parameter values to initialize the
    proposal covarianc for DRAM. This is done by first calculating the
    sensitivity matrix, :math:`\\chi`, through finite differences. Next, the
    Fisher Information matrix is calulated as :math:`\\chi^T\\chi`. Finally,
    the estimated covariance matrix is calculated as :math:`s^2*\\chi^T\\chi`,
    where s is the standard deviation estimate.

    Args:
        * **paramList** (:py:class:`list`): List of parameter names for
          refinement - size (q).
        * **start** (:py:class:`list`): List of initial parameter values
          in parameter space - size (q).
        * **init_z** (:class:`~numpy.ndarray`): Vector of initial parameter
          values in z-space - size (q,).
        * **Calc** (:class:`.Calculator`): calculator operator that interacts
          with the designated GPX file by referencing GSAS-II libraries.
        * **upper** (:class:`~numpy.ndarray`): Vector of upper limits on a
          uniform prior distribution in the parameter space - size (q,).
        * **lower** (:class:`~numpy.ndarray`): Vector of lower limits on a
          uniform prior distribution in the parameter space - size (q,).
        * **x** (:class:`~numpy.ndarray`): Vector of 2-theta values
          - size (n,). Will be created from the GPX file if not
          user-defined.
        * **y** (:class:`~numpy.ndarray`): Vector of diffraction pattern
          intensities - size (n,). Created from the GPX file if not
          user-defined.
        * **L** (:py:class:`int`): Number of cubic B-spline basis functions to
          model the background intensity. Default is 20.
        * **delta** (:py:class:`float`): Fraction by which the parameters are
          perturbed with respect to the initial values for finite difference
          calculation.

    Returns:
        * (:py:class:`dict`): Dictionary containing the estimated covariance
          matrix, estimated variance, and the eigenvalues and eigenvectors of
          the covariance matrix. The eigenvalues can be examined to determine
          potentially non-indentifiable parameters by locating the eigenvalues
          significantly close to zero in value.
    '''
    # Initialize inputs
    Calc.UpdateParameters(dict(zip(paramList, start)))
    f0 = Calc.Calculate()
    x, y = diffraction_file_data(x=x, y=y, Calc=Calc)
    n = np.shape(y)[0]
    p = np.shape(init_z)[0]
    gamma = np.ones(L)
    BG = np.matmul(calculate_bsplinebasis(x=x, L=L), gamma)

    # Finite difference calculation
    def finitediff(ii, init_z, paramList, f0, delta):
        q_star = np.copy(init_z)
        q_star[ii] = np.copy(init_z[ii])*(1+delta)
        params_perturb = z2par(q_star, lower, upper)
        Calc.UpdateParameters(dict(zip(paramList, params_perturb)))
        f1 = Calc.Calculate()
        diff = (f1-f0)/(delta*init_z[ii])
        return diff
    # Calculate sensitivity matrix
    sensitivity = np.zeros([n, p])
    for jj in range(0, np.shape(init_z)[0]):
        sensitivity[:, jj] = finitediff(jj, init_z, paramList, f0, delta)
    # Calculate fisher information matrix
    fisher = np.dot(sensitivity.transpose(), sensitivity)
    # Calcualte estimate for s^2
    R = y-BG-f0  # residuals
    s2 = (1./(n-p))*np.dot(R, R.transpose())
    # Calculate covariance matrix
    covariance = s2*np.linalg.inv(fisher)
    evals = np.linalg.eig(covariance)
    print("Covariance eignvalues:\n{}".format(evals[0]))
    return dict(cov=covariance, s2=s2, evals=evals)


def z2par(z, lower, upper, grad=False):
    '''
    Transform between the bounded parameter space and continuous z-space.

    Args:
        * **z** (:class:`~numpy.ndarray`): Array of parameter values in
          z-space - size (q,).
        * **lower** (:class:`~numpy.ndarray`): Vector of lower limits on a
          uniform prior distribution in the parameter space - size (q,).
        * **upper** (:class:`~numpy.ndarray`): Vector of upper limits on a
          uniform prior distribution in the parameter space - size (q,).
        * **grad** (:py:class:`bool`): If False (default), converts from the
          z-space to the parameter space. If True, function converts
          WHAT DOES THIS DO???

    Returns:
        * **par** (:class:`~numpy.ndarray`): Array of parameter values in
          parameter space from the given z values- size (q,).
    '''
    if (grad):
        d = (upper-lower)*norm.pdf(z)
        return d
    else:
        par = lower + (upper-lower)*norm.cdf(z)
        # Avoid parameters approaching the boundaries
        par[np.array([par[j] == upper[j] for j in range(len(par))])] -= 1e-10
        par[np.array([par[j] == lower[j] for j in range(len(par))])] += 1e-10
        return par


def prior_loglike(par, m0, sd0):
    '''
    Calculate the log likelihood of the parameters in z-space based on the
    prior distributions. The prior is a uniform distribution in parameter space
    which corresponds to the product of normal distributions for each value in
    z-space.

    Args:
        * **par** (:class:`~numpy.ndarray`): Array of parameter values in
          z-space - size (q,).
        * **m0** (:py:class:`float`): Mean of prior normal distribution on
          z. Default is 0.
        * **sd0** (:py:class:`float`): Standard deviation of prior normal
          distribution on z. Default is 1.

    Returns:
        * **prior** (:py:class:`float`): Value of prior distribution given
          current z-values.
    '''
    return -0.5*(sd0**(-2))*np.inner(par-m0, par-m0)


def log_post(y, x, BG, Calc, paramList, z, lower,
             upper, scale, tau_y, m0, sd0):
    '''
    Calculate the log of the posterior probabilities for a set of parameters.
    The posterior is defined as a log-likelihood of the product of normal
    distributions multiplied by the prior. For definition of the prior,
    see :meth:`~prior_loglike`.

    Args:
        * **y** (:class:`~numpy.ndarray`): Vector of diffraction pattern
          intensities - size (n,).
        * **x** (:class:`~numpy.ndarray`): Vector of 2-theta values - size
          (n,).
        * **BG** (:class:`~numpy.ndarray`): Vector of background intensity
          values - size (n,).
        * **Calc** (:class:`~.Calculator`): calculator operator that interacts
          with the designated GPX file by referencing GSAS-II libraries.
        * **paramList** (:py:class:`list`): List of parameter names for
          refinement - size (q,).
        * **z** (:class:`~numpy.ndarray`): Current parameter values in
          z-space - size (q,).
        * **lower** (:class:`~numpy.ndarray`): Vector of lower limits on a
          uniform prior distribution in the parameter space - size (q,).
        * **upper** (:class:`~numpy.ndarray`): Vector of upper limits on a
          uniform prior distribution in the parameter space - size (q,).
        * **scale** (:class:`~numpy.ndarray`): Vector that scales with the
          intensity of data, heteroscedastic. See function
          :meth:`~initialize_intensity_weight`
        * **tau_y** (:py:class:`float`): Model precision. Default initial
          valus is 1.
        * **m0** (:py:class:`float`): Mean of prior normal distribution on z.
          Default is 0.
        * **sd0** (:py:class:`float`): Standard deviation of prior normal
          distribution on z. Default is 1.

    Returns:
        * **posterior** (:py:class:`float`): Value of the prior times
          likelihood given current z-space candidate values.
    '''
    params = z2par(z=z, lower=lower, upper=upper)
    # Update the calculator to reflect the current parameter estimates
    Calc.UpdateParameters(dict(zip(paramList, params)))
    # Calculate residuals
    R = y-BG-Calc.Calculate()
    # Calculate weighted sum of squares error
    S = np.inner(R/np.sqrt(scale), R/np.sqrt(scale))
    # Add log-prior and log-likelihood values
    logpost = 0.5*tau_y*S - prior_loglike(par=z, m0=m0, sd0=sd0)
    return (-1)*logpost


def calculate_bsplinebasis(x, L):
    '''
    Calculate a B-spline basis for the 2-theta values.

    Args:
        * **x** (:class:`~numpy.ndarray`): Vector of 2-theta values
          - size (n,).
        * **L** (:py:class:`int`): Number of cubic B-spline basis functions to
          model the background intensity. Default is 20.

    Returns:
        * **B** (:class:`~numpy.ndarray`): B-spline basis for data
          - size (n, L)
    '''
    # Calculate a B-spline basis for the range of x
    unique_knots = np.percentile(a=x, q=np.linspace(0, 100, num=(L - 2)))
    knots = augknt(unique_knots, 3)
    objB = Bspline(knots, order=3)
    B = objB.collmat(x)
    return B


def diffraction_file_data(x, y, Calc):
    '''
    Extract the intensity values (y) and 2-theta angles (x) from the
    underlying GPX file or set the user-defined data values.

    Args:
        * **x** (:class:`~numpy.ndarray`): Predefined vector of 2-theta
          values, if it exists.
        * **y** (:class:`~numpy.ndarray`): Predefined vector of diffraction
          pattern intensities, if it exists.
        * **Calc** (:class:`~.Calculator`): calculator operator that interacts
          with the designated GPX file by referencing GSAS-II libraries.

    Returns:
        * 2-tuple containing the diffraction data. Tuple entries are

        #. **x** (:class:`~numpy.ndarray`): Vector of 2-theta values
           - size (n,).
        #. **y** (:class:`~numpy.ndarray`): Vector of diffraction pattern
           intensities - size (n,).
    '''
    # Assign the intensity vector (y) from the GPX file, if necessary
    if y is None:
        Index = np.where((Calc._tth > Calc._lowerLimit) &
                         (Calc._tth < Calc._upperLimit) == True)  # noqa: E712
        y = np.array(Calc._Histograms[list(Calc._Histograms.keys())[0]]['Data'][1][Index], copy=True)  # noqa: E501

    # Assign the grid of angles (x) from the GPX file, if no values are
    # provided.
    if x is None:
        x = np.array(Calc._tthsample, copy=True)
        # If values are provided externally, overwrite the _tthsample parameter
    else:
        # Update Calc internally for remainder of script
        Calc._tthsample = np.array(x, copy=True)
    return x, y


def smooth_ydata(x, y):
    '''
    Smooth diffraction data intensities at 2-theta values with
    Locally Weighted Scatterplot Smoothing (lowess) function from statsmodels_.

    .. _statsmodels: https://www.statsmodels.org/dev/generated/statsmodels
       .nonparametric.smoothers_lowess.lowess.html

    Args:
        * **y** (:class:`~numpy.ndarray`): Vector of diffraction pattern
          intensities - size (n,).
        * **x** (:class:`~numpy.ndarray`): Vector of 2-theta values from
          diffraction pattern- size (n,).

    Returns:
        * **y_sm** (:class:`~numpy.ndarray`): Vector of smoothed intensity
          data - size (n,).
    '''
    # Smooth the observed Ys on the Xs, patch for negative or 0 values
    y_sm = lowess(endog=y.reshape(y.size,), exog=x.reshape(x.size,),
                  frac=6.0/len(x), return_sorted=False)
    y_sm = np.array([max(0, sm) for sm in y_sm])
    return y_sm


def initialize_cov(initCov, q):
    '''
    If not previously defined, initialize the covariance for the proposal
    distribution.

    Args:
        * **initCov** (:class:`~numpy.ndarray`): Pre-defined initial covariance
          matrix in z-space - size (q, q).
        * **q** (:py:class:`int`): Number of parameters.

    Returns:
        * **varS1** (:class:`~numpy.ndarray`): Covariance matrix in z-space
          - size (q, q).
    '''
    if initCov is None:
        varS1 = np.diag(0.05*np.ones(q))
    elif initCov.shape == (q, q):
        varS1 = initCov
    else:
        raise ValueError("Specification for initCov is not valid."
                         + "  Please provide a (%d x %d) matrix." % (q, q))
    return varS1


def _initialize_output(iters, q, n_keep, L, update):
    # Initialize output objects
    all_Z = np.zeros((iters, q))
    keep_params = np.zeros((n_keep, q))
    keep_gamma = np.zeros((n_keep, L))
    keep_b = np.zeros(n_keep)
    keep_tau_y = np.zeros(n_keep)
    keep_tau_b = np.zeros(n_keep)
    accept_rate_S1 = np.zeros(n_keep//update)
    accept_rate_S2 = np.zeros(n_keep//update)
    return (all_Z, keep_params, keep_gamma, keep_b,
            keep_tau_y, keep_tau_b, accept_rate_S1, accept_rate_S2)


def update_background(B, var_scale, tau_y, tau_b, L, Calc, y):
    '''
    Update the basis function loadings and then background values.

    Args:
        * **B** (:py:class:`float`): B-spline basis for 2-theta range
          - size (n, L). See :meth:`~calculate_bsplinebasis`.
        * **var_scale** (:class:`~numpy.ndarray`): Vector of scaling factors
          corresponding to intensity data- size (n,).
          See function :meth:`~initialize_intensity_weight`
        * **tau_y** (:py:class:`float`): Model precision. See
          :meth:`~updat_tauy`
        * **tau_b** (:py:class:`float`): Loadings precision for background.
          See :meth:`~update_taub`
        * **L** (:py:class:`int`): Number of cubic B-spline basis functions to
          model the background intensity. Default is 20.
        * **Calc** (:class:`~.Calculator`): calculator operator that interacts
          with the designated GPX file by referencing GSAS-II libraries.
        * **y** (:class:`~numpy.ndarray`): Vector of diffraction pattern
          intensity data - size(nx1). See :meth:`~diffraction_file_data`.

    Returns:
        * 2-tuple containing the updated information for the background fit of
          the data. Tuple entries are

        #. **gamma** (:class:`~numpy.ndarray`): Vector of updated basis
           loadings - size (L,)
        #. **BG** (:class:`~numpy.ndarray`): Vector of updated background
           intensity values - size (n,).
    '''
    BtB = np.matmul(np.transpose(B)/var_scale, B)
    VV = np.linalg.inv(tau_y*BtB + tau_b*np.identity(L))
    err = (y-Calc.Calculate())/var_scale
    MM = np.matmul(VV, tau_y*np.sum(np.transpose(B)*err, axis=1))
    gamma = np.random.multivariate_normal(mean=MM, cov=VV)
    BG = np.matmul(B, gamma)
    return gamma, BG


def stage1_acceptprob(z, varS1, y, x, BG, Calc, paramList, lower, upper,
                      var_scale, tau_y, m0, sd0):
    '''
    Calculate the acceptance probability for Stage 1 of the DRAM algorithm.

    Args:
        * **z** (:class:`~numpy.ndarray`): Vector of current parameter
          values in z-space - size(q,).
        * **varS1** (:class:`~numpy.ndarray`): Current covariance matrix
          - size(q, q).
        * **y** (:class:`~numpy.ndarray`): Vector of diffraction pattern
          intensities - size (n,).
        * **x** (:class:`~numpy.ndarray`): Vector of 2-theta values
          - size (n,).
        * **BG** (:class:`~numpy.ndarray`): Vector of background intensity
          values - size (n,).
        * **Calc** (:class:`~.Calculator`): calculator operator that interacts
          with the designated GPX file by referencing GSAS-II libraries.
        * **paramList** (:py:class:`list`): List of parameter names for
          refinement - size (q,).
        * **lower** (:class:`~numpy.ndarray`): Vector of lower limits on a
          uniform prior distribution in the parameter space - size (q,).
        * **upper** (:class:`~numpy.ndarray`): Vector of upper limits on a
          uniform prior distribution in the parameter space - size (q,).
        * **var_scale** (:class:`~numpy.ndarray`): Vector that scales with the
          intensity of data, heteroscedastic. See function
          :meth:`~initialize_intensity_weight`
        * **tau_y** (:py:class:`float`): Model precision. Default initial
          valus is 1.
        * **m0** (:py:class:`float`): Mean of prior normal distribution on z.
          Default is 0.
        * **sd0** (:py:class:`float`): Standard deviation of prior normal
          distribution on z. Default is 1.

    Returns:
        * 4-tuple containing the stage 1 acceptance probability, candidates,
          and respective log posterior values. Tuple entries are

        #. **can_z1** (:class:`~numpy.ndarray`): Vector of candidate 1
           parameter values in z-space - size(q,). The candidate is
           constructed from a multivariate normal distribution with mean z
           and covariance of varS1. :math:`can\\_z1 = N_q(z,varS1)`.
        #. **can1_post** (:py:class:`float`): Value of the log posterior
           probability for candidate 1 parameter values, can_z1. Computed with
           :meth:`~log_post`.
        #. **cur_post** (:py:class:`float`): Value of the log posterior
           probability for the current parameter values, z. Computed with
           :meth:`~log_post`.
        #. **R1** (:py:class:`float`): Log of Stage 1 acceptance probability.
    '''
    # Draw a random candidate in z-space from a multivariate normal
    can_z1 = np.random.multivariate_normal(mean=z, cov=varS1)
    # Calculate the posterior probability of the candidate
    can1_post = log_post(y=y, x=x, BG=BG, Calc=Calc, paramList=paramList,
                         z=can_z1, lower=lower, upper=upper, scale=var_scale,
                         tau_y=tau_y, m0=m0, sd0=sd0)
    # Calculate the posterior probability of the current parameter values
    # in z-space
    cur_post = log_post(y=y, x=x, BG=BG, Calc=Calc, paramList=paramList,
                        z=z, lower=lower, upper=upper, scale=var_scale,
                        tau_y=tau_y, m0=m0, sd0=sd0)
    # Calculate the acceptance probability
    R1 = can1_post - cur_post
    return can_z1, can1_post, cur_post, R1


def stage2_acceptprob(can1_post, can2_post, cur_post,
                      can_z1, can_z2, z, varS1):
    '''
    Calculate the acceptance probability for Stage 2 of the DRAM algorithm.

    Args:
        * **z** (:class:`~numpy.ndarray`): Vector of current parameter
          values in z-space - size(q,).
        * **cur_post** (:py:class:`float`): Value of the log posterior
          probability for the current parameter values.
        * **varS1** (:class:`~numpy.ndarray`): Current covariance matrix
          - size(q, q).
        * **can_z1** (:class:`~numpy.ndarray`): Vector of candidate 1 parameter
          values in z-space - size(q,). The candidate is constructed from a
          multivariate normal distribution with mean z and covariance of varS1.
          :math:`can\\_z1 = N_q(z,varS1)`.
        * **can_z2** (:class:`~numpy.ndarray`): Vector of candidate 2 parameter
          values in z-space - size(q,). The candidate is constructed from a
          multivariate normal distribution with mean z and covariance of
          shrinkage*varS1. :math:`can\\_z2 = N_q(z,shrinkage*varS1)`
        * **can1_post** (:py:class:`float`): Value of the log posterior
          probability for candidate 1 parameter values from Stage 1. Computed
          with :meth:`~log_post`.
        * **can2_post** (:py:class:`float`): Value of the log posterior
          probability for candidate 2 parameter values from Stage 2. Computed
          with :meth:`~log_post`.

    Returns:
        * **R2** (:py:class:`float`): Log of Stage 2 acceptance probability.
    '''
    # Calculate the relative acceptance probabilities
    inner_n = 1 - np.min([1, np.exp(can1_post - can2_post)])
    inner_d = 1 - np.min([1, np.exp(can1_post - cur_post)])
    # Adjust factors for inner_n and inner_d to avoid approaching
    # the boundaries 0, 1
    inner_n = inner_n + 1e-10 if inner_n == 0 else inner_n
    inner_n = inner_n - 1e-10 if inner_n == 1 else inner_n
    inner_d = inner_d + 1e-10 if inner_d == 0 else inner_d
    inner_d = inner_d - 1e-10 if inner_d == 1 else inner_d
    # Caclculate stage 2 acceptance probability
    numer = (can2_post + mvnorm.logpdf(x=can_z1, mean=can_z2, cov=varS1)
             + np.log(inner_n))
    denom = (cur_post + mvnorm.logpdf(x=can_z1, mean=z, cov=varS1)
             + np.log(inner_d))
    R2 = numer - denom
    return R2


def adapt_covariance(i, adapt, s_p, all_Z, epsilon, q, varS1):
    '''
    Adapt the covariance matrix at the given adaption interval.

    Args:
        * **i** (:py:class:`int`): Iteration number.
        * **adapt** (:py:class:`int`): Adaption interval, user-defined.
          Default is 20.
        * **s_p** (:py:class:`float`): Scaling parameter for adapting the
          covariance. Default is :math:`\\frac{2.4^2}{q}` where q is the size
          of the parameter space.
        * **all_Z** (:class:`~numpy.ndarray`): Storage of z-space samples for
          each iteration - size(iters, q)
        * **epsilon** (:py:class:`float`): Constant to prevent singularity of
          adaptive covariance. Default is 0.0001.
        * **q** (:py:class:`int`): Number of parameters.
        * **varS1** (:class:`~numpy.ndarray`): Current covariance matrix
          - size(q, q).

    Returns:
        * **varS1** (:class:`~numpy.ndarray`): Adapted covariance matrix
          - size(q, q).
    '''
    if (0 < i) & (i % adapt == 0):
        varS1 = (
                s_p*np.cov(all_Z[range(i+1)].transpose())
                + s_p*epsilon*np.diag(np.ones(q))
                )
    else:
        varS1 = varS1
    return varS1


def update_taub(d_g, gamma, c_g, L):
    '''
    Update the background model precision.

    Args:
        * **d_g** (:py:class:`float`): Scale parameter for Gamma distribution
          for the error in the prior distribution for the basis function
          loadings. Default is 0.1.
        * **gamma** (:class:`~numpy.ndarray`): Basis function loadings
          - size (L,).
        * **c_g** (:py:class:`float`): Shape parameter for Gamma distribution
          for the error in the prior distribution for the basis function
          loadings. Default is 0.1.
        * **L** (:py:class:`int`): Number of cubic B-spline basis functions to
          model the background intensity. Default is 20.

    Returns:
        * **tau_b** (:py:class:`float`): Loadings precision for background.
    '''
    rate = d_g + 0.5*np.inner(gamma, gamma)
    tau_b = np.random.gamma(shape=(c_g + 0.5*L), scale=1/rate)
    return tau_b


def update_tauy(y, BG, Calc, var_scale, d_y, c_y, n):
    '''
    Update the model precision.

    Args:
        * **y** (:class:`~numpy.ndarray`): Vector of diffraction pattern
          intensities - size (n,).
        * **BG** (:class:`~numpy.ndarray`): Vector of background intensity
          values - size (n,).
        * **Calc** (:class:`~.Calculator`): calculator operator that interacts
          with the designated GPX file by referencing GSAS-II libraries.
        * **var_scale** (:class:`~numpy.ndarray`): Vector that scales with the
          intensity of data, heteroscedastic.
          See function :meth:`~initialize_intensity_weight`
        * **d_y** (:py:class:`float`): Scale parameter for Gamma distribution
          of the error variance. Default is 0.1
        * **c_y** (:py:class:`float`): Shape parameter for Gamma distribution
          of the error variance.
        * **n** (:py:class:`int`): Number data points
          (equivalent to the length of intensities vector, y).

    Returns:
        * **tau_y** (:py:class:`float`): Updated model precision.
    '''
    err = (y-BG-Calc.Calculate())/np.sqrt(var_scale)
    rate = d_y + 0.5*np.inner(err, err)
    tau_y = np.random.gamma(shape=(c_y + 0.5*n), scale=1/rate)
    return tau_y


def _print_update(curr_keep, update, n_keep, accept_S1, attempt_S1, accept_S2,
                  attempt_S2, accept_rate_S1, accept_rate_S2):
    # Print an update if necessary
    print("Collected %d of %d samples" % (curr_keep, n_keep))
    print('  %03.2f acceptance rate for Stage 1 (%d attempts)'
          % (accept_S1/attempt_S1, attempt_S1))
    if attempt_S2 > 0:
        rate_S2 = accept_S2/attempt_S2
    else:
        rate_S2 = 0
    print('  %03.2f acceptance rate for Stage 2 (%d attempts)'
          % (rate_S2, attempt_S2))
    accept_rate_S1[(curr_keep//update)-1] = accept_S1/attempt_S1
    accept_rate_S2[(curr_keep//update)-1] = rate_S2
    return accept_rate_S1, accept_rate_S2


def traceplots(plot, q, keep_params, curr_keep, paramList, n_keep, update):
    '''
    Produce traceplots of sampling at the update intervals. Plot in the console
    window and save final traceplot in the current folder.
    '''
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


def _check_parameter_specification(Calc, paramList):
    # Get indices of parameters to refine, even if they are "fixed" by bounds
    useInd = [np.asscalar(np.where(np.array(Calc._varyList) == par)[0])
              for par in paramList]
    if (any(np.array(Calc._varyList)[useInd] != paramList)):
        raise ValueError("Parameter list specification is not valid.")


def _check_parameter_initialization(paramList, init_z):
    # Make sure initial z values are given for every parameter in paramList
    if len(paramList) != len(init_z):
        raise ValueError("Initial value specification for Z is not valid.")


def initialize_intensity_weight(x, y, scaling_factor=1):
    '''
    Define a heteroscedastic vector to scale the residuals between the model
    and the data with the intensity of the smoothed data. The smoothed
    intensity data, y_sm, is obtained from :meth:`smooth_ydata`.

    Args:
        * **y** (:class:`~numpy.ndarray`): Vector of diffraction pattern
          intensities - size (n,).
        * **x** (:class:`~numpy.ndarray`): Vector of 2-theta values from
          diffraction pattern- size (n,).
        * **scaling factor** (:py:class:`float`): Contribution of smoothed
          intensity data. Default is 1.

    Returns:
        * **var_scale** (:class:`~numpy.ndarray`): Vector of scaling factors
          corresponding to intensity data - size (n,).
    '''
    y_sm = smooth_ydata(x=x, y=y)
    var_scale = scaling_factor*y_sm + 1  # Scale for y_sm/tau_y
    return var_scale


# MCMC function
def nlDRAM(GPXfile, paramList, variables, init_z, lower, upper, initCov=None,
           y=None, x=None, L=20, shrinkage=0.2, s_p=(2.4**2), epsilon=1e-4,
           m0=0, sd0=1, c_y=0.1, d_y=0.1, c_g=0.1, d_g=0.1, c_b=0.1, d_b=0.1,
           adapt=20, thin=1, iters=5000, burn=2000, update=500, plot=True,
           fix=False):
    '''
    Args:
        * **GPXfile** (:py:class:`str`):
          Filepath for the GPX file underlying the current data
        * **paramList** (:py:class:`list`):
          (q) list of GSASII parameter names in the same order
          as the upper and lower limits being provided
        * **variables** (:class:`~numpy.ndarray`):
          (q,) vector of parameter names that matches 'paramList'
        * **init_z** (:class:`~numpy.ndarray`):
          (q,) vector of initial values in the z-space
        * **lower** (:class:`~numpy.ndarray`):
          (q,) vector of lower bounds for the parameter values
        * **upper** (:class:`~numpy.ndarray`):
          (q,) vector of upper bounds for the parameter values

    Kwargs:
        * **initCov** (:class:`~numpy.ndarray`) - `None`: (q, q) matrix to
          be used as the covariance matrix for the proposal distribution,
          default value is None. Covariance matrix can be specified with the
          estimate covariance function. If there is no matrix specified, the
          function will use a diagonal matrix with 0.05 on the diagonal
        * **y** (:class:`~numpy.ndarray`) - `None`:
          (n,) vector of intensities. If no values
          are specified, the function uses the values from the provided GPX
          file
        * **x** (:class:`~numpy.ndarray`) - `None`:
          (n,) vector of angles (2*theta). If no
          values are specified, the function uses the values from the provided
          GPX file
        * **L** (:py:class:`float`) - `20`:
          number of B-spline basis functions to
          model the background intensity.
        * **shrinkage** (:py:class:`float`) - `0.2`:
          Governs covariance change between proposal stages,
          default is 0.2
        * **s_p** (:py:class:`float`): :math:`\\frac{2.4^2}{q}`
          Scaling parameter for the adaptive covariance, default is
          set to :math:`\\frac{2.4^2}{q}` as in Gelman (1995), where q is the
          dimension of the parameter space
        * **epsilon** (:py:class:`float`) - `0.0001`:
          Ridge constant to prevent singularity of the adaptive
          covariance.
        * **m0** (:py:class:`float`) - `0`: Governs the mean value on the
          latent z-space of the parameters.
        * **sd0** (:py:class:`float`) - `1`: Governs the standard deviation
          on the latent z-space of the parameters.
        * **c_y** (:py:class:`float`) - 0.1: Shape parameter for Gamma
          distribution of the error variance.
        * **d_y** (:py:class:`float`) - 0.1: Scale parameter for Gamma
          distribution of the error variance.
        * **c_g** (:py:class:`float`) - 0.1: Shape parameter for Gamma
          distribution for the error in the prior distribution for the
          basis function loadings.
        * **d_g** (:py:class:`float`) - 0.1: Scale parameter for Gamma
          distribution for the error in the prior distribution for the
          basis function loadings.
        * **c_b** (:py:class:`float`) - 0.1: Shape parameter for Gamma
          distribution for scale of the proportional constribution to the
          error variance.
        * **d_b** (:py:class:`float`) - 0.1: Scale parameter for Gamma
          distribution for scale of the proportional constribution to the
          error variance.
        * **adapt** (:py:class:`float`): `20`: Controls the adaptation period.
        * **thin** (:py:class:`float`) - `1`: Degree of thinning.
        * **iters** (:py:class:`float`) - `5000`: Number of total iterations
          to run.
        * **burn** (:py:class:`float`) - `2000`: Number of samples to consider
          as burn-in.
        * **update** (:py:class:`float`) - `500`: Period between updates
          printed to the console.
        * **plot** (:py:class:`bool`) - `True`: Indicator for whether or not to
          create trace plots as the sampler progresses.

    Returns:
        * 8-tuple containing the posterior samples for the parameters and
          the model timing, tuple entries are

        #. **keep_params** (:class:`~numpy.ndarray`): Matrix of posterior
           samples for the mean process parameters of interest - (nSamples, q)
        #. **curr_keep** (:py:class:`int`): Number of samples kept. Equivalent
           to (iters - burn) if thin=1.
        #. **varS1** (:class:`~numpy.ndarray`): Final adapated covariance
           matrix - size(q, q).
        #. **1.0/keep_tau_y** (:class:`~numpy.ndarray`): Vector of posterior
           samples for the overall model variance - size(nSamples,)
        #. **keep_gamma** (:class:`~numpy.ndarray`): Matrix of posterior
           samples for the basis function loadings modeling the background
           intensity - (nSamples, L)
        #. **mins** (:py:class:`float`): Number of minutes the sampler took to
           complete.
        #. **accept_rate_S1** (:py:class:`float`): Acceptance rate of stage 1
           DRAM.
        #. **accept_rate_S2** (:py:class:`float`): Acceptance rate of stage 2
           DRAM.
    '''
    Calc = gsas_calculator(GPXfile=GPXfile)
    Calc._varyList = variables
    # Set the scaling parameter based on the number of parameters
    s_p = ((2.4**2)/len(paramList))
    # Assign the intensity vector (y) and 2-theta angles (x) from the GPX file
    # if no values are provided
    x, y = diffraction_file_data(x=x, y=y, Calc=Calc)
    # Calculate a B-spline basis for the range of x
    B = calculate_bsplinebasis(x=x, L=L)
    # Save dimensions
    n = len(y)       # Number of data observations
    q = len(init_z)  # Number of parameters of interest
    # Smooth the observed intensities over the 2-theta data points with lowess
    var_scale = initialize_intensity_weight(x=x, y=y)
    _check_parameter_specification(Calc=Calc, paramList=paramList)
    _check_parameter_initialization(paramList=paramList, init_z=init_z)
    # Initialize parameter values
    z = np.array(init_z, copy=True)  # Latent process
    params = z2par(z=init_z,
                   lower=lower,
                   upper=upper)      # Parameters of interest
    tau_y = 1                        # Error variance for Y
    gamma = np.ones(L)               # Loadings
    tau_b = 1                        # Variance for loadings
    BG = np.matmul(B, gamma)         # Background intensity
    # Update the parameters in the GSAS calculator
    Calc.UpdateParameters(dict(zip(paramList, params)))
    # Initialize covariance for proposal distribution
    varS1 = initialize_cov(initCov=initCov, q=q)
    # Set up counters for the parameters of interest
    # Attempts / acceptances counters
    attempt_S1 = attempt_S2 = accept_S1 = accept_S2 = 0
    # Calculate the number of thinned samples to keep
    n_keep = np.floor_divide(iters - burn - 1, thin) + 1
    curr_keep = 0
    # Initialize output objects
    (all_Z, keep_params, keep_gamma, keep_b, keep_tau_y, keep_tau_b,
     accept_rate_S1, accept_rate_S2) = _initialize_output(
             iters=iters, q=q, n_keep=n_keep, L=L, update=update)
    # Begin 2-stage Delayed Rejection Adaptive Metropolis
    tick = timer()
    for i in range(iters):
        # Update basis function loadings and then background values
        gamma, BG = update_background(B, var_scale, tau_y,
                                      tau_b, L, Calc, y)
        attempt_S1 += 1
        # Stage 1 DRAM:
        can_z1, can1_post, cur_post, R1 = stage1_acceptprob(
                z=z, varS1=varS1, y=y, x=x, BG=BG, Calc=Calc,
                paramList=paramList, lower=lower, upper=upper,
                var_scale=var_scale, tau_y=tau_y, m0=m0, sd0=sd0)
        # Accept candidate if acceptance probability is greater than a random
        # draw and located away from the bounds
        if (
                (np.log(np.random.uniform()) < R1) &
                (np.sum(np.abs(can_z1) > 3) == 0)
                ):
            accept_S1 += 1
            z = np.array(can_z1, copy=True)                # Update latent
            params = z2par(z=z, lower=lower, upper=upper)  # Update parameters
            Calc.UpdateParameters(dict(zip(paramList, params)))
        # Continue to Stage 2 if candidate is rejected
        else:
            # Stage 2:
            attempt_S2 += 1
            # Propose the candidate
            can_z2 = np.random.multivariate_normal(mean=z, cov=shrinkage*varS1)
            if np.sum(np.abs(can_z2) > 3) == 0:  # Ensures away from the bounds
                can2_post = log_post(
                        y=y, x=x, BG=BG, Calc=Calc,
                        paramList=paramList, z=can_z2,
                        lower=lower, upper=upper, scale=var_scale,
                        tau_y=tau_y, m0=m0, sd0=sd0)
                # Calculate the acceptance probability
                R2 = stage2_acceptprob(
                        can1_post=can1_post,
                        can2_post=can2_post,
                        cur_post=cur_post,
                        can_z1=can_z1,
                        can_z2=can_z2,
                        z=z, varS1=varS1)
                # Accept the candidate acceptance probability is greater than
                # a random draw, otherwise reject
                if np.log(np.random.uniform()) < R2:
                    accept_S2 += 1
                    z = np.array(can_z2, copy=True)
                    params = z2par(z=z, lower=lower, upper=upper)
                    Calc.UpdateParameters(dict(zip(paramList, params)))
                del can_z2, can2_post, R2
            else:
                del can_z2
        del can_z1, can1_post, cur_post, R1
        # Store accepted candidates
        all_Z[i] = z
        # Adapt the proposal distribution covariance matrix
        varS1 = adapt_covariance(i=i, adapt=adapt, s_p=s_p, all_Z=all_Z,
                                 epsilon=epsilon, q=q, varS1=varS1)
        # Update tau_b
        tau_b = update_taub(d_g=d_g, gamma=gamma, c_g=c_g, L=L)
        # Update tau_y
        tau_y = update_tauy(
                y=y, BG=BG, Calc=Calc,
                var_scale=var_scale, d_y=d_y, c_y=c_y, n=n)
        # Keep track of everything
        if i >= burn:
            # Store posterior draws if appropriate
            if (i-burn) % thin == 0:
                keep_params[curr_keep] = params
                keep_gamma[curr_keep] = gamma
                # keep_b[curr_keep] = b
                keep_tau_y[curr_keep] = tau_y
                keep_tau_b[curr_keep] = tau_b
                curr_keep += 1
            if curr_keep % update == 0:
                # Print an update if necessary
                accept_rate_S1, accpet_rate_S2 = _print_update(
                        curr_keep=curr_keep, update=update, n_keep=n_keep,
                        accept_S1=accept_S1, attempt_S1=attempt_S1,
                        accept_S2=accept_S2, attempt_S2=attempt_S2,
                        accept_rate_S1=accept_rate_S1,
                        accept_rate_S2=accept_rate_S2)
                # Produce trace plots
                traceplots(plot=plot, q=q, keep_params=keep_params,
                           curr_keep=curr_keep, paramList=paramList,
                           n_keep=n_keep, update=update)
    tock = timer()
    # Gather output into a tuple
    output = (keep_params, curr_keep, varS1, 1.0/keep_tau_y, keep_gamma,
              (tock-tick)/60, accept_rate_S1, accept_rate_S2)
    return output
