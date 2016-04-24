# -*- coding: utf-8 -*-

import numpy as np
import emcee
import scipy
from scipy import stats
from time import time
from math import *

import corner
import matplotlib.pyplot as pl

start_time = time()
datacounts = np.loadtxt('data/826A_Stockholm_counts.txt',skiprows=2)
dataevents = np.loadtxt('data/826A_Stockholm_events.txt',skiprows=3)
F = np.loadtxt('data/inao.txt',skiprows=3)#np.loadtxt('iersst_nino4aa.txt',skiprows=6)

def lnprior(theta):
    a_l, b_l, a_s, b_s, a_x, b_x = theta
    if -5 < b_l < 5 and -5 < a_l < 5 and -5 < a_s < 5 and -5 < b_s < 5 and -5 < a_x < 5 and -5 < b_x < 5:
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x,z,F,FF):
    a_l, b_l, a_s, b_s, a_x, b_x = theta
    ll = np.exp(a_l*F+b_l)
    
    sig = np.exp(a_s*FF+b_s)
    
    xi = np.exp(a_x*FF+b_x)-1


    mu = 205.112847
    model = np.sum(scipy.stats.poisson.logpmf(x,ll))+np.sum(np.log(scipy.stats.genpareto.pdf(z,xi,loc=mu,scale=sig)))#np.log(np.sum(scipy.stats.poisson.pmf(x,ll)))+np.log(np.sum( 1/sig*(1+xi*(z-mu)/sig)**(-1/xi-1) ) ) #+np.sum(-log(sig)+(1/xi-1)*np.log(1-xi*z/sig)           
    return model
#@jit
def lnprob(theta, x,z,F,FF):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x,z,F,FF)

ndim, nwalkers = 6, 100



m_true = -0.9594
b_true = 4.294
N = 50


z = dataevents[:,3]
x = datacounts[:,2]#np.sort(10*np.random.rand(N))
k = []
kr = []
for i in range(len(dataevents[:,1])):
		k.append( int( dataevents[i,1 ] ) )
		kr.append( int( datacounts[i,0] ) )

#do things
FK = []
MK = np.reshape(k,-1)
F[69:,0]

km = F[69:,0]-1890
datev = dataevents[:,0]-1890
FF = []
for i in range(len(k)):
	FF.append(F[69+datev[i],k[i]])

FF2 = np.asarray(FF)


Fl = []
for r in range(len(F[68:187,0])):
    Fl.append(int(F[68+r,0]))

#for j in range(len(F1)):
#	for i in range(12):
#

F1 = np.reshape(F[68:187,1:],-1)#F[68:187,1:]#F[420:1848,1]



p0 = [np.random.rand(ndim) for i in range(nwalkers)]


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[x, z, F1,FF2] )

#sampler.run_mcmc(p0, 1000)

# Run 100 steps as a burn-in.
pos, prob, state = sampler.run_mcmc(p0, 40)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain, sample for 1000
# steps.

sampler.run_mcmc(p0, 500)

# Print out the mean acceptance fraction. In general, acceptance_fraction
# has an entry for each walker so, in this case, it is a 250-dimensional
# vector.
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

# Estimate the integrated autocorrelation time for the time series in each
# parameter.
print("Autocorrelation time:", sampler.get_autocorr_time())


##for i in range(ndim):
##    pl.figure()
##    pl.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
##    pl.title("Dimension {0:d}".format(i))


print "a_l: %f, b_l: %f, a_s: %f, b_s: %f, a_x: %f, b_x: %f" %(np.mean(sampler.flatchain[:,0]),np.mean(sampler.flatchain[:,1]),np.mean(sampler.flatchain[:,2]),np.mean(sampler.flatchain[:,3]),np.mean(sampler.flatchain[:,4]),np.mean(sampler.flatchain[:,5]))
#pl.show()


fig = corner.corner( sampler.flatchain[:,:],labels=["a_l","b_l","a_s","b_s","a_x","b_x"],range=[[-0.1, 0.2], \
[-1.85, -1.35],[-0.1, 0.25], [3.5, 4.25],[-0.15, 0.15], [-0.15, 0.2]], quantiles=[0.16, 0.5, 0.84],show_titles=True, labels_args={"fontsize": 40})

#truths=[mm[0],mm[1],mm[2]]
#fig.savefig("triangle.png")
#fig.show()

b_lfo = -1.56
b_sfo = 3.91
b_xfo = 0.04

print("--- %s seconds ---" % (time() - start_time))

