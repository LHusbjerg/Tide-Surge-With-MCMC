# -*- coding: utf-8 -*-

import numpy as np
import emcee
import scipy
from scipy import stats
from time import time
from math import *
from array import array
import corner
import matplotlib.pyplot as pl



datacounts = np.loadtxt('data/826A_Stockholm_counts.txt',skiprows=2)
dataevents = np.loadtxt('data/826A_Stockholm_events.txt',skiprows=3)
F = np.loadtxt('data/inao.txt',skiprows=3)#np.loadtxt('iersst_nino4aa.txt',skiprows=6)




def lnprior(theta):
    a_l, b_l, a_s, b_s, a_x, b_x = theta
    if -5 < b_l < 5 and -5 < a_l < 5 and -5 < a_s < 5 \
    and -5 < b_s < 5 and -5 < a_x < 5 and -5 < b_x < 5:
        return 0.0
    return -np.inf

def lnlike(theta, x,z,F,FF):
    a_l, b_l, a_s, b_s, a_x, b_x = theta
    ll = np.exp(a_l*F+b_l)
    
    sig = np.exp(a_s*FF+b_s)
    
    xi = np.exp(a_x*FF+b_x)-1
    mu = 205.112847
    model = np.sum(scipy.stats.poisson.logpmf(x,ll))+np.sum(np.log(scipy.stats.genpareto.pdf(z,xi,loc=mu,scale=sig)))#np.sum( x*np.log(ll)-ll)+np.log(np.sum(scipy.stats.poisson.pmf(x,ll)))+np.log(np.sum( 1/sig*(1+xi*(z-mu)/sig)**(-1/xi-1) ) ) #+np.sum(-log(sig)+(1/xi-1)*np.log(1-xi*z/sig)           
    return model

def lnprob(theta, x,z,F,FF):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x,z,F,FF)

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
#F[69,0] = 1890
#F[69,5], F[69,11], F[70,2],F[70,10],F[70,12],F[71,1],F[71,12],F[72,11],F[73,1],F[73,2],F[73,12], F[74,3],F[74,8],F[74,12],F[74,12]
##F[75,0] = 1896
#F[75,1],F[75,2],F[75,3],F[76,12],F[77,2],F[77,11],F[77,12],F[76,12]
#
km = F[69:,0]-1890
datev = dataevents[:,0]-1890
FF = []
for i in range(len(k)):
	FF.append(F[69+datev[i],k[i]])

FF2 = np.asarray(FF)


Fl = []
for r in range(len(F[68:187,0])):
    Fl.append(int(F[68+r,0]))


F1 = np.reshape(F[68:187,1:],-1)#F[68:187,1:]#F[420:1848,1]


b_lfo = -1.57
b_sfo = 3.88
b_xfo = 0.04
a_xfo = 0.02
a_lfo = 0.08
a_sfo = 0.07

mu = 205.112847

overFF = []
underFF = []
index = []
xif0p = []
lambf0p = []
sigf0p = []
xif0m = []
lambf0m = []
sigf0m = []
kkm = []
kkp = []
for i in range(len(FF2)):
    if FF2[i]-np.mean(FF2) < 0.0:
        underFF.append(FF2[i])
        index.append(-1.0)
        kkm.append(i)
    else:
        overFF.append(FF2[i])
        index.append(1.0)
        kkp.append(i)

for i in range(len(FF2)):
    if index[i] == 1:
        xif0p.append(np.exp(FF2[i]*a_xfo+b_xfo)-1)
        lambf0p.append(np.exp(FF2[i]*a_lfo+b_lfo))
        sigf0p.append(np.exp(FF2[i]*a_sfo+b_sfo))
    else:
        xif0m.append(np.exp(FF2[i]*a_xfo+b_xfo)-1)
        lambf0m.append(np.exp(FF2[i]*a_lfo+b_lfo))
        sigf0m.append(np.exp(FF2[i]*a_sfo+b_sfo))

xif0 = np.exp(b_xfo)-1
lambf0 = np.exp(b_lfo)
sigf0 = np.exp(b_sfo)
zsort = sorted(z)
years = dataevents[:,0]-1890+dataevents[:,1]/12

returnp0 = 1/(lambf0*(scipy.stats.genpareto.sf(z,xif0,loc=mu,scale=sigf0)))

returnp0plus = []
returnp0minus = []
for i in range(len(xif0p)):
    returnp0plus.append(1/(lambf0*(scipy.stats.genpareto.sf(z[kkp[i]],xif0p[i],loc=mu,scale=sigf0p[i]))))

for i in range(len(xif0m)):
    returnp0minus.append(1/(lambf0*(scipy.stats.genpareto.sf(z[kkm[i]],xif0m[i],loc=mu,scale=sigf0m[i]))))

#1/(lambf0*(scipy.stats.genpareto.sf(z,xif0,loc=mu,scale=sigf0)))

#(1/(lambf0*(1-scipy.stats.genpareto.cdf(zsort,xif0,loc=mu,scale=sigf0))))**(-1)
returnp0plusar = np.asarray(returnp0plus)
returnp0minusar = np.asarray(returnp0minus)
zp = []
zm = []
for i in range(len(kkp)):
    zp.append(z[kkp[i]])
for i in range(len(kkm)):
    zm.append(z[kkm[i]])

zpar = np.asarray(zp)
zmar = np.asarray(zm)
pl.plot(returnp0/12,z,'g^',returnp0plusar/12,zpar,'ro',returnp0minusar/12,zmar,'bs')
pl.xscale('log')
pl.xlabel('Return Period [yr]')
pl.ylabel('Magnitude [cm]')
pl.show()


#FITS N STUFF

##def poisson(k, lamb):
##    return (lamb**k/factorial(k)) * np.exp(-lamb)
##
##
##pl.hist(z, 100, color="k", histtype="step",range=[mu, 600],normed=True)
##
##poi = [0]*205
##for i in range(300,600):
##    poi.append(scipy.stats.genpareto.pdf(i,xif0,loc=mu,scale=sigf0))
##
##pl.plot(sorted(z),sorted(scipy.stats.genpareto.pdf(z,xif0,loc=mu,scale=sigf0), reverse=True))
##pl.ylabel('Normalised Probability')
##pl.xlabel('Event Magnitude')
##pl.show()
