from __future__ import print_function
import numpy as np
import sys
import os
from multiprocessing import Pool
import matplotlib, matplotlib.pyplot as plt
import time
import matplotlib.gridspec as gridspec
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats
from matplotlib.path import Path
from matplotlib import rc
from pylab import *
import pickle
import matplotlib.lines as mlines
from scipy.interpolate import interp1d
from scipy.integrate import simps
from joblib import Parallel, delayed
import hashlib
import wizcola
import wigglezold
from mcmc import *
import methods

class Fitted(object):
    def __init__(self, debug=False):
        self._debug = debug
    def getIndex(self, param):
        for i, x in enumerate(self.getParams()):
            if x[0] == param:
                return i + 3
        return None
    def debug(self, string):
        if self._debug:
            print(string)
            sys.stdout.flush()
    def getNumParams(self):
        raise NotImplementedError
    def getChi2(self, params):
        raise NotImplementedError
    def getParams(self):
        raise NotImplementedError

class CovarianceFitter(Fitted):
    def __init__(self, debug=False, cambFast=False, pk2xiFast=False):
        self._debug = debug
        self.params = [('omch2', 0.05, 2, '$\\Omega_c h^2$'),('alpha', 0.7, 1.3, r'$\alpha$'),('epsilon', -0.4, 0.4, r'$\epsilon$') ]
    def getParams(self):
        return self.params
    def getNumParams(self):
        return len(self.params)
    def setCovariance(self, cov):
        self.cov = cov
        self.invCov = np.linalg.inv(cov)
    def setData(self, data):
        self.data = data
        
    def getChi2(self, params):
        allParams = self.getParams()
        for i, p in enumerate(params):
            p = np.round(p, 5)
            if p <= allParams[i][1] or p >= allParams[i][2]:
                self.debug("Outside %s: %0.2f" % (allParams[i][0], p))
                return None
        model = np.array([[a]*3 for a in params]).flatten()
        chi2 = np.dot((self.data - model).T, np.dot(self.invCov, self.data - model))

        return chi2
        
        
class CosmoMonopoleFitter(Fitted):
    def __init__(self, debug=False, cambFast=False, pk2xiFast=False):
        self._debug = debug
        self.cambFast = cambFast
        self.pk2xiFast = pk2xiFast
        self.cambDefaults = {}
        self.cambDefaults['get_scalar_cls'] = False
        if cambFast:
            self.cambDefaults['do_lensing'] = False
            self.cambDefaults['accurate_polarization'] = False
            self.cambDefaults['high_accuracy_default'] = False
            
        self.cambParams = []
        self.fitParams = [('b2', 0.001, 3, '$b^2$') ]
        
        self.ss = np.linspace(10, 235, 100)
        self.generator = None
        
    def addCambParameter(self, name, minValue, maxValue, label):
        self.cambParams.append((name, minValue, maxValue, label))
        
    def getNumParams(self):
        return len(self.cambParams) + len(self.fitParams)
        
    def addDefaultCambParams(self, **args):
        self.cambDefaults.update(args)
        
    def getParams(self):
        return self.cambParams + self.fitParams
    def generateMus(self, n=100):
        self.mu = np.linspace(0,1,n)
        self.mu2 = np.power(self.mu, 2)
        self.muNA = self.mu[np.newaxis]
        self.mu2NA = self.mu2[np.newaxis]
        self.p2 = 0.5 * (3 * self.mu2 - 1)
        self.p4 = (35*self.mu2*self.mu2 - 30*self.mu2 + 3) / 8.0
        
    def setData(self, datax, monopole, quadrupole, totalCovariance, monopoleCovariance, quadrupoleCovariance, z, matchQuad=True, minS=40, maxS=180, poles=True, angular=True, logkstar=None, log=True, correction=False, fast=False):
        selection = (datax > minS) & (datax < maxS)
        selection2 = np.concatenate((selection, selection))
        self.rawX = datax
        self.dataZ = z
        self.dataX = datax[selection]
        self.monopole = monopole[selection]
        if quadrupole is not None:
            self.quadrupole = quadrupole[selection]
        self.rawMonopole = monopole
        self.rawQuadrupole = quadrupole
        self.matchQuad = matchQuad
        self.poles = poles
        self.angular = angular
        self.logkstar = logkstar
        self.correction = correction
        self.fast = fast
        if correction:
            self.fitParams.append(('a1', -200, 300, '$a_1$'))
        if logkstar is None:
            if log:
                self.fitParams.append(('kstar', -5, 0, '$\\log(k_*)$'))
            else:
                self.fitParams.append(('sigmav', 0, 10, '$\\sigma_v$'))
        if angular:
            self.fitParams.append(('beta', 0.1, 8, '$\\beta$'))
            self.fitParams.append(('lorentzian', 0.1, 10.0, '$\\sigma H(z)$'))
        if poles:
            self.generateMus()
            self.fitParams.append(('alpha', 0.7, 1.3, r'$\alpha$'))
            if matchQuad:
                self.fitParams.append(('epsilon', -0.4, 0.4, r'$\epsilon$'))
        else:
            self.generateMus(n=50)
            self.fitParams.append(('alphaPerp', 0.7, 1.3, r'$\alpha_\perp$'))
            self.fitParams.append(('alphaParallel', 0.7, 1.3, r'$\alpha_\parallel$'))
        
        if poles and not matchQuad:
            self.totalData = self.monopole
            self.dataCov = (monopoleCovariance[:,:,3])[:,selection][selection,:]
        else:
            self.totalData = np.concatenate((self.monopole, self.quadrupole))
            self.dataCov = (totalCovariance[:,:,3])[:,selection2][selection2,:]
            
        self.rawE = np.sqrt(monopoleCovariance[np.arange(monopoleCovariance.shape[0]), np.arange(monopoleCovariance.shape[1]), 2])
        self.dataE = self.rawE[selection]
        
        if quadrupole is not None:
            self.rawQE = np.sqrt(quadrupoleCovariance[np.arange(monopoleCovariance.shape[0]), np.arange(quadrupoleCovariance.shape[1]), 2])
            self.dataQE = self.rawQE[selection]
        else:
            self.quadrupole = None
            self.rawQE = None
            self.dataQE = None
        
        
        
            
        
    def getData(self):
        return (self.dataX, self.monopole, self.dataE, self.quadrupole, self.dataQE, self.dataZ)
        
    def getRawData(self):
        return (self.rawX, self.rawMonopole, self.rawE, self.rawQuadrupole, self.rawQE, self.dataZ)
        
    
    def getModel(self, params, modelss):
        allParams = self.getParams()
        for i, p in enumerate(params):
            p = np.round(p, 5)
            if p <= allParams[i][1] or p >= allParams[i][2]:
                self.debug("Outside %s: %0.2f" % (allParams[i][0], p))
                return None
        #cambParams = {k[0]: np.round(params[i],5) for (i,k) in enumerate(self.cambParams)}
        #fitDict = {k[0]:params[i + len(cambParams)]  for (i,k) in enumerate(self.fitParams)}
        cambParams = dict((k[0], np.round(params[i],5)) for (i,k) in enumerate(self.cambParams))
        #cambParams = {k[0]: np.round(params[i],5) for (i,k) in enumerate(self.cambParams)}
        fitDict = dict((k[0], params[i + len(cambParams)])  for (i,k) in enumerate(self.fitParams))
        #fitDict = {k[0]:params[i + len(cambParams)]  for (i,k) in enumerate(self.fitParams)}
        omch2 = cambParams.get('omch2')    
        
        if self.logkstar is None:
            if fitDict.get('sigmav') is not None:
                sigmav = fitDict['sigmav']
                kstar = 1 / (np.sqrt(2) * sigmav)
            else:
                kstar = np.exp(fitDict['kstar'])
        else:
            kstar = np.exp(self.logkstar)
        b2 = fitDict['b2']
        
        
        
        if self.generator is None:
            self.generator = methods.SlowGenerator(debug=True)
        (ks, pklin, pkratio) = self.generator.getOmch2AndZ(omch2, self.dataZ)
        
        pknw = methods.dewiggle(ks, pklin)

        
        weights = methods.getLinearNoWiggleWeight(ks, kstar)
        pkdw = pklin * weights + pknw * (1 - weights)
        pknl = pkdw * pkratio
        if self.correction:
            pknl = pknl + (fitDict['a1'] / ks)
        mpknl = b2 * pknl
        
        if self.angular:
            beta = fitDict['beta']
            loren = fitDict['lorentzian']
            
            ksmu = ks[np.newaxis].T.dot(self.muNA )
            ar = mpknl[np.newaxis].T.dot(np.power((1 + beta * self.mu2NA), 2)) / (1 + (loren * loren * ksmu * ksmu))
        else:
            ar = mpknl
            
        s0 = 0.32
        gamma = -1.36
        
        
        if self.poles:
            alpha = fitDict['alpha']
            if self.matchQuad:
                epsilon = fitDict['epsilon']
            else:
                epsilon = 0
            if self.angular:
                monopole = simps(ar, self.mu)
            else:
                monopole = mpknl
            quadrupole = simps(ar * self.p2 * 5.0, self.mu)       
            hexadecapole = simps(ar * self.p4 * 9.0, self.mu)
            d = 5  
            
            ximpa = methods.pk2xiGauss(ks, monopole, modelss * alpha, interpolateDetail=d)
            xiqpa = methods.pk2xiGaussQuad(ks, quadrupole, modelss * alpha, interpolateDetail=d)
            if not self.fast:
                xihpa = methods.pk2xiGaussHex(ks, hexadecapole, modelss * alpha, interpolateDetail=d)

            
            ds = 0.5
            dss = np.array([i+j for i in modelss for j in [-ds, ds]])
            dlogs = np.diff(np.log(dss))[::2]
            dxi0as = np.diff(methods.pk2xiGauss(ks, monopole, dss*alpha, interpolateDetail=d))[::2]
            dxi2as = np.diff(methods.pk2xiGaussQuad(ks, quadrupole, dss*alpha, interpolateDetail=d))[::2]
            if not self.fast:
                dxi4as = np.diff(methods.pk2xiGaussHex(ks, hexadecapole, dss*alpha, interpolateDetail=d))[::2]
                dxi4asdlogs = dxi4as / dlogs
            dxi0asdlogs = dxi0as / dlogs
            dxi2asdlogs = dxi2as / dlogs
           

            datapointsM = ximpa + 0.4 * epsilon * (3 * xiqpa + dxi2asdlogs)
            datapointsQ = 2 * epsilon * dxi0asdlogs  + (1 + (6.0 * epsilon / 7.0)) * xiqpa
            if not self.fast:
                datapointsQ += (4.0 * epsilon / 7.0) * ( dxi2asdlogs   + 5 * xihpa + dxi4asdlogs )

            growth = 1 + np.power(((modelss * alpha)/s0), gamma)
            datapointsM = datapointsM * growth
            datapointsQ = datapointsQ * growth

            
            if self.matchQuad:        
                return np.concatenate((datapointsM, datapointsQ))
            else:
                return datapointsM
                
        else:
            alphaPerp = fitDict['alphaPerp']
            alphaParallel = fitDict['alphaParallel']

            monopole = simps(ar, self.mu)
            quadrupole = simps(ar * self.p2 * 5, self.mu)
            hexapole = simps(ar * self.p4 * 9, self.mu)
            
            datapointsM = methods.pk2xiGauss(ks, monopole, self.ss) 
            datapointsQ = methods.pk2xiGaussQuad(ks, quadrupole, self.ss)
            datapointsH = methods.pk2xiGaussHex(ks, hexapole, self.ss)
            

            growth = 1 + np.power((self.ss/s0), gamma)

            datapointsM = datapointsM * growth
            datapointsQ = datapointsQ * growth
            datapointsH = datapointsH * growth
            
            sprime = modelss
            
            
            xi2dm = np.ones(self.mu.size)[np.newaxis].T.dot(datapointsM[np.newaxis])
            xi2dq = self.p2[np.newaxis].T.dot(datapointsQ[np.newaxis])
            xi2dh = self.p4[np.newaxis].T.dot(datapointsH[np.newaxis])
            xi2d = xi2dm + xi2dq + xi2dh

            mugrid = self.muNA.T.dot(np.ones((1, datapointsM.size)))
            ssgrid = self.ss[np.newaxis].T.dot(np.ones((1, self.mu.size))).T
            
            flatmu = mugrid.flatten()
            flatss = ssgrid.flatten()
            flatxi2d = xi2d.flatten()
            
            
            sqrtt = np.sqrt(alphaParallel * alphaParallel * self.mu2 + alphaPerp * alphaPerp * (1 - self.mu2))
            mus = alphaParallel * self.mu / sqrtt
            mu1 = self.mu[:self.mu.size/2]
            mu2 = self.mu[self.mu.size/2:]
            xiT = []
            xiL = []
            svals = np.array([])
            mvals = np.array([])
            for sp in sprime:
                svals = np.concatenate((svals, sp * sqrtt))
                mvals = np.concatenate((mvals, mus))
            
            xis = scipy.interpolate.griddata((flatmu, flatss), flatxi2d, (mvals, svals))
            for i, sp in enumerate(sprime):
                sz = sqrtt.size/2
                ii = 2 * i
                xis1 = xis[ii*sz : (ii+1)*sz]
                xis2 = xis[(ii+1)*sz : (ii+2)*sz]
                xiT.append(2 * simps(xis1, mu1))
                xiL.append(2 * simps(xis2, mu2))
           
            return np.concatenate((np.array(xiT), np.array(xiL)))


    def getChi2(self, params):
        datapoints = self.getModel(params, self.dataX)
        if datapoints is None:
            return None
        chi2 = np.dot((self.totalData - datapoints).T, np.dot(self.dataCov, self.totalData - datapoints))
        return chi2

class WizColaCosmoMonopoleFitter(CosmoMonopoleFitter):
    def __init__(self, cambFast=True, pk2xiFast=True, debug=True, bin=0, matchQuad=True, minS=25, maxS=180, correction=False, fast=False):
        super(self.__class__, self).__init__(cambFast=cambFast, pk2xiFast=pk2xiFast, debug=debug)
        
        wiz = wizcola.WizColaLoader.getMultipoles()
        monopoles = wiz.getAllMonopoles(bin)
        quadrupoles = wiz.getAllQuadrupoles(bin)
        z = wiz.getZ(bin)
        datax = monopoles[:, 0]
        monopoles = monopoles[:, 1:]
        quadrupoles = quadrupoles[:, 1:]
        monopole = np.average(monopoles, axis=1)
        quadrupole = np.average(quadrupoles, axis=1)
        monopoleCor = wiz.getMonopoleCovariance(bin)
        quadrupoleCor = wiz.getQuadrupoleCovariance(bin)
        cor = wiz.getCovariance(bin)
        monopoleCor[:,:,3] *= np.sqrt(600)
        monopoleCor[:,:,2] /= np.sqrt(600)
        quadrupoleCor[:,:,3] *= np.sqrt(600)
        quadrupoleCor[:,:,2] /= np.sqrt(600)
        cor[:,:,3] *= np.sqrt(600)
        cor[:,:,2] /= np.sqrt(600)
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        self.setData(datax, monopole, quadrupole, cor, monopoleCor, quadrupoleCor, z, minS=minS, maxS=maxS, matchQuad=matchQuad, correction=correction, fast=fast)

class WizColaSingleMultipole(CosmoMonopoleFitter):
    def __init__(self, realisation, debug=True, bin=0, matchQuad=True, minS=25, maxS=180, correction=False, fast=True):
        super(self.__class__, self).__init__(debug=debug)
        
        wiz = wizcola.WizColaLoader.getMultipoles()
        monopoles = wiz.getAllMonopoles(bin)
        quadrupoles = wiz.getAllQuadrupoles(bin)
        z = wiz.getZ(bin)
        datax = monopoles[:, 0]
        monopole = monopoles[:, 1+realisation]
        quadrupole = quadrupoles[:, 1+realisation]
        monopoleCor = wiz.getMonopoleCovariance(bin)
        quadrupoleCor = wiz.getQuadrupoleCovariance(bin)
        cor = wiz.getCovariance(bin)
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        print("Realisation %d and bin %d" % (realisation, bin))
        self.setData(datax, monopole, quadrupole, cor, monopoleCor, quadrupoleCor, z, minS=minS, maxS=maxS, matchQuad=matchQuad, correction=correction, fast=fast)
        
        
class WizColaSingleWedge(CosmoMonopoleFitter):
    def __init__(self, realisation, cambFast=False, pk2xiFast=False, debug=True, bin=0, minS=25, maxS=180):
        super(self.__class__, self).__init__(cambFast=cambFast, pk2xiFast=pk2xiFast, debug=debug)
        
        wiz = wizcola.WizColaLoader.getWedges()
        trans = wiz.getAllTransverse(bin)
        longs = wiz.getAllLongitudinal(bin)
        z = wiz.getZ(bin)
        datax = trans[:, 0]
        trans = trans[:, 1:]
        longs = longs[:, 1:]
        tran = trans[:, 1+realisation]
        long = trans[:, 1+realisation]
        transCor = wiz.getTransverseCovariance(bin)
        longCor = wiz.getLongitudinalCovariance(bin)
        cor = wiz.getCovariance(bin)
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        print("Wedge Realisation %d and bin %d" % (realisation, bin))
        self.setData(datax, tran, long, cor, transCor, longCor, z, minS=minS, maxS=maxS, poles=False)
        
        
        
class WizColaCosmoWedgeFitter(CosmoMonopoleFitter):
    def __init__(self, cambFast=False, pk2xiFast=False, debug=True, bin=0, minS=25, maxS=180):
        super(self.__class__, self).__init__(cambFast=cambFast, pk2xiFast=pk2xiFast, debug=debug)
        
        wiz = wizcola.WizColaLoader.getWedges()
        trans = wiz.getAllTransverse(bin)
        longs = wiz.getAllLongitudinal(bin)
        z = wiz.getZ(bin)
        datax = trans[:, 0]
        trans = trans[:, 1:]
        longs = longs[:, 1:]
        tran = np.average(trans, axis=1)
        longss = np.average(longs, axis=1)
        transCor = wiz.getTransverseCovariance(bin)
        longCor = wiz.getLongitudinalCovariance(bin)
        cor = wiz.getCovariance(bin)
        transCor[:,:,3] *= np.sqrt(600)
        transCor[:,:,2] /= np.sqrt(600)
        longCor[:,:,3] *= np.sqrt(600)
        longCor[:,:,2] /= np.sqrt(600)
        cor[:,:,3] *= np.sqrt(600)
        cor[:,:,2] /= np.sqrt(600)
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        self.setData(datax, tran, longss, cor, transCor, longCor, z, minS=minS, maxS=maxS, poles=False)
        
class WigglezOldMonopoleFitter(CosmoMonopoleFitter):
    def __init__(self, cambFast=False, pk2xiFast=False, debug=True, bin=0, minS=10, maxS=180, angular=False, log=False):
        super(self.__class__, self).__init__(cambFast=cambFast, pk2xiFast=pk2xiFast, debug=debug)
        wig = wigglezold.WigglezOldLoader.getInstance()
        z = wig.getZ(bin)
        data = wig.getMonopoles(bin)
        cor = wig.getCov(bin)
        datax = data[:,0]
        monopole = data[:,1]
        print(monopole.size)
        self.addCambParameter('omch2', 0.05, 0.2, '$\\Omega_c h^2$')
        self.setData(datax, monopole, None, cor, cor, None, z, minS=minS, maxS=maxS, matchQuad=False, angular=angular, log=log)
'''   
def doFit(cov, finalData):
    uid = "ztesting123"
    f = CovarianceFitter(debug=False)
    f.setCovariance(cov)
    f.setData(finalData)
    manager = CambMCMCManager(uid, f, debug=False)
    manager.configureMCMC(numCalibrations=11,calibrationLength=1000, thinning=2, maxSteps=200000)
    manager.configureSaving(stepsPerSave=50000)
    
    manager.doWalk(0)
    manager.doWalk(1)
    manager.doWalk(2)
    manager.consolidateData()
    manager.testConvergence()
    manager.getTamParameterBounds(numBins=25)
    manager.plotResults()
    return manager'''
# doFit(cov, finals[1])
    
        
def configureAndDoWalk(manager, fitter, uid, walk, chunkLength=None):
    manager.fitter = fitter
    manager.uid = uid
    np.random.seed((int(time.time()) + abs(hash(uid))) * (walk + 1))
    manager.doWalk(walk, chunkLength=chunkLength)
