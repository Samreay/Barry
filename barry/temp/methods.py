import numpy as np
from numpy import pi
from scipy.integrate import *
from scipy.interpolate import interp1d
import scipy.stats
import collections
import os
#sfrom camb4py import *
#import camb4py
from matplotlib.colors import LinearSegmentedColormap
import pickle

cms = 299792458.0
cks = cms / 1000.0

def getEz(om, ol, z):
    return np.sqrt((1.0 + z) * (1.0 + z) * (1.0 + z) * om + ol )
    
vgetEz = np.vectorize(getEz, excluded=['om', 'ol'])

def getRChi(om, ol, z, quality=100):
    zs = np.linspace(0, z, quality)
    integrand = 1.0 / vgetEz(om, ol, zs)
    return simps(integrand, zs)
    
def getDa(om, ol, z):
    return (cks/100) * getRChi(om, ol, z) / (1.0 + z)
    
def getHz(om, ol, z):
    return getEz(om, ol, z)

def getDv(om, ol, z):
    #return np.power(getRChi(om, ol, z)**2 * z / getEz(om, ol, z), 1.0/3.0) * cks/100
    da = getDa(om, ol, z)
    hz = getHz(om, ol, z)
    return np.power( (1.0+z)*(1.0+z)*da*da * cks*z/(100 * hz), 1.0/3.0)
    
def getDaVar(alpha, epsilon, sigmaaa, sigmaee, sigmaae):
    return np.power(sigmaaa/alpha,2) + np.power(sigmaee/(1 + epsilon), 2) - (2/(alpha * (1 + epsilon)))*sigmaae
    
def getHVar(alpha, epsilon, sigmaaa, sigmaee, sigmaae):
    return np.power(sigmaaa/alpha,2) + 4*np.power(sigmaee/(1 + epsilon), 2) + (4/(alpha * (1 + epsilon)))*sigmaae
 
def dewiggle(ks, pk, degree=13, sigma=1, weight=0.5):
    ks2 = np.log(ks)
    pk2 = np.log(pk)
    index = np.argmax(pk)
    maxk2 = ks2[index]
    gauss = np.exp(-0.5 * np.power(((ks2 - maxk2)/sigma), 2))   
    w = np.ones(pk.size) - weight * gauss
    z = np.polyfit(ks2, pk2, degree, w=w)
    p = np.poly1d(z)
    polyval = p(ks2)
    pkp = np.exp(polyval)
    return pkp

def pk2xi(ks, pks, ss, interpolateDetail=50):
    # Interpolation based on maximum k value to ensure each sinusoid can be numerically intergrated to high accuracy
    ks2 = np.logspace(np.log(np.min(ks) * 1.000001), np.log(np.max(ks) * 0.999999), interpolateDetail * ks.size, base=np.e)
    pks2 = np.exp(interp1d(ks, np.log(pks), kind='linear', copy=False, assume_sorted=True)(ks2)) # Lienar used as scipy's quad and cubic interp breaks for large arrays
    
    # Set up output array
    xis = np.zeros(ss.size)

    # Precompute k^2 P(k)    
    kkpks = ks2  * pks2 # * ks

    # Iterate over all values in desired output array of distances (s)
    for i, s in enumerate(ss):
        integrand = kkpks * np.sin(ks2 * s) / s
        xis[i] = trapz(integrand, ks2)
    xis /= (2 * pi * pi)
    return xis
    
def pk2xi3Gauss(ks, pks, ss, interpolateDetail=10, a=0.5):
    # Interpolation based on maximum k value to ensure each sinusoid can be numerically intergrated to high accuracy
    ks2 = np.logspace(np.log(np.min(ks) * 1.000001), np.log(np.max(ks) * 0.999999), interpolateDetail * ks.size, base=np.e)
    pks2 = np.exp(interp1d(ks, np.log(pks), kind='linear')(ks2)) # Linear used as scipy's quad and cubic interp breaks for large arrays
    
    # Set up output array
    xis = np.zeros(ss.size)
    # Precompute k^2 P(k)    
    kkpks = ks2  * pks2 * np.exp(-ks2 * ks2 * a * a)
    totalRange = np.max(ks2) - np.min(ks2)
    sOsc = 2 * np.pi / ss
    numOsc = np.floor(totalRange / sOsc)
    kRange = np.min(ks2) + numOsc * sOsc
    indexes = np.zeros(ss.size)
    for i, upper in enumerate(kRange):
        indexes[i] = np.argmax(ks2 > upper) - 1

    # Iterate over all values in desired output array of distances (s)
    for i, s in enumerate(ss):
        #max = np.min(np.max(ks2), 900*twoPi/s + np.min()
        #ks3 = interp1d(ks2)
        integrand = kkpks[:indexes[i]] * np.sin(ks2[:indexes[i]] * s) / s
        xis[i] = trapz(integrand, ks2[:indexes[i]])
    xis /= (2 * pi * pi)
    return xis
    
    
def pk2xi3(ks, pks, ss, interpolateDetail=100, numCycles=880):
    # Interpolation based on maximum k value to ensure each sinusoid can be numerically intergrated to high accuracy
    ks2 = np.logspace(np.log(np.min(ks) * 1.000001), np.log(np.max(ks) * 0.999999), interpolateDetail * ks.size, base=np.e)
    pks2 = np.exp(interp1d(ks, np.log(pks), kind='linear')(ks2)) # Linear used as scipy's quad and cubic interp breaks for large arrays
    
    # Set up output array
    xis = np.zeros(ss.size)
    
    # Precompute k^2 P(k)    
    kkpks = ks2  * pks2 # * ks
    
    totalRange = np.max(ks2) - np.min(ks2)
    sOsc = 2 * np.pi / ss
    numOsc = np.minimum(numCycles, np.floor(totalRange / sOsc))
    kRange = np.min(ks2) + numOsc * sOsc
    indexes = np.zeros(ss.size)
    for i, upper in enumerate(kRange):
        indexes[i] = np.argmax(ks2 > upper) - 1

    # Iterate over all values in desired output array of distances (s)
    for i, s in enumerate(ss):
        #max = np.min(np.max(ks2), 900*twoPi/s + np.min()
        #ks3 = interp1d(ks2)
        integrand = kkpks[:indexes[i]] * np.sin(ks2[:indexes[i]] * s) / s
        xis[i] = trapz(integrand, ks2[:indexes[i]])
    xis /= (2 * pi * pi)
    return xis
    
def pk2xi2(ks, pks, ss, pointsPerSinPeriod=100, maxCount=900):
    
    pks2 = pks * ks * ks
    maxk = np.max(ks)
    mink = np.min(ks)
    twoPi = 2 * np.pi
    dxi = 0
    xis = np.zeros(ss.size)
    
    for i, s in enumerate(ss):
        osc = twoPi / s
        klo = mink
        khi = klo + osc
        xi = 0
        count = 0
        while (khi < maxk):
            lklo = np.log(klo)
            lkhi = np.log(khi)
            lkSlice = np.linspace(lklo, lkhi, pointsPerSinPeriod)
            kSlice = np.exp(lkSlice)
            
            pkkk_slice = interp1d(ks, pks2)(kSlice)
            integrand_slice = pkkk_slice * np.sin(kSlice * s) / s
            dxi = simps(integrand_slice, lkSlice)
            xi += dxi
            klo = khi
            khi = klo + osc
            count += 1
            if count >= maxCount:
                break
        xis[i] = xi    
    return xis / (2 * np.pi * np.pi)

def getDeltaChi2(dof, sigma):
    ci = scipy.stats.chi2.cdf(sigma**2, 1)
    return scipy.stats.chi2.ppf(ci, dof)
    
    
    
def getKStar(ks, pows):
    f = 1 / (3 * np.pi)
    v = trapz(pows * ks, ks)
    final = 1 / np.sqrt(f * v)
    return final
    
def getLinearNoWiggleWeight(ks, kStar):
    return np.exp(- ks * ks / (2 * kStar * kStar))

def getLinearNoWiggleWeightSigmaV(ks, sigmaV):
    return np.exp(- ks * ks * sigmaV * sigmaV)


def pk2xiGauss(ks, pks, ss, interpolateDetail=5, a=0.5):
    ks2 = np.logspace(np.log(np.min(ks) * 1.000001), np.log(np.max(ks) * 0.999999), interpolateDetail * ks.size, base=np.e)
    pks2 = interp1d(ks, pks, kind='linear')(ks2)
    
    # Set up output array
    xis = np.zeros(ss.size)

    # Precompute k^2 P(k)and gauss
    kkpks = ks2  * pks2 * np.exp(-ks2 * ks2 * a * a)

    # Iterate over all values in desired output array of distances (s)
    for i, s in enumerate(ss):
        integrand = kkpks * np.sin(ks2 * s) / s
        xis[i] = trapz(integrand, ks2)
    xis /= (2 * pi * pi)
    return xis

def pk2xiGaussQuad(ks, pks, ss, interpolateDetail=5, a=0.5):
    ks2 = np.logspace(np.log(np.min(ks) * 1.000001), np.log(np.max(ks) * 0.999999), interpolateDetail * ks.size, base=np.e)
    pks2 = interp1d(ks, pks, kind='linear')(ks2)
    
    # Set up output array
    xis = np.zeros(ss.size)

    # Precompute k^2 P(k)and gauss
    kkpks = ks2 * ks2  * pks2 * np.exp(-ks2 * ks2 * a * a)

    # Iterate over all values in desired output array of distances (s)
    for i, s in enumerate(ss):
        sks2 = ks2 * s
        integrand = kkpks * ( 3*(np.sin(sks2)/(sks2*sks2*sks2) - np.cos(sks2)/(sks2*sks2)) - np.sin(sks2)/sks2 )
        xis[i] = trapz(integrand, ks2)
    xis /= -(2 * pi * pi)
    return xis
    
def pk2xiGaussHex(ks, pks, ss, interpolateDetail=5, a=0.5):
    ks2 = np.logspace(np.log(np.min(ks) * 1.000001), np.log(np.max(ks) * 0.999999), interpolateDetail * ks.size, base=np.e)
    pks2 = interp1d(ks, pks, kind='linear')(ks2)
    
    # Set up output array
    xis = np.zeros(ss.size)

    # Precompute k^2 P(k)and gauss
    kkpks = ks2 * ks2  * pks2 * np.exp(-ks2 * ks2 * a * a)

    # Iterate over all values in desired output array of distances (s)
    for i, s in enumerate(ss):
        sks2 = ks2 * s
        integrand = kkpks * ( 105*np.sin(sks2)/(sks2**5) - 105*np.cos(sks2)/(sks2**4) - 45*np.sin(sks2)/(sks2**3) + 10*np.cos(sks2)/(sks2*sks2) + np.sin(sks2)/sks2 )
        xis[i] = trapz(integrand, ks2)
    xis /= (2 * pi * pi)
    return xis
'''
class NonLinearGenerator(object):
    def __init__(self, debug=False, useFilesystem=True):
        self.saveFolder = 'C:\\nonLinearSaves4'
        self.nonLinearPrefix = 'nonLinear_%s'
        self.useFilesystem = useFilesystem
        self.removeFromTitle = []
        self.used = {}
    def getNonLinearFilename(self, **params):
        od = collections.OrderedDict(sorted(params.items()))
        string = "_".join(['%s_%s' % (k, str(v).replace('.', 'd')[:10]) for k, v in od.iteritems() if k not in self.removeFromTitle])
        f =  self.saveFolder + os.sep + (self.nonLinearPrefix % string)
        return f
    def getNonLinear(self, om, fb, h, sig8, ns, zz, mink, maxk):
        (ks, ratio) = SmithCorr()(om, fb, h, sig8, ns, zz, 1, 2, mink, maxk, 300)
        return (ks, ratio)
        
    def __call__(self, om, fb, h, sig8, ns, zz, mink, maxk):
    
        f = self.getNonLinearFilename(om=om, fb=fb, h=h, sig8=sig8, ns=ns, zz=zz, mink=mink, maxk=maxk)
        
        if self.used.get(f) is not None:
            return self.used.get(f)
        if self.useFilesystem:
            try:
                r = np.loadtxt(f)
                ks = r[:,0]
                ratio = r[:,1]
            except:
                (ks, ratio) = self.getNonLinear(om, fb, h, sig8, ns, zz, mink, maxk)
                self.used[f] = (ks, ratio)
                np.savetxt(f, np.column_stack((ks, ratio)))
                
        else:
            (ks, ratio) = self.getNonLinear(om, fb, h, sig8, ns, zz, mink, maxk)
            self.used[f] = (ks, ratio)
        return (ks, ratio)
        
class CambGenerator(object):
    def __init__(self, debug=False, useFilesystem=True):
        self.camb = camb4py.load(debug=debug)
        self.cambSaveFolder = 'C:\\cambSaves4'
        self.cambPrefix = 'camb_%s'
        self.removeFromTitle = ['sigV', 'alpha', 'b2', 'kstar', 'get_scalar_cls']
        self.useFilesystem = useFilesystem
        self.used = {}
    def getCambFilename(self, params):
        od = collections.OrderedDict(sorted(params.items()))
        string = "_".join(['%s_%s' % (k, str(v).replace('.', 'd')[:10]) for k, v in od.iteritems() if k not in self.removeFromTitle])
        f =  self.cambSaveFolder + os.sep + (self.cambPrefix % string)
        return f
    def getCamb(self, params):
        res = self.camb(**params)
        ks = res['transfer_matterpower']['k/h']
        pk = res['transfer_matterpower']['power']
        return (ks, pk)
    def __call__(self, **params):
        f = self.getCambFilename(params)
        if self.used.get(f) is not None:
            return self.used.get(f)
        if self.useFilesystem:
            try:
                r = np.loadtxt(f)
                ks = r[:,0]
                pk = r[:,1]
            except:
                (ks, pk) = self.getCamb(params)
                self.used[f] = (ks, pk)
                np.savetxt(f, np.column_stack((ks, pk)))
                
        else:
            (ks, pk) = self.getCamb(params)
            self.used[f] = (ks, pk)
        return (ks, pk)


'''
class SlowGenerator(object):
    def __init__(self, debug=False, generate=False, omch2Range=[0.05,0.25], zs=[0.44, 0.6, 0.73]):
        self.saveFolder = "C:"
        self.prefix = "wizcola.dat"
        self.removeFromTitle = ['ombh2']
        self.zs = zs
        self.omch2Range = omch2Range
        self.omch2s = np.arange(self.omch2Range[0], self.omch2Range[1], 1e-5)
        
        self.results = {}
        self.defaults = {'ns': 0.96, 'bb': 1, 'obh2': 0.0226, 'sig8': 0.8, 'h': 0.71}
        
        self.cambDefaults = {'hubble':71}
        self.cambDefaults['get_scalar_cls'] = False
        self.debug = debug
        #self.cambGenerator = CambGenerator()
        #self.nonLinearGenerator = NonLinearGenerator()
        self.loadChunck()
        #if generate:
        #    self.generate()
            
    def fix(self):
        for i, omch2 in enumerate(self.omch2s):
            key = '%0.5f' % omch2
            array = self.results[key]
            length = array.size / 5
            r = []
            for j in range(5):
                r.append(array[j * length:(j+1)*length])
            self.results[key] = r
            
        self.savePickle()
            
    def loadChunck(self):
        fname = self.saveFolder + os.sep + self.prefix
        if self.debug:
            print("Loading pickle from " + fname)
        try:
            with open(fname, 'rb') as f:
                self.results = pickle.load(f, encoding='latin1')
        except:
            print("Unable to load file. It may not exist yet")
            raise
        if self.debug:
            print("Loaded pickle")
            
    def getOmch2AndZ(self, omch2, z):
        omch2 = '%0.5f' % (np.round(omch2, decimals=5))
        if z not in self.zs:
            raise Exception("This z value has not been generated for")
        if self.results.get(omch2) is None:
            raise Exception("This omch2 has not been generated for: %s" % omch2)
        
        index = self.zs.index(z)
        res = self.results[omch2]
        return (res[0], res[1], res[2 + index])
        
    def getOmch2AndZs(self, omch2, zs):
        #omch2 = '%0.5f' % (np.round(omch2, decimals=5))
        omch2 = '%0.5f' % omch2
        if self.results.get(omch2) is None:
            raise Exception("This omch2 has not been generated for: %s" % omch2)
        for z in zs:
            if z not in self.zs:
                raise Exception("This z value has not been generated for")
        indexes = [self.zs.index(z) for z in zs]
        res = self.results[omch2]
        ks = res[0]
        pklin = res[1]
        pkratios = [res[2 + i] for i in indexes]
        return (ks, pklin, pkratios)
    '''def generate(self):
        start = time.time()
        for i, omch2 in enumerate(self.omch2s):
            if self.results.get('%0.5f' % omch2) is not None:
                continue
            
            ombh2 = self.defaults['obh2']
            
            oc = omch2 / self.defaults['h'] / self.defaults['h']
            ob = ombh2 / self.defaults['h'] / self.defaults['h']
            om = oc + ob
            
            d2 = self.cambDefaults.copy()
            d2.update({'omch2': omch2, 'ombh2': ombh2})        
            (ks, pklin) = self.cambGenerator(**d2)
            
            zarr = [ks, pklin]
            for z in self.zs:
                (karr, pkratio) = self.nonLinearGenerator(om, ob/om , self.defaults['h'], self.defaults['sig8'], self.defaults['ns'], z, np.min(ks), 1 + np.max(ks))
                zarr.append(interp1d(karr, pkratio, kind='linear')(ks))
                
            self.results['%0.5f' % omch2] = zarr
            
            
            fraction = np.where(self.omch2s==omch2)[0] * 1.0 / self.omch2s.size
            minElapsed = (time.time() - start)/60.0
            timeLeft = minElapsed * ((1.0 / fraction) - 1.0)
            print("Generating %0.5f, %0.3f%% complete. %0.2f minutes left" % (omch2, fraction * 100, timeLeft))            
            
            
            if i % 100 == 0:
                self.savePickle()
        self.savePickle()'''
        
    def savePickle(self):
        path = self.saveFolder + os.sep + self.prefix
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)
cm_data_mid = [[  1.46159096e-03,   4.66127766e-04,   1.38655200e-02],
       [  2.26726368e-03,   1.26992553e-03,   1.85703520e-02],
       [  3.29899092e-03,   2.24934863e-03,   2.42390508e-02],
       [  4.54690615e-03,   3.39180156e-03,   3.09092475e-02],
       [  6.00552565e-03,   4.69194561e-03,   3.85578980e-02],
       [  7.67578856e-03,   6.13611626e-03,   4.68360336e-02],
       [  9.56051094e-03,   7.71344131e-03,   5.51430756e-02],
       [  1.16634769e-02,   9.41675403e-03,   6.34598080e-02],
       [  1.39950388e-02,   1.12247138e-02,   7.18616890e-02],
       [  1.65605595e-02,   1.31362262e-02,   8.02817951e-02],
       [  1.93732295e-02,   1.51325789e-02,   8.87668094e-02],
       [  2.24468865e-02,   1.71991484e-02,   9.73274383e-02],
       [  2.57927373e-02,   1.93306298e-02,   1.05929835e-01],
       [  2.94324251e-02,   2.15030771e-02,   1.14621328e-01],
       [  3.33852235e-02,   2.37024271e-02,   1.23397286e-01],
       [  3.76684211e-02,   2.59207864e-02,   1.32232108e-01],
       [  4.22525554e-02,   2.81385015e-02,   1.41140519e-01],
       [  4.69146287e-02,   3.03236129e-02,   1.50163867e-01],
       [  5.16437624e-02,   3.24736172e-02,   1.59254277e-01],
       [  5.64491009e-02,   3.45691867e-02,   1.68413539e-01],
       [  6.13397200e-02,   3.65900213e-02,   1.77642172e-01],
       [  6.63312620e-02,   3.85036268e-02,   1.86961588e-01],
       [  7.14289181e-02,   4.02939095e-02,   1.96353558e-01],
       [  7.66367560e-02,   4.19053329e-02,   2.05798788e-01],
       [  8.19620773e-02,   4.33278666e-02,   2.15289113e-01],
       [  8.74113897e-02,   4.45561662e-02,   2.24813479e-01],
       [  9.29901526e-02,   4.55829503e-02,   2.34357604e-01],
       [  9.87024972e-02,   4.64018731e-02,   2.43903700e-01],
       [  1.04550936e-01,   4.70080541e-02,   2.53430300e-01],
       [  1.10536084e-01,   4.73986708e-02,   2.62912235e-01],
       [  1.16656423e-01,   4.75735920e-02,   2.72320803e-01],
       [  1.22908126e-01,   4.75360183e-02,   2.81624170e-01],
       [  1.29284984e-01,   4.72930838e-02,   2.90788012e-01],
       [  1.35778450e-01,   4.68563678e-02,   2.99776404e-01],
       [  1.42377819e-01,   4.62422566e-02,   3.08552910e-01],
       [  1.49072957e-01,   4.54676444e-02,   3.17085139e-01],
       [  1.55849711e-01,   4.45588056e-02,   3.25338414e-01],
       [  1.62688939e-01,   4.35542881e-02,   3.33276678e-01],
       [  1.69575148e-01,   4.24893149e-02,   3.40874188e-01],
       [  1.76493202e-01,   4.14017089e-02,   3.48110606e-01],
       [  1.83428775e-01,   4.03288858e-02,   3.54971391e-01],
       [  1.90367453e-01,   3.93088888e-02,   3.61446945e-01],
       [  1.97297425e-01,   3.84001825e-02,   3.67534629e-01],
       [  2.04209298e-01,   3.76322609e-02,   3.73237557e-01],
       [  2.11095463e-01,   3.70296488e-02,   3.78563264e-01],
       [  2.17948648e-01,   3.66146049e-02,   3.83522415e-01],
       [  2.24762908e-01,   3.64049901e-02,   3.88128944e-01]]
cm_data = [[ 0.26700401,  0.00487433,  0.32941519],
       [ 0.26851048,  0.00960483,  0.33542652],
       [ 0.26994384,  0.01462494,  0.34137895],
       [ 0.27130489,  0.01994186,  0.34726862],
       [ 0.27259384,  0.02556309,  0.35309303],
       [ 0.27380934,  0.03149748,  0.35885256],
       [ 0.27495242,  0.03775181,  0.36454323],
       [ 0.27602238,  0.04416723,  0.37016418],
       [ 0.2770184 ,  0.05034437,  0.37571452],
       [ 0.27794143,  0.05632444,  0.38119074],
       [ 0.27879067,  0.06214536,  0.38659204],
       [ 0.2795655 ,  0.06783587,  0.39191723],
       [ 0.28026658,  0.07341724,  0.39716349],
       [ 0.28089358,  0.07890703,  0.40232944],
       [ 0.28144581,  0.0843197 ,  0.40741404],
       [ 0.28192358,  0.08966622,  0.41241521],
       [ 0.28232739,  0.09495545,  0.41733086],
       [ 0.28265633,  0.10019576,  0.42216032],
       [ 0.28291049,  0.10539345,  0.42690202],
       [ 0.28309095,  0.11055307,  0.43155375],
       [ 0.28319704,  0.11567966,  0.43611482],
       [ 0.28322882,  0.12077701,  0.44058404],
       [ 0.28318684,  0.12584799,  0.44496   ],
       [ 0.283072  ,  0.13089477,  0.44924127],
       [ 0.28288389,  0.13592005,  0.45342734],
       [ 0.28262297,  0.14092556,  0.45751726],
       [ 0.28229037,  0.14591233,  0.46150995],
       [ 0.28188676,  0.15088147,  0.46540474],
       [ 0.28141228,  0.15583425,  0.46920128],
       [ 0.28086773,  0.16077132,  0.47289909],
       [ 0.28025468,  0.16569272,  0.47649762],
       [ 0.27957399,  0.17059884,  0.47999675],
       [ 0.27882618,  0.1754902 ,  0.48339654],
       [ 0.27801236,  0.18036684,  0.48669702],
       [ 0.27713437,  0.18522836,  0.48989831],
       [ 0.27619376,  0.19007447,  0.49300074],
       [ 0.27519116,  0.1949054 ,  0.49600488],
       [ 0.27412802,  0.19972086,  0.49891131],
       [ 0.27300596,  0.20452049,  0.50172076],
       [ 0.27182812,  0.20930306,  0.50443413],
       [ 0.27059473,  0.21406899,  0.50705243],
       [ 0.26930756,  0.21881782,  0.50957678],
       [ 0.26796846,  0.22354911,  0.5120084 ],
       [ 0.26657984,  0.2282621 ,  0.5143487 ],
       [ 0.2651445 ,  0.23295593,  0.5165993 ],
       [ 0.2636632 ,  0.23763078,  0.51876163],
       [ 0.26213801,  0.24228619,  0.52083736],
       [ 0.26057103,  0.2469217 ,  0.52282822],
       [ 0.25896451,  0.25153685,  0.52473609],
       [ 0.25732244,  0.2561304 ,  0.52656332],
       [ 0.25564519,  0.26070284,  0.52831152],
       [ 0.25393498,  0.26525384,  0.52998273],
       [ 0.25219404,  0.26978306,  0.53157905],
       [ 0.25042462,  0.27429024,  0.53310261],
       [ 0.24862899,  0.27877509,  0.53455561],
       [ 0.2468114 ,  0.28323662,  0.53594093],
       [ 0.24497208,  0.28767547,  0.53726018],
       [ 0.24311324,  0.29209154,  0.53851561],
       [ 0.24123708,  0.29648471,  0.53970946],
       [ 0.23934575,  0.30085494,  0.54084398],
       [ 0.23744138,  0.30520222,  0.5419214 ],
       [ 0.23552606,  0.30952657,  0.54294396],
       [ 0.23360277,  0.31382773,  0.54391424],
       [ 0.2316735 ,  0.3181058 ,  0.54483444],
       [ 0.22973926,  0.32236127,  0.54570633],
       [ 0.22780192,  0.32659432,  0.546532  ],
       [ 0.2258633 ,  0.33080515,  0.54731353],
       [ 0.22392515,  0.334994  ,  0.54805291],
       [ 0.22198915,  0.33916114,  0.54875211],
       [ 0.22005691,  0.34330688,  0.54941304],
       [ 0.21812995,  0.34743154,  0.55003755],
       [ 0.21620971,  0.35153548,  0.55062743],
       [ 0.21429757,  0.35561907,  0.5511844 ],
       [ 0.21239477,  0.35968273,  0.55171011],
       [ 0.2105031 ,  0.36372671,  0.55220646],
       [ 0.20862342,  0.36775151,  0.55267486],
       [ 0.20675628,  0.37175775,  0.55311653],
       [ 0.20490257,  0.37574589,  0.55353282],
       [ 0.20306309,  0.37971644,  0.55392505],
       [ 0.20123854,  0.38366989,  0.55429441],
       [ 0.1994295 ,  0.38760678,  0.55464205],
       [ 0.1976365 ,  0.39152762,  0.55496905],
       [ 0.19585993,  0.39543297,  0.55527637],
       [ 0.19410009,  0.39932336,  0.55556494],
       [ 0.19235719,  0.40319934,  0.55583559],
       [ 0.19063135,  0.40706148,  0.55608907],
       [ 0.18892259,  0.41091033,  0.55632606],
       [ 0.18723083,  0.41474645,  0.55654717],
       [ 0.18555593,  0.4185704 ,  0.55675292],
       [ 0.18389763,  0.42238275,  0.55694377],
       [ 0.18225561,  0.42618405,  0.5571201 ],
       [ 0.18062949,  0.42997486,  0.55728221],
       [ 0.17901879,  0.43375572,  0.55743035],
       [ 0.17742298,  0.4375272 ,  0.55756466],
       [ 0.17584148,  0.44128981,  0.55768526],
       [ 0.17427363,  0.4450441 ,  0.55779216],
       [ 0.17271876,  0.4487906 ,  0.55788532],
       [ 0.17117615,  0.4525298 ,  0.55796464],
       [ 0.16964573,  0.45626209,  0.55803034],
       [ 0.16812641,  0.45998802,  0.55808199],
       [ 0.1666171 ,  0.46370813,  0.55811913],
       [ 0.16511703,  0.4674229 ,  0.55814141],
       [ 0.16362543,  0.47113278,  0.55814842],
       [ 0.16214155,  0.47483821,  0.55813967],
       [ 0.16066467,  0.47853961,  0.55811466],
       [ 0.15919413,  0.4822374 ,  0.5580728 ],
       [ 0.15772933,  0.48593197,  0.55801347],
       [ 0.15626973,  0.4896237 ,  0.557936  ],
       [ 0.15481488,  0.49331293,  0.55783967],
       [ 0.15336445,  0.49700003,  0.55772371],
       [ 0.1519182 ,  0.50068529,  0.55758733],
       [ 0.15047605,  0.50436904,  0.55742968],
       [ 0.14903918,  0.50805136,  0.5572505 ],
       [ 0.14760731,  0.51173263,  0.55704861],
       [ 0.14618026,  0.51541316,  0.55682271],
       [ 0.14475863,  0.51909319,  0.55657181],
       [ 0.14334327,  0.52277292,  0.55629491],
       [ 0.14193527,  0.52645254,  0.55599097],
       [ 0.14053599,  0.53013219,  0.55565893],
       [ 0.13914708,  0.53381201,  0.55529773],
       [ 0.13777048,  0.53749213,  0.55490625],
       [ 0.1364085 ,  0.54117264,  0.55448339],
       [ 0.13506561,  0.54485335,  0.55402906],
       [ 0.13374299,  0.54853458,  0.55354108],
       [ 0.13244401,  0.55221637,  0.55301828],
       [ 0.13117249,  0.55589872,  0.55245948],
       [ 0.1299327 ,  0.55958162,  0.55186354],
       [ 0.12872938,  0.56326503,  0.55122927],
       [ 0.12756771,  0.56694891,  0.55055551],
       [ 0.12645338,  0.57063316,  0.5498411 ],
       [ 0.12539383,  0.57431754,  0.54908564],
       [ 0.12439474,  0.57800205,  0.5482874 ],
       [ 0.12346281,  0.58168661,  0.54744498],
       [ 0.12260562,  0.58537105,  0.54655722],
       [ 0.12183122,  0.58905521,  0.54562298],
       [ 0.12114807,  0.59273889,  0.54464114],
       [ 0.12056501,  0.59642187,  0.54361058],
       [ 0.12009154,  0.60010387,  0.54253043],
       [ 0.11973756,  0.60378459,  0.54139999],
       [ 0.11951163,  0.60746388,  0.54021751],
       [ 0.11942341,  0.61114146,  0.53898192],
       [ 0.11948255,  0.61481702,  0.53769219],
       [ 0.11969858,  0.61849025,  0.53634733],
       [ 0.12008079,  0.62216081,  0.53494633],
       [ 0.12063824,  0.62582833,  0.53348834],
       [ 0.12137972,  0.62949242,  0.53197275],
       [ 0.12231244,  0.63315277,  0.53039808],
       [ 0.12344358,  0.63680899,  0.52876343],
       [ 0.12477953,  0.64046069,  0.52706792],
       [ 0.12632581,  0.64410744,  0.52531069],
       [ 0.12808703,  0.64774881,  0.52349092],
       [ 0.13006688,  0.65138436,  0.52160791],
       [ 0.13226797,  0.65501363,  0.51966086],
       [ 0.13469183,  0.65863619,  0.5176488 ],
       [ 0.13733921,  0.66225157,  0.51557101],
       [ 0.14020991,  0.66585927,  0.5134268 ],
       [ 0.14330291,  0.66945881,  0.51121549],
       [ 0.1466164 ,  0.67304968,  0.50893644],
       [ 0.15014782,  0.67663139,  0.5065889 ],
       [ 0.15389405,  0.68020343,  0.50417217],
       [ 0.15785146,  0.68376525,  0.50168574],
       [ 0.16201598,  0.68731632,  0.49912906],
       [ 0.1663832 ,  0.69085611,  0.49650163],
       [ 0.1709484 ,  0.69438405,  0.49380294],
       [ 0.17570671,  0.6978996 ,  0.49103252],
       [ 0.18065314,  0.70140222,  0.48818938],
       [ 0.18578266,  0.70489133,  0.48527326],
       [ 0.19109018,  0.70836635,  0.48228395],
       [ 0.19657063,  0.71182668,  0.47922108],
       [ 0.20221902,  0.71527175,  0.47608431],
       [ 0.20803045,  0.71870095,  0.4728733 ],
       [ 0.21400015,  0.72211371,  0.46958774],
       [ 0.22012381,  0.72550945,  0.46622638],
       [ 0.2263969 ,  0.72888753,  0.46278934],
       [ 0.23281498,  0.73224735,  0.45927675],
       [ 0.2393739 ,  0.73558828,  0.45568838],
       [ 0.24606968,  0.73890972,  0.45202405],
       [ 0.25289851,  0.74221104,  0.44828355],
       [ 0.25985676,  0.74549162,  0.44446673],
       [ 0.26694127,  0.74875084,  0.44057284],
       [ 0.27414922,  0.75198807,  0.4366009 ],
       [ 0.28147681,  0.75520266,  0.43255207],
       [ 0.28892102,  0.75839399,  0.42842626],
       [ 0.29647899,  0.76156142,  0.42422341],
       [ 0.30414796,  0.76470433,  0.41994346],
       [ 0.31192534,  0.76782207,  0.41558638],
       [ 0.3198086 ,  0.77091403,  0.41115215],
       [ 0.3277958 ,  0.77397953,  0.40664011],
       [ 0.33588539,  0.7770179 ,  0.40204917],
       [ 0.34407411,  0.78002855,  0.39738103],
       [ 0.35235985,  0.78301086,  0.39263579],
       [ 0.36074053,  0.78596419,  0.38781353],
       [ 0.3692142 ,  0.78888793,  0.38291438],
       [ 0.37777892,  0.79178146,  0.3779385 ],
       [ 0.38643282,  0.79464415,  0.37288606],
       [ 0.39517408,  0.79747541,  0.36775726],
       [ 0.40400101,  0.80027461,  0.36255223],
       [ 0.4129135 ,  0.80304099,  0.35726893],
       [ 0.42190813,  0.80577412,  0.35191009],
       [ 0.43098317,  0.80847343,  0.34647607],
       [ 0.44013691,  0.81113836,  0.3409673 ],
       [ 0.44936763,  0.81376835,  0.33538426],
       [ 0.45867362,  0.81636288,  0.32972749],
       [ 0.46805314,  0.81892143,  0.32399761],
       [ 0.47750446,  0.82144351,  0.31819529],
       [ 0.4870258 ,  0.82392862,  0.31232133],
       [ 0.49661536,  0.82637633,  0.30637661],
       [ 0.5062713 ,  0.82878621,  0.30036211],
       [ 0.51599182,  0.83115784,  0.29427888],
       [ 0.52577622,  0.83349064,  0.2881265 ],
       [ 0.5356211 ,  0.83578452,  0.28190832],
       [ 0.5455244 ,  0.83803918,  0.27562602],
       [ 0.55548397,  0.84025437,  0.26928147],
       [ 0.5654976 ,  0.8424299 ,  0.26287683],
       [ 0.57556297,  0.84456561,  0.25641457],
       [ 0.58567772,  0.84666139,  0.24989748],
       [ 0.59583934,  0.84871722,  0.24332878],
       [ 0.60604528,  0.8507331 ,  0.23671214],
       [ 0.61629283,  0.85270912,  0.23005179],
       [ 0.62657923,  0.85464543,  0.22335258],
       [ 0.63690157,  0.85654226,  0.21662012],
       [ 0.64725685,  0.85839991,  0.20986086],
       [ 0.65764197,  0.86021878,  0.20308229],
       [ 0.66805369,  0.86199932,  0.19629307],
       [ 0.67848868,  0.86374211,  0.18950326],
       [ 0.68894351,  0.86544779,  0.18272455],
       [ 0.69941463,  0.86711711,  0.17597055],
       [ 0.70989842,  0.86875092,  0.16925712],
       [ 0.72039115,  0.87035015,  0.16260273],
       [ 0.73088902,  0.87191584,  0.15602894],
       [ 0.74138803,  0.87344918,  0.14956101],
       [ 0.75188414,  0.87495143,  0.14322828],
       [ 0.76237342,  0.87642392,  0.13706449],
       [ 0.77285183,  0.87786808,  0.13110864],
       [ 0.78331535,  0.87928545,  0.12540538],
       [ 0.79375994,  0.88067763,  0.12000532],
       [ 0.80418159,  0.88204632,  0.11496505],
       [ 0.81457634,  0.88339329,  0.11034678],
       [ 0.82494028,  0.88472036,  0.10621724],
       [ 0.83526959,  0.88602943,  0.1026459 ],
       [ 0.84556056,  0.88732243,  0.09970219],
       [ 0.8558096 ,  0.88860134,  0.09745186],
       [ 0.86601325,  0.88986815,  0.09595277],
       [ 0.87616824,  0.89112487,  0.09525046],
       [ 0.88627146,  0.89237353,  0.09537439],
       [ 0.89632002,  0.89361614,  0.09633538],
       [ 0.90631121,  0.89485467,  0.09812496],
       [ 0.91624212,  0.89609127,  0.1007168 ],
       [ 0.92610579,  0.89732977,  0.10407067],
       [ 0.93590444,  0.8985704 ,  0.10813094],
       [ 0.94563626,  0.899815  ,  0.11283773],
       [ 0.95529972,  0.90106534,  0.11812832],
       [ 0.96489353,  0.90232311,  0.12394051],
       [ 0.97441665,  0.90358991,  0.13021494],
       [ 0.98386829,  0.90486726,  0.13689671],
       [ 0.99324789,  0.90615657,  0.1439362 ]]
cm_data2 = [[  1.46159096e-03,   4.66127766e-04,   1.38655200e-02],
       [  2.26726368e-03,   1.26992553e-03,   1.85703520e-02],
       [  3.29899092e-03,   2.24934863e-03,   2.42390508e-02],
       [  4.54690615e-03,   3.39180156e-03,   3.09092475e-02],
       [  6.00552565e-03,   4.69194561e-03,   3.85578980e-02],
       [  7.67578856e-03,   6.13611626e-03,   4.68360336e-02],
       [  9.56051094e-03,   7.71344131e-03,   5.51430756e-02],
       [  1.16634769e-02,   9.41675403e-03,   6.34598080e-02],
       [  1.39950388e-02,   1.12247138e-02,   7.18616890e-02],
       [  1.65605595e-02,   1.31362262e-02,   8.02817951e-02],
       [  1.93732295e-02,   1.51325789e-02,   8.87668094e-02],
       [  2.24468865e-02,   1.71991484e-02,   9.73274383e-02],
       [  2.57927373e-02,   1.93306298e-02,   1.05929835e-01],
       [  2.94324251e-02,   2.15030771e-02,   1.14621328e-01],
       [  3.33852235e-02,   2.37024271e-02,   1.23397286e-01],
       [  3.76684211e-02,   2.59207864e-02,   1.32232108e-01],
       [  4.22525554e-02,   2.81385015e-02,   1.41140519e-01],
       [  4.69146287e-02,   3.03236129e-02,   1.50163867e-01],
       [  5.16437624e-02,   3.24736172e-02,   1.59254277e-01],
       [  5.64491009e-02,   3.45691867e-02,   1.68413539e-01],
       [  6.13397200e-02,   3.65900213e-02,   1.77642172e-01],
       [  6.63312620e-02,   3.85036268e-02,   1.86961588e-01],
       [  7.14289181e-02,   4.02939095e-02,   1.96353558e-01],
       [  7.66367560e-02,   4.19053329e-02,   2.05798788e-01],
       [  8.19620773e-02,   4.33278666e-02,   2.15289113e-01],
       [  8.74113897e-02,   4.45561662e-02,   2.24813479e-01],
       [  9.29901526e-02,   4.55829503e-02,   2.34357604e-01],
       [  9.87024972e-02,   4.64018731e-02,   2.43903700e-01],
       [  1.04550936e-01,   4.70080541e-02,   2.53430300e-01],
       [  1.10536084e-01,   4.73986708e-02,   2.62912235e-01],
       [  1.16656423e-01,   4.75735920e-02,   2.72320803e-01],
       [  1.22908126e-01,   4.75360183e-02,   2.81624170e-01],
       [  1.29284984e-01,   4.72930838e-02,   2.90788012e-01],
       [  1.35778450e-01,   4.68563678e-02,   2.99776404e-01],
       [  1.42377819e-01,   4.62422566e-02,   3.08552910e-01],
       [  1.49072957e-01,   4.54676444e-02,   3.17085139e-01],
       [  1.55849711e-01,   4.45588056e-02,   3.25338414e-01],
       [  1.62688939e-01,   4.35542881e-02,   3.33276678e-01],
       [  1.69575148e-01,   4.24893149e-02,   3.40874188e-01],
       [  1.76493202e-01,   4.14017089e-02,   3.48110606e-01],
       [  1.83428775e-01,   4.03288858e-02,   3.54971391e-01],
       [  1.90367453e-01,   3.93088888e-02,   3.61446945e-01],
       [  1.97297425e-01,   3.84001825e-02,   3.67534629e-01],
       [  2.04209298e-01,   3.76322609e-02,   3.73237557e-01],
       [  2.11095463e-01,   3.70296488e-02,   3.78563264e-01],
       [  2.17948648e-01,   3.66146049e-02,   3.83522415e-01],
       [  2.24762908e-01,   3.64049901e-02,   3.88128944e-01],
       [  2.31538148e-01,   3.64052511e-02,   3.92400150e-01],
       [  2.38272961e-01,   3.66209949e-02,   3.96353388e-01],
       [  2.44966911e-01,   3.70545017e-02,   4.00006615e-01],
       [  2.51620354e-01,   3.77052832e-02,   4.03377897e-01],
       [  2.58234265e-01,   3.85706153e-02,   4.06485031e-01],
       [  2.64809649e-01,   3.96468666e-02,   4.09345373e-01],
       [  2.71346664e-01,   4.09215821e-02,   4.11976086e-01],
       [  2.77849829e-01,   4.23528741e-02,   4.14392106e-01],
       [  2.84321318e-01,   4.39325787e-02,   4.16607861e-01],
       [  2.90763373e-01,   4.56437598e-02,   4.18636756e-01],
       [  2.97178251e-01,   4.74700293e-02,   4.20491164e-01],
       [  3.03568182e-01,   4.93958927e-02,   4.22182449e-01],
       [  3.09935342e-01,   5.14069729e-02,   4.23720999e-01],
       [  3.16281835e-01,   5.34901321e-02,   4.25116277e-01],
       [  3.22609671e-01,   5.56335178e-02,   4.26376869e-01],
       [  3.28920763e-01,   5.78265505e-02,   4.27510546e-01],
       [  3.35216916e-01,   6.00598734e-02,   4.28524320e-01],
       [  3.41499828e-01,   6.23252772e-02,   4.29424503e-01],
       [  3.47771086e-01,   6.46156100e-02,   4.30216765e-01],
       [  3.54032169e-01,   6.69246832e-02,   4.30906186e-01],
       [  3.60284449e-01,   6.92471753e-02,   4.31497309e-01],
       [  3.66529195e-01,   7.15785403e-02,   4.31994185e-01],
       [  3.72767575e-01,   7.39149211e-02,   4.32400419e-01],
       [  3.79000659e-01,   7.62530701e-02,   4.32719214e-01],
       [  3.85228383e-01,   7.85914864e-02,   4.32954973e-01],
       [  3.91452659e-01,   8.09267058e-02,   4.33108763e-01],
       [  3.97674379e-01,   8.32568129e-02,   4.33182647e-01],
       [  4.03894278e-01,   8.55803445e-02,   4.33178526e-01],
       [  4.10113015e-01,   8.78961593e-02,   4.33098056e-01],
       [  4.16331169e-01,   9.02033992e-02,   4.32942678e-01],
       [  4.22549249e-01,   9.25014543e-02,   4.32713635e-01],
       [  4.28767696e-01,   9.47899342e-02,   4.32411996e-01],
       [  4.34986885e-01,   9.70686417e-02,   4.32038673e-01],
       [  4.41207124e-01,   9.93375510e-02,   4.31594438e-01],
       [  4.47428382e-01,   1.01597079e-01,   4.31080497e-01],
       [  4.53650614e-01,   1.03847716e-01,   4.30497898e-01],
       [  4.59874623e-01,   1.06089165e-01,   4.29845789e-01],
       [  4.66100494e-01,   1.08321923e-01,   4.29124507e-01],
       [  4.72328255e-01,   1.10546584e-01,   4.28334320e-01],
       [  4.78557889e-01,   1.12763831e-01,   4.27475431e-01],
       [  4.84789325e-01,   1.14974430e-01,   4.26547991e-01],
       [  4.91022448e-01,   1.17179219e-01,   4.25552106e-01],
       [  4.97257069e-01,   1.19379132e-01,   4.24487908e-01],
       [  5.03492698e-01,   1.21575414e-01,   4.23356110e-01],
       [  5.09729541e-01,   1.23768654e-01,   4.22155676e-01],
       [  5.15967304e-01,   1.25959947e-01,   4.20886594e-01],
       [  5.22205646e-01,   1.28150439e-01,   4.19548848e-01],
       [  5.28444192e-01,   1.30341324e-01,   4.18142411e-01],
       [  5.34682523e-01,   1.32533845e-01,   4.16667258e-01],
       [  5.40920186e-01,   1.34729286e-01,   4.15123366e-01],
       [  5.47156706e-01,   1.36928959e-01,   4.13510662e-01],
       [  5.53391649e-01,   1.39134147e-01,   4.11828882e-01],
       [  5.59624442e-01,   1.41346265e-01,   4.10078028e-01],
       [  5.65854477e-01,   1.43566769e-01,   4.08258132e-01],
       [  5.72081108e-01,   1.45797150e-01,   4.06369246e-01],
       [  5.78303656e-01,   1.48038934e-01,   4.04411444e-01],
       [  5.84521407e-01,   1.50293679e-01,   4.02384829e-01],
       [  5.90733615e-01,   1.52562977e-01,   4.00289528e-01],
       [  5.96939751e-01,   1.54848232e-01,   3.98124897e-01],
       [  6.03138930e-01,   1.57151161e-01,   3.95891308e-01],
       [  6.09330184e-01,   1.59473549e-01,   3.93589349e-01],
       [  6.15512627e-01,   1.61817111e-01,   3.91219295e-01],
       [  6.21685340e-01,   1.64183582e-01,   3.88781456e-01],
       [  6.27847374e-01,   1.66574724e-01,   3.86276180e-01],
       [  6.33997746e-01,   1.68992314e-01,   3.83703854e-01],
       [  6.40135447e-01,   1.71438150e-01,   3.81064906e-01],
       [  6.46259648e-01,   1.73913876e-01,   3.78358969e-01],
       [  6.52369348e-01,   1.76421271e-01,   3.75586209e-01],
       [  6.58463166e-01,   1.78962399e-01,   3.72748214e-01],
       [  6.64539964e-01,   1.81539111e-01,   3.69845599e-01],
       [  6.70598572e-01,   1.84153268e-01,   3.66879025e-01],
       [  6.76637795e-01,   1.86806728e-01,   3.63849195e-01],
       [  6.82656407e-01,   1.89501352e-01,   3.60756856e-01],
       [  6.88653158e-01,   1.92238994e-01,   3.57602797e-01],
       [  6.94626769e-01,   1.95021500e-01,   3.54387853e-01],
       [  7.00575937e-01,   1.97850703e-01,   3.51112900e-01],
       [  7.06499709e-01,   2.00728196e-01,   3.47776863e-01],
       [  7.12396345e-01,   2.03656029e-01,   3.44382594e-01],
       [  7.18264447e-01,   2.06635993e-01,   3.40931208e-01],
       [  7.24102613e-01,   2.09669834e-01,   3.37423766e-01],
       [  7.29909422e-01,   2.12759270e-01,   3.33861367e-01],
       [  7.35683432e-01,   2.15905976e-01,   3.30245147e-01],
       [  7.41423185e-01,   2.19111589e-01,   3.26576275e-01],
       [  7.47127207e-01,   2.22377697e-01,   3.22855952e-01],
       [  7.52794009e-01,   2.25705837e-01,   3.19085410e-01],
       [  7.58422090e-01,   2.29097492e-01,   3.15265910e-01],
       [  7.64009940e-01,   2.32554083e-01,   3.11398734e-01],
       [  7.69556038e-01,   2.36076967e-01,   3.07485188e-01],
       [  7.75058888e-01,   2.39667435e-01,   3.03526312e-01],
       [  7.80517023e-01,   2.43326720e-01,   2.99522665e-01],
       [  7.85928794e-01,   2.47055968e-01,   2.95476756e-01],
       [  7.91292674e-01,   2.50856232e-01,   2.91389943e-01],
       [  7.96607144e-01,   2.54728485e-01,   2.87263585e-01],
       [  8.01870689e-01,   2.58673610e-01,   2.83099033e-01],
       [  8.07081807e-01,   2.62692401e-01,   2.78897629e-01],
       [  8.12239008e-01,   2.66785558e-01,   2.74660698e-01],
       [  8.17340818e-01,   2.70953688e-01,   2.70389545e-01],
       [  8.22385784e-01,   2.75197300e-01,   2.66085445e-01],
       [  8.27372474e-01,   2.79516805e-01,   2.61749643e-01],
       [  8.32299481e-01,   2.83912516e-01,   2.57383341e-01],
       [  8.37165425e-01,   2.88384647e-01,   2.52987700e-01],
       [  8.41968959e-01,   2.92933312e-01,   2.48563825e-01],
       [  8.46708768e-01,   2.97558528e-01,   2.44112767e-01],
       [  8.51383572e-01,   3.02260213e-01,   2.39635512e-01],
       [  8.55992130e-01,   3.07038188e-01,   2.35132978e-01],
       [  8.60533241e-01,   3.11892183e-01,   2.30606009e-01],
       [  8.65005747e-01,   3.16821833e-01,   2.26055368e-01],
       [  8.69408534e-01,   3.21826685e-01,   2.21481734e-01],
       [  8.73740530e-01,   3.26906201e-01,   2.16885699e-01],
       [  8.78000715e-01,   3.32059760e-01,   2.12267762e-01],
       [  8.82188112e-01,   3.37286663e-01,   2.07628326e-01],
       [  8.86301795e-01,   3.42586137e-01,   2.02967696e-01],
       [  8.90340885e-01,   3.47957340e-01,   1.98286080e-01],
       [  8.94304553e-01,   3.53399363e-01,   1.93583583e-01],
       [  8.98192017e-01,   3.58911240e-01,   1.88860212e-01],
       [  9.02002544e-01,   3.64491949e-01,   1.84115876e-01],
       [  9.05735448e-01,   3.70140419e-01,   1.79350388e-01],
       [  9.09390090e-01,   3.75855533e-01,   1.74563472e-01],
       [  9.12965874e-01,   3.81636138e-01,   1.69754764e-01],
       [  9.16462251e-01,   3.87481044e-01,   1.64923826e-01],
       [  9.19878710e-01,   3.93389034e-01,   1.60070152e-01],
       [  9.23214783e-01,   3.99358867e-01,   1.55193185e-01],
       [  9.26470039e-01,   4.05389282e-01,   1.50292329e-01],
       [  9.29644083e-01,   4.11479007e-01,   1.45366973e-01],
       [  9.32736555e-01,   4.17626756e-01,   1.40416519e-01],
       [  9.35747126e-01,   4.23831237e-01,   1.35440416e-01],
       [  9.38675494e-01,   4.30091162e-01,   1.30438175e-01],
       [  9.41521384e-01,   4.36405243e-01,   1.25409440e-01],
       [  9.44284543e-01,   4.42772199e-01,   1.20354038e-01],
       [  9.46964741e-01,   4.49190757e-01,   1.15272059e-01],
       [  9.49561766e-01,   4.55659658e-01,   1.10163947e-01],
       [  9.52075421e-01,   4.62177656e-01,   1.05030614e-01],
       [  9.54505523e-01,   4.68743522e-01,   9.98735931e-02],
       [  9.56851903e-01,   4.75356048e-01,   9.46952268e-02],
       [  9.59114397e-01,   4.82014044e-01,   8.94989073e-02],
       [  9.61292850e-01,   4.88716345e-01,   8.42893891e-02],
       [  9.63387110e-01,   4.95461806e-01,   7.90731907e-02],
       [  9.65397031e-01,   5.02249309e-01,   7.38591143e-02],
       [  9.67322465e-01,   5.09077761e-01,   6.86589199e-02],
       [  9.69163264e-01,   5.15946092e-01,   6.34881971e-02],
       [  9.70919277e-01,   5.22853259e-01,   5.83674890e-02],
       [  9.72590351e-01,   5.29798246e-01,   5.33237243e-02],
       [  9.74176327e-01,   5.36780059e-01,   4.83920090e-02],
       [  9.75677038e-01,   5.43797733e-01,   4.36177922e-02],
       [  9.77092313e-01,   5.50850323e-01,   3.90500131e-02],
       [  9.78421971e-01,   5.57936911e-01,   3.49306227e-02],
       [  9.79665824e-01,   5.65056600e-01,   3.14091591e-02],
       [  9.80823673e-01,   5.72208516e-01,   2.85075931e-02],
       [  9.81895311e-01,   5.79391803e-01,   2.62497353e-02],
       [  9.82880522e-01,   5.86605627e-01,   2.46613416e-02],
       [  9.83779081e-01,   5.93849168e-01,   2.37702263e-02],
       [  9.84590755e-01,   6.01121626e-01,   2.36063833e-02],
       [  9.85315301e-01,   6.08422211e-01,   2.42021174e-02],
       [  9.85952471e-01,   6.15750147e-01,   2.55921853e-02],
       [  9.86502013e-01,   6.23104667e-01,   2.78139496e-02],
       [  9.86963670e-01,   6.30485011e-01,   3.09075459e-02],
       [  9.87337182e-01,   6.37890424e-01,   3.49160639e-02],
       [  9.87622296e-01,   6.45320152e-01,   3.98857472e-02],
       [  9.87818759e-01,   6.52773439e-01,   4.55808037e-02],
       [  9.87926330e-01,   6.60249526e-01,   5.17503867e-02],
       [  9.87944783e-01,   6.67747641e-01,   5.83286889e-02],
       [  9.87873910e-01,   6.75267000e-01,   6.52570167e-02],
       [  9.87713535e-01,   6.82806802e-01,   7.24892330e-02],
       [  9.87463516e-01,   6.90366218e-01,   7.99897176e-02],
       [  9.87123759e-01,   6.97944391e-01,   8.77314215e-02],
       [  9.86694229e-01,   7.05540424e-01,   9.56941797e-02],
       [  9.86174970e-01,   7.13153375e-01,   1.03863324e-01],
       [  9.85565739e-01,   7.20782460e-01,   1.12228756e-01],
       [  9.84865203e-01,   7.28427497e-01,   1.20784651e-01],
       [  9.84075129e-01,   7.36086521e-01,   1.29526579e-01],
       [  9.83195992e-01,   7.43758326e-01,   1.38453063e-01],
       [  9.82228463e-01,   7.51441596e-01,   1.47564573e-01],
       [  9.81173457e-01,   7.59134892e-01,   1.56863224e-01],
       [  9.80032178e-01,   7.66836624e-01,   1.66352544e-01],
       [  9.78806183e-01,   7.74545028e-01,   1.76037298e-01],
       [  9.77497453e-01,   7.82258138e-01,   1.85923357e-01],
       [  9.76108474e-01,   7.89973753e-01,   1.96017589e-01],
       [  9.74637842e-01,   7.97691563e-01,   2.06331925e-01],
       [  9.73087939e-01,   8.05409333e-01,   2.16876839e-01],
       [  9.71467822e-01,   8.13121725e-01,   2.27658046e-01],
       [  9.69783146e-01,   8.20825143e-01,   2.38685942e-01],
       [  9.68040817e-01,   8.28515491e-01,   2.49971582e-01],
       [  9.66242589e-01,   8.36190976e-01,   2.61533898e-01],
       [  9.64393924e-01,   8.43848069e-01,   2.73391112e-01],
       [  9.62516656e-01,   8.51476340e-01,   2.85545675e-01],
       [  9.60625545e-01,   8.59068716e-01,   2.98010219e-01],
       [  9.58720088e-01,   8.66624355e-01,   3.10820466e-01],
       [  9.56834075e-01,   8.74128569e-01,   3.23973947e-01],
       [  9.54997177e-01,   8.81568926e-01,   3.37475479e-01],
       [  9.53215092e-01,   8.88942277e-01,   3.51368713e-01],
       [  9.51546225e-01,   8.96225909e-01,   3.65627005e-01],
       [  9.50018481e-01,   9.03409063e-01,   3.80271225e-01],
       [  9.48683391e-01,   9.10472964e-01,   3.95289169e-01],
       [  9.47594362e-01,   9.17399053e-01,   4.10665194e-01],
       [  9.46809163e-01,   9.24168246e-01,   4.26373236e-01],
       [  9.46391536e-01,   9.30760752e-01,   4.42367495e-01],
       [  9.46402951e-01,   9.37158971e-01,   4.58591507e-01],
       [  9.46902568e-01,   9.43347775e-01,   4.74969778e-01],
       [  9.47936825e-01,   9.49317522e-01,   4.91426053e-01],
       [  9.49544830e-01,   9.55062900e-01,   5.07859649e-01],
       [  9.51740304e-01,   9.60586693e-01,   5.24203026e-01],
       [  9.54529281e-01,   9.65895868e-01,   5.40360752e-01],
       [  9.57896053e-01,   9.71003330e-01,   5.56275090e-01],
       [  9.61812020e-01,   9.75924241e-01,   5.71925382e-01],
       [  9.66248822e-01,   9.80678193e-01,   5.87205773e-01],
       [  9.71161622e-01,   9.85282161e-01,   6.02154330e-01],
       [  9.76510983e-01,   9.89753437e-01,   6.16760413e-01],
       [  9.82257307e-01,   9.94108844e-01,   6.31017009e-01],
       [  9.88362068e-01,   9.98364143e-01,   6.44924005e-01]]

viridis2 = LinearSegmentedColormap.from_list(__file__, cm_data2)
viridis = LinearSegmentedColormap.from_list(__file__, cm_data)
viridis3 = LinearSegmentedColormap.from_list(__file__, np.concatenate((cm_data2[55:-90][::-1], cm_data[20:])))

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(__file__, cdict)

    return newcmap
#
#def getAngularCorrelationFunction(ss, power, angle, beta):
#    """ angle from line-of-sight in RADIANS """ 
#    mu = np.cos(angle)
#    
#    p0 = 1
#    p2 = (3 * mu * mu - 1) / 2.0
#    p4 = (35 * mu**4 - 30 * mu * mu + 3) / 8.0
#    
#    
#    e0 = (1 + 2 * beta / 3.0 + beta * beta / 5) * er
