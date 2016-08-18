import sys
import os
from glob import glob
from math import sqrt, ceil, pi, cos
import re

import numpy as np

from astrometry.util.fits import fits_table
from astrometry.util.util import Tan
from astrometry.libkd import spherematch
from astrometry.util.starutil_numpy import arcsec2dist, arcsec2deg
import astrometry

def memusage():
    import resource
    import gc

    gc.collect()

    if len(gc.garbage):
        print 'Garbage list:'
        for obj in gc.garbage:
            print obj

    # print heapy.heap()
    #ru = resource.getrusage(resource.RUSAGE_BOTH)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    pgsize = resource.getpagesize()
    print 'Memory usage:'
    print 'page size', pgsize
    mb = int(np.ceil(ru.ru_maxrss * pgsize / 1e6))
    unit = 'MB'
    f = 1.
    if mb > 1024:
        f = 1024.
        unit = 'GB'
    print 'max rss: %.1f %s' % (mb/f, unit)
    #print 'shared memory size:', (ru.ru_ixrss / 1e6), 'MB'
    #print 'unshared memory size:', (ru.ru_idrss / 1e6), 'MB'
    #print 'unshared stack size:', (ru.ru_isrss / 1e6), 'MB'
    #print 'shared memory size:', ru.ru_ixrss
    #print 'unshared memory size:', ru.ru_idrss
    #print 'unshared stack size:', ru.ru_isrss
    procfn = '/proc/%d/status' % os.getpid()
    try:
        t = open(procfn).readlines()
        #print 'proc file:', t
        d = dict([(line.split()[0][:-1], line.split()[1:]) for line in t])
        #print 'dict:', d
        for key in ['VmPeak', 'VmSize', 'VmRSS', 'VmData', 'VmStk' ]: # VmLck, VmHWM, VmExe, VmLib, VmPTE
            #print key, ' '.join(d.get(key, []))
            #print 'd:', d
            va = d.get(key,[])
            if len(va) < 2:
                continue
            v = float(va[-2])
            unit = va[-1]
            if unit == 'kB' and v > 1024:
                unit = 'MB'
                v /= 1024.
            if unit == 'MB' and v > 1024:
                unit = 'GB'
                v /= 1024.
            print key, '%.1f %s' % (v, unit)
            
    except:
        pass



def cosd(d):
    return np.cos(np.deg2rad(d))

class Field(object):
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
    def copy(self):
        print 'Field copy(): dir', dir(self)
        d = {}
        for k,v in self.__dict__.items():
            if k.startswith('__'):
                continue
            if hasattr(v, 'copy'):
                d[k] = v.copy()
            else:
                d[k] = v
        return Field(**d)



def match_many(TT, rad, midra,middec, kdfns=None):
    # Project around central RA,Dec
    fakewcs = Tan(midra, middec, 0, 0, 0, 0, 0, 0, 0, 0)
    kds = []

    # Ugly hackery due to poor spherematch.py API: open/create vs close/free
    closekds = []
    freekds = []

    # HACK -- can we use threads here?  (Can't use multiprocessing
    # since kd-trees refer to local memory).

    print 'Building kd-trees...'
    for i,T in enumerate(TT):
        if kdfns is not None and os.path.exists(kdfns[i]):
            kd = spherematch.tree_open(kdfns[i])
            print 'Loaded kd-tree from', kdfns[i]
            closekds.append(kd)
        else:
            ok,ix,iy = fakewcs.radec2iwc(T.ra, T.dec)
            X = np.vstack((ix, iy)).T
            kd = spherematch.tree_build(X)
            freekds.append(kd)
            if kdfns is not None and kdfns[i] is not None:
                spherematch.tree_save(kd, kdfns[i])
                print 'Saved kd-tree to', kdfns[i]
        kds.append(kd)

    print 'Matching trees...'
    kdrad = arcsec2deg(rad)
    matches = []
    for i,kdi in enumerate(kds):
        for j in range(i+1, len(kds)):
            kdj = kds[j]
            I,J,d = spherematch.trees_match(kdi, kdj, kdrad)
            print 'Match', i, 'to', j, '-->', len(I)
            if len(I) == 0:
                continue
            m = Match(TT[i], TT[j], rad, I=I, J=J, dists=d)
            matches.append((i, j, m))
    print 'Freeing trees...'
    for kd in closekds:
        spherematch.tree_close(kd)
    for kd in freekds:
        spherematch.tree_free(kd)
    print 'Done'
    return matches
    

def _makematch((i,j,Ti,Tj,rad)):
    print 'Matching', i, 'to', j, '...'
    M = Match(Ti,Tj,rad)
    print 'Matching', i, 'to', j, '...',
    print 'Got', len(M.I), 'matches'
    return i,j,M


def findprimaries(TT, rad, mp):
    # start by initializing all to have primary = True
    print 'Initializing primary arrays...'
    for i,Ti in enumerate(TT):
        Ti.primary = np.ones(len(Ti), bool)

    print 'Setting up matching arguments...'
    # look at pairs of fields... (in parallel)
    ranges = []
    for Ti in TT:
        ranges.append((Ti.ra.min(), Ti.ra.max(), Ti.dec.min(), Ti.dec.max()))
    margs = []
    for i,Ti in enumerate(TT):
        irlo,irhi,idlo,idhi = ranges[i]
        for j in range(i+1, len(TT)):
            jrlo,jrhi,jdlo,jdhi = ranges[j]
            if (jrlo > irhi) or (jdlo > idhi) or (jrhi < irlo) or (jdhi < idlo):
                #print 'No overlap between', i, 'and', j, ':', ranges[i], ranges[j]
                continue
            Tj = TT[j]
            margs.append((i, j, Ti, Tj, rad))
    print 'Set up matching arguments...', len(margs)
    print 'Matching...'
    matches = mp.map(_makematch, margs)
    print 'Matching done'

    for i,j,M in matches:
        if len(M.I) == 0:
            continue
        Tj = TT[j]
        # arbitrarily declare the lower-index field to be 'primary'
        # Does this simple approach work?
        # if A->B->C, A is primary, B gets set un-primary, C gets set un-primary,
        # and if later A->C, C gets un-primary again, but that's fine.
        Tj.primary[M.J] = False

    return matches


def write_update_script(wcsfn, outfn, scriptfn, f=sys.stderr, inext=0, outext=0,
                        verbose=True):
    hdr0 = pyfits.open(wcsfn)[inext].header
    hdr1 = pyfits.open(outfn)[outext].header
    if verbose:
        print >>sys.stderr, "\n# Old file: %s\n# New file: %s" % (wcsfn, outfn)
    for k,v in hdr1.items():
        s = None
        v0 = hdr0.get(k, None)
        if k in ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND']:
            s = '# Key "%s" ignored' % k
        elif v == v0:
            # skip
            s = '# Key "%s" unchanged: "%s"' % (k, str(v))
        else:
            if verbose:
                print 'key', k, 'val', v, 'type', type(v)
            if type(v) is float:
                sv = '%.10G' % v
            else:
                sv = str(v)
            s = 'hedit %s %s %s ver- add+' % (scriptfn, k, sv)
        f.write(s + '\n')
    f.flush()


def magcuts(T, mag1cut=None, mag2cut=None, copy=True):
    if mag1cut is not None and mag2cut is not None:
        T = T[(T.mag1 < mag1cut) * (T.mag2 < mag2cut)]
    elif mag1cut is not None:
        T = T[T.mag1 < mag1cut]
    elif mag2cut is not None:
        T = T[T.mag2 < mag2cut]
    elif copy:
        T = T.copy()
    return T

def getwcsoutline(wcs, w=None, h=None):
    if w is None:
        # Tan
        if hasattr(wcs, 'imagew'):
            w = wcs.imagew
        # Sip
        if hasattr(wcs, 'wcstan') and hasattr(wcs.wcstan, 'imagew'):
            w = wcs.wcstan.imagew
    if h is None:
        if hasattr(wcs, 'imageh'):
            h = wcs.imageh
        # Sip
        if hasattr(wcs, 'wcstan') and hasattr(wcs.wcstan, 'imageh'):
            h = wcs.wcstan.imageh
    rd = np.array([wcs.pixelxy2radec(x,y) for x,y in
                   [(1,1), (w,1), (w,h), (1,h), (1,1)]])
    return (rd[:,0], rd[:,1])

def set_fp_err():
    #np.seterr(all='raise')
    np.seterr(all='warn')

def plotmatchdisthist(M, mas=True, nbins=100, doclf=True, color='b', **kwa):
    import pylab as plt
    if doclf:
        plt.clf()
    R = np.sqrt(M.dra_arcsec**2 + M.ddec_arcsec**2)
    if mas:
        R *= 1000.
        rng = [0, M.rad*1000.]
    else:
        rng = [0, M.rad]
    print 'Match distances: median', np.median(R), 'arcsec'
    n,b,p = plt.hist(R, nbins, range=rng, histtype='step', color=color, **kwa)
    if mas:
        plt.xlabel('Match distance (mas)')
    else:
        plt.xlabel('Match distance (arcsec)')
    plt.xlim(*rng)
    return n,b,p

def findFile(fn, path=''):
    if path == '':
        g  = fn
    else:
        g = os.path.join(path, fn)
    G = glob(g)
    if len(G) == 0:
        print 'Could not find file "%s" in path "%s"' % (fn,path)
        return None
    if len(G) > 1:
        print 'Warning: found', len(G), 'matches to filename pattern', g
        print 'Keeping', G[0]
    return G[0]

def get_bboxes(wcs):
    bboxes = []
    for wcsi in wcs:
        r,d = [],[]
        W,H = wcsi.imagew, wcsi.imageh
        for x,y in [ (0,0), (W,0), (W,H), (0,H), (0,0) ]:
            rr,dd = wcsi.pixelxy2radec(x,y)
            r.append(rr)
            d.append(dd)
        bboxes.append((r,d))
    return bboxes

def resetplot():
    import matplotlib
    import pylab as plt
    kw = {}
    for p in ['bottom', 'top', 'left', 'right', 'hspace', 'wspace']:
        kw[p] = matplotlib.rcParams['figure.subplot.' + p]
    plt.subplots_adjust(**kw)
                                    

def plotaffinegrid(affines, exag=1e3, affineOnly=True, R=0.025, tpre='', bboxes=None):
    import pylab as plt
    NR = 3
    NC = int(ceil(len(affines)/3.))
    #R = 0.025 # 1.5 arcmin
    #for (exag,affonly) in [(1e2, False), (1e3, True), (1e4, True)]:
    plt.clf()
    for i,aff in enumerate(affines):
        plt.subplot(NR, NC, i+1)
        dl = aff.refdec - R
        dh = aff.refdec + R
        rl = aff.refra  - R / aff.rascale
        rh = aff.refra  + R / aff.rascale
        RR,DD = np.meshgrid(np.linspace(rl, rh, 11),
                            np.linspace(dl, dh, 11))
        plotaffine(aff, RR.ravel(), DD.ravel(), exag=exag, affineOnly=affineOnly,
                   doclf=False,
                   units='dots', width=2, headwidth=2.5, headlength=3, headaxislength=3)
        if bboxes is not None:
            for bb in bboxes:
                plt.plot(*bb, linestyle='-', color='0.5')
            plt.plot(*bboxes[i], linestyle='-', color='k')
        setRadecAxes(rl,rh,dl,dh)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.yticks([])
        plt.title('field %i' % (i+1))
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
    if affineOnly:
        tt = tpre + 'Affine part of transformations'
    else:
        tt = tpre + 'Transformations'
    plt.suptitle(tt + ' (x %g)' % exag)


def plotaffine(aff, RR, DD, exag=1000, affineOnly=False, doclf=True, **kwargs):
    import pylab as plt
    if doclf:
        plt.clf()
    if affineOnly:
        dr,dd = aff.getAffineOffset(RR, DD)
    else:
        rr,dd = aff.apply(RR, DD)
        dr = rr - RR
        dd = dd - DD
    #plt.plot(RR, DD, 'r.')
    #plt.plot(RR + dr*exag, DD + dd*exag, 'bx')
    plt.quiver(RR, DD, exag*dr, exag*dd,
               angles='xy', scale_units='xy', scale=1,
               pivot='middle', color='b', **kwargs)
               #pivot='tail'
    ax = plt.axis()
    plt.plot([aff.getReferenceRa()], [aff.getReferenceDec()], 'r+', mew=2, ms=5)
    plt.axis(ax)
    esuf = ''
    if exag != 1.:
        esuf = ' (x %g)' % exag
    plt.title('Affine transformation found' + esuf)


def mahalanobis_distsq(X, mu, C):
    #print 'X', X.shape
    #print 'mu', mu.shape
    #print 'C', C.shape
    assert(C.shape == (2,2))
    det = C[0,0]*C[1,1] - C[0,1]*C[1,0]
    Cinv = np.array([[ C[1,1]/det, -C[0,1]/det],
             [-C[1,0]/det,  C[0,0]/det]])
    #print 'Cinv', Cinv.shape
    #print 'Cinv * C:', np.dot(Cinv, C)
    #print 'C * Cinv:', np.dot(C, Cinv)
    d = X - mu
    #print 'd', d.shape
    Cinvd = np.dot(Cinv, d.T)
    #print 'Cinvd', Cinvd.shape
    M = np.sum(d * Cinvd.T, axis=1)
    #print 'M', M.shape
    return M

def gauss2d_from_M2(M2, C):
    M = -0.5 * M2
    rtn = np.zeros_like(M)
    I = (M > -700)  # -> ~ 1e-304
    det = C[0,0]*C[1,1] - C[0,1]*C[1,0]
    #print 'det', det
    rtn[I] = 1./(2.*pi*sqrt(abs(det))) * np.exp(M[I])
    return rtn

def gauss2d_cov(X, mu, C):
    M2 = mahalanobis_distsq(X, mu, C)
    return gauss2d_from_M2(M2, C)

def em_cov_step(X, mu, C, background, B, Creg=1e-6):
    #print '   em_cov_step: mu', mu, 'cov', C.ravel(), 'background fraction', B
    # E:
    # fg = p( Y, Z=f | theta ) = p( Y | Z=f, theta ) p( Z=f | theta )
    fg = gauss2d_cov(X, mu, C) * (1. - B)
    # bg = p( Y, Z=b | theta ) = p( Y | Z=b, theta ) p( Z=b | theta )
    bg = background * B
    assert(all(np.isfinite(fg)))
    assert(all(np.isfinite(np.atleast_1d(bg))))
    assert(all(fg >= 0))
    assert(bg >= 0)
    # normalize:
    # fore = p( Z=f | Y, theta )
    fore = fg / (fg + bg)
    # back = p( Z=b | Y, theta )
    back = bg / (fg + bg)
    assert(all(np.isfinite(fore)))
    assert(all(np.isfinite(back)))
    assert(all(fore >= 0))
    assert(all(fore <= 1))
    assert(all(back >= 0))
    assert(all(back <= 1))
    # M:
    # maximize mu, C:
    if np.sum(fore) == 0:
        print 'em_cov_step: no foreground weight'
        return None
    mu = np.sum(fore[:,np.newaxis] * X, axis=0) / np.sum(fore)

    #print 'X', X.shape
    #print 'mu', mu.shape
    dx = (X - mu)[:,0]
    dy = (X - mu)[:,1]
    #print 'dx', dx.shape
    #print 'dy', dy.shape
    C = np.zeros((2,2))
    #print 'C', C.shape

    # this produced underflow in the numerator multiply once...
    np.seterr(all='warn')
    #try:
    C[0,0] = np.sum(fore * (dx**2)) / np.sum(fore)
    #except:
    #print 'fore:', fore
    #   print 'dx:', dx
    #   print 'sum fore:', np.sum(fore)
    #   print 'numerator:', np.sum(fore * (dx**2))
        
    C[1,0] = C[0,1] = np.sum(fore * (dx*dy)) / np.sum(fore)
    C[1,1] = np.sum(fore * (dy**2)) / np.sum(fore)

    # regularize... we are working in arcsec here; we add in a small
    # regularizer to the covariance.
    C += np.eye(2) * Creg**2

    if np.linalg.det(C) < 1e-30:
        print '  -> det(C) =', np.linalg.det(C), ', bailing out.'
        return None

    #print 'mu', mu
    #print 'C', C.ravel()
    # maximize B.
    # B = p( Z=b | theta )
    B = np.mean(back)
    # avoid multiplying 0 * -inf = NaN
    I = (fg > 0)
    lfg = np.zeros_like(fg)
    lfg[I] = np.log(fg[I])
    lbg = np.log(bg * np.ones_like(fg))
    lbg[np.flatnonzero(np.isfinite(lbg) == False)] = 0.
    # Total expected log-likelihood
    Q = np.sum(fore*lfg + back*lbg)
    return (mu, C, B, Q, fore)


def gauss2d(X, mu, sigma):
    # single-component:
    # 
    # # prevent underflow in exp
    # e = -np.sum((X - mu)**2, axis=1) / (2.*sigma**2)
    # rtn = np.zeros_like(e)
    # I = (e > -700)  # -> ~ 1e-304
    # rtn[I] = 1./(2.*pi*sigma**2) * np.exp(e[I])
    # return rtn

    N,two = X.shape
    assert(two == 2)
    C,two = mu.shape
    assert(two == 2)
    rtn = np.zeros((N,C))
    for c in range(C):
        e = -np.sum((X - mu[c,:])**2, axis=1) / (2.*sigma[c]**2)
        I = (e > -700)  # -> ~ 1e-304
        rtn[I,c] = 1./(2.*pi*sigma[c]**2) * np.exp(e[I])
    return rtn


def em_step(X, weights, mu, sigma, background, B):
    '''
    mu: shape (C,2) or (2,)
    sigma: shape (C,) or scalar
    weights: shape (C,) or 1.
    C: number of Gaussian components

    X: (N,2)
    '''
    mu_orig = mu

    mu = np.atleast_2d(mu)
    sigma = np.atleast_1d(sigma)
    weights = np.atleast_1d(weights)
    weights /= np.sum(weights)
    
    #print 'em_step: X', X.shape, 'mu', mu.shape, 'sigma', sigma.shape, 'background', background, 'B', B
    print '    em_step: weights', weights, 'mu', mu, 'sigma', sigma, 'background fraction', B
    # E:
    # fg = p( Y, Z=f | theta ) = p( Y | Z=f, theta ) p( Z=f | theta )
    fg = gauss2d(X, mu, sigma) * (1. - B) * weights
    # fg shape is (N,C)
    # bg = p( Y, Z=b | theta ) = p( Y | Z=b, theta ) p( Z=b | theta )
    bg = background * B
    assert(all(np.isfinite(fg.ravel())))
    assert(all(np.isfinite(np.atleast_1d(bg))))
    # normalize:

    sfg = np.sum(fg, axis=1)
    # fore = p( Z=f | Y, theta )
    fore = fg / (sfg + bg)[:,np.newaxis]
    # back = p( Z=b | Y, theta )
    back = bg / (sfg + bg)
    assert(all(np.isfinite(fore.ravel())))
    assert(all(np.isfinite(back.ravel())))

    # M:
    # maximize mu, sigma:
    #mu = np.sum(fore[:,np.newaxis] * X, axis=0) / np.sum(fore)
    mu = np.dot(fore.T, X) / np.sum(fore)
    # 2.*sum(fore) because X,mu are 2-dimensional.
    #sigma = np.sqrt(np.sum(fore[:,np.newaxis] * (X - mu)**2) / (2.*np.sum(fore)))
    C = len(sigma)
    for c in range(C):
        sigma[c] = np.sqrt(np.sum(fore[:,c][:,np.newaxis] * (X - mu[c,:])**2) / (2. * np.sum(fore[:,c])))
    #print 'mu', mu, 'sigma', sigma
    if np.min(sigma) == 0:
        return (mu, sigma, B, -1e6, np.zeros(len(X)))
    assert(np.all(sigma > 0))

    # maximize weights:
    weights = np.mean(fore, axis=0)
    weights /= np.sum(weights)

    # maximize B.
    # B = p( Z=b | theta )
    B = np.mean(back)

    # avoid multiplying 0 * -inf = NaN
    I = (fg > 0)
    lfg = np.zeros_like(fg)
    lfg[I] = np.log(fg[I])

    lbg = np.log(bg * np.ones_like(fg))
    lbg[np.flatnonzero(np.isfinite(lbg) == False)] = 0.

    # Total expected log-likelihood
    Q = np.sum(fore*lfg + back[:,np.newaxis]*lbg)

    if len(mu_orig.shape) == 1:
        return (1., mu[0,:], sigma[0], B, Q, fore[0,:])

    return (weights, mu, sigma, B, Q, fore)

class Alignment(object):
    '''
    A class to represent the alignment between two catalogs.
    '''
    def __init__(self, tableA, tableB, searchradius=1.,
                 histbins=21, cov=True, cutrange=None, match=None,
                 maxB=None, minEMsteps=5,
                 initsigma=None, rascale=None,
                 ngaussians=1):
        '''
        

        self.match.I: matched pairs
        self.subset : region around the peak in dRA,Ddec
        - bool, len(subset) == len(match.I)
        - sum(subset) == len(fore)
        self.fore: float, foreground probabilities, for the "subset" objects
        '''
        self.TA = tableA
        self.TB = tableB
        if rascale is None:
            d = np.mean(tableA.dec)
            #print 'Alignment: using mean Dec', d
            self.rascale = cosd(d)
            #print '-> rascale', self.rascale
        self.searchradius = searchradius
        self.histbins = histbins
        self.cov = cov
        self.ngauss = ngaussians
        self.cutrange = cutrange
        self.match = match
        self.maxB = maxB
        self.minsteps = minEMsteps
        self.initsigma = initsigma

    def getEllipseString(self, fmt='%.1f', mas=True):
        sigs = self.getEllipseSize()
        if mas:
            sigs[0] *= 1000.
            sigs[1] *= 1000.
        s = ((fmt + ' x ' + fmt) % (sigs[0], sigs[1]))
        if mas:
            s += ' mas'
        else:
            s += ' arcsec'
        return s

    # In arcsec
    def getEllipseSize(self):
        if self.cov:
            U,S,V = np.linalg.svd(self.C)
            eigs = np.sqrt(S)
            return eigs
        else:
            if self.ngauss == 1:
                return [self.sigma, self.sigma]
            else:
                return [self.sigma[0], self.sigma[0]]

    def getContours(self, nsigma=1, steps=100, c=0):
        if self.cov:
            if not hasattr(self, 'C'):
                return None
            C = self.C
            U,S,V = np.linalg.svd(C)
            angles = np.linspace(0, 2.*pi, steps)
            xy = np.vstack((np.cos(angles), np.sin(angles)))
            sxy = np.sqrt(S[:,np.newaxis]) * xy * nsigma
            usxy = np.dot(U, sxy)
            return (self.mu[0] + usxy[0,:], self.mu[1] + usxy[1,:])
        else:
            angles = np.linspace(0, 2.*pi, steps)
            std = self.sigma
            if self.ngauss == 1:
                return (self.mu[0] + self.sigma * nsigma * np.cos(angles),
                        self.mu[1] + self.sigma * nsigma * np.sin(angles))
            else:
                return (self.mu[c,0] + self.sigma[c] * nsigma * np.cos(angles),
                        self.mu[c,1] + self.sigma[c] * nsigma * np.sin(angles))

    def getMatchWeights(self):
        W = np.zeros_like(self.match.dra_arcsec)
        W[self.subset] = self.fore
        return W

    def getNfore(self):
        return sum(self.fore)

    def arcsecshift(self):
        return self.mu.copy()

    def getshift(self):
        # from arcsec back to deg.
        return np.array([self.mu[0]/self.rascale, self.mu[1]])/3600.

    def findMatches(self, **kwargs):
        # For brick 21, fields 3-4, the shift is about 1"
        rad = self.searchradius
        #print 'Matching with radius', rad, 'arcsec...'
        M = Match(self.TA, self.TB, rad, **kwargs)
        #print 'Found %i matches' % (len(M.I))
        self.match = M

    def shift(self):
        rad = self.searchradius
        if not hasattr(self, 'match') or self.match is None:
            self.findMatches(rascale=self.rascale)
        M = self.match
        if len(M.I) == 0:
            return None

        # Assume a Gaussian peak plus flat background.
        # Estimate peak location with EM.
        dra,ddec = M.dra_arcsec, M.ddec_arcsec
        # histogram to get a first estimate of the peak.
        bb = np.linspace(-rad, rad, self.histbins+1)
        H,xe,ye = np.histogram2d(dra, ddec, bins=(bb,bb))
        # peak
        mx = np.argmax(H)
        xmax = xe[int(mx / (len(ye)-1))] + 0.5*(xe[1]-xe[0])
        ymax = ye[int(mx % (len(ye)-1))] + 0.5*(ye[1]-ye[0])

        mu = np.array([xmax,ymax])
        if self.initsigma is not None:
            sigma = self.initsigma
        else:
            sigma = (xe[1]-xe[0])/2.
        # Background fraction
        B = 0.5
        lastQ = None

        # Cut to matches within a small distance of the peak.
        if self.cutrange is None:
            # default: two bins (ie, keep radius of (2/histbins ~ 10% of search radius))
            #(xe[2]-xe[0])
            # NO, make it bigger!
            cutrange = self.cutrange = rad/2.
        else:
            cutrange = self.cutrange
        assert(cutrange <= self.searchradius)
        self.cutcenter = mu.copy()
        # Re-center if the cut circle goes outside the search circle.
        # (otherwise the EM is biased because not enough background matches are observed)
        if sqrt(sum(self.cutcenter**2)) + cutrange > self.searchradius:
            ccr = sqrt(sum(self.cutcenter**2))
            newccr = self.searchradius - cutrange
            self.cutcenter *= (newccr / ccr)

        X = np.vstack((dra,ddec)).T
        I = (np.sum((X - self.cutcenter)**2, axis=1) < cutrange**2)
        X = X[I,:]
        background = 1/(pi*cutrange**2)

        self.subset = I
        W = 1.

        if self.cov:
            C = np.array([[sigma**2, 0],[0, sigma**2]])
            for i in range(101):
                if self.maxB is not None:
                    B = min(self.maxB, B)
                EM = em_cov_step(X, mu, C, background, B)
                if EM is None:
                    
                    return None
                (mu, C, B, Q, fore)  = EM
                #print 'Q', Q
                if lastQ is not None and Q - lastQ < 1 and i >= (self.minsteps-1):
                    #print 'Q did not improve enough, bailing'
                    break
                lastQ = Q
            nil = em_cov_step(X, mu, C, background, B)
            if self.maxB is not None:
                B = min(self.maxB, B)
            fore = nil[-1]
            self.C = C

        else:
            if self.ngauss > 1:
                mu = mu[np.newaxis,:].repeat(self.ngauss, axis=0)
                sigma = sigma * 1.5 ** np.arange(self.ngauss)
                W = np.ones(self.ngauss) / float(self.ngauss)

            for i in range(101):
                if self.maxB is not None:
                    B = min(self.maxB, B)
                (W, mu, sigma, B, Q, fore) = em_step(X, W, mu, sigma, background, B)
                #print '  Q', Q
                if lastQ is not None and Q - lastQ < 1:
                    break
                lastQ = Q
                if np.min(sigma) == 0:
                    break
    
            # assert(sigma < 0.1)
            # print 'Found match sigma:', sigma
            if np.min(sigma) == 0:
                print 'EM failed'
                return None

            # em_step returns the foreground/background assignments of the *previous*
            # settings... here we grab the assignments for the final step.
            if self.maxB is not None:
                B = min(self.maxB, B)
            nil = em_step(X, W, mu, sigma, background, B)
            fore = nil[-1]
            # this is used for getMatchWeights(), so sum the foreground components.
            fore = np.sum(fore, axis=1)
            self.sigma = sigma
    
        self.H = H
        self.xe, self.ye = xe, ye
        # the EM data
        self.X = X
        self.fore = fore
        self.mu = mu
        self.weights = W
        if self.maxB is not None:
            B = min(self.maxB, B)
        self.bg = background * B
        self.B = B
        return True

    def getModel(self, X, Y):
        mod = np.zeros_like(X)
        mod += self.bg
        XY = np.vstack((X,Y)).T
        if self.cov:
            mod += (1. - self.B) * gauss2d_cov(XY, self.mu, self.C)
        else:
            mod += (1. - self.B) * gauss2d(XY, self.mu, self.sigma)
        return mod


def plotresids3(Tme, Tother, M, **kwargs):
    ra  = (Tme.ra [M.I] + Tother.ra[M.J] )/2.
    dec = (Tme.dec[M.I] + Tother.dec[M.J])/2.
    dra  = M.dra_arcsec
    ddec = M.ddec_arcsec
    plotresids2(ra, dec, dra, ddec, **kwargs)

def plotresids(Tme, M, **kwargs):
    ra = Tme.ra[M.I]
    dec = Tme.dec[M.I]
    dra  = M.dra_arcsec
    ddec = M.ddec_arcsec
    plotresids2(ra, dec, dra, ddec, **kwargs)

def plotresids2(ra, dec, dra, ddec, title=None, mas=True, bins=200,
                dralabel=None, ddeclabel=None, ralabel=None, declabel=None,
                dlim=None, doclf=True, **kwargs):
    import pylab as plt
    from astrometry.util.plotutils import plothist
    scale = 1.
    if mas:
        scale = 1000.
        dralab = 'dRA (mas)'
        ddeclab = 'dDec (mas)'
    else:
        dralab = 'dRA (arcsec)'
        ddeclab = 'dDec (arcsec)'

    ra_kwargs = kwargs.copy()
    dec_kwargs = kwargs.copy()

    if dlim is not None and not 'range' in kwargs:
        rarange = (ra.min(), ra.max())
        decrange = (dec.min(), dec.max())
        drange = (-dlim, dlim)
        ra_kwargs['range'] = (rarange, drange)
        dec_kwargs['range'] = (decrange, drange)

    plt.clf()
    plt.subplot(2,2,1)
    plothist(ra, dra * scale, bins, doclf=False, docolorbar=False, **ra_kwargs)
    if dlim is not None:
        plt.ylim(-dlim, dlim)
    #plt.xlabel('RA (deg)')
    if dralabel is not None:
        plt.ylabel(dralabel)
    else:
        plt.ylabel(dralab)
    # flip RA
    ax = plt.axis()
    raticks,ratlabs = plt.xticks()
    if len(raticks) > 3:
        # FIXME -- might want to choose between the options....
        raticks = raticks[::2]
        plt.xticks(raticks)
    plt.axis([ax[1],ax[0],ax[2],ax[3]])

    plt.subplot(2,2,2)
    plothist(dec, dra * scale, bins, doclf=False, docolorbar=False, **dec_kwargs)
    if dlim is not None:
        plt.ylim(-dlim, dlim)
    #plt.xlabel('Dec (deg)')
    #plt.ylabel(dralab)
    ax = plt.axis()
    decticks,dectlabs = plt.xticks()
    if len(decticks) > 3:
        decticks = decticks[::2]
        plt.xticks(decticks)
    plt.axis(ax)

    plt.subplot(2,2,3)
    plothist(ra, ddec * scale, bins, doclf=False, docolorbar=False, **ra_kwargs)
    if dlim is not None:
        plt.ylim(-dlim, dlim)
    if ralabel is not None:
        plt.xlabel(ralabel)
    else:
        plt.xlabel('RA (deg)')
    if ddeclabel is not None:
        plt.ylabel(ddeclabel)
    else:
        plt.ylabel(ddeclab)
    # flip RA
    ax = plt.axis()
    plt.xticks(raticks)
    plt.axis([ax[1],ax[0],ax[2],ax[3]])

    plt.subplot(2,2,4)
    plothist(dec, ddec * scale, bins, doclf=False, docolorbar=False, **dec_kwargs)
    if dlim is not None:
        plt.ylim(-dlim, dlim)
    if declabel is not None:
        plt.xlabel(declabel)
    else:
        plt.xlabel('Dec (deg)')
    #plt.ylabel(ddeclab)
    ax = plt.axis()
    plt.xticks(decticks)
    plt.axis(ax)
    if title is not None:
        plt.suptitle(title)


class Affine(object):
    @staticmethod
    def fromTable(tab):
        '''
        Returns: a list of Affine objects
        Given: a FITS table.
        '''
        ### FIXME -- polynomial!
        affines = [ Affine(tab.dra[i], tab.ddec[i],
                           [ tab.tra_dra[i], tab.tra_ddec[i],
                             tab.tdec_dra[i], tab.tdec_ddec[i] ],
                           tab.refra[i], tab.refdec[i])
                    for i in range(len(tab)) ]
        return affines

    @staticmethod
    def toTable(affines, T=None):
        ### FIXME -- polynomial!
        if T is None:
            T = tabledata()
        N = len(affines)
        T.dra  = [ a.getShiftDeg()[0] for a in affines ]
        T.ddec = [ a.getShiftDeg()[1] for a in affines ]
        T.tra_dra   = [ a.getAffine(0)  for a in affines ]
        T.tra_ddec  = [ a.getAffine(1)  for a in affines ]
        T.tdec_dra  = [ a.getAffine(2)  for a in affines ]
        T.tdec_ddec = [ a.getAffine(3)  for a in affines ]
        T.refra  = [ a.getReferenceRadec()[0] for a in affines ]
        T.refdec = [ a.getReferenceRadec()[1] for a in affines ]

        a0 = affines[0]
        if ((a0.refxy is not None) and
            (a0.cdmatrix is not None) and
            (a0.sipterms is not None)):

            T.sip_refxy = np.array([ a.refxy for a in affines ])
            T.sip_cd = np.array([ a.cdmatrix for a in affines ])
            T.sip_terms = np.array([ a.sipterms for a in affines ])

        return T


    '''
    dra, ddec: in degrees; not isotropic

    [ r* ] = r + sra  + [1/rascale] * [ Tra_ra   Tra_dec  ] * [ (r - r_ref) * rascale ]
    [ d* ] = d + sdec + [1        ]   [ Tdec_ra  Tdec_dec ]   [ (d - d_ref)           ]
    
    T is [ Tra_ra, Tra_dec, Tdec_ra, Tdec_dec ]; isotropic
    '''
    def __init__(self, dra=0., ddec=0.,
                 T=[0,0,0,0],
                 refra=0, refdec=0,
                 # This was an earlier attempt to do polynomial distortion corrections
                 # in RA,Dec space rather than in pixel space as in SIP.
                 rapoly=None, decpoly=None,
                 # These are the SIP values
                 refxy=None, cdmatrix=None, sipterms=None
                 ):
        self.dra = dra
        self.ddec = ddec
        self.T = T
        if refra is not None and refdec is not None:
            self.setReferenceRadec(refra, refdec)
        self.rapoly  = rapoly
        self.decpoly = decpoly

        self.refxy = refxy
        self.cdmatrix = cdmatrix
        self.sipterms = sipterms

    def copy(self):
        return Affine(self.dra, self.ddec, [self.T[i] for i in range(4)],
                      self.refra, self.refdec, self.rapoly, self.decpoly,
                      self.refxy, self.cdmatrix, self.sipterms)

    def averageWith(self, other):
        '''
        Returns a *NEW* affine that is the average of this with *other*.
        '''
        assert(other.refra == self.refra)
        assert(other.refdec == self.refdec)
        assert(other.refxy == self.refxy)
        assert(other.cdmatrix == self.cdmatrix)

        sipterms = None
        assert((self.sipterms is None and other.sipterms is None) or
               (self.sipterms is not None and other.sipterms is not None))
        if self.sipterms is not None:
            sipterms = (np.array(self.sipterms) + np.array(other.sipterms)) / 2.

        return Affine((np.array(self.dra) + np.array(other.dra)) / 2.,
                      (np.array(self.ddec) + np.array(other.ddec)) / 2.,
                      (np.array(self.T) + np.array(other.T)) / 2.,
                      self.refra, self.refdec, self.rapoly, self.decpoly,
                      self.refxy, self.cdmatrix, sipterms)

    def __repr__(self):
        return str(self)

    def __str__(self):
        T = self.T
        dra_arcsec,ddec_arcsec = self.getShiftArcsec()
        s = ('Affine: shift %g,%g arcsec, T [ %g, %g, %g, %g ], ref point (%g,%g)' %
             (dra_arcsec, ddec_arcsec,
              T[0], T[1], T[2], T[3], self.refra, self.refdec))
        if self.rapoly is not None:
            s += ', rapoly ' + str(self.rapoly) + ', decpoly ' + str(self.decpoly)
        if self.sipterms is not None:
            s += ', SIP terms ' + str(self.sipterms)
        return s
    
    def getApproxRotation(self):
        # do an SVD to express T as a rotation matrix.
        M = np.array([[self.T[0]+1., self.T[1]], [self.T[2], self.T[3]+1.]])
        U,S,V = np.linalg.svd(M)
        print 'approx rotation:'
        print 'M='
        print M
        print 'U='
        print U
        print 'V='
        print V
        print 'S='
        print S
        r1 = np.rad2deg(np.arctan2(U[0,1], U[0,0]))
        r2 = np.rad2deg(np.arctan2(V[0,1], V[0,0]))
        print 'r1', r1
        print 'r2', r2
        print 'rotation', r1-r2
        return r1-r2

    def changeReferencePoint(self, newrefra, newrefdec):
        #print 'Changing reference point:'
        #print self
        #print 'to new ref point', (newrefra, newrefdec)
        S = self.rascale
        Snew = cosd(newrefdec)
        dR = newrefra  - self.refra
        dD = newrefdec - self.refdec
        T = self.T
        # Push the reference-point change through the affine...
        sr = ((dR * S) * T[0] + dD * T[1]) / S
        sd =  (dR * S) * T[2] + dD * T[3]
        self.dra  += sr
        self.ddec += sd
        # Push the (very slight) scale change into the matrix.
        # T[0] is multiplied and divide, so it cancels
        # T[1] sees the "output" effect
        # T[2] sees the "input" effect
        # T[3] is unaffected
        T[2] *= (S / Snew)
        T[1] *= (Snew / S)
        self.refra = newrefra
        self.refdec = newrefdec
        self.rascale = cosd(newrefdec)

    def setRascale(self, S):
        self.rascale = S

    def getRascale(self):
        return self.rascale

    def setAffine(self, Trara, Tradec, Tdecra, Tdecdec):
        self.T = [ Trara, Tradec, Tdecra, Tdecdec ]

    def getAffine(self, i):
        return self.T[i]

    def setRotation(self, rot, smallangle=True):
        '''
        Rotation angle in degrees
        '''
        rad = np.deg2rad(rot)
        if smallangle:
            # bring rad close to zero.
            rad = np.fmod(rad, 2.*pi)
            if rad > pi:
                rad -= 2.*pi
            if rad < -pi:
                rad += 2.*pi
            self.T = [ 0., -rad, rad, 0. ]
        else:
            cr = np.cos(rad)
            sr = np.sin(rad)
            self.T = [ cr - 1, -sr, sr, cr - 1 ]

    def applyScale(self, scale):
        #self.T = [self.T[i] * (scale-1.) for i in range(4)]
        #other = Affine(refra=self.refra, refdec=self.refdec)
        #other.setScale(scale)
        #self.add(other)
        O1, O2, O3, O4 = self.T
        T1, T2, T3, T4 = (scale-1., 0., 0., scale-1.)
        self.T = [ T1 + O1 + O1*T1 + O2*T3,
                   T2 + O2 + O1*T2 + O2*T4,
                   T3 + O3 + O3*T1 + O4*T3,
                   T4 + O4 + O3*T2 + O4*T4 ]

    ### NOTE, this overrides any rotation!
    def setScale(self, scale):
        #self.T = [scale-1.,scale-1.,scale-1.,scale-1.]
        self.T = [scale-1., 0., 0., scale-1.]

    def setShift(self, dra, ddec):
        self.dra = dra
        self.ddec = ddec

    def getReferenceRadec(self):
        return (self.refra, self.refdec)

    def getReferenceRa(self):
        return self.refra
    def getReferenceDec(self):
        return self.refdec

    def setReferenceRadec(self, r, d):
        self.refra = r
        self.refdec = d
        self.rascale = cosd(d)
        #print 'Affine: setting ref dec=', d, 'rascale', self.rascale

    def getShiftDeg(self):
        return (self.dra, self.ddec)

    def getShiftArcsec(self):
        ''' In isotropic coords'''
        return (self.dra * 3600. * self.rascale, self.ddec * 3600.)

    def apply(self, ra, dec, x=None, y=None):
        dr,dd = self.offset(ra, dec, x=x, y=y)
        return (ra + dr, dec + dd)

    def offset(self, ra, dec, x=None, y=None, ignoreSip=False):
        tr,td = self.getAffineOffset(ra, dec)
        dr = self.dra  + tr
        dd = self.ddec + td

        if self.rapoly is not None:
            pr,pd = self.getPolynomialOffset(ra, dec)
            dr += pr
            dd += pd

        if self.sipterms is not None:
            if ignoreSip:
                print 'Ignoring SIP terms in Affine.offset()'
            else:
                assert((x is not None) and (y is not None))
                print 'Affine.offset(): Applying SIP terms'
                dx,dy = self.offsetSipXy(x, y)
                (cd1,cd2,cd3,cd4) = self.cdmatrix
                dr += cd1 * dx + cd2 * dy
                dd += cd3 * dx + cd4 * dy
        return (dr, dd)

    def offsetSipXy(self, x, y):
        k = 0
        dx = np.zeros(x.shape, float)
        dy = np.zeros_like(dx)
        print 'refxy', self.refxy
        x0,y0 = self.refxy
        #print 'dx,dy', dx.shape, dy.shape
        for order in range(2, 11):
            for xorder in range(0, order+1):
                yorder = order - xorder
                sipmult = ((x-x0) ** xorder * (y-y0) ** yorder)
                #print 'sipmult:', sipmult.shape
                aterm = self.sipterms[k]
                bterm = self.sipterms[k+1]
                #print 'aterm', aterm
                #print 'bterm', bterm
                dx += sipmult * aterm
                dy += sipmult * bterm
                k += 2
            if k >= len(self.sipterms):
                break
        return dx,dy
    
    def getAffineOffset(self, ra, dec):
        dr = (ra  - self.refra ) * self.rascale
        dd = (dec - self.refdec)
        tr = (self.T[0] * dr + self.T[1] * dd) / self.rascale
        td = (self.T[2] * dr + self.T[3] * dd)
        return tr,td

    def getPolynomialOffset(self, ra, dec):
        dr = (ra  - self.refra ) * self.rascale
        dd = (dec - self.refdec)
        pr = np.zeros_like(ra)
        pd = np.zeros_like(dec)
        k = 0
        ## HACK
        for o in range(2, 10):
            for deco in range(o+1):
                rao = o - deco
                rcoeff = self.rapoly[k]
                dcoeff = self.decpoly[k]
                k += 1
                pr += rcoeff * dr**rao * dd**deco
                pd += dcoeff * dr**rao * dd**deco
            ## ASSUME that rapoly/decpoly have a sensible number of terms.
            if k >= len(self.rapoly):
                break
        return pr,dd

    def applyToWcs(self, cd, crval):
        T = self.T
        Ta = np.array([[ 1.+T[0], T[1] ], [ T[2], 1.+T[3] ]])
        #print 'T array:', Ta
        #print 'CD:', cd
        cd2 = np.dot(Ta, cd)
        #print 'CD2:', cd2
        crval2 = crval + np.array([self.dra, self.ddec])
        return cd2, crval2

    def applyToWcsObject(self, wcs):
        '''
        'wcs' must be a astrometry.util.util.Sip or Tan.
        '''
        if type(wcs) is astrometry.util.util.Sip:
            wcs = wcs.wcstan
        assert(type(wcs) is astrometry.util.util.Tan)

        self.changeReferencePoint(wcs.crval[0], wcs.crval[1])
        crval = np.array([wcs.crval[0], wcs.crval[1]])
        cd = np.array([[ wcs.cd[0], wcs.cd[1] ],
                       [ wcs.cd[2], wcs.cd[3]]])
        cd2,crval2 = self.applyToWcs(cd, crval)
        wcs.set_crval(*crval2)
        wcs.set_cd(cd2[0,0], cd2[0,1], cd2[1,0], cd2[1,1])

    def add(self, other):
        '''
        Set my transformation to be the result of applying me, then other.

        I do:

        [ r* ] = r + sra  + [1/rascale] * [ Tra_ra   Tra_dec  ] * [ (r - r_ref) * rascale ]
        [ d* ] = d + sdec + [1        ]   [ Tdec_ra  Tdec_dec ]   [ (d - d_ref)           ]


        R is r_ref
        D is d_ref
        S is rascale
        ... These are all the same for the two affines.

        Apply me (_1) to r_a,d_a to get r_b,d_b.

        r_b = r_a + sr_1 +  [ 1/S ] * T_1 * (r_a - R)*S
        d_b   d_a + sd_1 +  [ 1   ]         (d_a - D)

        Then apply other (_2) to r_b,d_b to get r_c,d_c

        r_c = r_b + sr_2 +  [ 1/S ] * T_2 * (r_b - R)*S
        d_c   d_b + sd_2 +  [ 1   ]         (d_b - D)

        .   = ( r_a + sr_1 + [1/S] T_1 (r_a - R)*S ) + sr_2 + [1/S] T_2 (r_b - R)*S
        .     ( d_a   sd_1   [1  ]     (d_a - D)   )   sd_2   [1  ]     (d_b - D)

        Expanding the last term,

        .   = 1/S T_2 (r_a - R)*S  +  1/S T_2 sr_1 * S  +  1/S T_2 (1/S T_1 * (r_a - R)*S)*S
        .     1       (d_a - D)       1       sd_1         1       (          (d_a - D)  )

        We get some 1/S*S cancellation and:

        r_c = r_a + (sr_1 + sr_2) + [1/S] (T_1 + T_2 + T_2 * T_1) (r_a - R)*S
        d_c = d_a + (sd_1 + sd_2)   [1  ]                         (d_a - D)

        '''
        assert(self.refra == other.refra)
        assert(self.refdec == other.refdec)
        assert(self.rascale == other.rascale)

        if self.sipterms is not None:
            if other.sipterms is not None:
                print 'Adding SIP terms:', self.sipterms, other.sipterms
                assert(self.refxy == other.refxy)
                assert(self.cdmatrix == other.cdmatrix)
                newterms = np.zeros(max(len(self.sipterms), len(other.sipterms)))
                newterms[:len(self.sipterms)] += self.sipterms
                newterms[:len(other.sipterms)] += other.sipterms
                print 'Setting new SIP terms:', newterms
                self.sipterms = newterms

        else:
            if other.sipterms is not None:
                # FIXME -- copy() ?
                self.sipterms = other.sipterms
                self.refxy = other.refxy
                self.cdmatrix = other.cdmatrix


        O1, O2, O3, O4 = other.T
        newdra  = (self.dra  + other.dra +
                   (self.dra * self.rascale * O1 + self.ddec * O2) / self.rascale)
        newddec = (self.ddec + other.ddec +
                   (self.dra * self.rascale * O3 + self.ddec * O4))
        self.dra  = newdra
        self.ddec = newddec

        T1, T2, T3, T4 = self.T

        # T1 + T2 + T2*T1 (confusingly, T2=other.T and T1=me.T)
        self.T = [ T1 + O1 + O1*T1 + O2*T3,
                   T2 + O2 + O1*T2 + O2*T4,
                   T3 + O3 + O3*T1 + O4*T3,
                   T4 + O4 + O3*T2 + O4*T4 ]


class CamMeta(object):
    pass

def describeFilters(cam, Tme=None, delete_old_mag_cols=False):
    ''' Returns a CamMeta '''
    meta = CamMeta()
    meta.cam = cam
    if cam == 'ACS':
        if Tme is not None:
            Tme.mag1 = Tme.mag1_acs
            Tme.mag2 = Tme.mag2_acs
            if delete_old_mag_cols:
                Tme.delete_column('mag1_acs')
                Tme.delete_column('mag2_acs')
        meta.magcols = [ 'mag1', 'mag2' ]
        meta.fnames = [ 'ACS F475W', 'ACS F814W' ]
        meta.filters = [ 475, 814 ]
        meta.flambdas = [ 475., 814. ]

    elif cam == 'IR':
        if Tme is not None:
            Tme.mag1 = Tme.mag1_ir
            Tme.mag2 = Tme.mag2_ir
            if delete_old_mag_cols:
                Tme.delete_column('mag1_ir')
                Tme.delete_column('mag2_ir')
        meta.magcols = [ 'mag1', 'mag2' ]
        meta.fnames = [ 'IR F110W', 'IR F160W' ]
        meta.filters = [ 110, 160 ]
        meta.flambdas = [ 1100., 1600. ]

    elif cam == 'UV':
        if Tme is not None:
            Tme.mag1 = Tme.mag1_uvis
            Tme.mag2 = Tme.mag2_uvis
            if delete_old_mag_cols:
                Tme.delete_column('mag1_uvis')
                Tme.delete_column('mag2_uvis')
        meta.magcols = [ 'mag1', 'mag2' ]
        meta.fnames = [ 'UVIS F275W', 'UVIS F336W' ]
        meta.filters = [ 275, 336 ]
        meta.flambdas = [ 275., 336. ]

    elif cam in ['CFHT', 'CFHT2']:
        if Tme is not None:
            Tme.mag1 = Tme.i
        meta.magcols = [ 'i' ]
        meta.fnames = [ 'CFHT i' ]
        meta.flambdas = [ 763. ] # HACK -- this is the SDSS i-band midpoint, roughly
        meta.filters = [ 763 ]
        return meta

    else:
        print 'Unknown camera: "%s"' % cam
        return None

    if Tme is not None:
        Tme.color = Tme.mag1 - Tme.mag2
    return meta

def getNearMags(me, ref):
    ''' Takes two CamMeta objects; returns the indices of their nearby filters '''
    bestij = None
    bestdiff = 1e6
    for i,lam1 in enumerate(me.flambdas):
        for j,lam2 in enumerate(ref.flambdas):
            dl = abs(lam1 - lam2)
            if dl < bestdiff:
                bestij = (i,j)
                bestdiff = dl
    i,j = bestij
    return i, j


def loadBrick(cam, path='.', duprad=None, merge=True, itab=None):
    if cam == 'CFHT':
        cfht = fits_table('data/cfht/cfht.fits')
        print 'Got', len(cfht), 'CFHT sources'
        return cfht

    if cam == 'CFHT2':
        cfht = fits_table('data/cfht/cfht-2mass.fits')
        print 'Got', len(cfht), 'CFHT2 sources'
        return cfht

    Tme = []
    print 'Reading', len(itab), 'files'
    affines = Affine.fromTable(itab)
    for i in range(len(itab)):
        fn = itab.filename[i]
        print 'Reading', fn
        G = glob(os.path.join(path, fn))
        if len(G) == 0:
            print 'Could not find file', fn, 'in path', path
            sys.exit(-1)
        fn = G[0]
        T = fits_table(fn)
        T.ra,T.dec = affines[i].apply(T.ra, T.dec)
        T.fieldid = np.zeros(len(T), int) + itab.fieldid[i]
        Tme.append(T)

    if not merge:
        # just return the list of tables
        return Tme

    # Merge the files.
    Tallme = Tme.pop()
    while len(Tme):
        Ti = Tme.pop()
        Tallme.append(Ti)
    Tme = Tallme
    print 'Merged all', cam, 'fields to make', len(Tme), 'sources'

    if duprad == 0.0:
        # don't do any matching
        return Tme

    # Keep a copy before we cut...
    Tall = Tme

    if duprad is None:
        if cam == 'ACS':
            duprad = 0.02
        else:
            duprad = 0.03

    M = Match(Tme, Tme, duprad, notself=True)
    print 'With symmetric matches, got', len(M.I), 'matches within', duprad, 'arcsec'
    # Trim symmetric matches -- i->j and j->i
    II = (M.I < M.J)
    M.cut(II)
    print 'After trimming, got', len(M.I), 'matches'
    # Now all i < j

    # Don't allow matches within the same fieldid.
    II = (Tme.fieldid[M.I] != Tme.fieldid[M.J])
    M.cut(II)
    print 'After cut on fieldid:', len(M.I)
    
    # Next, in each matched pair we drop the star with the higher index (j)
    keep = np.ones(len(Tme), bool)
    keep[M.J] = False

    Tall.primary = keep
    match = np.zeros(len(Tall), int) - 1
    # For each star that will be dropped, we want to point back to the primary.
    # This doesn't quite do it: if A->B->C, B will be dropped but C will point
    # to it.  We have to iterate...
    match[M.J] = M.I
    drop = np.logical_not(keep)
    assert(all(match[drop] > -1))
    while True:
        assert(all(match[drop] > -1))
        # drop: bool: will the star be dropped
        # D: int: indices of stars to be dropped
        # match[D]: int: the indices of the matched stars of dropped stars
        # drop[match[D]]: bool: is the matched star going to be dropped?
        D = np.flatnonzero(drop)
        DMD = drop[match[D]]
        if sum(DMD) == 0:
            break
        # Which ones need to be iterated?
        I = np.flatnonzero(DMD)
        # Grab their indices
        II = D[I]
        # Iterate
        match[II] = match[match[II]]
    Tall.match = match

    print 'Cut from', len(Tme), 'to',
    Tme = Tme[keep]
    print len(Tme)

    return Tme, Tall, M

def plotfitquality(H, xe, ye, A):
    '''
    H,xe,ye from plotalignment()
    '''
    import pylab as plt
    xe /= 1000.
    ye /= 1000.
    xx = (xe[:-1] + xe[1:])/2.
    yy = (ye[:-1] + ye[1:])/2.
    XX,YY = np.meshgrid(xx, yy)
    XX = XX.ravel()
    YY = YY.ravel()
    XY = np.vstack((XX,YY)).T
    Mdist = np.sqrt(mahalanobis_distsq(XY, A.mu, A.C))
    assert(len(H.ravel()) == len(Mdist))
    mod = A.getModel(XX, YY)
    R2 = XX**2 + YY**2
    mod[R2 > (A.match.rad)**2] = 0.
    mod *= (H.sum() / mod.sum())
    plt.clf()
    rng = (0, 7)
    plt.hist(Mdist, 100, weights=H.ravel(), histtype='step', color='b', label='data', range=rng)
    plt.hist(Mdist, 100, weights=mod, histtype='step', color='r', label='model', range=rng)
    plt.xlabel('| Chi |')
    plt.ylabel('Number of matches')
    plt.title('Gaussian peak fit quality')
    plt.legend(loc='upper right')
    

def plotalignment(A, nbins=200, M=None, rng=None, doclf=True, docolorbar=True,
                  docutcircle=True, docontours=True, dologhist=False,
                  doaxlines=False, imshowargs={}):
    import pylab as plt
    from astrometry.util.plotutils import plothist, loghist
    if doclf:
        plt.clf()
    if M is None:
        M = A.match
    if dologhist:
        f = loghist
    else:
        f = plothist
    H,xe,ye = f(M.dra_arcsec*1000., M.ddec_arcsec*1000., nbins,
                range=rng, doclf=doclf, docolorbar=docolorbar,
                imshowargs=imshowargs)
    ax = plt.axis()
    if A is not None:
        # The EM fit is based on a subset of the matches;
        # draw the subset cut circle.
        if docutcircle:
            angle = np.linspace(0, 2.*pi, 360)
            plt.plot((A.cutcenter[0] + A.cutrange * np.cos(angle))*1000.,
                     (A.cutcenter[1] + A.cutrange * np.sin(angle))*1000., 'r-')
        if docontours:
            for i,c in enumerate(['b','c','g']*2):
                if i == A.ngauss:
                    break
                for nsig in [1,2]:
                    XY = A.getContours(nsig, c=i)
                    if XY is None:
                        break
                    X,Y = XY
                    plt.plot(X*1000., Y*1000., '-', color=c)#, alpha=0.5)
    if doaxlines:
        plt.axhline(0., color='b', alpha=0.5)
        plt.axvline(0., color='b', alpha=0.5)
    plt.axis(ax)
    plt.xlabel('dRA (mas)')
    plt.ylabel('dDec (mas)')
    return H,xe,ye

def histlog10(x, **kwargs):
    import pylab as plt
    I = (x > 0)
    L = np.log10(x[I])
    plt.clf()
    plt.hist(L, **kwargs)

class Match(object):
    '''
    Fields:

    .I: indices in the first table of the matches
    .J:   "            second         "
    .dra, .ddec: differences (T1 - T2) of RA,Dec, in deg.
    .dra_arcsec, .ddec_arcsec: differences (T1-T2) of Ra,Dec, in isotropic arcsec.
    
    '''
    def __init__(self, *args, **kwargs):
        '''
        args: T1, T2, rad
        rad: match radius in arcsec
        '''
        if len(args) == 0:
            return
        (T1, T2, rad) = args

        self.rascale = kwargs.get('rascale', None)
        if self.rascale is None:
            d = np.mean(T2.dec)
            #print 'Match: mean dec', d
            self.rascale = cosd(d)
            #print 'Match: using rascale from mean Dec', self.rascale
        nearest = kwargs.get('nearest', False)
        notself = kwargs.get('notself', False)
        unsymm  = kwargs.get('unsymm', False)

        self.rad = rad

        I = kwargs.get('I', None)
        J = kwargs.get('J', None)
        D = kwargs.get('dists', None)

        if I is None or J is None or D is None:
            # Do tangent-plane projection around mid RA,Dec;
            # this will fail on RA=0--360 wrap-around!
            r = 0.5 * (T2.ra.min()  + T2.ra.max() )
            d = 0.5 * (T2.dec.min() + T2.dec.max())
            fakewcs = Tan(r, d, 0, 0, 0, 0, 0, 0, 0, 0)
            ok,ix,iy = fakewcs.radec2iwc(T1.ra, T1.dec)
            X1 = np.vstack((ix, iy)).T
            #X1 = np.vstack((T1.ra*self.rascale, T1.dec)).T
            if T2 is T1:
                X2 = X1
            else:
                ok,ix,iy = fakewcs.radec2iwc(T2.ra, T2.dec)
                X2 = np.vstack((ix,iy)).T
                #X2 = np.vstack((T2.ra*self.rascale, T2.dec)).T
            del ix
            del iy

            r = arcsec2deg(rad)
            if nearest:
                #pickle_to_file((X1, X2, r), 'nearest.pickle')
                #sys.exit(-1)
                print 'search rad arcsec', rad
                print 'dist:', r
                (inds,dists2) = spherematch.nearest(X2, X1, r, notself)
                dists = np.sqrt(dists2)
                # spherematch.nearest(T1,T2) returns, for each point in T2,
                # the index in T1 of the nearest point, or -1.

                # With the args swapped,
                # inds is the length of X1
                # flatnonzero(inds != -1) are indices of matches in X1
                # inds[inds != -1] are indices of matches in X2

                print 'nearest: X1', X1.shape, 'X2', X2.shape
                print 'inds', inds.shape, 'dists', dists.shape
                I = np.flatnonzero(inds != -1)
                print ' number != -1:', len(I)
                # We want "inds" to be  ( index in X1, index in X2 ) * nmatches
                #inds = np.vstack((np.flatnonzero(I), inds[I])).T
                #dists = dists[I]
                #print 'inds shape', inds.shape
                self.I = I
                self.J = inds[I]
                self.dists = dists[I]
                #print 'dists shape', self.dists.shape

                assert(all(self.I > -1))
                assert(all(self.I < len(T1)))
                assert(all(self.J > -1))
                assert(all(self.J < len(T2)))
                assert(all(self.dists <= r))
                assert(len(self.I) == len(self.J))
                assert(len(self.dists) == len(self.I))
                realdiffs = X1[self.I,:] - X2[self.J,:]
                realdists = np.sum(realdiffs**2, axis=1)
                assert(all(realdists <= r))

            else:
                (inds,dists) = spherematch.match(X1, X2, r, notself)
                self.dists = dists[:,0]
                self.I = inds[:,0]
                self.J = inds[:,1]
                if T2 is T1 and unsymm:
                    K = self.I < self.J
                    self.dists = self.dists[K]
                    self.I = self.I[K]
                    self.J = self.J[K]

                #print 'Match: X1', X1.min(axis=0), X1.max(axis=0), 'X2', X2.min(axis=0), X2.max(axis=0)
                #D = (X1[self.I] - X2[self.J])
                # print 'deltas', D.min(axis=0), D.max(axis=0)

        else:
            self.I = I
            self.J = J
            self.dists = D

        self.recompute_dradec(T1, T2)

    def recompute_dradec(self, T1, T2):
        self.dra  = T1[self.I].ra  - T2[self.J].ra
        self.ddec = T1[self.I].dec - T2[self.J].dec
        self.dra_arcsec = self.dra * self.rascale * 3600.
        self.ddec_arcsec = self.ddec * 3600.
        #print 'Match: dra,ddec ranges:', self.dra.min(), self.dra.max(), self.ddec.min(), self.ddec.max()
        #print '  ratios (dec/ra)', self.ddec.min()/self.dra.min(), self.ddec.max()/self.dra.max()
        #print '  Rascale', self.rascale
        #print '  arcsec:', self.dra_arcsec.min(), self.dra_arcsec.max(), self.ddec_arcsec.min(), self.ddec_arcsec.max()

    def getdistsq_arcsec(self):
        return self.dra_arcsec**2 + self.ddec_arcsec**2

    def getdist_arcsec(self):
        return np.sqrt(self.getdistsq_arcsec())

    def getdist_mas(self):
        return 1000. * self.getdist_arcsec()

    def getcut(self, I):
        m = Match()
        for k in ['rascale', 'rad']:
            setattr(m, k, getattr(self, k))
        for k in ['dists', 'I', 'J', 'dra', 'ddec', 'dra_arcsec', 'ddec_arcsec']:
            X = getattr(self, k)
            X = X[I]
            setattr(m, k, X)
        return m

    def cut(self, I):
        N = None
        for n in ['dists', 'I', 'J', 'dra', 'ddec', 'dra_arcsec', 'ddec_arcsec']:
            X = getattr(self, n)
            X = X[I]
            setattr(self, n, X)
            if N is None:
                N = len(X)
            else:
                assert(N == len(X))

def getGstFiles(cam, brick, path=None):
    if path is None:
        path = 'data/pipe/*/proc'
    cammap = {'ACS':'WFC'}
    cam = cammap.get(cam, cam)
    fns = glob(os.path.join(path, '*B%02i*%s*.gst.fits' % (brick, cam)))
    fns.sort()
    return fns

def readGsts(fns, keepcols=[]):
    TT = []

    cutcols = ['mag1_std', 'chi1', 'sharp1', 'round1',
               'crowd1', 'snr1', 'flag1', 'mag2_std', 
               'chi2', 'sharp2', 'round2', 'crowd2', 'snr2', 'flag2']
    for k in keepcols:
        cutcols.remove(k)

    for fn in fns:
        T = fits_table(fn)
        print fn, ' --> ', len(T), 'stars'
        # remove columns we don't use (to save some memory)
        # 'mag1_err', 'mag2_err',
        for k in cutcols:
            T.delete_column(k)
        TT.append(T)
    return TT

