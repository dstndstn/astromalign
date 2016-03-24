if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

from optparse import OptionParser
from glob import glob
import os
import time

from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.plotutils import antigray
from astrometry.util.ttime import *

from math import cos, pi, floor, ceil, log10, sqrt

import matplotlib
import numpy as np
import pylab as plt

import scipy.sparse
import scipy.sparse.linalg

from astrom_common import *

def totalstars(TT):
    return sum([len(T) for T in TT])

class Intra(object):
    # self.edges:
    # 0-5: i, j, matched RAs, matched Decs, matched dRAs, matched dDecs
    # 6-8: aligned dRADec (mu in deg), mean RA, mean Dec,
    #   9: muarcsec
    # 10-11: matched dRAs (arcsec), matched dDecs (arcsec)
    # 12: A.subset
    # 13-14: M.I, M.J
    # 15: A.fore
    # 16,17,18,19: good-RAs, good-Decs, good-dRAs, good-dDecs
    # 20,21,22,23: bad-""
    # 24,25: good, bad

    # mu in deg

    def __init__(self, N):
        self.edges = []
        self.alignments = []
        self.affines = [ Affine() for i in range(N) ]

    def __str__(self):
        return 'Intra: affines [\n  ' + ',\n  '.join(str(a) for a in self.affines) + ' ]'

    def __repr__(self):
        return str(self)

    def update_all_dradecs(self, CC):
        for i,Ci in enumerate(CC):
            for j,Cj in enumerate(CC):
                if j <= i:
                    continue
                A = self.AA[i].get(j, None)
                if A is None:
                    continue
                A.match.recompute_dradec(Ci, Cj)

    # shared SIP terms
    def set_joint_sip_terms(self, sip):
        self.joint_sip = sip

    # per-field SIP terms
    def set_sip_terms(self, sip):
        self.sip = sip

    def get_sip_terms(self, i):
        if hasattr(self, 'joint_sip'):
            return self.joint_sip
        if hasattr(self, 'sip'):
            return self.sip[i]
        return None

    def set_shifts(self, dra, ddec):
        for a,dr,dd in zip(self.affines, dra, ddec):
            a.setShift(dr, dd)

    def get_affine(self, i):
        return self.affines[i]

    def get_affines(self):
        return self.affines

    def add_to_affines(self, affs):
        assert(len(self.affines) == len(affs))
        for a,ame in zip(affs, self.affines):
            a.add(ame)

    def set_affine(self, T1, T2, T3, T4):
        for a,t1,t2,t3,t4 in zip(self.affines, T1,T2,T3,T4):
            a.setAffine(t1,t2,t3,t4)

    def get_rascale(self, fieldi):
        return self.affines[fieldi].getRascale()

    def get_rascales(self):
        return [a.getRascale() for a in self.affines]

    def set_rascales(self, rascale):
        for a in self.affines:
            a.setRascale(rascale)

    def set_rotation(self, R):
        for a,r in zip(self.affines, R):
            a.setRotation(r)

    def set_reference_radec(self, fieldi, r, d):
        return self.affines[fieldi].setReferenceRadec(r, d)

    def get_reference_radecs(self):
        return [self.affines[fieldi].getReferenceRadec()
                for fieldi in range(len(self.affines))]

    def get_reference_radec(self, fieldi):
        return self.affines[fieldi].getReferenceRadec()

    def get_ra_shift_deg(self, fieldi):
        ra,dec = self.affines[fieldi].getShiftDeg()
        return ra
    def get_dec_shift_deg(self, fieldi):
        ra,dec = self.affines[fieldi].getShiftDeg()
        return dec

    def get_radec_shift_arcsec(self, fieldi):
        return self.affines[fieldi].getShiftArcsec()

    def offset(self, fieldi, ra, dec, ignoreSip=True):
        return self.affines[fieldi].offset(ra,dec, ignoreSip=ignoreSip)

    def apply(self, fieldi, ra, dec):
        return self.affines[fieldi].apply(ra,dec)

    def applyTo(self, TT):
        assert(len(self.affines) == len(TT))
        for i,Ti in enumerate(TT):
            (cr,cd) = self.offset(i, Ti.ra, Ti.dec)
            Ti.ra  += cr
            Ti.dec += cd

    def applyToWcsObjects(self, WCS):
        for i,wcs in enumerate(WCS):
            self.affines[i].applyToWcsObject(wcs)

    def add(self, other):
        for a, o in zip(self.affines, other.affines):
            a.add(o)

    def toTable(self):
        return Affine.toTable(self.affines)

    def get_all_dradec(self, apply=False):
        alldra = []
        allddec = []
        for ei in range(len(self.edges)):
            (ra,dec,dra,ddec) = self.get_edge_dradec_arcsec(ei, corrected=apply)
            alldra.append(dra)
            allddec.append(ddec)
        return np.hstack(alldra), np.hstack(allddec)

    def trimBeforeSaving(self):
        # Dump all but the results ( enough to support offset() )
        self.edges = []

    def nedges(self):
        return len(self.edges)

    def add_alignment(self, a):
        self.alignments.append(a)

    def add_edge(self, *args):
        S = args[12]
        good = np.zeros_like(S)
        fore = args[15]
        good[S] = (fore > 0.5)
        G = [args[i][good] for i in range(2,6)]
        bad = np.logical_not(good)
        B = [args[i][bad] for i in range(2,6)]
        self.edges.append(args + tuple(G) + tuple(B) + (good, bad))

    def radec_axes(self, TT, fieldi):
        T = TT[fieldi]
        rl,rh = T.ra.min(), T.ra.max()
        dl,dh = T.dec.min(), T.dec.max()
        setRadecAxes(rl,rh, dl,dh)
        
    def edge_ij(self, edgei):
        return (self.edges[edgei])[0:2]

    def edge_IJ(self, edgei):
        return (self.edges[edgei])[13:15]

    def edge_good(self, edgei):
        return self.edges[edgei][24]

    # (RAs, Decs, dRAs, dDecs),  all in deg.
    def edge_matches(self, edgei, goodonly=False, badonly=False):
        if goodonly:
            return self.edges[edgei][16:20]
        elif badonly:
            return self.edges[edgei][20:24]
        else:
            return self.edges[edgei][2:6]
        
    def get_edge_dradec_deg(self, edgei, corrected=False, goodonly=False, badonly=False):
        i,j = self.edge_ij(edgei)
        (matchRA, matchDec, matchdRA, matchdDec) = self.edge_matches(edgei, goodonly, badonly)
        dr = matchdRA.copy()
        dd = matchdDec.copy()
        if corrected:
            (cri,cdi) = self.offset(i, matchRA, matchDec)
            (crj,cdj) = self.offset(j, matchRA, matchDec)
            dr += cri - crj
            dd += cdi - cdj
        return (matchRA, matchDec, dr, dd)

    def get_edge_dradec_arcsec(self, edgei, corrected=False, goodonly=False,
                               badonly=False):
        (ra, dec, dra, ddec) = (
            self.get_edge_dradec_deg(edgei, corrected=corrected,
                                     goodonly=goodonly, badonly=badonly))
        return (ra, dec, dra * rascale * 3600., ddec * 3600.)

    def get_total_matches(self):
        return sum([len(e[2]) for e in self.edges])

    # float array, foreground weights for matches in the 'subset'
    def get_edge_fg(self, edgei):
        return self.edges[edgei][15]

    # foreground weights for all matches (not just the 'subset')
    def get_edge_all_weights(self, ei):
        S = self.get_edge_subset(ei)
        w = np.zeros(len(S))
        w[S] = self.get_edge_fg(ei)
        return w

    # boolean array, len(matches for edge ei)
    def get_edge_subset(self, edgei):
        return self.edges[edgei][12]

    def plotallstars(self, TT):
        if totalstars(TT) > 10000:
            plothist(np.hstack([T.ra for T in TT]), np.hstack([T.dec for T in TT]),
                     101, imshowargs=dict(cmap=antigray), dohot=False)
        else:
            for T in TT:
                plt.plot(T.ra, T.dec, '.', alpha=0.1, color='0.5')

    def plotmatchedstars(self, edgei):
        X = self.edges[edgei]
        #(i, j, matchRA, matchDec, matchdRA, matchdDec, mu) = X[:7]
        (matchRA, matchDec, dr,dd) = self.edge_matches(edgei, badonly=True)
        plt.plot(matchRA, matchDec, 'k.', alpha=0.5)
        (matchRA, matchDec, dr,dd) = self.edge_matches(edgei, goodonly=True)
        plt.plot(matchRA, matchDec, '.', color=(0.5,0,0), alpha=0.5)

    def plotallmatches(self):
        for ei in range(len(self.edges)):
            self.plotmatchedstars(ei)

    # one plot per edge
    def edgeplot(self, TT, ps):
        for ei,X in enumerate(self.edges):
            (i, j) = X[:2]
            Ta = TT[i]
            Tb = TT[j]
            plt.clf()
            if len(Ta) > 1000:
                nbins = 101
                ra = np.hstack((Ta.ra, Tb.ra))
                dec = np.hstack((Ta.dec, Tb.dec))
                H,xe,ye = np.histogram2d(ra, dec, bins=nbins)
                (matchRA, matchDec, dr,dd) = self.edge_matches(ei, goodonly=True)
                G,xe,ye = np.histogram2d(matchRA, matchDec, bins=(xe,ye))
                assert(G.shape == H.shape)
                img = antigray(H / H.max())
                img[G>0,:] = matplotlib.cm.hot(G[G>0] / H[G>0])
                ax = setRadecAxes(xe[0], xe[-1], ye[0], ye[-1])
                plt.imshow(img, extent=(min(xe), max(xe), min(ye), max(ye)),
                           aspect='auto', origin='lower', interpolation='nearest')
                plt.axis(ax)

            else:
                self.plotallstars([Ta,Tb])
                self.plotmatchedstars(ei)
                plt.xlabel('RA (deg)')
                plt.ylabel('Dec (deg)')
            ps.savefig()

    # one plot per edge
    def edgescatter(self, ps):
        for ei,X in enumerate(self.edges):
            i,j = X[:2]
            matchdRA, matchdDec = X[10:12]
            mu = X[9]
            A = self.alignments[ei]

            plt.clf()
            if len(matchdRA) > 1000:
                plothist(matchdRA, matchdDec, 101)
            else:
                plt.plot(matchdRA, matchdDec, 'k.', alpha=0.5)
            plt.axvline(0, color='0.5')
            plt.axhline(0, color='0.5')
            plt.axvline(mu[0], color='b')
            plt.axhline(mu[1], color='b')
            for nsig in [1,2]:
                X,Y = A.getContours(nsigma=nsig)
                plt.plot(X, Y, 'b-')
            plt.xlabel('delta-RA (arcsec)')
            plt.ylabel('delta-Dec (arcsec)')
            plt.axis('scaled')
            ps.savefig()

    def vecplot(self, TT, apply=False, scale=100, outlines=None,
                arrowminmatches=100):
        # find RA,Dec range of stars
        ral,rah = 1000,-1000
        decl,dech = 1000,-1000
        for i in range(len(TT)):
            print 'File', i, 'RA', TT[i].ra.min(), TT[i].ra.max(), 'DEC', TT[i].dec.min(), TT[i].dec.max()
            ral = min(ral, TT[i].ra.min())
            rah = max(rah, TT[i].ra.max())
            decl = min(decl, TT[i].dec.min())
            dech = max(dech, TT[i].dec.max())
        print 'ra,dec range', ral,rah,decl,dech

        #ps3 = PlotSequence('tst-', format='%02i')

        plt.clf()
        self.plotallstars(TT)
        ###
        # ps3.savefig()
        ax = plt.axis()
        if outlines is not None:
            for rr,dd in outlines:
                plt.plot(rr, dd, 'k-', alpha=0.4)
        ###
        # ps3.savefig()

        for ei in range(len(self.edges)):
            self.plotmatchedstars(ei)
            ###
            # ps3.savefig()

        for ei,X in enumerate(self.edges):
            (i, j) = self.edge_ij(ei)
            (matchRA, matchDec, dr,dd) = self.edge_matches(ei, goodonly=True)
            Nmatches = len(matchRA)
            if Nmatches == 0:
                continue
            assert(all(np.isfinite(matchRA)))
            assert(all(np.isfinite(matchDec)))

            #print 'edge from', i, 'to', j, 'has', len(matchRA), 'matches'
            mu = X[6].copy()
            Ta = TT[i]
            Tb = TT[j]
            era,edec = np.mean(matchRA), np.mean(matchDec)
            ara,adec = np.mean(Ta.ra), np.mean(Ta.dec)
            bra,bdec = np.mean(Tb.ra), np.mean(Tb.dec)

            plt.text(era, edec, '%i' % Nmatches, color='g', zorder=11)
            plt.text(ara, adec, '%i' % i, color='k', zorder=11)
            plt.text(bra, bdec, '%i' % j, color='k', zorder=11)

            if Nmatches < arrowminmatches:
                continue

            if apply:
                # ignoreSIP = True
                (ri,di) = self.offset(i, era, edec)
                (rj,dj) = self.offset(j, era, edec)
                mu[0] += ri - rj
                mu[1] += di - dj
            mu *= scale
            lw = max(1, log10(Nmatches))
            W = ax[1]-ax[0]
            aargs = dict(width=W * 2e-3, head_width=W * 4e-3, lw=lw, zorder=10)
            #aargs = dict(width=0.005, head_width=0.04, lw=lw, zorder=10)
            plt.arrow(era, edec, mu[0], mu[1], color='r', **aargs)
            plt.arrow(ara, adec, mu[0], mu[1], color='b', alpha=0.7, **aargs)
            plt.arrow(bra, bdec, -mu[0], -mu[1], color='b', alpha=0.7, **aargs)
            plt.plot([ara,bra,era], [adec,bdec,edec], 'k.', zorder=11)

            ###
            # ps3.savefig()

        setRadecAxes(ral, rah, decl, dech)
        ###
        # ps3.savefig()

    def scatterplot(self, TT, apply=False, markshifts=True, mas=False,
                    force_hist=False, range=None):

        if len(self.edges) == 0:
            print 'Cannot create scatterplot() without edge data'
            return False

        rng = range
        import __builtin__
        range = __builtin__.range
        plt.clf()
        H = None
        dohist = force_hist or (self.get_total_matches() > 10000)
        if mas:
            S = 1000.
        else:
            S = 1.
        if dohist:
            if rng is not None:
                mn,mx = rng
            else:
                mn = -self.matchrad_arcsec * S
                mx =  self.matchrad_arcsec * S
            bins = np.linspace(mn, mx, 201)

        sr,sd = [],[]

        for ei in range(len(self.edges)):
            (ra,dec,bdRA,bdDec) = self.get_edge_dradec_arcsec(ei, corrected=apply, badonly=True)
            (ra,dec,gdRA,gdDec) = self.get_edge_dradec_arcsec(ei, corrected=apply, goodonly=True)
            if len(gdRA) == 0:
                continue
            if not apply and markshifts:
                i,j = self.edge_ij(ei)
                drai,ddeci = self.get_radec_shift_arcsec(i)
                draj,ddecj = self.get_radec_shift_arcsec(j)
                dra  = draj  - drai
                ddec = ddecj - ddeci
                sr.append(dra * S)
                sd.append(ddec* S)
            
            if dohist:
                Hi,xe,ye = np.histogram2d(gdRA *S, gdDec *S, bins=bins)
                if H is None:
                    H = Hi
                else:
                    H += Hi
            else:
                plt.plot(bdRA *S, bdDec *S, 'k.', alpha=0.1)
                plt.plot(gdRA *S, gdDec *S, 'r.', alpha=0.1)


        if dohist:
            plt.imshow(H.T, extent=(min(xe), max(xe), min(ye), max(ye)),
                   aspect='auto', origin='lower', interpolation='nearest')
            plt.hot()
            plt.colorbar()

        # compute ||dra,ddec||^2 weighted by p(fg) -- estimated size of residuals
        sumd2 = 0
        sumfg = 0
        sumra = 0
        sumdec = 0
        for ei in range(len(self.edges)):
            (ra,dec,dra,ddec) = self.get_edge_dradec_arcsec(ei, corrected=apply)
            sub = self.get_edge_subset(ei)
            dra,ddec = dra[sub],ddec[sub]
            fg = self.get_edge_fg(ei)
            sumd2 += np.sum(fg * (dra**2 + ddec**2))
            sumfg += np.sum(fg)
            sumra  += np.sum(fg * dra)
            sumdec += np.sum(fg * ddec)

        mra,mdec = sumra/sumfg * S, sumdec/sumfg * S
        # 2: per-coordinate
        md = sqrt(sumd2/(sumfg*2.)) * S
        ax = plt.axis()
        if len(sr):
            plt.plot(sr, sd, 'g+', mew=2, zorder=10)
        plt.plot([mra],[mdec], 'b+')
        angles = np.linspace(0, 2.*pi, 100)
        plt.plot(mra + md * np.cos(angles), mdec + md * np.sin(angles), 'b-')
        plt.plot(mra + 2. * md * np.cos(angles),
                 mdec + 2. * md * np.sin(angles), 'b-')
        plt.title('%i matches, mean (%.1g, %.1g) mas, std %.1f mas' %
                  (int(round(sumfg)), 1000.*mra/S, 1000.*mdec/S, 1000.*md/S))

        plt.axvline(0, color='0.5')
        plt.axhline(0, color='0.5')
        if mas:
            plt.xlabel('delta RA (mas)')
            plt.ylabel('delta Dec (mas)')
        else:
            plt.xlabel('delta RA (arcsec)')
            plt.ylabel('delta Dec (arcsec)')
        if not dohist and rng is not None:
            plt.axis([rng[0], rng[1], rng[0], rng[1]])
        else:
            plt.axis(ax)
        plt.axis('scaled')

    def xyoffsets(self, TT, apply=False, scale=100., binsize=None, wcs=None, subplotsize=None, ps=None):
        plt.clf()

        if subplotsize is None:
            subplotsize = (3,3)

        (subw,subh) = subplotsize

        print 'x,y size', (TT[0].x.max() - TT[0].x.min()), (TT[0].y.max() - TT[0].y.min())

        if binsize is None:
            binsize = 200

        for i,T in enumerate(TT):
            print 'xyoffsets: field', i

            if ps is None:
                plt.subplot(subw,subh, i+1)
            else:
                plt.clf()

            if wcs is not None:
                thiswcs = wcs[i]
            else:
                thiswcs = None

                
            for ei in range(len(self.edges)):
                ii,jj = self.edge_ij(ei)
                I,J = self.edge_IJ(ei)
                if i == ii:
                    K = I
                    s = 1.
                elif i == jj:
                    K = J
                    s = -1.
                else:
                    continue
                x,y = T.x[K], T.y[K]
                (r,d, dr,dd) = self.get_edge_dradec_deg(ei, corrected=apply, goodonly=True)
                good = self.edge_good(ei)
                x,y = x[good],y[good]
                #print len(x), len(r), len(dr)
                assert(len(x) == len(r))
                assert(len(r) == len(dr))
                
                dr *= s
                dd *= s

                x0 = min(x)
                y0 = min(y)
                NX = 1 + int(floor((max(x) - x0) / binsize))
                NY = 1 + int(floor((max(y) - y0) / binsize))
                NB = NX * NY

                bi = (
                    (np.floor(x - x0) / binsize).astype(int) +
                    (np.floor(y - y0) / binsize).astype(int)*NX )

                if thiswcs:
                    RR,DD = [],[]
                    w,h = thiswcs.get_width(), thiswcs.get_height()
                    for x,y in [ (0,0), (w,0), (w,h), (0,h), (0,0) ]:
                        ri,di = thiswcs.pixelxy2radec(x,y)
                        RR.append(ri)
                        DD.append(di)
                    plt.plot(RR, DD, '-', color='0.5')

                px = []
                py = []
                dx = []
                dy = []

                for b in np.unique(bi): #range(NB):
                    #ix = b % NX
                    #iy = b / NX
                    #cx = x0 + (ix + 0.5) * binsize
                    #cy = y0 + (iy + 0.5) * binsize
                    I = (bi == b)
                    if sum(I) == 0:
                        continue
                    mra,mdec = np.mean(r[I]), np.mean(d[I])
                    mdra,mddec = np.mean(dr[I]), np.mean(dd[I])
                    #plt.plot([mra], [mdec], 'k.', zorder=10, ms=2)
                    #plt.plot([mra, mra + mdra*scale], [mdec, mdec + mddec*scale],
                    #'r-', zorder=11)

                    px.append(mra)
                    py.append(mdec)
                    dx.append((mra, mra + mdra * scale))
                    dy.append((mdec, mdec + mddec * scale))

                plt.plot(px, py, 'k.', zorder=10, ms=2)
                plt.plot(np.array(dx).T, np.array(dy).T,
                         'r-', zorder=11)

            self.radec_axes(TT, i)

            if ps is not None:
                ps.savefig()
            else:
                plt.xticks([],[])
                plt.yticks([],[])

    def quiveroffsets(self, TT, apply=False):
        plt.clf()
        self.plotallstars(TT)
        self.plotallmatches()
        for ei in range(len(self.edges)):
            (matchRA, matchDec, dr, dd) = self.get_edge_dradec_deg(ei, corrected=apply, goodonly=True)
            scale = 100.
            plt.plot(np.vstack((matchRA,  matchRA  + dr*scale)),
                 np.vstack((matchDec, matchDec + dd*scale)),
                 'r-', alpha=0.5)
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')


    # rad in arcsec
    def hsvoffsets(self, TT, rad, apply=False):
        print 'hsv offsets plot'
        plt.clf()

        for ix,X in enumerate(self.edges):
            X = self.get_edge_dradec_arcsec(ix, corrected=apply, goodonly=True)
            (matchRA, matchDec, dra, ddec) = X

            print 'matchRA,Dec:', len(matchRA), len(matchDec)
            print 'dra,dec:', len(dra), len(ddec)

            for ra,dec,dr,dd in zip(matchRA, matchDec, dra, ddec):
                angle = arctan2(dd, dr) / (2.*pi)
                angle = fmod(angle + 1, 1.)
                mag = hypot(dd, dr)
                mag = min(1, mag/(0.5*rad))
                rgb = colorsys.hsv_to_rgb(angle, mag, 0.5)
                plt.plot([ra], [dec], '.', color=rgb, alpha=0.5)

        # legend in top-right corner.
        ax=plt.axis()
        xlo,xhi = plt.gca().get_xlim()
        ylo,yhi = plt.gca().get_ylim()
        # fraction
        keycx = xlo + 0.90 * (xhi-xlo)
        keycy = ylo + 0.90 * (yhi-ylo)
        keyrx = 0.1 * (xhi-xlo) / 1.4 # HACK
        keyry = 0.1 * (yhi-ylo)
        nrings = 5
        for i,(rx,ry) in enumerate(zip(np.linspace(0, keyrx, nrings), np.linspace(0, keyry, nrings))):
            nspokes = ceil(i / float(nrings-1) * 30)
            angles = np.linspace(0, 2.*pi, nspokes, endpoint=False)
            for a in angles:
                rgb = colorsys.hsv_to_rgb(a/(2.*pi), float(i)/(nrings-1), 0.5)
                plt.plot([keycx + rx*sin(a)], [keycy + ry*cos(a)], '.', color=rgb, alpha=1.)
        plt.axis(ax)
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')

    def get_mags(self, TT, apply, magcol='mag1', magerrcol='mag1_err'):
        alldd = []
        allmagI = []
        allmagJ = []
        allmagerrI = []
        allmagerrJ = []
        for ei in range(len(self.edges)):
            good = self.edge_good(ei)
            ii,jj = self.edge_ij(ei)
            I,J = self.edge_IJ(ei)
            (r,d, dra,ddec) = self.get_edge_dradec_arcsec(ei, corrected=apply,
                                                          goodonly=True)
            dd = np.sqrt(dra**2 + ddec**2)
            alldd.append(dd)
            # 'I' is an int array
            # 'good' is a bool array of size == size(I)
            # number of True elements in 'good' == len(dra)
            magI = TT[ii][I][good].get(magcol)
            magJ = TT[jj][J][good].get(magcol)
            allmagI.append(magI)
            allmagJ.append(magJ)
            if magerrcol in TT[ii].columns():
                allmagerrI.append(TT[ii].get(magerrcol)[I][good])
                allmagerrJ.append(TT[jj].get(magerrcol)[J][good])

        alldd = np.hstack(alldd)
        allmagI = np.hstack(allmagI)
        allmagJ = np.hstack(allmagJ)
        allmagerrI = np.hstack(allmagerrI)
        allmagerrJ = np.hstack(allmagerrJ)
        return (allmagI, allmagJ, alldd, magcol, allmagerrI, allmagerrJ)

    def magmagplot(self, TT, magcol, filtname, weighted=True):
        plt.clf()
        m1 = []
        m2 = []
        ww = []
        for ei in range(self.nedges()):
            i,j = self.edge_ij(ei)
            I,J = self.edge_IJ(ei)
            Ti = TT[i][I]
            Tj = TT[j][J]
            mag1 = Ti.get(magcol)
            mag2 = Tj.get(magcol)
            weights = self.get_edge_all_weights(ei)
            K = (mag1 < 50) * (mag2 < 50)
            m1.append(mag1[K])
            m2.append(mag2[K])
            ww.append(weights[K])
        m1 = np.hstack(m1)
        m2 = np.hstack(m2)
        ww = np.hstack(ww)
        if weighted:
            loghist(m1, m2, weights=ww)
        else:
            loghist(m1, m2)
        plt.xlabel('%s (mag)' % filtname)
        plt.ylabel('%s (mag)' % filtname)
        return ww

    def magplot(self, TT, apply=False, hist=False, errbars=False,
                step=0.5, maxerr=None, magcol='mag1', magerrcol='mag1_err'):
        ''' maxerr: cut any errors larger than this, in arcsec. '''
        allmagI,allmagJ,alldd,magcol, allmagerrI,allmagerrJ = self.get_mags(TT, apply, magcol=magcol, magerrcol=magerrcol)
        allmag = (allmagI+allmagJ) / 2.
        allmagerr = (allmagerrI+allmagerrJ) / 2.

        if maxerr is not None:
            I = (alldd <= maxerr)
            allmag = allmag[I]
            allmagerr = allmagerr[I]
            alldd = alldd[I]

        I = (allmag < 50.)
        allmag = allmag[I]
        allmagerr = allmagerr[I]
        alldd = alldd[I]

        # work in milli-arcsec.
        alldd *= 1000.
        plt.clf()
        #plt.plot(allmag1, alldd, 'm.', zorder=9, alpha=0.5)
        #plt.plot(allmag2, alldd, 'm.', zorder=9, alpha=0.5)
        #plt.plot(allmag, alldd, 'r.', zorder=10, alpha=0.5)
        loghist(allmag, alldd, 100, imshowargs=dict(cmap=antigray), hot=False)

        plt.xlabel('Mag ' + magcol)
        plt.ylabel('Match distance (milli-arcsec)')
        ax = plt.axis()
        
        bind = []
        binerr = []
        if hist or errbars:
            mn = int(floor(allmag.min()))
            mx = int(ceil(allmag.max()))
            binx = []
            biny = []
            binyerr = []
            # "step" is mag bin size
            for m in np.arange(mn,mx,  step):
                I = ((allmag > m) * (allmag < (m+step)))
                d = alldd[I]
                bind.append(d)
                binerr.append(np.median(allmagerr[I]))
                binx.append(m+step/2.)
                if len(d) == 0:
                    biny.append(0)
                    binyerr.append(0)
                else:
                    biny.append(np.mean(d))
                    binyerr.append(np.std(d))

        if hist:
            # histogram per 1-mag big
            bins = np.linspace(0, ax[3], 20)
            for d,m in zip(bind,binx):
                if len(d) == 0:
                    continue
                H,e = np.histogram(d, bins=bins)
                pk = H.max()
                n,b,p = plt.hist(d, bins=bins,
                                 weights=np.ones_like(d) * 0.5/pk,
                                 orientation='horizontal', bottom=m, zorder=15,
                                 ec='b', fc='none')

        if errbars:
            I = np.array([(len(d) >= 4) for d in bind])
            binx = np.array(binx)
            biny = np.array(biny)
            binyerr = np.array(binyerr)
            binerr = np.array(binerr)
            p1,p2,p3 = plt.errorbar(binx[I], biny[I], binyerr[I], fmt='o',
                                    barsabove=True, zorder=15)
            for p in p2 + p3:
                p.set_zorder(15)

            if False:
                ax1 = plt.gca()
                plt.twinx()
                plt.plot(binx[I], binerr[I], 'ro', zorder=16)
                plt.sca(ax1)
            
        if False:
            # contours
            B = 50
            H,xe,ye = np.histogram2d(allmag, alldd,
                                     bins=(np.linspace(ax[0],ax[1],B),
                                           np.linspace(ax[2],ax[3],B)))
            #plt.imshow(H.T, extent=ax, #(min(xe), max(xe), min(ye), max(ye)),
            #          aspect='auto', origin='lower', interpolation='nearest')
            #plt.colorbar()
            mx = H.max()
            print 'max histogram:', mx
            step = max(mx / 5, 10)
            plt.contour(H.T, extent=ax, colors='k', zorder=15,
                        levels=np.arange(step, mx, step))
                               
        plt.axis(ax)
        return biny,binerr

def n_sip_terms(sip_order):
    assert(sip_order >= 2)
    n = 0
    for order in range(2, sip_order+1):
        n += 2 * (order + 1)
    return n

# for parallelization of matching in 'intrabrickshift'...
def alignfunc(args):
    try:
        return _real_alignfunc(args)
    except:
        import traceback
        print 'Exception in alignfunc:'
        traceback.print_exc()
        raise

def _real_alignfunc(args):
    (Ti, Tj, matchradius, align_kwargs, mag1diff, mag2diff, i, j,
     mdhrad, alignplotargs, silent) = args

    weightrange = align_kwargs.pop('weightrange', None)

    if i == j:
        jname = 'ref'
    else:
        jname = j

    if not silent:
        print 'Matching', i, 'to', jname, 'with radius', matchradius, 'arcsec...'
    A = Alignment(Ti, Tj, searchradius = matchradius,
                  **align_kwargs)
    # apply maximum magnitude difference cuts
    if mag1diff is not None or mag2diff is not None:
        M = Match(Ti, Tj, matchradius)
        if mag1diff is not None:
            magdiff = np.abs(Ti.mag1[M.I] - Tj.mag1[M.J])
            K = np.flatnonzero(magdiff <= mag1diff)
            M.cut(K)
        if mag2diff is not None:
            magdiff = np.abs(Ti.mag2[M.I] - Tj.mag2[M.J])
            K = np.flatnonzero(magdiff <= mag2diff)
            M.cut(K)
        A.match = M
    if A.shift() is None:
        if not silent:
            print 'Failed to find a shift between files', i, 'and', j
        return i,j,None
    if not silent:
        print 'Matching', i, 'to', jname, ': got', len(A.match.I)

    # use EM results to weight the matches
    # A.match (a Match object) are the matches
    # A.subset (a slice) are the matches used in EM peak-fitting
    # A.fore are the foreground probabilities for the matches.

    M = A.match

    # Save histogram of matches (before the cut)
    hist2dargs = alignplotargs
    if not 'range' in hist2dargs:
        RR = matchradius * 1000.
        hist2dargs.update(range=((-RR,RR),(-RR,RR)))
    (H,xe,ye) = np.histogram2d(M.dra_arcsec * 1000., M.ddec_arcsec*1000., **hist2dargs)
    alplot = dict(H=H.T, extent=(min(xe), max(xe), min(ye), max(ye)),
                  aspect='auto', interpolation='nearest', origin='lower',
                  esize=A.getEllipseSize(),
                  estring=A.getEllipseString(),
                  econ1=A.getContours(1),
                  econ2=A.getContours(2),
                  ecen=A.arcsecshift(),
                  foresum = np.sum(A.fore),
                  )

    matchra, matchdec = Ti.ra[M.I], Ti.dec[M.I]
    # Cut to the objects used in the EM fit
    S = A.subset

    # possibly cut on A.fore
    w = np.sqrt(A.fore)
    if weightrange is not None:
        maxw = np.max(w)
        K = np.flatnonzero(w >= (maxw * weightrange))
        if not silent:
            print 'Cut to', len(K), 'of', len(w), 'matches with weights greater than fraction', weightrange, 'of max'
        # A.subset is a bool array
        assert(S.dtype == bool)
        S = np.flatnonzero(S)[K]
        w = w[K]

    matchra,matchdec = matchra[S], matchdec[S]

    alplot.update(MIall=M.I, MJall=M.J)

    M.cut(S)
    # dra,ddec are in degrees.
    dra,ddec = M.dra,M.ddec
    # make the errors isotropic
    dra *= A.rascale

    mdh = np.histogram(M.getdist_mas(), bins=100, range=(0, mdhrad*1000.))

    alplot.update(MI=M.I, MJ=M.J)

    return i,j, (dra,ddec,matchra,matchdec,w, M.I, M.J, mdh, alplot)


def intrabrickshift(TT, matchradius = 1.,
                    do_rotation=False, do_affine=False,
                    refradecs = None,
                    mag1diff = None, mag2diff = None,
                    sip_order = None,
                    sip_groups = None,
                    refxys = None,
                    cdmatrices = None,
                    align_kwargs = {},
                    lsqr_kwargs = {},
                    ref=None, refrad=0.,
                    # Cuts to apply to the catalogs before matching to ref.
                    refmag1cut=None, refmag2cut=None,
                    mp=None,
                    cam='(cam)', refcam='(refcam)',
                    mdhrad=None,
                    alignplotargs={},
                    # If non-None: a 2-d numpy array of booleans, indicating whether
                    # fields i,j overlap.
                    overlaps=None,
                    # Save alignment results?
                    save_aligngrid = False,
                    ):
    '''
    match radius is in arcsec

    refxys: list, one element per field, of (crpix0, crpix1)
    cdmatrices: list, one per field, of (cd11, cd12, cd21, cd22)
          = [ d(isotropic RA)/dx, d(iRA)/dy, dDec/dx, dDec/dy ]
    sip_groups: list of integers, same length as TT and starting at
         zero; fields with the same sipgroup will share SIP polynomial
         coefficients.

    AA: [ [Alignment],] objects; AA[i][j] is Alignment between fields
    i and j, or None for no matches.
    '''
    do_sip = False

    t0 = Time()
    
    if sip_order is not None:
        do_sip = True
        do_affine = True
        assert(refxys is not None)
        assert(cdmatrices is not None)
        assert(sip_groups is not None)
        assert(len(sip_groups) == len(TT))
        assert(len(refxys) == len(TT))
        assert(len(cdmatrices) == len(TT))
        Nsip = n_sip_terms(sip_order)
        Nsipgroups = len(np.unique([sg for sg in sip_groups if sg != -1]))
        print 'sip groups:', np.unique([sg for sg in sip_groups if sg != -1])
        print 'Solving for', Nsip, 'SIP terms for each of', Nsipgroups, 'groups'

    if do_rotation and do_affine:
        print 'Both do_rotation and do_affine are set -- doing affine.'
        do_affine = True
        do_rotation = False

    intra = Intra(len(TT))
    intra.matchrad_arcsec = matchradius
    #intra.AA = {}


    ramin,ramax = [],[]
    decmin,decmax = [],[]
    for i,Ti in enumerate(TT):
        ramin.append(Ti.ra.min())
        ramax.append(Ti.ra.max())
        decmin.append(Ti.dec.min())
        decmax.append(Ti.dec.max())

        if refradecs is None:
            intra.set_reference_radec(i, (ramin[i]+ramax[i])/2., (decmin[i]+decmax[i])/2.)
        else:
            intra.set_reference_radec(i, *refradecs[i])

    # global parameters
    Nglobal = 0
    if do_sip:
        Nglobal = Nsip * Nsipgroups

    # number of params to solve for, per field
    if do_affine:
        Nparams = 6
    elif do_rotation:
        Nparams = 3
    else:
        Nparams = 2

    # If rotation:
    #   parameter order: (shiftRA, shiftDec, theta)_(field 0)
    # If affine:
    #   parameter order: (shiftRA, shiftDec,
    #                     Tra_dra, Tra_ddec, Tdec_dra, Tdec_ddec)_(field 0)
    offset = { 'ra': 0, 'dec': 1,
               'rot': 2,
               'Tra_dra': 2, 'Tra_ddec':3, 'Tdec_dra':4, 'Tdec_ddec': 5,
               }

    Nfields = len(TT)

    if mdhrad is None:
        mdhrad = matchradius

    silent = not (mp is None)
    print 'silent?', silent

    # In order to use parallel "map", we build a list of arguments and
    # unpack results below.
    margs = []
    for i,Ti in enumerate(TT):
        for j in range(i, len(TT)):
            if j == i:
                if ref is None:
                    continue
                # Match to reference catalog.
                #print 'File', i, 'has', len(Ti)
                P = Ti
                # "refmag[12]cut" are just mis-named; they are how to cut the non-ref
                # camera to match to ref.
                if refmag1cut is not None:
                    P = P[P.mag1 < refmag1cut]
                    print 'Cut to', len(P), 'passing mag1 cut at', refmag1cut
                if refmag2cut is not None:
                    P = P[P.mag2 < refmag2cut]
                    print 'Cut to', len(P), 'passing mag2 cut at', refmag2cut

                margs.append((P, ref, refrad, {}, None, None, i, j, mdhrad,
                              alignplotargs, silent))

            else:
                Tj = TT[j]
                #print
                #print 'Looking for overlap between files', i, 'and', j

                ## FIXME -- Could use WCS geometry here to do better overlap-checking!

                if overlaps is not None:
                    if not overlaps[i,j]:
                        #print 'According to overlaps array, skipping %i -- %i' % (i,j)
                        continue

                if ((ramin[i] > ramax[j]) or (ramin[j] > ramax[i]) and
                    (decmin[i] > decmax[j]) or (decmin[j] > decmax[i])):
                    #print 'No overlap of RA,Dec ranges'
                    continue
                margs.append((Ti, Tj, matchradius, align_kwargs, mag1diff, mag2diff, i, j,
                              mdhrad, alignplotargs, silent))

    print 'Intrabrickshift: before alignment:'
    print Time()-t0

    if mp is None:
        aligns = map(alignfunc, margs)
    else:
        aligns = mp.map(alignfunc, margs)

        ## Boo hiss, map_async returns a single AsyncResult object, not a list of them
        # ares = mp.map_async(alignfunc, margs)
        # N = len(ares)
        # sares = set(ares)
        # while len(sares):
        #   done = set()
        #   for x in sares:
        #       if x.ready():
        #           done.add(x)
        #   sares.difference_update(done)
        #   print 'Alignments: %i of %i done, waiting for %i.' % (len(done), N, len(sares))
        #   if len(sares) == 0:
        #       break
        #   time.sleep(1.)
        # aligns = [x.get() for x in ares]

    del margs

    print 'Intrabrickshift: after alignment:'
    Time()-t0

    # build 2-D map aligngrid.
    aligngrid = {}
    # Match-distance histogram grid
    mdhgrid = {}
    # Alignment plot summary grid
    alplotgrid = {}
    for i,j,X in aligns:
        if X is None:
            continue
        if not i in aligngrid:
            aligngrid[i] = {}
            mdhgrid[i] = {}
            alplotgrid[i] = {}

        aligngrid[i][j] = X[:-2]
        mdhgrid[i][j] = X[-2]
        alplotgrid[i][j] = X[-1]
    del aligns

    intra.mdhgrid = mdhgrid
    # Summary information about Alignment between pairs of fields
    intra.alplotgrid = alplotgrid

    if save_aligngrid:
        intra.aligngrid = aligngrid

    NC = Nfields * Nparams + Nglobal
    print 'NC', NC
    colnorm = np.zeros(NC)
    sprows = []
    spcols = []
    spvals = []
    allresids = []
    row0 = 0
    def addelements(v, r, c):
        '''
        v: array of matrix elements
        r: array of matrix rows (integers)
        c: int, matrix column
        '''
        assert(len(v) == NR)
        assert(len(v) == len(r))
        assert(type(c) is int)
        assert(c >= 0)
        assert(c < NC)
        assert(len(colnorm) == NC)
        vals.append(v)
        rows.append(r)
        cols.append([c]*len(v))
        colnorm[c] += np.sum(v**2)

    for i,Ti in enumerate(TT):
        for j in range(i, len(TT)):
            jmoves = True
            X = aligngrid.get(i,{}).get(j,None)
            if X is None:
                continue
            if i == j:
                jmoves = False
                Tj = None #?
            else:
                Tj = TT[j]

            (dra,ddec, matchra,matchdec, w, MIS, MJS) = X

            refrai,refdeci = intra.get_reference_radec(i)
            if jmoves:
                refraj,refdecj = intra.get_reference_radec(j)

            # Distances from the reference points of fields i/j, in deg.
            # We make these isotropic too (as required by Affine)
            idra, iddec = matchra - refrai, matchdec - refdeci
            idra *= intra.get_rascale(i)
            if jmoves:
                jdra, jddec = matchra - refraj, matchdec - refdecj
                jdra *= intra.get_rascale(j)

            if do_sip:
                idx = Ti.x[MIS] - refxys[i][0]
                idy = Ti.y[MIS] - refxys[i][1]
                assert(len(idx) == len(dra))
                assert(len(idy) == len(idx))
                if jmoves:
                    jdx = Tj.x[MJS] - refxys[j][0]
                    jdy = Tj.y[MJS] - refxys[j][1]
                    assert(len(jdx) == len(idx))
                    assert(len(jdy) == len(idx))

            assert(len(dra) == len(ddec))
            assert(len(matchra) == len(matchdec))
            assert(len(dra) == len(matchra))
            assert(len(idra) == len(iddec))
            assert(len(idra) == len(dra))
            if jmoves:
                assert(len(jdra) == len(jddec))
                assert(len(idra) == len(jdra))

            # We build up the sparse matrix of derivatives of residuals
            # in RA,Dec wrt parameters.
            # The rows are the residuals
            # The cols are the parameters

            # Parameters: for each field, shiftRA, shiftDec,
            # [rotation or affine]

            # Residuals: we stack all delta-RA then all delta-Dec
            # for this edge.

            # Recall that each residual is affected by the parameters
            # from the two fields (if j moves)

            # number of residuals for this edge.
            NR = len(dra)

            assert(len(w) == len(dra))

            vals = []
            rows = []
            cols = []

            # The columns are the parameters, for fields i and j
            # II is the start of the block of "Nparams" parameters of field i
            II = i * Nparams
            JJ = j * Nparams

            # Global parameters go after the per-field parameters.
            GG = Nfields * Nparams
            
            # The rows are the residual terms:
            drarows  = row0 +      np.arange(NR)
            ddecrows = row0 + NR + np.arange(NR)
            row0 += 2 * NR
            # Their order is determined by this...
            allresids.append(dra  * w)
            allresids.append(ddec * w)
            assert(row0 == sum([len(a) for a in allresids]))

            # Above, we scaled dra to make the residuals isotropic.
            # We will compute the parameters in isotropic coords, matching Affine.

            # delta RA / (shift RA_i)    -- this is "isotropic RA"
            addelements( w.copy(), drarows,  II + offset['ra'])
            # delta Dec / (shift Dec_i)
            addelements( w.copy(), ddecrows, II + offset['dec'])

            if jmoves:
                # delta RA / (shift RA_j)
                addelements(-w, drarows,  JJ + offset['ra'])
                # delta Dec / shift (Dec_j)
                addelements(-w, ddecrows, JJ + offset['dec'])

            if do_sip:
                k = 0
                for order in range(2, sip_order+1):
                    for xorder in range(0, order+1):
                        yorder = order - xorder
                        # Here we compute the derivatives for the SIP terms
                        # A_xorder_yorder and B_xorder_yorder.

                        # These SIP coefficients affect both fields I and J.
                        siptermi = (idx ** xorder * idy ** yorder)
                        # Need to push this through CD matrices for I,
                        # J to see how it affects dRA,dDec.
                        cdi = cdmatrices[i]
                        si = sip_groups[i]
                        if jmoves:
                            siptermj = (jdx ** xorder * jdy ** yorder)
                            cdj = cdmatrices[j]
                            sj = sip_groups[j]

                        if si != -1:
                            # da: SIP term affecting X position (A_*)
                            dra_da_i  =  siptermi * cdi[0]
                            ddec_da_i =  siptermi * cdi[2]
                            # db: SIP term affecting Y position (B_*)
                            dra_db_i  =  siptermi * cdi[1]
                            ddec_db_i =  siptermi * cdi[3]
                            # "A" term is at position GG + Nsip * (si/sj) + k
                            # "B" term is at position  that + 1
                            addelements(w *  dra_da_i, drarows,  GG + Nsip * si + k)
                            addelements(w *  dra_db_i, drarows,  GG + Nsip * si + k + 1)
                            addelements(w * ddec_da_i, ddecrows, GG + Nsip * si + k)
                            addelements(w * ddec_db_i, ddecrows, GG + Nsip * si + k + 1)

                        if jmoves and sj != -1:
                            dra_da_j  = -siptermj * cdj[0]
                            ddec_da_j = -siptermj * cdj[2]
                            dra_db_j  = -siptermj * cdj[1]
                            ddec_db_j = -siptermj * cdj[3]
                            addelements(w *  dra_da_j, drarows,  GG + Nsip * sj + k)
                            addelements(w *  dra_db_j, drarows,  GG + Nsip * sj + k + 1)
                            addelements(w * ddec_da_j, ddecrows, GG + Nsip * sj + k)
                            addelements(w * ddec_db_j, ddecrows, GG + Nsip * sj + k + 1)

                        k += 2
                assert(k == Nsip)
                    
            if do_affine:
                # Here, affine element Tra_ddec (called "T_RA_i / ddec" here)
                # multiplies "iddec" to affect "drarows".

                # These comments are only useful to dstn
                # dRA / ( (dT_RA_i / dRA) )  -- Affine T[0] element
                addelements( idra * w,  drarows,  II + offset['Tra_dra'])
                # delta RA / ( T_RA_i / dDec )
                addelements( iddec * w, drarows,  II + offset['Tra_ddec'])
                # delta Dec / ( T_Dec_i / dRA )
                addelements( idra * w,  ddecrows, II + offset['Tdec_dra'])
                # delta Dec / ( T_Dec_i / dDec )
                addelements( iddec * w, ddecrows, II + offset['Tdec_ddec'])

                if jmoves:
                    # delta RA / ( T_RA_j / dRA )
                    addelements(-jdra * w,  drarows,  JJ + offset['Tra_dra'])
                    # delta RA / ( T_RA_j / dDec )
                    addelements(-jddec * w, drarows,  JJ + offset['Tra_ddec'])
                    # delta Dec / ( T_Dec_j / dRA )
                    addelements(-jdra * w,  ddecrows, JJ + offset['Tdec_dra'])
                    # delta Dec / ( T_Dec_j / dDec )
                    addelements(-jddec * w, ddecrows, JJ + offset['Tdec_ddec'])

            elif do_rotation:
                # delta RA / theta i
                #   ->  (-iddec / rascale) is the change in RA; then scale by rascale * w
                addelements(-iddec * w, drarows,  II + offset['rot'])
                # delta Dec / theta i
                addelements(  idra * w, ddecrows, II + offset['rot'])

                if jmoves:
                    # delta RA / theta j
                    addelements(+jddec * w, drarows,  JJ + offset['rot'])
                    # delta Dec / theta j
                    addelements(-jdra * w, ddecrows, JJ + offset['rot'])

            if len(rows):
                sprows.extend(rows)
                spcols.extend(cols)
                spvals.extend(vals)

    del aligngrid


    print 'Intrabrickshift: after building sparse:'
    Time()-t0

    # Divide each column (parameter derivative) by its L2 norm
    print 'Scaling columns...'
    colnorm = np.sqrt(colnorm)
    for cc,vv in zip(spcols, spvals):
        # assert(all cc are the same int)
        # If this assert triggers, you filled a column with zeros.
        assert(colnorm[cc[0]] >= 0)
        vv /= colnorm[cc[0]]

    sprows = np.hstack(sprows)
    spcols = np.hstack(spcols)
    spvals = np.hstack(spvals)
    allresids = np.hstack(allresids)

    print 'sparse matrix:', len(sprows), len(spcols), len(spvals)
    print 'rhs:', len(allresids)

    print 'Intrabrickshift: after building sparse (2):'
    Time()-t0

    NR = len(allresids)
    print 'Building sparse matrix...', (NR,NC)
    #print 'max row', np.max(sprows)
    #print 'max col', np.max(spcols)
    M = scipy.sparse.csr_matrix((spvals, (sprows, spcols)), shape=(NR,NC))

    del spvals
    del sprows
    del spcols

    print 'Intrabrickshift: after building sparse (3):'
    Time()-t0

    # lsqr can trigger floating-point errors
    np.seterr(all='warn')
    X = scipy.sparse.linalg.lsqr(M, -allresids, show=True, **lsqr_kwargs)
    set_fp_err()
    (J, istop, niters, r1norm, r2norm, anorm, acond, arnorm, xnorm, var) = X
    print 'column-scaled J norm:', np.sqrt(np.sum(J**2))
    print 'initial r2 norm:', np.sqrt(np.sum(allresids**2))
    print 'final   r2 norm:', r2norm
    # 
    J = np.array(J)
    J /= colnorm
    #print 'Column norms:', colnorm
    bad = (colnorm == 0.0)
    if any(bad):
        J[bad] = 0.0
        print 'Ignoring zero-norm columns; setting J=0'
    print 'unscaled J norm', np.sqrt(np.sum(J**2))

    print 'Intrabrickshift: after lsqr:'
    Time()-t0
    del M
    del allresids
    print 'Intrabrickshift: after lsqr (2):'
    Time()-t0

    # Chop off the global parameters.
    if Nglobal:
        Jglobals = J[-Nglobal:]
        J = J[:-Nglobal]
    else:
        Jglobals = []

    if do_sip:
        Jsip = Jglobals[:Nsip*Nsipgroups]
        Jglobals = Jglobals[Nsip*Nsipgroups:]
        print 'Jsip:', Jsip
        # Split into the SIP terms for each group;
        # sips[group] are the SIP terms for SIP group "group".
        sips = [ Jsip[Nsip*i: Nsip*(i+1)] for i in range(Nsipgroups) ]
        print 'sips:', sips
        #
        sipterms = []
        for sg in sip_groups:
            if sg == -1:
                sipterms.append(None)
            else:
                sipterms.append(sips[sip_groups[sg]])
        #intra.set_sip_terms([sips[sip_groups[i]] for i in range(Nfields)])
        intra.set_sip_terms(sipterms)
        print 'Intra SIP terms:', intra.sip

    # We better have grabbed all the globals
    assert(len(Jglobals) == 0)

    # The parameter unpacking is determined by the 'cols' offsets above.
    # The parameters are in blocks of "Nparams" columns for each field, so to pull
    # out the RA shift for all fields, stride through in steps of Nparams.

    # Note that here we rescale from isotropic back to real RA
    sra  = np.array( J[offset['ra']  ::Nparams] ) / np.array(intra.get_rascales())
    sdec = np.array( J[offset['dec'] ::Nparams] )
    intra.set_shifts(sra, sdec)

    if do_affine:
        intra.set_affine(np.array( J[offset['Tra_dra']   ::Nparams] ),
                         np.array( J[offset['Tra_ddec']  ::Nparams] ),
                         np.array( J[offset['Tdec_dra']  ::Nparams] ),
                         np.array( J[offset['Tdec_ddec'] ::Nparams] ))
    elif do_rotation:
        rot = np.rad2deg(J[offset['rot'] ::Nparams])
        intra.set_rotation(rot)

    if do_sip:
        for i in range(len(TT)):
            aff = intra.get_affine(i)
            aff.refra,aff.refdec = refradecs[i]
            aff.refxy = refxys[i]
            aff.cdmatrix = cdmatrices[i]
            aff.sipterms = intra.sip[i]
        
    for i,(sr,sd) in enumerate(zip(sra, sdec)):
        print
        print ('Field index %i: shift by (%.1f, %.1f) milli-arcsec' %
               (i, sr*3600.*1000., sd*3600.*1000.))
        if do_rotation:
            print '  Rotate by %.1f arcsec' % (rot[i] * 3600.)
        if do_affine:
            print '  Affine:', intra.get_affine(i)

    return intra


def doplots(intra, TT, ps, wcs, xyplots, edgeplots, eps, rad, smallrad, vecscale,
            radcircles, markshifts=True, magname=None):

    # Ridiculously, colorbar can cause floating-point errors (on purposes, it seems)
    err = np.geterr()
    np.seterr(all='warn')

    if len(TT) == 18:
        subp = (3,6)
    else:
        subp = None

    for app in [ False, True]:
        intra.vecplot(TT, apply=app, scale=vecscale)
        plt.title('x %f' % vecscale)
        ps.savefig()

    rax = [-rad,rad]*2
    sax = [-smallrad,smallrad]*2

    for app in [ False, True ]:
        intra.scatterplot(TT, apply=app, markshifts=markshifts)
        angles = np.linspace(0, 2.*pi, 180)
        if radcircles:
            for r in np.arange(0, rad+0.4, 0.5):
                plt.plot(r * np.sin(angles), r * np.cos(angles), 'g-')
        plt.axis(rax)
        ps.savefig()

    # then zoom...
    plt.axis(sax)
    ps.savefig()

    cam = TT[0].cam
    tt = '%s-%s matches' % (cam,cam)

    intra.magplot(TT, apply=True, hist=True, step=1., maxerr=smallrad)
    plt.title(tt)
    if magname is not None:
        plt.xlabel(magname + ' (mag)')
    ps.savefig()

    intra.magplot(TT, apply=True, errbars=True, maxerr=smallrad)
    plt.title(tt)
    if magname is not None:
        plt.xlabel(magname + ' (mag)')
    ps.savefig()

    if xyplots:
        intra.xyoffsets(TT, wcs=wcs, subplotsize=subp)
        ps.savefig()
        intra.xyoffsets(TT, wcs=wcs, apply=True, subplotsize=subp)
        ps.savefig()

    if edgeplots:
        print 'edgeplot...'
        intra.edgeplot(TT, eps)
        print 'edgescatter...'
        intra.edgescatter(eps)

    np.seterr(**err)


# fields: eg, fieldlist.getACS(1)
def intra_main(fields,
               mag1cut=None,
               radius=2,
               # Maximum magnitude difference for matches.
               mag1diff = None, mag2diff = None
               ):
    paths = [f.gst for f in fields]
    TT = readGsts(paths)

    for T,cam in zip(TT, [f.cam for f in fields]):
        cammeta = describeFilters(cam, T)
        T.cam = cam
    magname = cammeta.fnames[0]
    print 'Mag1 name:', magname

    TT1 = TT
    if mag1cut is not None:
        TT1 = [ Ti[Ti.mag1 < mag1cut] for Ti in TT ]
        print 'After mag1 cut at', mag1cut, 'got', sum([len(Ti) for Ti in TT1])
                
    intra = intrabrickshift(TT1, matchradius=radius,
                            mag1diff=mag1diff, mag2diff=mag2diff)

    return intra, TT



def main():
    import sys

    parser = OptionParser(usage='%(program) [options] <field-list.py filename>')
    parser.add_option('-r', '--searchrad', dest='rad', type='float', help='Search radius (default 2")', default=2.)
    parser.add_option('-s', '--smallrad', dest='smallrad', type='float', default=0.2)
    parser.add_option('--mag1cut', dest='mag1cut', type='float', help='Magnitude #1 cut for first-round (full-radius) search.')
    parser.add_option('--mag2cut', dest='mag2cut', type='float', help='Magnitude #2 cut for first-round (full-radius) search.')
    parser.add_option('--edge-plots', dest='edgeplots', default=False, action='store_true')
    parser.add_option('--xy-plots', dest='xyplots', default=False, action='store_true')
    parser.add_option('--rotation', dest='rotation', default=False, action='store_true')
    parser.add_option('--affine', dest='affine', default=False, action='store_true')
    parser.add_option('--plots', dest='basefn', help='Plot base filename')
    parser.add_option('--output', '-o', dest='outfn', help='Output filename')
    parser.add_option('-p', '--path', dest='path', help='Path to .gst.fits files (default: "data/pipe/*/proc")', default='data/pipe/*/proc')
    parser.add_option('-w', '--wcspath', dest='wcspath', help='Path to .wcs files (default: "data/pipe/*/proc")', default='data/pipe/*/proc')
    parser.add_option('-e', '--ext', dest='ext', help='Read WCS from this FITS extension number (default: 0)', default=0, type='int')

    opt,args = parser.parse_args()

    if opt.outfn is None:
        parser.print_help()
        print 'Need output filename (--output)'
        sys.exit(-1)

    fns = args
    if len(fns) != 1:
        parser.print_help()
        print 'Need a field list file.'
        sys.exit(-1)

    fn = fns[0]
    print 'Executing config file', fn
    loc = locals()
    execfile(fn, globals(), loc)
    fields = loc['fields']

    print 'Got fields:'
    for f in fields:
        print '  ', f

    fns = [f.gst for f in fields]
    wcsfiles = [f.wcs for f in fields]
    wcs = [Tan(findFile(fn, opt.wcspath), opt.ext) for fn in wcsfiles]
    bboxes = get_bboxes(wcs)

    paths = [findFile(fn, opt.path) for fn in fns]
    TT = readGsts(paths)
    for T,cam in zip(TT, [f.cam for f in fields]):
        cammeta = describeFilters(cam, T)
        T.cam = cam
    magname = cammeta.fnames[0]
    print 'Mag1 name:', magname

    rad = opt.rad
    smallrad = opt.smallrad
    tinyrad = 0.05

    if opt.basefn is not None:
        ps = PlotSequence(opt.basefn+'-', format='%02i')
        ps2 = PlotSequence(opt.basefn + '-edge-', format='%03i')
        plots = True
    else:
        ps = None
        ps2 = None
        plots = False

    # Don't do rotation and affine in the first round...
    TT1 = TT
    if opt.mag1cut is not None:
        TT1 = [ Ti[Ti.mag1 < opt.mag1cut] for Ti in TT ]
        print 'After mag1 cut at', opt.mag1cut, 'got', sum([len(Ti) for Ti in TT1])
    if opt.mag2cut is not None:
        TT1 = [ Ti[Ti.mag2 < opt.mag2cut] for Ti in TT ]
        print 'After mag2 cut at', opt.mag2cut, 'got', sum([len(Ti) for Ti in TT1])
                
    intra = intrabrickshift(TT1, matchradius=rad)

    if plots:
        doplots(intra, TT1, ps, wcs, opt.xyplots, opt.edgeplots,
                ps2, rad, smallrad, 100., True, magname=magname)

    # Save some memory...
    intra.trimBeforeSaving()

    # Apply offsets in-place
    intra.applyTo(TT)

    # Round 2:

    # Use the same reference RA,Dec as in the first round.
    refradecs = intra.get_reference_radecs()
    # First do shifts, then affines.
    intra2 = intrabrickshift(TT, matchradius=smallrad, refradecs=refradecs)

    if plots:
        plotaffinegrid(intra.affines, exag=1e2, affineOnly=False, tpre='Step 1: ',
                       bboxes=bboxes)
        ps.savefig()
        plotaffinegrid(intra2.affines, exag=1e3, affineOnly=False, tpre='Step 2: ',
                       bboxes=bboxes)
        ps.savefig()
        resetplot()
        doplots(intra2, TT, ps, wcs, opt.xyplots, opt.edgeplots, ps2,
                smallrad, tinyrad, 1000., False, magname=magname)

    intra2.trimBeforeSaving()

    intras = [intra2]

    if opt.rotation or opt.affine:
        # Apply
        intra2.applyTo(TT)

        for step in [3,4]:
            # Now do affines
            intra3 = intrabrickshift(TT, matchradius=smallrad, refradecs=refradecs,
                                     do_rotation = opt.rotation,
                                     do_affine = opt.affine)
            intras.append(intra3)

            if plots:
                plotaffinegrid(intra3.affines, exag=1e3, affineOnly=False,
                               tpre='Step %i: ' % step, bboxes=bboxes)
                ps.savefig()
                plotaffinegrid(intra3.affines, exag=1e3, affineOnly=True,
                               tpre='Step %i: ' % step, bboxes=bboxes)
                ps.savefig()
                plotaffinegrid(intra3.affines, exag=1e4, affineOnly=False,
                               tpre='Step %i: ' % step, bboxes=bboxes)
                ps.savefig()
                plotaffinegrid(intra3.affines, exag=1e4, affineOnly=True,
                               tpre='Step %i: ' % step, bboxes=bboxes)
                ps.savefig()
                resetplot()
                doplots(intra3, TT, ps, wcs, opt.xyplots, opt.edgeplots, ps2,
                        smallrad, tinyrad, 1000., False, markshifts=False,
                        magname=magname)

            intra3.trimBeforeSaving()
            intra3.applyTo(TT)


    for iN in intras:
        intra.add(iN)
    T = intra.toTable()
    T.filename = fns
    T.fieldid = [getFieldId(f.cam, f.brick, f.field) for f in fields]
    T.wcsfilename = wcsfiles
    T.writeto(opt.outfn)


if __name__ == '__main__':
    main()
    #import cProfile
    #from datetime import datetime
    #T = datetime.utcnow()
    #cProfile.run('main()', 'profile-%s.dat' % T.isoformat())


