if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

from optparse import OptionParser
from math import sqrt, floor, ceil, pi
from glob import glob

import numpy as np
import pylab as plt
from scipy import linalg

from astrom_common import *
from astrom_intra import Intra
from astrom_merge import mergeBrick
#from astrom_merge2 import mergeBrick2
from astrometry.util.plotutils import antigray

def loadBrickCached(cam, brick, mergedfn=None, ps=None, **kwargs):
    if cam in ['CFHT', 'CFHT2']:
        return loadBrick(cam, **kwargs)
    T = mergeBrick(cam, brick, mergedfn, ps, **kwargs)
    if 'primary' in T.columns():
        T = T[T.primary]
        print 'After cutting on primary:', len(T)
    return T


def main():
    import sys

    parser = OptionParser(usage='%(program) [options] <gst.fits filenames>')
    parser.add_option('-b', '--brick', dest='brick', type='int', help='Brick')
    parser.add_option('-c', '--cam', dest='cam', help='Camera -- ACS, IR or UV', action=None)
    parser.add_option('--ref', dest='ref', help='Reference "camera" -- CFHT, ACS, IR or UV', action=None)
    parser.add_option('--refmerged', dest='refmergedfn', help='File to read/write merged reference sources from/into')
    #parser.add_option('--refitab', dest='refitab', help='Reference source table')
    parser.add_option('--refmagcut', dest='refmagcut', type='float', help='Reference mag cut')
    parser.add_option('-p', '--path', dest='path', help='Path to .gst.fits files (default: "data/pipe/*/proc")')
    parser.add_option('-r', '--radius', dest='radius', type='float', help='Search radius (default 1")', default=1.)
    parser.add_option('-m', '--magcut', dest='magcut', type='float', help='mag cut (default: 22 for ACS, 21 for IR)')
    parser.add_option('-R', '--rotation', dest='rotation', type='float', help='Apply this rotation correction (default=0 deg)', default=0.)
    parser.add_option('-s', '--smallrad', dest='smallrad', type='float', help='Small search radius (default 0.1")', default=0.1)
    parser.add_option('-E', '--emrad', dest='emrad', type='float', help='Radius for EM (default: searchrad)')
    parser.add_option('--merged', dest='mergedfn', help='File to read/write merged sources from/into')
    #parser.add_option('--itab', dest='itab', help='Target source table')

    parser.add_option('-G', '--grid', dest='grid', action='store_true', default=False,
                      help='Show a grid of the 18 fields in this brick.')

    parser.add_option('-B', '--basefn', dest='basefn',
                      help='Base output filename for plots')

    parser.add_option('--rot-lo', dest='rotlo', type='float',
                      help='Search rotations from --rot-lo to --rot-hi in steps of --rot-step')
    parser.add_option('--rot-hi', dest='rothi', type='float')
    parser.add_option('--rot-step', dest='rotstep', type='float', default=0.01)

    parser.add_option('--output', '-o', dest='outfn', help='Output filename (affine FITS)', default=None)

    opt,args = parser.parse_args()

    if opt.brick is None or opt.cam is None:
        parser.print_help()
        print 'Need --brick and --cam'
        sys.exit(-1)

    if opt.emrad is None:
        opt.emrad = opt.radius

    #if opt.itab is not None:
    #   opt.itab = fits_table(opt.itab)
    #if opt.refitab is not None:
    #   opt.refitab = fits_table(opt.refitab)

    if opt.basefn is None:
        basefn = 'inter-%02i-%s-%s' % (opt.brick, opt.cam, opt.ref)
    else:
        basefn = opt.basefn
    ps = PlotSequence(basefn+'-', format='%02i')

    Tme = loadBrickCached(opt.cam, opt.brick, path=opt.path, mergedfn=opt.mergedfn,
                          #itab=opt.itab,
                          ps=ps)
    me = describeFilters(opt.cam, Tme)

    Tref = loadBrickCached(opt.ref, opt.brick, path=opt.path, mergedfn=opt.refmergedfn,
                           #itab=opt.refitab,
                           ps=ps)
    ref = describeFilters(opt.ref, Tref)

    i,j = getNearMags(me, ref)
    Tme.cam = opt.cam
    Tme.mag = Tme.get('mag%i' % (i+1))
    Tme.filter = me.fnames[i]
    Tref.cam = opt.ref
    Tref.mag = Tref.get('mag%i' % (j+1))
    Tref.filter = ref.fnames[j]

    if opt.magcut is not None:
        I = (Tme.mag < opt.magcut)
        Tme = Tme[I]
        print 'Got', len(Tme), 'after mag cut (at', opt.magcut, ')'

    if opt.refmagcut is not None:
        I = (Tref.mag < opt.refmagcut)
        Tref = Tref[I]
        print 'Got', len(Tref), 'reference after mag cut (at %g)' % opt.refmagcut
        
    rl,rh = Tme.ra.min(), Tme.ra.max()
    dl,dh = Tme.dec.min(), Tme.dec.max()
    dmid = (dl+dh)/2.
    rmid = (rl+rh)/2.

    def rotate_radec(rot, ra, dec, refra, refdec):
        trans = Affine()
        trans.setRotation(rot, smallangle=False)
        trans.setReferenceRadec(refra, refdec)
        newra,newdec = trans.apply(ra, dec)
        return newra, newdec, trans

    rot = 0
    trans0 = None

    if opt.rotation != 0.:
        rot = opt.rotation
        # rotate.
        print 'Applying rotation correction of', rot, 'deg'
        Tme.ra, Tme.dec, trans0 = rotate_radec(rot, Tme.ra, Tme.dec, rmid, dmid)

    elif opt.rotlo is not None and opt.rothi is not None:
        lo = opt.rotlo
        hi = opt.rothi
        step = opt.rotstep
        print 'Trying rotations between', lo, 'and', hi, 'in steps of', step
        variances = []
        rots = np.arange(lo, hi+step/2., step)
        for rot in rots:
            print 'Rotation', rot
            Tm = Tme.copy()
            Tm.ra, Tm.dec, nil = rotate_radec(rot, Tm.ra, Tm.dec, rmid, dmid)
            print 'Matching...'
            M = Match(Tm, Tref, opt.radius)
            print 'Got %i matches' % len(M.I)

            nbins = 200
            H,xe,ye = plothist(M.dra_arcsec, M.ddec_arcsec, nbins)
            plt.xlabel('dRA (arcsec)')
            plt.ylabel('dDec (arcsec)')
            plt.title('Rotated by %g deg' % rot)
            ps.savefig()

            plotresids(Tm, M, 'Rotated by %g deg' % rot, bins=100)
            ps.savefig()

            # Trim the circle to avoid edge effects, and then measure the variance.
            X,Y = np.meshgrid(np.arange(nbins), np.arange(nbins))
            R2 = (X - nbins/2.)**2 + (Y - nbins/2.)**2
            I = (R2 < (0.95 * (nbins/2)**2))
            v = np.var(H[I])
            print 'Variance:', v
            variances.append(v)

        plt.clf()
        plt.plot(rots, variances, 'r-')
        plt.xlabel('Rotation (deg)')
        plt.ylabel('Variance in dRA,dDec histogram')
        ps.savefig()
    
        I = np.argmax(variances)
        rot = rots[I]

        print 'Applying rotation correction of', rot, 'deg'
        Tme.ra, Tme.dec, trans0 = rotate_radec(rot, Tme.ra, Tme.dec, rmid, dmid)

    if trans0 is not None:
        print 'Setting initial rotation affine transformation:'
        print trans0

    A = alignAndPlot(Tme, Tref, opt.radius, ps, emrad=opt.emrad, doweighted=False)
    #print 'Cov:', A.C

    trans = findAffine(Tme, Tref, A, (rmid,dmid))

    RR,DD = np.meshgrid(np.linspace(rl, rh, 20),
                        np.linspace(dl, dh, 20))
    RR = RR.ravel()
    DD = DD.ravel()

    plotaffine(trans, RR, DD, exag=1.)
    setRadecAxes(rl,rh,dl,dh)
    ps.savefig()

    plotaffine(trans, RR, DD, exag=100.)
    setRadecAxes(rl,rh,dl,dh)
    ps.savefig()

    exag = 1000.
    plotaffine(trans, RR, DD, exag, affineOnly=True)
    ps.savefig()
    
    Tme.ra,Tme.dec = trans.apply(Tme.ra, Tme.dec)

    # Do it again!

    A2 = alignAndPlot(Tme, Tref, opt.smallrad, ps, doweighted=False, emrad=opt.smallrad)
    trans2 = findAffine(Tme, Tref, A2, (rmid,dmid))
    Tme.ra,Tme.dec = trans2.apply(Tme.ra, Tme.dec)

    # For the 'after' plots

    A3 = alignAndPlot(Tme, Tref, opt.smallrad, ps, doweighted=False, emrad=opt.smallrad)

    # Save
    if opt.outfn:
        if trans0 is None:
            trans.add(trans2)
        else:
            trans0.add(trans)
            trans0.add(trans2)
            trans = trans0
        T = Affine.toTable([trans])
        T.writeto(opt.outfn)


def findAffine(Tme, Tref, A, refradec, affine=True, order=1):
    '''
    Computes an Affine transformation between two aligned catalogs.

    *Tme*: catalog to align
    *Tref*: reference catalog
    *A*: an Alignment object matching these two catalogs
    *refradec*: tuple (refra, refdec) of the reference point about which to
        rotate.
    *affine*: if True, produce an affine transformation; otherwise, just a shift
    *order*: polynomial distortion order.

    Returns:
    *Affine* object.
    '''
    refra,refdec = refradec
    rascale = np.cos(np.deg2rad(refdec))

    srdeg,sddeg = A.getshift()

    if not affine:
        affine = Affine(dra = -srdeg, ddec = -sddeg,
                        refra = refra, refdec = refdec)
        return affine
    assert(order >= 1)

    sr,sd = A.arcsecshift()
    w = np.sqrt(A.fore)
    M = A.match
    dra  = M.dra_arcsec [A.subset] - sr
    ddec = M.ddec_arcsec[A.subset] - sd
    ra  = Tme.ra [M.I[A.subset]]
    dec = Tme.dec[M.I[A.subset]]

    comps = [np.ones_like(ra) * w]
    for o in range(1, order+1):
        for deco in range(o+1):
            rao = o - deco
            rr = (ra  - refra )*rascale
            dd = (dec - refdec)
            # rr and dd are in isotropic degrees
            comps.append((rr ** rao) * (dd ** deco) * w)
            print 'ra order', rao, 'dec order', deco
    # In the linear case (order=1), the terms are listed as rao=1 then deco=1

    Amat = np.vstack(comps).T
    Amat = np.matrix(Amat)
    # dra,ddec are in isotropic degrees
    b1 = -dra  / 3600. * w
    b2 = -ddec / 3600. * w
    X1 = linalg.lstsq(Amat, b1)
    X2 = linalg.lstsq(Amat, b2)

    X1 = X1[0]
    X2 = X2[0]
    e,a,b = X1[:3]
    f,c,d = X2[:3]
    #print 'a,b,c,d', a,b,c,d
    #print 'e,f', e,f
    if order >= 2:
        rapoly  = X1[3:]
        decpoly = X2[3:]
    else:
        rapoly = decpoly = None
        
    affine = Affine(dra = e/rascale - srdeg, ddec = f - sddeg,
                    T = [ a, b, c, d ],
                    refra = refra, refdec = refdec,
                    rapoly=rapoly, decpoly=decpoly)
    return affine



'''
Returns the Alignment object A.
'''
def alignAndPlot(Tme, Tref, rad, ps, doweighted=True, emrad=None, nearest=False, **kwargs):
    aliargs = dict(cutrange=emrad)
    aliargs.update(kwargs)
    A = Alignment(Tme, Tref, searchradius=rad, **aliargs)
    if nearest:
        # There is something badly wrong with spherematch.nearest().
        assert(False)
        A.findMatches(nearest=True)
        M = A.match
        print 'dra,ddec arcsec:', M.dra_arcsec[:100], M.ddec_arcsec[:100]

    if A.shift() is None:
        print 'Shift not found!'
        return None
    M = A.match
    print 'Shift:', A.arcsecshift()
    sr,sd = A.arcsecshift()

    sumd2 = np.sum(A.fore * ((M.dra_arcsec [A.subset] - sr)**2 +
                             (M.ddec_arcsec[A.subset] - sd)**2))
    sumw  = np.sum(A.fore)
    # / 2. to get std per coord.
    std   = sqrt(sumd2 / (sumw * 2.))
    angles = np.linspace(0, 2.*pi, 100)

    modstr = ''
    if A.cov:
        eigs = A.getEllipseSize() * 1000.
        if eigs[0] > 100:
            modstr = '%.0fx%.0f' % (eigs[0], eigs[1])
        else:
            modstr = '%.1fx%.1f' % (eigs[0], eigs[1])
    else:
        modstr = '%.1f' % (1000. * A.sigma)

    W = np.zeros_like(A.subset).astype(float)
    W[A.subset] = A.fore

    rl,rh = Tme.ra.min(), Tme.ra.max()
    dl,dh = Tme.dec.min(), Tme.dec.max()

    if doweighted:
        rounds = [ {}, { 'weights': W } ]
    else:
        rounds = [ {} ]

    for i,args in enumerate(rounds):
        tsuf = '' if i == 0 else ' (weighted)'
        N = len(M.dra_arcsec) if i == 0 else sumw

        plotresids(Tme, M, '%s-%s match residuals%s' % (Tme.cam, Tref.cam, tsuf),
                   bins=100, **args)
        ps.savefig()

        dst = 1000. * np.sqrt(M.dra_arcsec ** 2 + M.ddec_arcsec ** 2)
        loghist(Tme.mag[M.I], dst, 100, **args)
        plt.xlabel(Tme.filter)
        plt.ylabel('Match residual (mas)')
        ps.savefig()

        loghist(Tref.mag[M.J], dst, 100, **args)
        plt.xlabel(Tref.filter)
        plt.ylabel('Match residual (mas)')
        ps.savefig()

        H,xe,ye = plotalignment(A)
        # show EM circle
        ax = plt.axis()
        angles = np.linspace(0, 2.*pi, 100)
        c = A.cutcenter
        r = A.cutrange
        plt.plot(c[0] + r * np.cos(angles), c[1] + r * np.sin(angles), 'g--')
        plt.axis(ax)
        plt.title('%s-%s (%i matches, std %.1f mas, model %s)%s' %
                  (Tme.cam, Tref.cam, int(sumw), std*1000., modstr, tsuf))
        ps.savefig()

        bins = 200
        edges = np.linspace(-rad, rad, bins)
        DR,DD = np.meshgrid(edges, edges)
        em = A.getModel(DR.ravel(), DD.ravel()).reshape(DR.shape)
        em *= len(M.dra_arcsec) * (edges[1]-edges[0])**2
        R2 = DR**2 + DD**2
        em[R2 > (A.match.rad)**2] = 0.
        plt.clf()
        plt.imshow(em, extent=(-rad, rad, -rad, rad),
                   aspect='auto', interpolation='nearest', origin='lower',
                   vmin=H.min(), vmax=H.max())
        plt.hot()
        plt.colorbar()
        plt.xlabel('dRA (arcsec)')
        plt.ylabel('dDec (arcsec)')
        plt.title('EM model')
        ps.savefig()

        plotfitquality(H, xe, ye, A)
        ps.savefig()




        rng = ((-5*std, 5*std), (-5*std, 5*std))
        myargs = args.copy()
        myargs.update({'range':rng})
        plothist(M.dra_arcsec - sr, M.ddec_arcsec - sd, 200, **myargs)
        ax = plt.axis()
        plt.xlabel('dRA (arcsec)')
        plt.ylabel('dDec (arcsec)')
        for nsig in [1,2]:
            X,Y = A.getContours(nsig)
            plt.plot(X-sr, Y-sd, 'b-')
        plt.axis(ax)
        plt.title('%s-%s (matches: %i, std: %.1f mas, model %s)%s' %
                  (Tme.cam, Tref.cam, int(sumw), std*1000., modstr, tsuf))
        ps.savefig()

        #plothist(Tme.mag[M.I], Tref.mag[M.J], 100, **args)
        #plt.xlabel('%s %s (mag)' % (Tme.cam, Tme.filter))
        #plt.ylabel('%s %s (mag)' % (Tref.cam, Tref.filter))
        #fn = '%s-%s.png' % (basefn, chr(ploti))
        #plt.title('%s-%s%s' % (Tme.cam, Tref.cam, tsuf))
        #plt.savefig(fn)
        #print 'saved', fn
        #ploti += 1

        loghist(Tme.mag[M.I], Tref.mag[M.J], 100, **args)
        plt.xlabel('%s (mag)' % (Tme.filter))
        plt.ylabel('%s (mag)' % (Tref.filter))
        plt.title('%s-%s%s' % (Tme.cam, Tref.cam, tsuf))
        ps.savefig()

        plothist(Tme.ra[M.I], Tme.dec[M.I], 100, range=((rl,rh),(dl,dh)))
        setRadecAxes(rl,rh,dl,dh)
        plt.title('%s-%s: %i matches%s' % (Tme.cam, Tref.cam, N, tsuf))
        ps.savefig()

    return A

if __name__ == '__main__':
    main()
