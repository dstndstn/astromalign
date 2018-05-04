from __future__ import print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

from optparse import OptionParser
from math import sqrt, floor, ceil, pi

import numpy as np
import pylab as plt

from astrom_common import *

def mergeBrick(cam, brick, mergedfn=None, ps=None, force=False, magcuts=[], **kwargs):
    if mergedfn is not None:
        mfn = mergedfn
    else:
        mfn = 'merged-%s-%02i.fits' % (brick, cam)

    if not force and os.path.exists(mfn):
        print('Reading', mfn)
        T = fits_table(mfn)
        print('Got', len(T), cam, 'sources')
        return T

    (T, Tall, M) = loadBrick(cam, **kwargs)
    print('Saving to', mfn)
    T.writeto(mfn)

    if ps is None:
        return T

    itab = kwargs.get('itab', None)
    if itab is not None:
        affines = Affine.fromTable(itab)
        for (exag,affonly) in [(1e2, False), (1e3, True), (1e4, True)]:
            plotaffinegrid(affines, exag=exag, affineOnly=affonly)
            ps.savefig()
        resetplot()
            
    rl,rh = Tall.ra.min(), Tall.ra.max()
    dl,dh = Tall.dec.min(), Tall.dec.max()
    rdrange = ((rl,rh),(dl,dh))

    F = describeFilters(cam, Tall)

    Imatched = np.zeros(len(Tall), bool)
    Imatched[M.I] = True
    for mag in [None]+magcuts:
        if mag is None:
            Icut = np.ones(len(Tall), bool)
            cutstr = ''
        else:
            Icut = (Tall.mag1 < mag)
            cutstr = ' (%s < %g)' % (F.fnames[0], mag)

        plothist(Tall.ra[Icut], Tall.dec[Icut], 200, range=rdrange)
        setRadecAxes(rl,rh,dl,dh)
        plt.title('All stars' + cutstr)
        ps.savefig()

        plothist(Tall.ra[Icut * Imatched], Tall.dec[Icut * Imatched], 200, range=rdrange)
        setRadecAxes(rl,rh,dl,dh)
        plt.title('Duplicate sources' + cutstr)
        ps.savefig()

        plothist(Tall.ra[Icut * np.logical_not(Imatched)],
                 Tall.dec[Icut * np.logical_not(Imatched)], 200, range=rdrange)
        setRadecAxes(rl,rh,dl,dh)
        plt.title('Merged' + cutstr)
        ps.savefig()

        #plotresids(Tall, M, title='Residuals: %s' % cam + cutstr)
        #ps.savefig()

    plotresids(Tall, M, title='Residuals: %s' % cam)
    ps.savefig()

    #plt.clf()
    #plothist(M.dra_arcsec*1000., M.ddec_arcsec*1000., 100)
    #ps.savefig()

    # run Alignment to get the EM-fit Gaussian size
    A = Alignment(Tall, Tall, match=M, cov=True, cutrange=M.rad)
    A.shift()
    H,xe,ye = plotalignment(A)
    eigs = A.getEllipseSize() * 1000.
    plt.title('Model: %.1f x %.1f mas' % (eigs[0], eigs[1]))
    ps.savefig()

    plotfitquality(H, xe, ye, A)
    ps.savefig()

    #plt.clf()
    #plt.imshow(mod.reshape(H.shape), 
    #              extent=(min(xe), max(xe), min(ye), max(ye)),
    #          aspect='auto', interpolation='nearest', origin='lower')
    #ps.savefig()

    

    return T

               
def main():
    import sys
    parser = OptionParser(usage='%(program) [options]')
    parser.add_option('-b', '--brick', dest='brick', type='int', help='Brick')
    parser.add_option('-c', '--cam', dest='cam', help='Camera -- ACS, IR or UV', action=None)
    parser.add_option('--merged', dest='mergedfn', help='File to read/write merged sources from/into')
    parser.add_option('--force', dest='force', default=False, action='store_true', help='Ignore previous merged file, and re-merge.')
    parser.add_option('--itab', dest='itab', help='Target source table')
    parser.add_option('-B', '--basefn', dest='basefn', help='Base output filename for plots')
    parser.add_option('--magcut', dest='magcut', action='append', type='float', default=[], help='Plot source density at this mag cut (this arg can be repeated)')

    opt,args = parser.parse_args()

    if opt.brick is None or opt.cam is None:
        parser.print_help()
        print('Need --brick and --cam')
        sys.exit(-1)

    fns = None
    if opt.itab is not None:
        opt.itab = fits_table(opt.itab)
        fns = opt.itab.filename

    if opt.basefn is None:
        basefn = 'merge-%s-%02i' % (opt.cam, opt.brick)
    else:
        basefn = opt.basefn
    ps = PlotSequence(basefn+'-', format='%02i')

    T = mergeBrick(opt.cam, opt.brick, mergedfn=opt.mergedfn, ps=ps, fns=fns, itab=opt.itab, force=opt.force, magcuts=opt.magcut)
    return 0

if __name__ == '__main__':
    main()
