#! /usr/bin/env python
from __future__ import print_function

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import os
import sys
import re
from tempfile import *
import colorsys

import numpy as np

from collections import Counter

from astrom_common import *
from astrom_intra import intrabrickshift
from astrom_inter import findAffine

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.miscutils import *
from astrometry.blind.plotstuff import *
from astrometry.util.multiproc import multiproc

def find_overlaps(r0,r1,d0,d1,NG,outlines):
    '''
    (ignored) NG: number of grid points, eg 100

    outlines: Outlines of the images in RA,Dec

    Returns:  overlaps, areas

    overlaps: 2-d numpy array, overlaps[i,j] is the fraction of points in i
        that are also in j.

    areas: approximate area of each image, in square degrees
    '''

    N = len(outlines)
    areas = np.zeros(N)
    overlaps = np.zeros((N,N))

    for i,out in enumerate(outlines):
        areas[i] = polygon_area(out)

    outlines2 = [np.array(list(zip(*out))) for out in outlines]
    for i,out1 in enumerate(outlines):
        for j,out2 in enumerate(outlines):
            if j <= i:
                continue
            rr1,dd1 = out1
            rr2,dd2 = out2
            if min(rr1) > max(rr2) or min(rr2) > max(rr1):
                continue
            if min(dd1) > max(dd2) or min(dd2) > max(dd1):
                continue
            outA,outB = outlines2[i], outlines2[j]
            if not polygons_intersect(outA, outB):
                continue
            cp = clip_polygon(outA, outB)
            if len(cp) == 0:
                continue
            cp = np.array(cp)
            area = polygon_area((cp[:,0], cp[:,1]))
            overlaps[i,j] = area / areas[i]
            overlaps[j,i] = area / areas[j]
    return overlaps, areas

def wavelength(f):
    '''
    Maps a string "F110W" to wavelength in int nanometers: 1100
    '''
    fmap = { 110:1100, 160:1600 }
    # trim F...W
    if not ((f[0] in ['F','f']) and (f[-1] in ['w','W','N','n'])):
        print('WARNING: wavelength("%s"): expected F###W' % f)
    f = int(f[1:-1])
    f = fmap.get(f, f)
    return f

def argsort_filters(filters):
    # Reorder them by wavelength...
    nf = []
    for f in filters:
        if f == 'ref':
            nf.append(0.)
        else:
            nf.append(wavelength(f))
    I = np.argsort(nf)
    return I

def filters_legend(lp, filters): #, **kwa):
    I = argsort_filters(filters)
    #plt.legend([lp[i] for i in I], [filters[i] for i in I], **kwa)
    return [lp[i] for i in I], [filters[i] for i in I]

def _alfunc(args):
    (Ti, Tj, R, i, j) = args
    print('Matching', i, 'to', j)
    M = Match(Ti, Tj, R)
    if len(M.I) == 0:
        return None
    A = Alignment(Ti, Tj, R, cutrange=R, match=M)
    if A.shift() is None:
        return None
    return A

def readfltgst(fltfn, gstfn, wcsexts):
    info = parse_flt_filename(fltfn)
    chip = info.get('chip')
    filt = info.get('filt')
    filt = filt.upper()
    name = info.get('name')
    hdr = read_header_as_dict(fltfn, 0)
    exptime = hdr['EXPTIME']
    if chip:
        cname = '%s_%i' % (name,chip)
    else:
        cname = name

    wcs = None
    for ext in wcsexts:
        try:
            wcs = Tan(fltfn, ext)
            break
        except:
            print('Failed to read WCS header from extension', ext, 'of', fltfn)
            #import traceback
            #traceback.print_exc()

    print('Read WCS header from', fltfn)

    outline = getwcsoutline(wcs)

    try:
        T = fits_table(gstfn)
        print('Read gst file', gstfn, '->', len(T), 'stars')
    except:
        print('WARNING: failed to read FITS file', gstfn)
        import traceback
        traceback.print_exc()
        return None

    cols = T.get_columns()
    cols = [c.lower() for c in cols]
    #print 'Columns:', cols
    if 'mag1_acs' in cols:
        magnm = 'mag1_acs'
    elif 'mag1_uvis' in cols:
        magnm = 'mag1_uvis'
    elif 'mag1_ir' in cols:
        magnm = 'mag1_ir'
    elif 'mag1_wfpc2' in cols:
        magnm = 'mag1_wfpc2'
    else:
        assert(False)
    T.magnm = magnm
    T.mag = T.get(magnm)

    return T, outline, (chip, filt, name, cname, exptime)

def readfltgsts(fltfns, gstfns, wcsexts, Nkeep, Nuniform):
    TT = []
    outlines = []
    chips = []
    names = []
    cnames = []
    filts = []
    exptimes = []
    Nall = []
    for fltfn,gstfn in zip(fltfns, gstfns):
        print('gst', gstfn)
        print('flt', fltfn)
        T, outline, meta = readfltgst(fltfn, gstfn, wcsexts)
        (chip, filt, name, cname, exptime) = meta
        if Nkeep and len(T) < Nkeep:
            print('WARNING: gst file', gstfn, 'contains "only"', len(T), 'stars')

        print('outline in RA,Dec:', outline)
        rr,dd = outline
        ra,dec = np.mean(rr), np.mean(dd)
        # check for clockwise polygons
        dra = rr - ra
        ddec = dd - dec
        crossprod = dra[0] * ddec[1] - dra[1] * ddec[0]
        if crossprod > 0:
            print('counter-clockwise outline -- reversing')
            outline = np.array(list(reversed(rr))), np.array(list(reversed(dd)))

        Nall.append(len(T))

        if Nuniform:
            xmax = T.x.max()
            nx = int(np.round(xmax / Nuniform))
            dx = xmax / nx
            ymax = T.y.max()
            ny = int(np.round(ymax / Nuniform))
            dy = ymax / ny
            Nper = int(Nkeep / (nx*ny))
            print('Uniform: xymax (%.1f,%.1f)' % (xmax,ymax), 'nx,ny', nx,ny,
                  'nkeep', Nkeep, ', n stars per bin:', Nper)
            xbin = (T.x / dx).astype(int).clip(0, nx-1)
            ybin = (T.y / dy).astype(int).clip(0, ny-1)

            II = []
            for y in range(ny):
                I = np.flatnonzero(ybin == y)
                xb = xbin[I]
                for x in range(nx):
                    J = np.flatnonzero(xb == x)
                    K = I[J]
                    KK = np.argsort(T.mag[K])
                    Ithis = K[KK[:Nper]]
                    if len(Ithis) < Nper:
                        Ithis = np.append(Ithis, np.zeros(Nper - len(Ithis), int) - 1)
                    II.append(Ithis)
            # We stack them in this way so that the index is sorted to have the
            # brightest star per cell first, then the second-brightest, etc.
            I = np.vstack(II).T.ravel()
            T.cut(I[ I>=0 ])

        elif Nkeep:
            T.cut(np.argsort(T.mag)[:Nkeep])
        TT.append(T)
        filts.append(filt)
        chips.append(chip)
        names.append(name)
        cnames.append(cname)
        exptimes.append(exptime)
        outlines.append(outline)

    filts = np.array(filts)
    names = np.array(names)
    cnames = np.array(cnames)
    chips = np.array(chips)
    exptimes = np.array(exptimes)

    r0 = min([T.ra.min()  for T in TT])
    r1 = max([T.ra.max()  for T in TT])
    d0 = min([T.dec.min() for T in TT])
    d1 = max([T.dec.max() for T in TT])
    
    print('Read', len(TT), 'fields:')
    for i,(cn,N,filt,t) in enumerate(zip(cnames,Nall,filts,exptimes)):
        print('  ', i+1, cn, filt, 'exposure %.1f sec' % t, N, 'stars')
    return (TT, outlines,
            (chips, names, cnames, filts, exptimes, Nall, (r0,r1,d0,d1)),
            )


def alignment_plots(afffn, name, Nkeep, Nuniform, R, NG, minoverlap,
                    perfield, nocache, mp, wcsexts,
                    lexsort=True, reffn=None, refrad=0.5,
                    cutfunction=None):
    import pylab as plt
    from astrometry.util.plotutils import (
        PlotSequence, loghist, plothist, setRadecAxes)
    
    Taff = fits_table(afffn)
    # Trim extra spaces off of filenames (spaces added by some FITS readers)
    Taff.flt = np.array([s.strip() for s in Taff.flt])
    Taff.gst = np.array([s.strip() for s in Taff.gst])
    
    affs = Affine.fromTable(Taff)

    TT, outlines, meta = readfltgsts(Taff.flt, Taff.gst, wcsexts, Nkeep,
                                     Nuniform)

    if cutfunction is not None:
        TT,outlines,meta = cutfunction(TT, outlines, meta)

    (chips, names, cnames, filts, exptimes, Nall, rd) = meta
    r0,r1,d0,d1 = rd

    for aff,T in zip(affs,TT):
        T.ra,T.dec = aff.apply(T.ra, T.dec)

    Tref = None
    if reffn:
        print('Reading reference catalog', reffn)
        Tref = fits_table(reffn)
        print('Got', len(Tref))
        print('Cutting to RA,Dec range', r0,r1,d0,d1)
        Tref.cut((Tref.ra  > r0) * (Tref.ra  < r1) *
                 (Tref.dec > d0) * (Tref.dec < d1))
        print('Cut to', len(Tref))
        #ikwargs.update(ref=Tref, refrad=refrad)

    nstars = np.array([len(T) for T in TT])
    wavelengths = np.array([wavelength(f) for f in filts])

    # Reorder...
    if lexsort:
        I = np.lexsort((names, -exptimes, chips, wavelengths))
        # Re-sort
        affs = [affs[i] for i in I]
        outlines = [outlines[i] for i in I]
        TT = [TT[i] for i in I]
        filts = filts[I]
        names = names[I]
        chips = chips[I]
        exptimes = exptimes[I]
        nstars = nstars[I]
        cnames = cnames[I]
    # arrays that we're not going to bother re-sorting
    del wavelengths

    uf = np.unique(filts)
    uf = uf[argsort_filters(uf)]
    print('(sorted) Filters:', uf)

    print('Exposure times:', np.unique(exptimes))

    cc,ss,fcmap,fsmap = get_symbols_for_filts(uf)

    fcmap['ref'] = 'k'
    fsmap['ref'] = '.'
    
    N = len(TT)
    OO,areas = find_overlaps(r0,r1,d0,d1,NG,outlines)

    summary = not perfield

    ps1 = PlotSequence(name + '-summary', format='%02i')
    ps = ps1
    
    # Map
    lp = {}
    plt.clf()
    for (r,d),chip,filt in zip(outlines, chips, filts):
        p1 = plt.plot(r, d, '-', color=fcmap[filt], alpha=0.5, lw=2)
        if not filt in lp:
            lp[filt] = p1[0]
        #plt.text(np.mean(r), np.mean(d), '%i'%chip)
    setRadecAxes(r0,r1,d0,d1)
    plt.legend([lp[f] for f in uf], uf)
    ps.savefig()
    
    eepfn = name+'-ee.pickle'
    if summary and os.path.exists(eepfn) and (not nocache):
        print('Reading cache file', eepfn)
        EE = unpickle_from_file(eepfn)
    else:
        EE = []

        ps = PlotSequence(name + '-perimage', format='%03i')

        for i,Ti in enumerate(TT):
            AA = []
            IJ = []
            iname = os.path.basename(Taff.gst[i]).replace('.gst.fits', '')
            args = []
            for j,Tj in enumerate(TT):
                if i == j:
                    if Tref:
                        args.append((Ti, Tref, refrad, i, i))
                        # So that the reference field always appears first in the
                        #  overlap-sorted list.
                        OO[i,i] = 1.0
                    continue
                if summary and j < i:
                    continue
                if OO[i,j] <= minoverlap:
                    print('Overlap between fields', i, 'and', j, 'is', OO[i,j], '-- not trying to match them')
                    continue
                args.append((Ti, Tj, R, i, j))
            MR = mp.map(_alfunc, args)
            goodj = []
            for A,a in zip(MR,args):
                if A is None:
                    continue
                j = a[-1]
                AA.append(A)
                IJ.append((i,j))
                goodj.append(j)

            for (ii,j),A in zip(IJ,AA):
                esize = np.sqrt(np.product(A.getEllipseSize()))
                x,y = A.getContours()
                EE.append((x,y,i,j, len(A.match.I), esize))

            if summary:
                continue

            N = len(AA)
            cols = int(np.ceil(np.sqrt(len(AA))))
            rows = int(np.ceil(N / float(cols)))
            plt.clf()
            plt.subplots_adjust(hspace=0.01, wspace=0.01,
                                left=0.1, right=0.96,
                                bottom=0.1, top=0.90)

            if False:
                plt.clf()
                plothist(Ti.ra, Ti.dec, 200, range=((r0,r1),(d0,d1)),
                         imshowargs=dict(cmap=antigray), dohot=False, docolorbar=False)
                lp = {}
                for j,(r,d) in enumerate(outlines):
                    if j in goodj:
                        filt = filts[j]
                        p1 = plt.plot(r,d,'-', color=fcmap[filt], lw=2, alpha=0.5)
                        if not filt in lp:
                            lp[filt] = p1[0]
                    else:
                        plt.plot(r,d,'-', color='0.5', lw=2, alpha=0.5)
                # plot bad ones on top
                for j,(r,d) in enumerate(outlines):
                    if not j in goodj:
                        plt.plot(r,d,'-', color='0.5', lw=2, alpha=0.5)
                setRadecAxes(r0,r1,d0,d1)
                plt.legend([lp[f] for f in uf], uf)
                plt.suptitle('%s: %s matched fields' % (name, iname))
                ps.savefig()
    
            if False:
                plt.clf()
                lp = {}
                for (ii,j),A in zip(IJ,AA):
                    filt = filts[j]
                    x,y = A.getContours()
                    p1 = plt.plot(x*1000., y*1000., '-', color=fcmap[filt], lw=2, alpha=0.5)
                    if not filt in lp:
                        lp[filt] = p1[0]
                    esize = np.sqrt(np.product(A.getEllipseSize()))
                    EE.append((x,y,i,j, len(A.match.I), esize))
                plt.legend([lp[f] for f in uf], uf)
                plt.suptitle('%s: %s alignments' % (name, iname))
                plt.axhline(0, color='k', alpha=0.5)
                plt.axvline(0, color='k', alpha=0.5)
                plt.xlabel('dRA (mas)')
                plt.ylabel('dDec (mas)')
                ps.savefig()


            # Sort by overlap
            K = np.argsort([-OO[i,j] for ii,j in IJ])

            plt.clf()
            nbins = 50
            #for j,A in enumerate(AA):
            for ik,k in enumerate(K):
                ii,j = IJ[k]
                A = AA[k]
                plt.subplot(rows, cols, 1+ik)
                if ii == j:
                    RR = refrad * 1000.
                    cname = 'Ref'
                    filt = ''
                    btxt = None
                else:
                    RR = R*1000.
                    cname = cnames[j]
                    filt = filts[j]
                    btxt = '%2i %%' % (int(OO[i,j] * 100.))
                plotalignment(A, nbins=nbins, rng=[(-RR,RR)]*2,
                              doclf=False, docolorbar=False,
                              docutcircle=False)
                plt.title('')
                ax = plt.axis()
                plt.axhline(0, color='b', alpha=0.5)
                plt.axvline(0, color='b', alpha=0.5)
                if j != ((rows-1)*cols):
                    plt.xlabel('')
                    plt.ylabel('')
                    plt.xticks([])
                    plt.yticks([])
                plt.axis(ax)
                plt.text(RR, RR, '%s %s' % (cname, filt),
                         va='top', color='y', ha='right')
                if btxt:
                    plt.text(-RR,-RR, btxt, va='bottom', ha='left', color='y')
            plt.suptitle('%s: %s' % (name, iname))
            ps.savefig()

            if False:
                plt.clf()
                for k,(A,(i,j)) in enumerate(zip(AA, IJ)):
                    plt.subplot(rows, cols, 1+k)
    
                    RR = R*1000.
                    M = A.match
                    Ti = TT[i]
                    Tj = TT[j]
                    loghist(Ti.get(Ti.magnm)[M.I], Tj.get(Tj.magnm)[M.J], nbins,
                            doclf=False, docolorbar=False)
                    plt.xticks([])
                    plt.yticks([])
                plt.suptitle('%s: %s: mag-mags of matches' % (name, iname))
                ps.savefig()

        pickle_to_file(EE, eepfn)


    plt.subplots_adjust(hspace=0.25, wspace=0.25,
                        left=0.1, right=0.96,
                        bottom=0.1, top=0.90)

    N = len(TT)
    ngrid = np.zeros((N,N))
    esgrid = np.zeros((N,N))
    kc = 0
    ffmap = {}
    lp = []
    lt = []
    for x,y,i,j,n,esize in EE:
        ngrid[j,i] = n
        esgrid[i,j] = esize
        if i > j:
            continue
        ngrid[i,j] = n
        esgrid[j,i] = esize

    if False:
        # All-filter-pairs ellipses -- too busy
        plt.clf()
        for x,y,i,j,n,esize in EE:
            filt1 = filts[i]
            filt2 = filts[j]
            c = ffmap.get((filt1,filt2), None)
            keeplp = False
            if c is None:
                c = cc[kc % len(cc)]
                ffmap[(filt1,filt2)] = c
                ffmap[(filt2,filt1)] = c
                kc += 1
                keeplp = True
                lt.append('%s-%s' % (filt1,filt2))
            p1 = plt.plot(x*1000, y*1000, '-', color=c, alpha=0.1, lw=2)
            if keeplp:
                lp.append(p1[0])
        I = np.argsort(lt)
        plt.legend([lp[i] for i in I], [lt[i] for i in I])
        plt.axis([-25,25,-25,25])
        plt.axis('equal')
        ps.savefig()

    ps = ps1
    plt.clf()
    plt.imshow(np.minimum(OO, 1.), interpolation='nearest', origin='lower')
    plt.hot()
    plt.colorbar()
    plt.title('Fraction of spatial overlap between image pairs')
    plt.xticks(np.arange(N), cnames, rotation=90, fontsize=8)
    plt.yticks(np.arange(N), cnames, fontsize=8)
    ps.savefig()

    navailgrid = np.minimum(*np.meshgrid(nstars,nstars))
    fgrid = ngrid / navailgrid
    dgrid_valid = np.logical_and((navailgrid * OO) > 0, OO > minoverlap)
    dgrid = np.zeros_like(fgrid)
    K = np.flatnonzero(dgrid_valid)
    dgrid.flat[K] = (ngrid.flat[K] / (navailgrid * OO).flat[K])

    plt.clf()
    plt.imshow(fgrid, interpolation='nearest', origin='lower')
    plt.hot()
    plt.colorbar()
    plt.title('Fraction of matches between image pairs')
    plt.xticks(np.arange(N), cnames, rotation=90, fontsize=8)
    plt.yticks(np.arange(N), cnames, fontsize=8)
    ps.savefig()
    
    plt.clf()
    plt.imshow(dgrid, interpolation='nearest', origin='lower')
    plt.hot()
    plt.colorbar()
    plt.title('Density of matches between image pairs')
    plt.xticks(np.arange(N), cnames, rotation=90, fontsize=8)
    plt.yticks(np.arange(N), cnames, fontsize=8)
    ps.savefig()

    # plt.clf()
    # plt.plot(OO.ravel(), ngrid.ravel(), 'r.')
    # plt.xlabel('Percent overlap')
    # plt.ylabel('Number of matches')
    # ps.savefig()
    # 
    # plt.clf()
    # plt.plot(OO.ravel(), esgrid.ravel(), 'r.')
    # plt.xlabel('Percent overlap')
    # plt.ylabel('Match ellipse size')
    # ps.savefig()

    for olo,ohi in [(None,None), (0.25,None), (0.1, 0.25), (None,0.1)]:
        plt.clf()
        NF = len(uf)
        cols = int(np.ceil(np.sqrt(NF)))
        rows = int(np.ceil(NF / float(cols)))
        lp = {}
        for ii,f1 in enumerate(uf):
            I = (filts == f1)
            # i is ee[2], j is ee[3]
            # Matching ellipses involving filter f1
            EEi = [ee for ee in EE if I[ee[2]] or I[ee[3]]]
            plt.subplot(rows, cols, ii+1)
            for f2 in uf:
                J = (filts == f2)
                for x,y,i,j,n,esize in EEi:
                    # Find matching ellipses between filters f1 and f2
                    if not ((J[j] and I[i]) or (J[i] and I[j])):
                        continue

                    # Check overlap fractions
                    if J[j] and I[i]:
                        ii,jj = i,j
                    else:
                        ii,jj, = j,i

                    if ii == jj:
                        OO[ii,jj] = 1.0
                    #print 'fields', ii,jj, '-> overlap', OO[ii,jj]

                    if olo and OO[ii,jj] < olo:
                        continue
                    if ohi and OO[ii,jj] > ohi:
                        continue

                    f2now = f2
                    if ii == jj:
                        f2now = 'ref'

                    iname = os.path.basename(Taff.gst[i]).replace('.gst.fits','')
                    jname = os.path.basename(Taff.gst[j]).replace('.gst.fits','')
                    print('Filters', f1, f2, 'images', iname, jname, ': %.1f mas' % (esize*1000.))

                    c = fcmap[f2now]
                    p1 = plt.plot(x*1000, y*1000, '-', color=c, alpha=0.1, lw=2)
                    if not f2now in lp:
                        p1 = plt.plot([], [], '-', color=c, lw=2)
                        lp[f2now] = p1[0]
            RR = 20
            #plt.axis([-RR,RR,-RR,RR])
            #plt.axis('scaled')
            #plt.axis([-RR,RR,-RR,RR])
            ax = plt.axis()
            plt.axis('scaled')
            M = max([np.abs(a) for a in ax])
            plt.axis([-M,M,-M,M])
            plt.title(f1 + ' match ellipses')
        I = argsort_filters(lp.keys())
        vals = list(lp.values())
        keys = list(lp.keys())
        plt.figlegend([vals[i] for i in I], [keys[i] for i in I], 'upper right')

        ostr = ''
        if olo and ohi:
            ostr = ', overlap [%i %%, %i %%]' % (int(olo * 100.), int(ohi * 100.))
        elif olo:
            ostr = ', overlap > %i %%' % (int(olo * 100.))
        elif ohi:
            ostr = ', overlap < %i %%' % (int(ohi * 100.))
        plt.suptitle('%s: match ellipses%s' % (name,ostr))
        ps.savefig()

    plt.clf()
    nbad = np.zeros(N)
    ntotal = np.zeros(N)

    for i,f1 in enumerate(uf):
        I = (filts == f1)
        plt.subplot(len(uf), 1, i+1)

        A = areas[I]
        print('Areas:', A)
        A = np.mean(A)
        print('Mean area in sq deg:', A)
        A *= (3600.**2)
        fprate = (np.pi * R**2) / A
        print('False positive match rate (single star):', fprate)
        # There are Nkeep "targets" and Nkeep "darts"
        Nfp = fprate * (Nkeep**2)
        print('Expected number of false positives:', Nfp)

        lp,lt = [],[]
        for f2 in uf:
            J = (filts == f2)

            ob,fb,obr,fbr = [],[],[],[]
            for ii in np.flatnonzero(I):
                for jj in np.flatnonzero(J):
                    if ii == jj:
                        obr.append(OO[ii,jj])
                        fbr.append(fgrid[ii,jj])
                    else:
                        ob.append(OO[ii,jj])
                        fb.append(fgrid[ii,jj])

            cc = fcmap['ref']
            p1 = plt.plot(np.array(obr) * 100., np.clip(fbr, 0, 1.),
                          fsmap['ref'], color=cc, mec=cc, mfc='none')
            if i == 0:
                lp.append(p1[0])
                lt.append('ref')

            cc = fcmap[f2]
            p1 = plt.plot(np.array(ob) * 100., np.clip(fb, 0., 1.),
                          fsmap[f2], color=cc, mec=cc, mfc='none')
            lp.append(p1[0])
            lt.append(f2)

            # OLD
            J = (filts == f2)
            oblock = OO   [I,:][:,J]
            fblock = fgrid[I,:][:,J]
            # cc = fcmap[f2]
            # p1 = plt.plot(oblock.ravel() * 100., fblock.ravel(),
            #             fsmap[f2], color=cc, mec=cc, mfc='none')
            # lp.append(p1[0])
            # lt.append(f2)

            nblock = ngrid[I,:][:,J]
            K = np.flatnonzero(oblock > 0.1)
            if len(K):
                # print smallest nblock / (Nfp*oblock) for K
                factorfp = nblock.flat[K] / (Nfp * oblock.flat[K])
                B = np.argsort(factorfp)
                print('Filters', f1, '--', f2)
                print('Number of matches as factor of FP rate:', factorfp[B])

                iy,ix = np.unravel_index(K[factorfp < 10.], oblock.shape)
                iy = np.flatnonzero(I)[iy]
                ix = np.flatnonzero(J)[ix]
                for a,b in zip(iy,ix):
                    print('  ', cnames[a], '--', cnames[b])
                    print('    ', filts[a], filts[b])
                    nbad[a] += 1
                    nbad[b] += 1

                iy,ix = np.unravel_index(K, oblock.shape)
                iy = np.flatnonzero(I)[iy]
                ix = np.flatnonzero(J)[ix]
                for a,b in zip(iy,ix):
                    ntotal[a] += 1
                    ntotal[b] += 1

        plt.ylabel('Fraction of matches')
        if i == 0:
            plt.figlegend(lp, lt, 'upper left')
        ymax = plt.axis()[3]
        plt.plot([0, 100.], [0, 1.0], 'k-', alpha=0.5)
        plt.plot([0, 100.], [0, 0.5], 'k-', alpha=0.5)
        plt.plot([0, 100.], [0, Nfp/Nkeep], 'r--', alpha=0.5)
        plt.axhline(0, color='k', alpha=0.5)
        plt.axvline(minoverlap * 100, color='k', alpha=0.5)
        plt.axvline(0, color='k', alpha=0.5)
        plt.xlim(-3, 103)
        plt.title(f1)
        plt.ylim(-0.03 * ymax, ymax)
    plt.xlabel('Percent overlap')
    plt.suptitle('%s: fraction of matches' % name)
    ps.savefig()

    # retrofit
    #I = np.arange(len(nstars))
    #dgrid_valid[I,I] = True

    plt.clf()
    for i,f1 in enumerate(uf):
        I = (filts == f1)
        plt.subplot(len(uf), 1, i+1)
        lp,lt = [],[]
        for f2 in uf:
            J = (filts == f2)

            ob,db,obr,dbr = [],[],[],[]
            for ii in np.flatnonzero(I):
                for jj in np.flatnonzero(J):
                    if not dgrid_valid[ii,jj]:
                        #if ii == jj:
                        #   #print 'dgrid invalid: navailgrid', navailgrid[ii,jj],
                        #   #print 'OO', OO[ii,jj]
                        continue
                    if ii == jj:
                        obr.append(OO[ii,jj])
                        dbr.append(dgrid[ii,jj])
                    else:
                        ob.append(OO[ii,jj])
                        db.append(dgrid[ii,jj])

            cc = fcmap['ref']
            p1 = plt.plot(np.array(obr) * 100., np.clip(dbr, 0.,1.),
                          fsmap['ref'], color=cc, mec=cc, mfc='none')
            if i == 0:
                lp.append(p1[0])
                lt.append('ref')

            cc = fcmap[f2]
            p1 = plt.plot(np.array(ob) * 100., db,
                          fsmap[f2], color=cc, mec=cc, mfc='none')
            lp.append(p1[0])
            lt.append(f2)

            # oblock = OO   [I,:][:,J]
            # dblock = dgrid[I,:][:,J]
            # K = np.flatnonzero(dgrid_valid[I,:][:,J])
            # cc = fcmap[f2]
            # p1 = plt.plot(oblock.flat[K] * 100., dblock.flat[K],
            #             fsmap[f2], color=cc, mec=cc, mfc='none')
            # lp.append(p1[0])
        plt.ylabel('Density of matches')
        if i == 0:
            plt.figlegend(lp, lt, 'upper right')
        ymax = plt.axis()[3]
        plt.axhline(1.0, color='k', alpha=0.5)
        plt.axhline(0.5, color='k', alpha=0.5)
        plt.axhline(Nfp/Nkeep, color='r', alpha=0.5)
        plt.axhline(0, color='k', alpha=0.5)
        plt.axvline(minoverlap * 100, color='k', alpha=0.5)
        plt.axvline(0, color='k', alpha=0.5)
        plt.xlim(-3, 103)
        plt.title(f1)
        plt.ylim(-0.03 * ymax, ymax)
    plt.xlabel('Percent overlap')
    plt.suptitle('%s: density of matches' % name)
    ps.savefig()

    # Worst fields:
    fbad = nbad / ntotal
    I = np.argsort(-fbad)
    for i in I:
        print(cnames[i], filts[i], '%.2f bad' % fbad[i], '(%i / %i)' % (nbad[i], ntotal[i]))

    plt.clf()
    for i,f1 in enumerate(uf):
        I = (filts == f1)
        plt.subplot(len(uf), 1, i+1)
        lp,lt = [],[]
        for f2 in uf:
            J = (filts == f2)

            ob,eb,obr,ebr = [],[],[],[]
            for ii in np.flatnonzero(I):
                for jj in np.flatnonzero(J):
                    if ii == jj:
                        obr.append(OO[ii,jj])
                        ebr.append(esgrid[ii,jj])
                    else:
                        ob.append(OO[ii,jj])
                        eb.append(esgrid[ii,jj])
            cc = fcmap['ref']
            p1 = plt.plot(np.array(obr) * 100., ebr,
                          fsmap['ref'], color=cc, mec=cc, mfc='none')
            if i == 0:
                lp.append(p1[0])
                lt.append('ref')

            cc = fcmap[f2]
            p1 = plt.plot(np.array(ob) * 100., eb,
                          fsmap[f2], color=cc, mec=cc, mfc='none')
            lp.append(p1[0])
            lt.append(f2)

            # oblock = OO    [I,:][:,J]
            # eblock = esgrid[I,:][:,J]
            # cc = fcmap[f2]
            # p1 = plt.plot(oblock.ravel() * 100., eblock.ravel() * 1000.,
            #             fsmap[f2], color=cc, mec=cc, mfc='none')
            # lp.append(p1[0])
        plt.ylabel('Matching ellipse size (mas)')
        if i == 0:
            plt.figlegend(lp, lt, 'upper right')
        plt.axvline(minoverlap * 100, color='k', alpha=0.5)
        plt.axvline(0, color='k', alpha=0.5)
        plt.xlim(-3, 103)
        plt.title(f1)
        ymax = plt.axis()[3]
        plt.ylim(-0.03 * ymax, ymax)
    plt.xlabel('Percent overlap')
    plt.suptitle('%s: match ellipse sizes' % name)
    ps.savefig()



    # Density of matches vs exposure times

    plt.clf()
    axes = []
    for i,f1 in enumerate(uf):
        I = (filts == f1)
        plt.subplot(len(uf), 1, i+1)
        lp,lt = [],[]
        for f2 in uf:
            J = (filts == f2)
            dblock = dgrid[I,:][:,J]
            cc = fcmap[f2]
            e2 = np.outer(exptimes[I], exptimes[J])
            # don't plot points that have zero overlap.
            K = np.flatnonzero(dgrid_valid[I,:][:,J])
            p1 = plt.semilogx(e2.flat[K], dblock.flat[K],
                              fsmap[f2], color=cc, mec=cc, mfc='none')
            lp.append(p1[0])
        plt.ylabel('Density of matches')
        if i == 0:
            #plt.legend(lp, uf, loc='upper left')
            plt.figlegend(lp, uf, 'upper right')
        #plt.xlim(-1, 101)
        plt.title(f1)
        plt.axhline(0, color='k', alpha=0.5)
        ymax = plt.axis()[3]
        plt.ylim(-0.03 * ymax, ymax)
        axes.append(plt.axis())
    # Expand axes to common range
    axes = np.array(axes)
    ax0 = np.max(axes, axis=0)
    ax1 = np.min(axes, axis=0)
    ax = [ax0[0], ax1[1], ax0[2], ax1[3]]
    for i,f1 in enumerate(uf):
        plt.subplot(len(uf), 1, i+1)
        plt.axis(ax)
    plt.xlabel('Exposure time * Exposure time (s^2)')
    plt.suptitle('%s: matches vs exposure time' % name)
    ps.savefig()


    ## By filter and exposure time, plot match fractions.
    I = np.arange(len(TT))
    
    # how many exposures of each filter are there?
    cnt = Counter()
    for f in filts:
        cnt[f] += 1

    if False:
        plt.clf()
        left = 0
        n = 0
        k0 = 0
        lastfilt = None
        subplot = 1
        lp = {}
        axes = []
        xt = []
        for k,i in enumerate(I):
            f1 = filts[i]
            lastone = (k == (len(I)-1)) or (filts[I[k+1]] != f1)

            if f1 != lastfilt:
                # in subplots per filter?
                plt.subplot(len(uf), 1, subplot)
                subplot += 1
                left = 0
                xt = []

                k0 = k
                lastfilt = f1
                n = cnt[f1]
                #plt.gca().add_artist(Rectangle((left-0.4, 0), n-0.2, 1.,
                #                    color=fcmap[f1], alpha=0.2, zorder=10))
                #left += n

            for f2 in uf:
                J = np.flatnonzero(filts == f2)
                K = np.flatnonzero(dgrid_valid[i,J])
                J = J[K]
                cc = fcmap[f2]
                y = dgrid[i, J]
                p1 = plt.plot(left + (k-k0)*np.ones_like(y), y,
                              fsmap[f2], color=cc, mec=cc, mfc='none',
                              zorder=20)
                if not f2 in lp:
                    lp[f2] = p1[0]

            xt.append(cnames[i])

            if lastone:
                #print 'xticks from', left, 'num', n, ':', xt
                #plt.xticks(np.arange(len(xt)), xt, fontsize=8)
                for ix,t in enumerate(xt):
                    plt.text(ix-0.05, 0, '  '+t, rotation=90, fontsize=10, va='bottom', ha='right',
                             color='k')
                plt.xticks([])
                plt.xlim(-0.5, len(xt)-0.5)
                plt.ylabel('Density of matches')
                plt.title(f1)
                ymax = plt.axis()[3]
                plt.ylim(-0.03 * ymax, ymax)
                axes.append(plt.axis())
                plt.axhline(0, color='k', alpha=0.5)

        I = argsort_filters(lp.keys())
        plt.figlegend([lp.values()[i] for i in I], [lp.keys()[i] for i in I], 'upper right')
        #plt.figlegend(lp, uf, 'upper right')
        # Expand axes to common range (vert only)
        axes = np.array(axes)
        ax0 = np.max(axes, axis=0)
        ax1 = np.min(axes, axis=0)
        ax = [ax0[0], ax1[1], ax0[2], ax1[3]]
        for i,f1 in enumerate(uf):
            plt.subplot(len(uf), 1, i+1)
            #plt.axis(ax)
            plt.ylim(ax[2],ax[3])
        plt.suptitle('%s: matches by image' % name)
        ps.savefig()

    
    plt.clf()
    left = 0
    n = 0
    k0 = 0
    lastfilt = None
    lp = {}
    axes = []
    xt = []
    I = np.arange(len(TT))
    for k,i in enumerate(I):
        f1 = filts[i]
        if f1 != lastfilt:
            k0 = k
            lastfilt = f1
            left += n
            n = cnt[f1]
        for f2 in uf:
            J = np.flatnonzero(filts == f2)
            # K = np.flatnonzero(dgrid_valid[i,J])
            # J = J[K]
            # cc = fcmap[f2]
            # y = dgrid[i, J]
            # p1 = plt.plot(left + (k-k0)*np.ones_like(y), y,
            #             fsmap[f2], color=cc, mec=cc, mfc='none',
            #             zorder=20)
            # if not f2 in lp:
            #   lp[f2] = p1[0]

            y,ry = [],[]
            for j in J:
                if not dgrid_valid[i,j]:
                    continue
                if i == j:
                    ry.append(dgrid[i,j])
                else:
                    y.append(dgrid[i,j])
            cc = fcmap['ref']
            p1 = plt.plot(left + (k-k0)+np.zeros(len(ry)), ry,
                          fsmap['ref'], color=cc, mec=cc, mfc='none',
                          zorder=20)
            if not 'ref' in lp:
                lp['ref'] = p1[0]
            cc = fcmap[f2]
            p1 = plt.plot(left + (k-k0)+np.zeros(len(y)), y,
                          fsmap[f2], color=cc, mec=cc, mfc='none',
                          zorder=20)
            if not f2 in lp:
                lp[f2] = p1[0]

        xt.append(cnames[i])
    plt.xticks(np.arange(len(xt)), xt, fontsize=10, rotation=90)
    plt.xlim(-0.5, len(xt)-0.5)
    plt.ylabel('Density of matches')
    plt.title(f1)
    ymax = plt.axis()[3]
    y0,y1 = (-0.03 * ymax, ymax)
    plt.axhline(0, color='k', alpha=0.5)

    left = 0
    for f1 in uf:
        from matplotlib.patches import Rectangle
        n = cnt[f1]
        plt.gca().add_artist(Rectangle((left-0.4, y0), n-0.2, y1-y0,
                                       color=fcmap[f1], alpha=0.1, zorder=10))
        left += n

    plt.ylim(y0,y1)
    vals = list(lp.values())
    keys = list(lp.keys())
    I = argsort_filters(keys)
    plt.figlegend([vals[i] for i in I], [keys[i] for i in I], 'upper right')
    plt.title('%s: matches by image' % name)
    ps.savefig()


    ####
    L = locals()
    rtn = {}
    for k in ['affs', 'TT', 'outlines', 'chips', 'names', 'cnames',
              'filts', 'exptimes', 'Nall', 'rd']:
        rtn[k] = L[k]
    return rtn


    # for fig in [2,3]:
    #   plt.figure(fig, figsize=(10,10))
    #   plt.clf()
    #   plt.subplots_adjust(hspace=0.01, wspace=0.01,
    #                       left=0.1, right=0.96,
    #                       bottom=0.1, top=0.90)
    # r,c = len(TT), len(TT)-1
    # NR,NC = 10,10
    # for r0 in range(1 + r/NR):
    #   for c0 in range(1 + c/NC):
    #       plt.clf()
    #       for i in range(r0, min(r0+NR, len(TT))):
    #           Ti = TT[i]
    #           #for i,Ti in enumerate(TT[r0:][:NR]):
    #           #for j,Tj in enumerate(TT[c0:i][:NC]):
    #           for j in range(c0, min(c0+NC, len(TT), i)):
    #               Tj = TT[j]
    #               print 'Matching', i, 'to', j
    #               A = Alignment(Ti, Tj, R, cutrange=R)
    #               if A.shift() is None:
    #                   continue
    # 
    #               plt.figure(2)
    #               #plt.subplot(r, c, 1 + i*r + j)
    #               plt.subplot(NR, NC, 1 + (i-r0)*NC + (j-c0))
    #               RR = R*1000.
    #               plotalignment(A, rng=[(-RR,RR)]*2, doclf=False, docolorbar=False,
    #                             docutcircle=False)
    # 
    #               plt.figure(3)
    #               #plt.subplot(r, c, 1 + i*r + j)
    #               plt.subplot(NR, NC, 1 + (i-r0)*NC + (j-c0))
    #               M = A.match
    #               loghist(Ti.get(Ti.magnm)[M.I], Tj.get(Tj.magnm)[M.J], 100,
    #                       doclf=False, docolorbar=False)
    #       
    #       for fig in [2,3]:
    #           plt.figure(fig)
    #           ps.savefig()
        

def parse_flt_filename(fltfn):
    if '_c0m.chip' in fltfn:
        # WFPC2
        m = re.search(r'proc/(?P<name>.{9})_(?P<filt>[Ff].{3}[WwNn])_c0m\.chip(?P<chip>[\d])\.fits', fltfn)
        if m is None:
            print('WARNING: failed to match regex', rex, 'for fltfn', fltfn)
            return {}
        chip = int(m.group('chip'))
    elif 'chip' in fltfn:
        rex = r'proc/(?P<name>.{9,10})_(?P<filt>[Ff].{3}[WwNn])_fl.\.chip(?P<chip>[\d])\.fits'
        m = re.search(rex, fltfn)
        if m is None:
            print('WARNING: failed to match regex', rex, 'for fltfn', fltfn)
            return {}
        chip = int(m.group('chip'))
    else:
        m = re.search(r'proc/(?P<name>.{9})_(?P<filt>[Ff].{3}[WwNn])_fl.\.fits', fltfn)
        if m is None:
            print('WARNING: failed to match regex', rex, 'for fltfn', fltfn)
            return {}
        chip = 0

    return { 'chip':chip, 'name': m.group('name'), 'filt': m.group('filt') }



def plot_alignment_grid(allA, RR, Rref, cnames, filts, overlaps, thisi, outlines):
    import pylab as plt
    from astrometry.util.plotutils import setRadecAxes
    
    N = len(allA) + 1
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / float(cols)))

    plt.clf()
    plt.subplots_adjust(hspace=0.01, wspace=0.01,
                        left=0.1, right=0.96,
                        bottom=0.1, top=0.90)

    # map
    jset = set([j for j,A in allA])

    plt.subplot(rows, cols, 1)
    rr,dd = [],[]
    for j,(r,d) in enumerate(outlines):
        if j == thisi:
            zo = 20
            c = 'r'
            a = 1.0
        elif j in jset:
            zo = 15
            c = 'b'
            a = 0.5
        else:
            zo = 10
            c = '0.5'
            a = 0.5
        
        p1 = plt.plot(r, d, '-', color=c, alpha=1, zorder=zo, lw=2)
        rr.append(r)
        dd.append(d)
    rr = np.array(rr).ravel()
    dd = np.array(dd).ravel()
    setRadecAxes(rr.min(), rr.max(), dd.min(), dd.max())
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.yticks([])

    # sort
    #I = np.argsort([j for j,A in allA])
    # Sort by % overlap.
    I = np.argsort([-(overlaps[j] + 1000.*(j == thisi)) for j,A in allA])
    
    print('Plotting alignments with', cnames[thisi])
    allA = [allA[j] for j in I]
    for k,(j,A) in enumerate(allA):
        plt.subplot(rows, cols, k + 2)
        print('  number %i: row %i, col %i: %s, overlap %.2f' % (k+1, 1+((k+1)/cols), 1+((k+1)%cols), cnames[j], overlaps[j]))
        A = A.copy()
        #es = A.pop('estring')
        econ1 = A.pop('econ1')
        econ2 = A.pop('econ2')
        ecen = A.pop('ecen')
        estring = A.pop('estring')
        esize = A.pop('esize')
        H = A.pop('H')
        foresum = A.pop('foresum')

        ref = (j == thisi)
        R = Rref if ref else RR

        keepkeys = ['extent', 'aspect', 'interpolation', 'origin']
        dropkeys = []
        for k in A:
            if not k in keepkeys:
                dropkeys.append(k)
        for k in dropkeys:
            del A[k]

        #mx = max(H.max() / 2, 5)
        #plt.imshow(H, vmin=0, vmax=mx, **A)
        #mx = max(H.max() / 2, 5)
        plt.imshow(H, vmin=0, **A)
        plt.hot()
        ax = plt.axis()
        # contours
        for X,Y in [econ1,econ2]:
            plt.plot(X*1000., Y*1000., '-', color=(0,1,0), alpha=0.5)
        plt.axhline(0, color='b', alpha=0.5)
        plt.axvline(0, color='b', alpha=0.5)
        angles = np.linspace(0, 2.*np.pi, 50, endpoint=True)

        plt.plot(np.sin(angles)*R, np.cos(angles)*R, 'b-')

        cx,cy = ecen * 1000.
        excl = R * 0.2
        plt.plot([cx,cx], [-R*2, cy-excl], '-', color=(0,1,0))
        plt.plot([cx,cx], [cy+excl, R*2],  '-', color=(0,1,0))
        plt.plot([-R*2, cx-excl], [cy,cy], '-', color=(0,1,0))
        plt.plot([cx+excl, R*2],  [cy,cy], '-', color=(0,1,0))

        if ref:
            txt = 'Ref'
        else:
            txt = '%s %s' % (cnames[j], filts[j])

        plt.text(R,R, txt, va='top', ha='right', color='y', fontsize=8)
        plt.text(-R,-R, '%i / %i' % (int(foresum), H.sum()),
                 va='bottom', ha='left', color='y', fontsize=8)
        esz = 1000. * np.sqrt(esize[0] * esize[1])
        plt.text(R,-R, '%.0f mas' % esz,
                 va='bottom', ha='right', color='y', fontsize=8)
        # plt.xticks([])
        # plt.yticks([])
        plt.xticks([-R, 0, R])
        plt.yticks([-R, 0, R])
        #plt.axis(ax)
        plt.axis('scaled')
        plt.axis([-R,R,-R,R])

def get_symbols_for_filts(uf):
    # Plot colors
    #cc = ['r','g','b','m','k']
    # blue to red
    cc = [colorsys.hsv_to_rgb(h, 1., 1.)
          for h in np.linspace(0.666, 0., len(uf), endpoint=True)]
    # normalize
    cc = [x / np.sum(x) for x in cc]
    # darken green
    for x in cc:
        x[1] *= 0.7
    # Plot symbols
    ss = ['*','x','o','^','s']
    # filter-to-color and filter-to-symbol maps
    fcmap = dict([(f,cc[i%len(cc)]) for i,f in enumerate(uf)])
    fsmap = dict([(f,ss[i%len(ss)]) for i,f in enumerate(uf)])

    return cc, ss, fcmap, fsmap


def align_dataset(name, dirs, mp, alplots, NG, minoverlap,
                  NkeepRads, Nuniform,
                  #((Nkeep1,R1),(Nkeep2,R2),(Nkeep3,R3)),
                  shfn=None, reffn=None, refrad=1.,
                  refrads=None,
                  Tref=None,
                  wcsexts=[0],
                  st=False,
                  # kwargs to pass to Alignment
                  akwargs={},
                  # kwargs to pass to intrabrickshift
                  ikwargs={},
                  merge_chips=False,
                  cutfunction=None,
                  # expected size of the peak
                  targetrad = 0.1,
                  ):
    from astrometry.util.plotutils import PlotSequence

    dataset = name

    if not 'weightrange' in akwargs:
        akwargs.update(weightrange=1e-6)

    def getfltgsts(dirs):
        fltfns = []
        gstfns = []
        # glob
        dd = []
        for d in dirs:
            dd.extend(glob(d))
        dirs = dd
        dirs.sort()
        for dirnm in dirs:
            print('  dir:', dirnm)
            # GST filenames are like:
            # 11360_30-DOR_IR_ib6wr8kgq_F110W.gst.fits
            if st:
                tag = 'st'
            else:
                tag = 'gst'
            gstglob = os.path.join(dirnm, '*.%s.fits' % tag)
            gstfn = glob(gstglob)
            print('  %s:'%tag, gstglob, '->', gstfn)
            # FLT filenames are like:
            # ib6wr8kgq_f110w_flt.fits   OR
            # j8f864a2q_F606W_flt.chip1.fits OR
            # ib6wr8kgq_f110w_flc.fits   ("flc" rather than "flt")
            fltglob1 = os.path.join(dirnm, '*_fl?.chip?.fits')
            fltfn1 = glob(fltglob1)
            print('  flt pattern 1:', fltglob1, '->', fltfn1)
            fltglob2 = os.path.join(dirnm, '*_fl?.fits')
            fltfn2 = glob(fltglob2)
            print('  flt pattern 2:', fltglob2, '->', fltfn2)
            # WFPC2
            fltglob3 = os.path.join(dirnm, '*_c0m.chip?.fits')
            fltfn3 = glob(fltglob3)
            print('  flt pattern 3:', fltglob3, '->', fltfn3)
            fltfn = fltfn1 + fltfn2 + fltfn3
            #print(fltfn)
            if len(gstfn) != 1 or len(fltfn) != 1:
                print('WARNING: found wrong number of [g]st/flt files (not 1); skipping')
                continue
            assert(len(gstfn) == 1)
            assert(len(fltfn) == 1)
            gstfn = gstfn[0]
            fltfn = fltfn[0]
            print(tag, gstfn)
            print('flt', fltfn)
            fltfns.append(fltfn)
            gstfns.append(gstfn)
        return fltfns,gstfns

    def plot_all_alignments(ap, RR, Rref, round, cnames, filts, ps, overlaps, outlines,
                            Nkeep):
        import pylab as plt
        # symmetrize
        keys = list(ap.keys())
        for i in keys:
            AA = ap[i]
            for j,A in AA.items():
                if not j in ap:
                    ap[j] = {}
                ap[j][i] = A
        for i in ap.keys():
            tt = 'Round %i (R=%i mas, Ref=%i mas): %s: %s' % (round, RR, Rref, cnames[i], filts[i])
            allA = list(ap[i].items())
            plot_alignment_grid(allA, RR, Rref, cnames, filts, overlaps[i,:], i, outlines)
            plt.suptitle(tt)
            ps.savefig()

        if False:
            # Plot match indices
            I = np.argsort([-overlaps[i,j] for j,A in allA])
            allA = [allA[j] for j in I]
            N = len(allA) + 1
            cols = int(np.ceil(np.sqrt(N)))
            rows = int(np.ceil(N / float(cols)))
            plt.clf()
            for k,(j,A) in enumerate(allA):
                plt.subplot(rows, cols, k+2)
                plt.plot(A['MIall'], A['MJall'], '.', color='0.5', alpha=0.3)
                plt.plot(A['MI'], A['MJ'], 'r.', alpha=0.7)
                plt.text(0, Nkeep, '%s %s' % (cnames[j], filts[j]),
                         va='top', ha='left', color='k', fontsize=8)
                marg = Nkeep*0.03
                plt.axis([-marg, Nkeep, -marg, Nkeep])
                plt.xticks([])
                plt.yticks([])
            plt.suptitle(tt)
            ps.savefig()
        return

    Nkeep = max([n for n,r in NkeepRads])

    fltfns,gstfns = getfltgsts(dirs)
    print('fltfns', fltfns)
    print('gstfns', gstfns)

    TT, outlines, meta = readfltgsts(fltfns, gstfns, wcsexts, Nkeep, Nuniform)
    if cutfunction is not None:
        TT,outlines,meta = cutfunction(TT, outlines, meta)
    (chips, names, cnames, filts, exptimes, Nall, rd) = meta
    r0,r1,d0,d1 = rd

    print('names', names)
    print('cnames', cnames)
    print('chips', chips)

    if merge_chips:
        # Find chip-pairs
        namemap = {}
        for i,n1 in enumerate(names):
            if n1 in namemap:
                namemap[n1].append(i)
            else:
                namemap[n1] = [i]

        mTT = []
        moutlines = []
        mchips = []
        mnames = []
        mcnames = []
        mfilts = []
        mexptimes = []
        mNall = []
        mgstfns = []
        mfltfns = []
        nmerged = []
        orig_TT = TT
        
        kk = namemap.keys()
        kk.sort()
        for name in kk:
            ii = namemap[name]
            print('name', name, 'ii', ii)
            mTT.append(merge_tables([TT[i] for i in ii]))
            orig_TT.extend([TT[i] for i in ii])
            nmerged.append(len(ii))
            rr,dd = [],[]
            for i in ii:
                r,d = outlines[i]
                rr.append(r)
                dd.append(d)
            moutlines.append((np.hstack(rr), np.hstack(dd)))
            i0 = ii[0]
            if len(ii) > 1:
                #mchips.append(0)
                mcnames.append(names[i0])
            else:
                #mchips.append(chips[i0])
                mcnames.append(mcnames[i0])
            mnames.append(names[i0])
            mfilts.append(filts[i0])
            mexptimes.append(exptimes[i0])
            mNall.append(sum([Nall[i] for i in ii]))
            mgstfns.extend([gstfns[i] for i in ii])
            mfltfns.extend([fltfns[i] for i in ii])
            mchips.extend([chips[i] for i in ii])

        TT = mTT
        outlines = moutlines
        chips = mchips
        names = mnames
        cnames = mcnames
        filts = mfilts
        exptimes = mexptimes
        Nall = mNall
        fltfns = mfltfns
        gstfns = mgstfns

    # HACK, allow mag1diff to work...
    for T in TT:
        T.mag1 = T.mag

    uf = np.unique(filts)
    print('Filters:', uf)

    cc,ss,fcmap,fsmap = get_symbols_for_filts(uf)

    ps = PlotSequence('align-' + dataset, format='%03i')

    magrange = (15, 27)

    # if alplots:
    #   plt.clf()
    #   for T,filt in zip(TT, filts):
    #       # plt.hist(T.get(T.magnm), 100, range=(10,30), histtype='step', alpha=0.5,
    #       #        color=fcmap[filt])
    #       H,e = np.histogram(T.get(T.magnm), bins=100, range=magrange)
    #       m = e[:-1] + (e[1]-e[0])/2.
    #       plt.plot(m, H, '-', alpha=0.5, color=fcmap[filt])
    #   plt.xlabel('mag')
    #   ps.savefig()

    overlaps,areas = find_overlaps(r0,r1,d0,d1, NG, outlines)
    tryoverlaps = (overlaps > minoverlap)

    # if alplots:
    #   plt.clf()
    #   for i,(T,filt) in enumerate(zip(TT, filts)):
    #       # plt.hist(T.get(T.magnm), 100, range=(10,30), histtype='step', alpha=0.5,
    #       #        color=fcmap[filt])
    #       Nkeep1,nil = NkeepRads[0]
    # 
    #       mm = T.get(T.magnm)[:Nkeep1]
    #       med,p10,p90 = [np.percentile(mm, p) for p in [50, 10, 90]]
    #       print '  ', cnames[i], filts[i], 'exp %.1f sec' % exptimes[i],
    #       print 'median %.1f' % med, '10-90th: %.1f, %.1f' % (p10, p90)
    #       H,e = np.histogram(mm, bins=100, range=magrange)
    #       m = e[:-1] + (e[1]-e[0])/2.
    #       plt.plot(m, H, '-', alpha=0.5, color=fcmap[filt])
    #   plt.xlabel('mag')
    #   plt.title('Mags: Nkeep1 = %i' % Nkeep1)
    #   ps.savefig()

    ikwargs.update(do_affine=True, mp=mp,
                   #alignplotargs=dict(bins=25),
                   alignplotargs=dict(bins=50),
                   overlaps=tryoverlaps)

    if Tref is None and reffn:
        print('Reading reference catalog', reffn)
        Tref = fits_table(reffn)
        print('Got', len(Tref))
    if Tref:
        print('Cutting to RA,Dec range', r0,r1,d0,d1)
        Tref.cut((Tref.ra  > r0) * (Tref.ra  < r1) *
                 (Tref.dec > d0) * (Tref.dec < d1))
        print('Cut to', len(Tref))
        ikwargs.update(ref=Tref, refrad=refrad)


    refrd = None
    affs = None
    for roundi,(Nk,R) in enumerate(NkeepRads):
        first = (roundi == 0)
        last = (roundi == len(NkeepRads)-1)

        print('Round', roundi+1, ', cutting to', Nk)
        TT1 = [T[:Nk] for T in TT]
        nb = int(np.ceil(R / targetrad))
        #nb = max(nb, 11)
        nb = max(nb, 5)
        if nb % 2 == 0:
            nb += 1
        print('Nbins:', nb)
        # mdhr = 0.1
        # mdhrad=mdhr, 

        if refrads is not None:
            assert(len(refrads) > roundi)
            refrad = refrads[roundi]
            print('Reference-catalog matching radius:', refrad)
            ikwargs.update(refrad=refrad)

        # FIXME -- separate reference catalog nbins / histbins?

        i1 = intrabrickshift(TT1, matchradius=R, refradecs=refrd,
                             align_kwargs=dict(histbins=nb, **akwargs),
                             **ikwargs)
        if alplots:
            ap = i1.alplotgrid
            plot_all_alignments(ap, R*1000, refrad*1000, roundi+1, cnames, filts, ps,
                                overlaps, outlines, Nk)
        else:
            ps.skip(len(TT) * 2)

        for T,aff in zip(TT,i1.affines):
            T.ra,T.dec = aff.apply(T.ra, T.dec)

        if affs is None:
            affs = i1.affines
        else:
            for a,a2 in zip(affs, i1.affines):
                a.add(a2)
            

        refrd = i1.get_reference_radecs()


    # Apply & run one last time (but don't apply), just for the "after" plot.
    # if alplots4:
    #   for T,aff in zip(TT,affs3):
    #       T.ra,T.dec = aff.apply(T.ra, T.dec)
    #   i4 = intrabrickshift(TT3, matchradius=R3, mdhrad=mdhr, refradecs=refrd,
    #                        align_kwargs = akwargs,
    #                        **ikwargs)
    #   plot_all_alignments(i4.alplotgrid, R3*1000., 4, cnames, filts, ps,
    #                       overlaps, outlines, Nkeep3)
    # else:
    #   ps.skip(len(TT) * 2)

    if merge_chips:
        # repeat affs "nmerged" times
        maffs = []
        for n,aff in zip(nmerged, affs):
            maffs.extend([aff]*n)
        affs = maffs
        assert(len(affs) == len(fltfns))
        assert(len(affs) == len(gstfns))
        assert(len(affs) == len(chips))

        # unmerge TT
        TT = []
        for aff,T in zip(affs, orig_TT):
            T.ra,T.dec = aff.apply(T.ra, T.dec)
            TT.append(T)
            
    T = Affine.toTable(affs)
    T.flt = fltfns
    T.gst = gstfns
    T.chip = chips
    T.writeto('affines-%s.fits' % dataset)

    if shfn is None:
        shfn = '%s.sh' % dataset
        fscript = open(shfn, 'w')

    for i,(fn,name,chip) in enumerate(zip(fltfns, names, chips)):
        for ext in wcsexts:
            try:
                wcs = Sip(fn, ext)
                wcsext = ext
                break
            except:
                print('Failed to read WCS header from extension', ext, 'of', fn)
                #import traceback
                #traceback.print_exc()
        aff = affs[i].copy()
        tan = wcs.wcstan
        aff.applyToWcsObject(tan)
        tt = NamedTemporaryFile()
        outfn = tt.name
        wcs.write_to(outfn)
        #scriptfn = '%s[%i]' % (os.path.basename(fn), wcsext)
        extmap = {1:1, 2:4, 0:1, }

        print('FLT filename:', fn)
        print('Name:', name)
        print('Chip:', chip)

        # WFPC2
        if '_c0m.chip' in fn:
            scriptfn = '%s_c0m.fits[%i]' % (name, chip)
            write_update_script(fn, outfn, scriptfn, fscript, inext=wcsext)
            scriptfn = '%s_c1m.fits[%i]' % (name, chip)
            write_update_script(fn, outfn, scriptfn, fscript, inext=wcsext)
        else:
            scriptfn = '%s_flt.fits[%i]' % (name, extmap[chip])
            write_update_script(fn, outfn, scriptfn, fscript, inext=wcsext)

    fscript.close()

    return T, TT


if __name__ == '__main__':
    import pylab as plt
    plt.figure(figsize=(10,10))

    from astrometry.util.ttime import *
    Time.add_measurement(MemMeas)

    from optparse import OptionParser

    parser = OptionParser(usage='%prog [options] <name> <data-directories>')
    parser.add_option('-p', dest='plots', action='store_true',
                      help='Produce summary plots, skipping alignment if output files exist')
    parser.add_option('-P', dest='fieldplots', action='store_true',
                      help='Produce per-image summary plots; implies -p')
    parser.add_option('-n', dest='nkeep', type=int, default=10000,
                      help='Number of stars to keep per image when plotting, default %default')
    parser.add_option('-r', dest='rad', type=float, default=0.1,
                      help='Matching radius in arcsec when plotting, default %default')

    for p,dn,dr,drr in zip([1,2,3,4], [1000, 2000, 10000,0], [4.0, 1.0, 0.1, 0.1], [4.0, 1.0, 1.0, 1.0]):
        parser.add_option('--n%i'%p, dest='nkeep%i' % p, type=int, default=dn,
                          help='Number of stars to keep per image in phase %i, default %%default' % p)
        parser.add_option('--r%i'%p, dest='rad%i' % p, type=float, default=dr,
                          help='Matching radius in arcsec for phase %i, default %%default' % p)
        parser.add_option('--rr%i'%p, '--refrad%i'%p, dest='refrad%i' % p, type=float, default=drr,
                          help='Reference-catalog matching radius in arcsec for phase %i, default %%default' % p)

    parser.add_option('-u', dest='uniform', default=0, type=int,
                      help='Spatial uniform cut into boxes about this size in pixels (eg, 200)')

    parser.add_option('--ap', '--alplots', dest='alplots', action='store_true',
                      help='Produce plots during alignment')
        
    parser.add_option('-C', '--no-cache', dest='nocache', action='store_true',
                      help='Do not use any cache files')
    
    parser.add_option('-g', dest='ng', type=int, default=100,
                      help='Number of grid points for RA and Dec to use when determining image overlaps; default %default')
    parser.add_option('-o', dest='minoverlap', type=float, default=0.01,
                      help='Minimum fraction of image overlap in order to try matching two fields; default %default')
    parser.add_option('-t', '--threads', dest='threads', type=int, default=8,
                      help='Number of threads to use during matching; default %default')

    parser.add_option('--ref', dest='refcat', help='Filename of reference catalog to use')
    #parser.add_option('--refrad', dest='refrad', type=float, help='Reference catalog matching radius')
    parser.add_option('--wcsext', '-e', dest='wcsexts', type=int, default=[0,1],
                      action='append',
                      help='FITS Extension from which to read WCS header, default %default')
    parser.add_option('--st', dest='st', action='store_true',
                      help='Use .st.fits rather than .gst.fits files?')

    opt,args = parser.parse_args()

    if len(args) < 2:
        parser.print_help()
        sys.exit(-1)

    name = args[0]
    dirs = args[1:]

    mp = multiproc(opt.threads)
    
    if opt.fieldplots:
        opt.plots = True
    
    align = True
    afffn = 'affines-%s.fits' % name
    if opt.plots:
        print('Looking for affines filename', afffn)
        if os.path.exists(afffn):
            if opt.nocache:
                print('Affines file', afffn, 'exists, but --no-cache was set')
            else:
                align = False
                print('File exists; not running alignment code')
        else:
            print('File not found; running alignment first')

    if align:
        NR = [(opt.nkeep1, opt.rad1),
              (opt.nkeep2, opt.rad2),
              (opt.nkeep3, opt.rad3)]
        refrads = [opt.refrad1,
                   opt.refrad2,
                   opt.refrad3,]
        if opt.nkeep4:
            NR.append((opt.nkeep4, opt.rad4))
            refrads.append(opt.refrad4)

        align_dataset(name, dirs, mp, opt.alplots,
                      opt.ng, opt.minoverlap,
                      NR,
                      opt.uniform,
                      reffn=opt.refcat,
                      refrads=refrads,
                      wcsexts=opt.wcsexts,
                      st=opt.st)

    if opt.plots:
        alignment_plots(afffn, name, opt.nkeep, opt.uniform, opt.rad, opt.ng,
                        opt.minoverlap, opt.fieldplots, opt.nocache, mp,
                        opt.wcsexts,
                        reffn=opt.refcat)

    
    # #dirs = glob('data/single/%s/proc' % dataset)
    # 
    # for name in ['NGC0104', 'NGC5927', 'NGC6341', 'NGC6528', 'NGC6752' ]:
    #   dataset = '12116_%s_*' % name
    #   dict(R1=4.0, Nkeep1=1000,
    #        R2=0.5, Nkeep2=2000,)))    
    # 
    # 
    #   name = 'NGC1851'
    #   fltfns,gstfns = getfltgsts('10775_NGC1851_WFC*')
    #   fltfns2,gstfns2 = getfltgsts('12311_NGC-1851_UVIS*')
    # 
    # 
    #   name = '30DOR'
    #   dataset = '11360_30-DOR*'
    #   fltfns,gstfns = getfltgsts(dataset)
    #   fltgst.append((name, fltfns,gstfns, dict(Nkeep1 = 1000,
    #                                            R2=0.5, Nkeep2 = 2000)))
    # 
    #   name = 'Hercules'
    #   dataset = '12549_HERCULES*'
    #   fltfns,gstfns = getfltgsts(dataset)
    #   fltgst.append((name, fltfns,gstfns,  dict(Nkeep1 = 3000, #R1=4.0,
    #                                             R2=0.5, Nkeep2 = 3000)))
    #   
