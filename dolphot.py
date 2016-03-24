def get_dolphot_idc(cam, ccdchip, filt, forward=True):
    IDC = fits_table('dolphot-idc-%s.fits' % (cam.lower()))
    IDC.order = 4
    if forward:
        dirn = 'FORWARD'
    else:
        dirn = 'INVERSE'
    row = np.flatnonzero((IDC.direction == dirn) *
                         (IDC.detchip == ccdchip) *
                         (IDC.filter == filt))
    assert(len(row) == 1)
    idc = IDC[row[0]]
    return idc

def get_dolphot_shifts(fx, fy, rx, ry, RW, RH, idc, scale):
    #fx = fx.copy()
    #fy = fy.copy()
    #rx = rx.copy()
    #ry = ry.copy()
    # -convert from FITS to dolphot-convention pixels
    fx -= 0.5
    fy -= 0.5
    rx -= 0.5
    ry -= 0.5
    # Ref: subtract half ref img size
    rx -= RW / 2.
    ry -= RH / 2.
    rx *= scale
    ry *= scale
    # Img: subtract IDC xref,yref;
    fx -= idc.xref
    fy -= idc.yref
    #print 'fx,fy 1', fx,fy
    # push through IDC.
    fx,fy = apply_idc(idc, fx, fy)
    #print 'fx,fy 2', fx,fy
    # convert back to pixels via scale
    fx /= idc.scale
    fy /= idc.scale
    #print 'fx,fy 3', fx,fy
    #print 'rx,ry', rx,ry
    #print 'sx,sy', fx-rx, fy-ry
    return fx-rx, fy-ry

def apply_idc(idc, x, y):
    fx,fy = np.zeros_like(x), np.zeros_like(y)
    for i in range(1, idc.order+1):
        for j in range(i+1):
            xpow = j
            ypow = i - j
            cx = idc.get('cx%i%i' % (i,j))
            cy = idc.get('cy%i%i' % (i,j))
            dx = cx * (x)**xpow * (y)**ypow
            dy = cy * (x)**xpow * (y)**ypow
            fx += dx
            fy += dy
    return fx,fy


def parse_phot_info(txt, fnregex=None):
    '''
    Returns [ (name, chip, dx, dy, scale), ... ]
    '''
    if fnregex is None:
        fnregex = r'(?P<id>[a-zA-Z0-9]{9,10})_F[\d]{3}W_flt(.chip(?P<chip>[12]))?'
    fnre = re.compile(fnregex)
    # as per http://docs.python.org/library/re.html
    floatre = r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'
    alignre = re.compile((r'\s(' + floatre.replace('(','(?:') + ')')*5)

    flts = []
    for m in fnre.finditer(txt):
        print 'got match with id', m.group('id'), 'and chip', m.group('chip')
        if m.group('chip') is not None and len(m.group('chip')):
            chip = int(m.group('chip'))
        else:
            chip = None
        flts.append((m.group('id'), chip))

    aligns = []
    lines = txt.split('\n')
    started = False
    for x in lines:
        if 'Alignment' in x:
            started = True
        if not started:
            continue
        if 'Aperture corrections' in x:
            break
        m = alignre.match(x)
        if not m:
            continue
        #print m.groups()
        aligns.append([float(y) for y in m.groups()])

    print 'got', len(flts), 'FLT chips and', len(aligns), 'alignments.'
    
    if len(flts) != len(aligns):
        return None
    rtn = []
    for f,a in zip(flts,aligns):
        # name, chip, dx, dy, scale
        rtn.append((f[0], f[1], a[0], a[1], a[2]))
    return rtn


