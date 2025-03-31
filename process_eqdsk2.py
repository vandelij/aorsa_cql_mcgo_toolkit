import os
import re
import string
import logging
import shutil
from numpy import *
from collections import OrderedDict
import numpy as np


def string_split_ig(s, sep):
    return [x for x in re.split(sep, s) if x != '']
    
def eval_form2020(s):
    #form2020= '(5e16.9)'
    d = 16
    ret = []
    for i in range(5):
        #     print s[d*i:d*(i+1)],s[d*i:d*(i+1)].strip().__repr__()
        if s[d*i:d*(i+1)].strip() != '':
            ret.append(float(s[d*i:d*(i+1)]))
    return ret


def readandstrip(f):
    line = ''
    count = 0
    while line == '':
        line = f.readline()
        line = line.rstrip("\r\n")
        count = count + 1
        if count > 5:
            return ''

    return line


def make_form2020(array):
    line0 = []
    line = ''
    i = 0
    while i < len(array):
        line = line+'{: 13.9e}'.format(array[i])
        i = i+1
        if (i % 5) == 0:
            line0.append(line)
            line = ''

    if len(line) != 0:
        line0.append(line)

    return line0


def load_array_form2020(f, size):
    k = 0
    arr = []
    while (k < size):
        arr = arr+eval_form2020(readandstrip(f))
        k = k+5
    return np.array(arr)


def load_matrix_form2020(f, size1, size2):
    k = 0
    arr = []
    while k < size1*size2:
        arr = arr+eval_form2020(readandstrip(f))
        k = k+5
    return np.array(arr).reshape(size1, size2)


def add_extra_data(val):
    fpol = val['fpol']
    isPlasma = val['isPlasma']
    psirz = val['psirz']
    ssimag = val['ssimag']
    ssibdry = val['ssibdry']
    ssibry = val['ssibry']

    xpsi = (val['psirz']-val['ssimag'])/(val['ssibdry']-val['ssimag'])
    for m, xv in enumerate(val['rgrid']):
        for n, yv in enumerate(val['zgrid']):
            if not isPlasma[n, m]:
                xpsi[n, m] = 1.0
    rgrid = val['rgrid']
    zgrid = val['zgrid']
    xx, yy = np.meshgrid(rgrid, zgrid)

    xxx = np.linspace(0, 1, fpol.shape[0])
    fp = np.interp(xpsi.flatten(), xxx, val['fpol'])
    fp = fp.reshape(xx.shape)
    fc = np.interp(xpsi.flatten(), xxx, val['ffprim'])
    fc = fc.reshape(xx.shape)
    pr = np.interp(xpsi.flatten(), xxx, val['pres'])
    pr = pr.reshape(xx.shape)
    pc = np.interp(xpsi.flatten(), xxx, val['pprime'])
    pc = pc.reshape(xx.shape)
    qr = np.interp(xpsi.flatten(), xxx, val['qpsi'])
    qr = qr.reshape(xx.shape)

    pr[isPlasma != True] = 0.0
    pc[isPlasma != True] = 0.0
    fc[isPlasma != True] = 0.0
    val["pressrz"] = pr
    val["qrz"] = qr
    val["btrz"] = fp/xx

    dpsidz, dpsidr = np.gradient(psirz)
    brrz = -dpsidz/(zgrid[1]-zgrid[0])/xx
    bzrz = dpsidr/(rgrid[1]-rgrid[0])/xx
    val["brrz"] = brrz
    val["bzrz"] = bzrz

    mu0 = 4e-7*3.1415926535
    val["jtrz"] = (xx*pc+fc/xx/mu0)/1e6  # 1e6=(MA/m2)

    k = (val["zmaxis"] - rgrid[0])/(rgrid[1] - rgrid[0])
    from scipy.interpolate import interp2d

    f = interp2d(rgrid, zgrid, psirz, kind='cubic')
    val['npsimid'] = np.array([(f(r, val["zmaxis"]) - ssimag)/(ssibry - ssimag)
                               for r in rgrid]).flatten()
    f1 = interp2d(rgrid, zgrid, val["btrz"], kind='cubic')
    f2 = interp2d(rgrid, zgrid, val["bzrz"], kind='cubic')
    val['gammamid'] = np.array([np.arctan(f2(r, val["zmaxis"])
                                          / f1(r, val["zmaxis"]))*180/3.1415926
                                for r in rgrid]).flatten()
    val['bzmid'] = np.array([f2(r, val["zmaxis"])
                             for r in rgrid]).flatten()
    val['btmid'] = np.array([f1(r, val["zmaxis"])
                             for r in rgrid]).flatten()
    f = interp2d(rgrid, zgrid, val["jtrz"], kind='cubic')
    val['jtmid'] = np.array([f(r, val["zmaxis"])
                             for r in rgrid]).flatten()
    f = interp2d(rgrid, zgrid, val["pressrz"], kind='cubic')
    val['pressmid'] = np.array([f(r, val["zmaxis"])
                                for r in rgrid]).flatten()
    f = interp2d(rgrid, zgrid, val["qrz"], kind='cubic')
    val['qmid'] = np.array([f(r, val["zmaxis"])
                            for r in rgrid]).flatten()


def getGfileDict(eqdsk_file):

    def is_ascii(s):
        return all([ord(c) < 128 for c in s])
        
    f = open(eqdsk_file, 'r')

    line = f.readline()
    line = line.rstrip("\r\n")
    tmp = string_split_ig(line[49:], ' |,')

    header = line[:48]
    idum = int(tmp[0])
    mw = int(tmp[1])
    mh = int(tmp[2])
    mw2 = mw
    if len(tmp) == 4:
        try:
            mw2 = int(tmp[3])
            if mw2 == 0:
                mw2 = mw
        except:
            pass

    xdim, zdim, rzero, rgrid1, zmid = eval_form2020(readandstrip(f))
    rmaxis, zmaxis, ssimag, ssibdry, bcentr = eval_form2020(readandstrip(f))
    cpasma, ssimag, xdum, rmaxis, xdum = eval_form2020(readandstrip(f))
    zmaxis, xdum, ssibry, xdum, xdum = eval_form2020(readandstrip(f))

    fpol = load_array_form2020(f, mw2)
    pres = load_array_form2020(f, mw2)
    ffprim = load_array_form2020(f, mw2)
    pprime = load_array_form2020(f, mw2)

    psirz = load_matrix_form2020(f, mh, mw)
    qpsi = load_array_form2020(f, mw2)

    try:
        nbbbs, limitr = string_split_ig(readandstrip(f), ' |,')
    # print nbbbs, limitr
        nbbbs = int(nbbbs)
        limitr = int(limitr)
        if nbbbs != 0:
            rzbbbs = load_matrix_form2020(f, nbbbs, 2)

        xylim = load_matrix_form2020(f, limitr, 2)
    except:
        limitr = 0
        nbbbs = 0
        rzbbbs = np.zeros((2, 1))
        xylim = np.zeros((2, 1))

    rgrid = rgrid1 + xdim*np.arange(mw)/(mw-1.)
    zgrid = zmid-0.5*zdim + zdim*np.arange(mh)/(mh-1.)

    psirzraw = psirz.copy()
    if cpasma > 0:
        sss = -1
        ssimag = ssimag*sss
        ssibdry = ssibdry*sss
        ssibry = ssibry*sss
        psirz = psirz*sss
        ffprim = ffprim*sss
        pprime = pprime*sss

    gfileDict = {}

    gfileDict["header"] = header
    gfileDict["idum"] = idum
    gfileDict["mw"] = mw
    gfileDict["mh"] = mh
    gfileDict["xdim"] = xdim
    gfileDict["zdim"] = zdim
    gfileDict["rzero"] = rzero
    gfileDict["rgrid1"] = rgrid1
    gfileDict["zmid"] = zmid
    gfileDict["rmaxis"] = rmaxis
    gfileDict["zmaxis"] = zmaxis
    gfileDict["ssimag"] = ssimag
    gfileDict["ssibdry"] = ssibdry
    gfileDict["bcentr"] = bcentr
    gfileDict["cpasma"] = cpasma
    gfileDict["ssibry"] = ssibry

    gfileDict["rgrid"] = rgrid
    gfileDict["zgrid"] = zgrid
    gfileDict["psirz"] = psirz
    gfileDict["psirzraw"] = psirzraw
    gfileDict["fpol"] = fpol
    gfileDict["pres"] = pres
    gfileDict["ffprim"] = ffprim
    gfileDict["pprime"] = pprime
    gfileDict["qpsi"] = qpsi
    gfileDict["nbbbs"] = nbbbs
        
    
    if nbbbs > 0:
        gfileDict["rbbbs"] = rzbbbs[:, 0]
        gfileDict["zbbbs"] = rzbbbs[:, 1]
    else:
        from python_lib.analysis.efit_tools import find_psi_contour
        rzbbbs = find_psi_contour(rgrid, zgrid, psirz, rmaxis, zmaxis,
                                  ssibry, return_all=False)

        nbbbs = rzbbbs.shape[0]
        gfileDict["nbbbs"] = nbbbs
        gfileDict["rbbbs"] = rzbbbs[:, 0]
        gfileDict["zbbbs"] = rzbbbs[:, 1]
    gfileDict["nlim"] = limitr
    if limitr > 0:
        gfileDict["xlim"] = xylim[:, 0]
        gfileDict["ylim"] = xylim[:, 1]
    else:
        gfileDict["xlim"] = []
        gfileDict["ylim"] = []
    # else:
    #   return nm
    # namelist section
    sec = None
    end_flag = False
    """
    while 1:
        line0 = f.readline()
        if not is_ascii(line0):
            continue
        if not line0:
            break
        line0 = line0.rstrip("\r\n")
        line0 = ' '.join(line0.split())
        if line0.endswith('/'):
            line = line0[:-1]
        else:
            line = line0

        if line == '':
            continue
        if line.startswith('&'):
            s = string_split_ig(line, ' |,')
            sec = s[0][1:]
            print('making new sec ', sec, line.__repr__(), s)
            nm[sec] = OrderedDict()
            if len(s) > 1:
                line = ' '.join(s[1:])
            else:
                continue
        if sec is None:
            continue  # skip unitl first &*** starts
        sall = string_split_ig(line, ' |,')

        i = 0
        while i < len(sall):
            if sall[i].find('=') != -1:
                if sall[i] == '=':    # '='
                    sall[i-1] = sall[i-1]+'='
                    del sall[i]
                    continue
                if sall[i].startswith('='):  # '=value'
                    sall[i-1] = sall[i-1]+'='
                    sall[i] = sall[i][1:]

                    continue
                if sall[i].endswith('='):
                    i = i+1
                    continue  # 'name='
                k = sall[i].split('=')
                sall[i] = k[0]+'='
                sall.insert(i+1, k[1])
            i = i+1

        for s in sall:
            if s.find('=') != -1:
                k = s.split('=')
                varname = k[0]
#          s[0]=k[1]
#          print s
#          if s[0] is '': del s[0]
                if debug != 0:
                    print(('create dict key', sec, varname))
                # print 'create dict key', sec, varname
                nm[sec][varname] = []
                if s.endswith('/'):
                    sec = None
                continue
        # for lines without 'xxxx = '
            # print s, sec, varname
            nm[sec][varname] = nm[sec][varname]+parseStr(s)
        if line0.endswith('/'):
            sec = None
    """
    xx, yy = np.meshgrid(rgrid, zgrid)
#   isPlasma = xx.copy()
    sss = len(rgrid)*len(zgrid)
    isPlasma = np.array([False]*sss).reshape(len(zgrid), len(rgrid))
    for m, xv in enumerate(rgrid):
        for n, yv in enumerate(zgrid):
            dx = rzbbbs[:, 0] - xv
            dy = rzbbbs[:, 1] - yv
            d1 = np.sqrt(dx[:-1]**2 + dy[:-1]**2)
            d2 = np.sqrt(dx[1:]**2 + dy[1:]**2)

            d = (dx[:-1]*dy[1:] - dx[1:]*dy[:-1])/d1/d2
            d = d[np.abs(d) < 1.0]
            xxxx = sum(np.arcsin(d))
            isPlasma[n, m] = (np.abs(xxxx) > 3)
#              print isPlasma[n, m] > 3
#   gfileDict['isPlasma0'] = isPlasma
#   print 'here'
#   isPlasma =  np.abs(isPlasma) > 3
    gfileDict["isPlasma"] = isPlasma
    
    
        
    add_extra_data(gfileDict)
        
    return gfileDict