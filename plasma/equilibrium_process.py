#!/usr/bin/env python
#
#   Try reading g-eqdsk file with re (regex) module
#   instead of the non-existant fortran format code
#   python feature.
#
#   WARNING: this code has only been testing on the 
#   two files listed below and my regex skills are
#   quite poor (as is my python ) so you have been 
#   warned.
#
#   DLG - 14-Aug-08
#
#   JCW - 27-Jan-10
#
# some improvements to regular expression and implemented direct casting
# of string arrays to floating arrays.
# Regexp are a bit fragile here. Should take advantage of the known data
# ordering with n.fromfile
#
#   JCW - 02-May-12 
#
#   Make into a function call returning a structure with optional outputs
#
# some eqdsk files have data in 16.8, some in 16.9

#Courtesy of OMFIT eq_JET.py
# Need following data:
# GEQDSK NAME                                                          - EQUIVALENT EFIT PPF NAME:
# RDIM: Horizontal dimension in meter of computational box             - FROM PSIR - R Grid
# ZDIM: Vertical dimension in meter of computational box               - FROM PSIZ - Z Grid
# RLEFT: Minimum R in meter of rectangular computational box           - AS ABOVE
# ZMID: Z of center of computational box in meter                      - AS ABOVE
# RMAXIS: R of magnetic axis in meter                                  - RMAG
# ZMAXIS: Z of magnetic axis in meter                                  - ZMAG
# SIMAG: poloidal flux at magnetic axis in Weber /rad                  - FAXS (W/rad)
# SIBRY: poloidal flux at the plasma boundary in Weber /rad            - FBND (W/rad)
# RCENTR: R in meter of vacuum toroidal magnetic field                 - 2.96m
# BCENTR: Vacuum toroidal magnetic field in Tesla at RCENTR            - BVAC
# CURRENT: Plasma current in Ampere                                    - (XIP = measured) XIPC - Calculated plasma current
# FPOL: Poloidal current function in m-T,  F = RBT  on  flux grid      - F
# PRES: Plasma pressure in nt / m2 on uniform flux grid                - P
# FFPRIM: FF'(PSI) in (mT)2 / (Weber /rad) on uniform flux grid        - DFDP * Mu0
# PPRIME: P'(PSI) in (nt /m2) / (Weber /rad) on uniform flux grid      - DPDP
# PSIZR: Poloidal flux in Weber / rad on the rectangular grid points   - PSI
# QPSI: q values on uniform flux grid from axis to boundary            - Q
# NBBBS: Number of boundary points                                     - NBND
# LIMITR: Number of limiter points                                     - length(RLIM)
# RBBBS: R of boundary points in meter                                 - RBND
# ZBBBS: Z of boundary points in meter                                 - ZBND
# RLIM: R of surrounding limiter contour in meter                      - RLIM
# ZLIM: Z of surrounding limiter contour in meter                      - ZLIM

def readGEQDSK(filename='eqdsk.dat', dointerior=False, doplot=None, width=9, dolimiter=None, ax=None, dodebug=False):
    import re
    import numpy as n
    import pylab as p

    file = open (filename)
    data    = file.read ()

    dimensionsRE    = re.compile ( ' {1,3}\d?\d?\d?\d\d' ) # Equivilant to i5 fortran code, JCW these should be i4
    dimensionsRE5    = re.compile ( ' {1,3}\d?\d?\d?\d' ) # Equivilant to i5 fortran code
    headerRE    = re.compile ( '^.*\\n') # First line
    if width==9:
        valuesRE   = re.compile ( '([ \-]\d\.\d{9}[eEdD][\+\-]\d\d)' )   # Equivilant to e16.9 fortran code
    else:
        valuesRE   = re.compile ( '([ \-]\d\.\d{8}[eEdD][\+\-]\d\d)' )   # Equivilant to e16.8 fortran code

#bbbsRE  = re.compile ( '( {1,3}\d?\d?\d?\d\d {1,3}\d?\d?\d?\d\d)' )   # Candidate dimension lines (2i5 fortran code)
    bbbsRE  = re.compile ( r'(?m)^.{10}\n' ) #there should be only one 10 character line
    #print('bbbsRE: ', bbbsRE) #TODO: delete mee
    dataStr     = valuesRE.findall ( data )
    headerStr   = headerRE.findall ( data )
    bbbStr  = bbbsRE.findall ( data )
    print(bbbStr)

    file.close ()
    if len(bbbStr) > 0:
        nbbbsStr    = dimensionsRE5.findall ( bbbStr[0] )
        #print('nbbbsStr: ', nbbbsStr) #TODO: delete mee
    else:
        print('no bounding box found. should be Line with 2 integers length of 10 characters')
        return -1
        
    nWnHStr = dimensionsRE.findall ( headerStr[0] )

    nW  = int ( nWnHStr[-2] )
    nH  = int ( nWnHStr[-1] )

    nbbbs   = int ( nbbbsStr[-2] )
    limitr   = int( nbbbsStr[-1] )

   
    rdim    = float ( dataStr[0] )
    zdim    = float ( dataStr[1] )

    if dodebug==True: print("Data string header:", dataStr[0:20] )
    if dodebug==True: print("Dimensions:", nW, nH, nbbbs, limitr, rdim, zdim )
    if dodebug==True: print("Size of data:", len(dataStr), 20+nW*5+nW*nH+2*nbbbs+2*limitr  )

    rcentr  = float ( dataStr[2] )
    rleft   = float ( dataStr[3] )
    zmid    = float ( dataStr[4] )

    rmaxis  = float ( dataStr[5] )
    zmaxis  = float ( dataStr[6] )
    simag   = float ( dataStr[7] )
    sibry   = float ( dataStr[8] )
    bcentr  = float ( dataStr[9] )

    current = float ( dataStr[10] )

    fpol    = n.zeros ( nW )
    pres    = n.zeros ( nW )
    ffprim  = n.zeros ( nW )
    pprime  = n.zeros ( nW )
    psizr   = n.zeros ( ( nW, nH ) )
    qpsi    = n.zeros ( nW )
    rbbbs   = n.zeros ( nbbbs )
    zbbbs   = n.zeros ( nbbbs )
    rlim    = n.zeros ( limitr )
    zlim    = n.zeros ( limitr )


#   If you know how to cast a list of strings to
#   a numpy array without a loop please let me 
#   know, as these loops should not be required.

#   1D arrays

    for i in n.arange ( nW ) : 
    
        fpol[i] = dataStr[int(i+20)]
        pres[i] = dataStr[int(i+20+nW)]
        ffprim[i] = dataStr[int(i+20+2*nW)]
        pprime[i] = dataStr[int(i+20+3*nW)]
        qpsi[i] = dataStr[int(i+20+4*nW+nW*nH)]

    if dodebug: print('one D arrays: ', fpol[-1],pres[-1], ffprim[-1], pprime[-1], qpsi[-1] )
    for i in n.arange ( nbbbs ) :
    
        rbbbs[i]    = dataStr[int(i*2+20+5*nW+nW*nH)]
        zbbbs[i]    = dataStr[int(i*2+1+20+5*nW+nW*nH)]
  

    for i in n.arange ( limitr ) :
       
        rlim[i] = dataStr[int(i*2+20+5*nW+nW*nH+2*nbbbs)] 
        zlim[i] = dataStr[int(i*2+1+20+5*nW+nW*nH+2*nbbbs)] 

#   2D array

    for i in n.arange ( nW ) :
        for j in n.arange ( nH ) :
            psizr[i,j] = dataStr[int(i+20+4*nW+j*nW)]

    rStep   = rdim / ( nW - 1 )
    zStep   = zdim / ( nH - 1 )
    fStep   = -( simag - sibry ) / ( nW - 1 )

    r   = n.arange ( nW ) * rStep + rleft
    z   = n.arange ( nH ) * zStep + zmid - zdim / 2.0

    fluxGrid    = n.arange ( nW ) * fStep + simag

#   Find indices of points inside and outside
#   the rbbbs/zbbbs boundary.
    import matplotlib.path as mplPath
    import numpy as np
    lcf=mplPath.Path( np.column_stack( (rbbbs,zbbbs) ) )
    iiInsideA   = n.zeros ( psizr.shape )
    iiInside = -1
    iiOutside = -1
    if (dointerior):
        for i in n.arange ( nW ) :
            for j in n.arange ( nH ) :
                if lcf.contains_point( (r[i],z[i]) ):
                    iiInsideA[i,j] = 1
                #q1  = n.size ( n.where ( ( r[i] - rbbbs > 0 ) & ( z[j] - zbbbs > 0 ) ) )
                #q2  = n.size ( n.where ( ( r[i] - rbbbs > 0 ) & ( z[j] - zbbbs <= 0 ) ) )
                #q3  = n.size ( n.where ( ( r[i] - rbbbs <= 0 ) & ( z[j] - zbbbs > 0 ) ) )
                #q4  = n.size ( n.where ( ( r[i] - rbbbs <= 0 ) & ( z[j] - zbbbs <= 0 ) ) )

                #if ( q1 > 0 ) & ( q2 > 0 ) & ( q3 > 0 ) & ( q4 > 0 ) :
                #    iiInsideA[i,j]  = 1
                
        iiInside    = n.where ( iiInsideA > 0 )
        iiOutside   = n.where ( iiInsideA == 0 )

#    print nW, nH, nbbbs, limitr
#    print rdim, zdim, rcentr, rleft, zmid
#    print rmaxis, zmaxis, simag, sibry, bcentr

#   Plot output
    fig='No figure'
    if (doplot):
        N=10
        if not isinstance(doplot,bool):
            if isinstance(doplot,int):
                 N=doplot
        if ax is None:
            fig = p.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            p.contour ( r, z, psizr.T, N )
            p.plot ( rbbbs, zbbbs, 'k', linewidth = 3 )
            if (dolimiter):
                p.plot ( rlim, zlim, 'g', linewidth = 4 )
            p.show ()
        else:
            ax.contour (r, z, psizr.T, N )
            ax.plot ( rbbbs, zbbbs, 'k', linewidth = 3 )
            if (dolimiter):
                ax.plot ( rlim, zlim, 'g', linewidth = 4 ) 

    #checks
    # rmaxis =/ rcentr
    eqdsk = {'nW':nW, 'nH':nH, 'nbbbs':nbbbs, 'limitr':limitr, 'rdim':rdim,
             'zdim':zdim, 'rcentr':rcentr, 'rleft':rleft, 'zmid':zmid, 
             'rmaxis':rmaxis, 'zmaxis':zmaxis, 'simag':simag, 'sibry':sibry,
             'bcentr':bcentr, 'current':current, 'fpol':fpol, 'pres':pres,
             'ffprim':ffprim, 'pprime':pprime, 'psizr':psizr, 'qpsi':qpsi, 'rbbbs':rbbbs,
             'zbbbs':zbbbs, 'rlim':rlim, 'zlim':zlim, 'r':r, 'z':z,
             'fluxGrid':fluxGrid, 'iiInside':iiInside}

    return eqdsk,fig


def getModB(eq):
    """
    Calculate the magnitude of the magnetic field on the RZ mesh.


        |B| = \sqrt(Fpol^2+(d\Psi/dZ)^2+(d\Psi/dR)^2)/R

    where Fpol== R*Bphi , Bpol = |grad Psi|/R
    """
    import numpy as np
    from scipy import interpolate

    #poloidal component
    R=eq.get('r')
    Z=eq.get('z')
    Rv,Zv=np.meshgrid(R,Z) #these are R and Z on RZ mesh
    psiRZ=np.transpose(eq.get('psizr'))
    spline_psi = interpolate.RectBivariateSpline(R,Z,psiRZ.T,bbox=[np.min(R),np.max(R),np.min(Z),np.max(Z)],kx=5,ky=5)
    psi_int_r=spline_psi.ev(Rv,Zv,dx=1)
    psi_int_z=spline_psi.ev(Rv,Zv,dy=1)
    grad_psi=np.sqrt(psi_int_z**2+psi_int_r**2)
    
    #toroidal component
    #get Fpol and interpolate to RZ mesh to get fpolRZ
    fpol=eq.get('fpol')
    psi=eq.get('fluxGrid') #mesh for fpol
    #fpolRZ=fpol(psiRZ)
    # scipy 0.18    spline_fpol=interpolate.CubicSpline(psi,fpol,bc_type='natural')
    #fill value is B0*R0
    spline_fpol=interpolate.interp1d(psi,fpol,bounds_error=False,fill_value=fpol[-1],kind='cubic')
    fpolRZ=[] #np.zeros(psiRZ.shape)
    for psirow in psiRZ:
        fpolRZ.append( spline_fpol(psirow) )
    fpolRZ=np.array(fpolRZ) #Fpol numpy array on RZ mesh

    modgradpsi=np.sqrt(grad_psi**2+fpolRZ**2)
    modB=modgradpsi/Rv
    if R[0]==0.0: #If origin is included in domain, be careful with |B| on axis.
        modB[:,0]=(np.diff(modgradpsi,axis=1)/(R[1]-R[0]))[:,0]
    #Add components
    return modB,grad_psi,fpolRZ,Rv,Zv


def getLCF(eq):
    #find which contour in LCF, same as rbbbs? 

    import matplotlib.path as mplPath
    import numpy as np
    import pylab as p

    R=eq.get('r')
    Z=eq.get('z')
    psiRZ=np.transpose(eq.get('psizr'))    
    CSlcf=p.contour(R,Z,psiRZ,levels=[eq['sibry']-.01])
    cntr=(eq['rmaxis'],eq['zmid'])
    lcf=(0,0)
    for p in CSlcf.collections[0].get_paths():
        v = p.vertices
        x = v[:,0]
        y = v[:,1]
        bbPath = mplPath.Path(np.column_stack( (x,y)))
        if bbPath.contains_point(cntr):
            lcf=(x,y)
            return lcf

def writeEQDSK(eq,fname):
    """Write out the equilibrium in G-EQDSK format.
    Code courtesy of eq_JET.py from OMFIT"""

    import fortranformat as ff
    import numpy as np

    # Open file for writing
    f = open(fname, 'w')

    nr = eq['nW']
    nz = eq['nH']    

    # Get eq at this timeslice
    rdim    = eq['rdim']
    zdim    = eq['zdim']
    rcentr  = eq['rcentr']
    rleft   = eq['rleft']
    zmid    = eq['zmid']
    rmaxis  = eq['rmaxis']
    zmaxis  = eq['zmaxis']
    simag   = eq['simag']
    sibry   = eq['sibry']
    bcentr  = eq['bcentr']
    current = eq['current']
    xdum    = 0.0

    def GetSlice( data, N, ti ):
        return data[ ti * N : ( ti + 1 ) * N ]

    # FPOL eq
    fpol =  eq['fpol']

    # Pressure eq
    pressure = eq['pres']

    # FFPRIM eq
    ffprim = eq['ffprim']

    # PPRIME eq
    pprime = eq['pprime']

    # PSI eq
    psi = np.transpose(eq['psizr'])

    # Q eq
    q = eq['qpsi']
    
    # Plasma Boundary
    Rbnd = eq['rbbbs']
    Zbnd = eq['zbbbs']
    n_bnd = eq['nbbbs'] 

    # Limiter eq
    Rlim = eq['rlim']
    Zlim = eq['zlim']
    limitr = len(Rlim)

    # Write Eqdsk from -----------------------------------

    f2020=ff.FortranRecordWriter('5e16.9')
    f2022=ff.FortranRecordWriter('2i5')
        
    def writeVar(handle,varList):
        f.write(handle.write(varList))
        f.write("\n")

    def writeOrderedPairs(handle,var1,var2):
        longArrayOfPairs=[]
        for pv,_ in enumerate(var1):
            longArrayOfPairs.append(var1[pv])
            longArrayOfPairs.append(var2[pv])

        writeVar(handle,longArrayOfPairs)
        
    A52 = 'plasma.py_v1.0_:_01:01:17'.ljust(48)
    f.write(A52[0:48])
    writeVar(ff.FortranRecordWriter('3i4'), [0,nr,nz] )
    writeVar(f2020,[rdim,zdim,rcentr,rleft,zmid])
    writeVar(f2020,[rmaxis,zmaxis,simag,sibry,bcentr])
    writeVar(f2020,[current,simag,xdum,rmaxis,xdum]) 
    writeVar(f2020,[zmaxis,xdum,sibry,xdum,xdum])  
    writeVar(f2020,fpol)
    writeVar(f2020,pressure)
    writeVar(f2020,ffprim)
    writeVar(f2020,pprime)
    writeVar(f2020,psi.flatten())
    writeVar(f2020,q)
    writeVar(f2022,[n_bnd,limitr])
    writeOrderedPairs(f2020,Rbnd,Zbnd)
    writeOrderedPairs(f2020,Rlim,Zlim)
    
    f.close()


def rescaleB(newR,eq,filename):
    import copy

    R0=eq['rmaxis']
    f=newR/R0

    neweq=copy.deepcopy(eq)
    neweq['psizr']=eq['psizr']*f
    neweq['fpol']=eq['fpol']*f

    neweq['bcentr']=eq['bcentr']*f
    neweq['ffprim']=eq['ffprim']*f*f
    neweq['simag']=eq['simag']*f    
    neweq['sibry']=eq['sibry']*f 
    neweq['current']=eq['current']*f 
    neweq['fluxgGrid']=eq['fluxGrid']*f

    writeEQDSK(neweq,filename)


