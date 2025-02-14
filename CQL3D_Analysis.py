##################################
#   Classes for working with both setup and post-proccesing of CQL3D 
#   Author: Jacob van de Lindt
#   Date: 2/13/2025
#################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm
from matplotlib import ticker, cm 
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os, sys
import netCDF4
import f90nml as f90

# import John's toolkit area
import plasma


class CQL3D_Post_Process():
    """
    Class to post-process the various output files from an Aorsa run 
    """
    def __init__(self, cql3d_nc_file, cql3d_krf_file=None, eqdsk_file=None, cql_input_file=None):
        self.cql3d_nc_file = cql3d_nc_file
        self.cql3d_krf_file = cql3d_krf_file
        self.eqdsk_file = eqdsk_file
        self.cql_input_file=cql_input_file

        # load up eqdsk using john's methods
        self.eqdsk, fig = plasma.equilibrium_process.readGEQDSK(eqdsk_file, doplot=False)

        self.R_wall = self.eqdsk['rlim']
        self.Z_wall = self.eqdsk['zlim']

        self.R_lcfs = self.eqdsk['rbbbs']
        self.Z_lcfs = self.eqdsk['zbbbs']

        # read .nc file and create usfull data 
        self.cql_nc = netCDF4.Dataset(self.cql3d_nc_file,'r')

        if self.cql3d_krf_file != None:
            self.cqlrf_nc = netCDF4.Dataset(self.cql3d_krf_file,'r')

        if self.cql_input_file != None:
            self.cql_nml = f90.read(self.cql_input_file)



    def plot_equilibrium(self, figsize, levels):
        psizr = self.eqdsk['psizr']
        plt.figure(figsize=figsize)
        plt.axis('equal')
        img = plt.contour( self.eqdsk['r'], self.eqdsk['z'], psizr.T, levels=levels)
        plt.plot(self.eqdsk['rlim'], self.eqdsk['zlim'], color='black', linewidth=3)
        plt.plot(self.eqdsk['rbbbs'], self.eqdsk['zbbbs'], color='black', linewidth=3)
        plt.colorbar(img)
        plt.show()

    def print_keys(self):
        print('The cql3d.nc file has keys')
        print(self.cql_nc.keys())
        if self.cql3d_krf_file != None:
            print('\n The cql3d_krf.nc file has keys')
            print(self.cqlrf.keys())

    def parse_cql_nc(self):
        self.rya = np.ma.getdata(self.cql_nc.variables["rya"][:])

        #pitch angles mesh at which f is defined in radians.
        #Note that np.ma.getdata pulls data through mask which
        # rejects bad data (NAN, etc)
        self.pitchAngleMesh = np.ma.getdata(self.cql_nc.variables["y"][:])

        #normalized speed mesh of f
        self.normalizedVel = self.cql_nc.variables["x"][:]

        self.enerkev = self.cql_nc.variables["enerkev"][:]

        #flux surface average energy per particle in keV 
        self.energy = self.cql_nc.variables["energy"][:]

        self.ebkev = self.cql_nml['frsetup']['ebkev'][0] # TODO for now jsut take the first beam energy