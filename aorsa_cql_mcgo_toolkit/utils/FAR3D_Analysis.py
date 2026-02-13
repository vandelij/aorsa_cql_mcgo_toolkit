##################################
#   Classes for working with both setup and post-proccesing of MCGO with P2F
#   Author: Jacob van de Lindt
#   Date: 2/5/2026
#################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm
from matplotlib import ticker, cm
from scipy.interpolate import interp1d, RectBivariateSpline, PchipInterpolator, RegularGridInterpolator
from scipy.optimize import curve_fit
import os, sys
import netCDF4
import f90nml as f90
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import re
# import John's toolkit area
import plasma
from plasma import equilibrium_process
import textwrap

# import Grant's eqdsk processor for getting B info
from process_eqdsk2 import getGfileDict

class Far3D_Analysis:
    """
    Class to post-process output from Far3d 
    """

    def __init__(self, eqdsk_file=None, species='d'):
        self.eqdsk_file = eqdsk_file
        self.species = species
        # load up eqdsk using john's methods
        if self.eqdsk_file is not None:
            self.process_eqdsk()     


        # assign mass from supported mcgo species
        self.species_dict = {}
        self.species_dict['d'] = {'mass':3.343583e-27, 'charge':1.6022e-19}

        self.data_dict = {} # will hold all of the far3d output data
        self.case_txt_dict = {} # will hold the profiles for the case to aid in setup 

        self.headers = flux_file_columns = [
                        "Rho(norml. sqrt. toroid. flux)",
                        "q",
                        "Beam Ion Density(10^13 cm^-3)",
                        "Ion Density(10^13 cm^-3)",
                        "Elec Density(10^13 cm^-3)",
                        "Impurity Density(10^13 cm^-3)",
                        "Beam Ion Effective Temp(keV)",
                        "Ion Temp(keV)",
                        "Electron Temp(keV)",
                        "Beam Pressure(kPa)",
                        "Thermal Pressure(kPa)",
                        "Equil.Pressure(kPa)",
                        "Zeff",
                        "Tor Rot(kHz)",
                        "Tor Rot(10^5 m/s)",
                        "RF Ion Density(10^13 cm^-3)",
                        "RF Ion Effective Temp(keV)",
                        "RF Pressure(kPa)"
                    ]

    def process_eqdsk(self):
        # unpack the equilibrium magnetics
        self.eqdsk, fig = plasma.equilibrium_process.readGEQDSK(
            self.eqdsk_file, doplot=False
        )
        self.eqdsk_with_B_info = getGfileDict(self.eqdsk_file)

        rgrid = self.eqdsk_with_B_info["rgrid"]
        zgrid = self.eqdsk_with_B_info["zgrid"]
        self.R_wall = self.eqdsk["rlim"]
        self.Z_wall = self.eqdsk["zlim"]

        self.R_lcfs = self.eqdsk["rbbbs"]
        self.Z_lcfs = self.eqdsk["zbbbs"]
        self.Rcenter = self.eqdsk['rmaxis']
        self.Zcenter = self.eqdsk['zmaxis']
        B_zGrid = self.eqdsk_with_B_info["bzrz"]
        B_TGrid = self.eqdsk_with_B_info["btrz"]
        B_rGrid = self.eqdsk_with_B_info["brrz"]

        # get the total feild strength
        Bstrength = np.sqrt(
            np.square(B_zGrid) + np.square(B_TGrid) + np.square(B_rGrid)
        )

        # create a function that can grab the B-feild magnitude at any r, z coordiante pair.
        self.getBStrength = RectBivariateSpline(rgrid, zgrid, Bstrength.T)

        # create the normalized flux function #TODO confirm that the user is using this flux coord for mapping
        psizr = self.eqdsk_with_B_info["psirz"]
        psi_mag_axis = self.eqdsk_with_B_info["ssimag"]
        psi_boundary = self.eqdsk_with_B_info["ssibry"]
        self.psirzNorm = (psizr - psi_mag_axis) / (psi_boundary - psi_mag_axis)
        self.getpsirzNorm = RectBivariateSpline(rgrid, zgrid, self.psirzNorm.T)

    def build_B_midplane_mag_interpolator(self):
        self.B_midplane_mag_interpolator = PchipInterpolator(
            self.rho_grid, self.getBStrength(self.R_f_grid, self.eqdsk['zmaxis'], grid=False)
        )

    def plot_equilibrium(self, figsize=(5,5), levels=10, fontsize=20, return_plot=False):
        fig, ax = plt.subplots(figsize=figsize)
        # psizr = self.eqdsk["psizr"]
        # psi_mag_axis = self.eqdsk["simag"]
        # psi_boundary = self.eqdsk["sibry"]

        # ## THIS NEEDS TO BE TOROIDAL RHO
        # # normalize the psirz so that the norm is 1 on boundary and zero on axis
        # psirzNorm = (psizr - psi_mag_axis)/(psi_boundary-psi_mag_axis)
        # ax.axis("equal")
        # # img = ax.contour(self.eqdsk["r"], self.eqdsk["z"], psirzNorm.T, levels=levels, colors='black')
        ax.set_aspect('equal')
        img = ax.contour(
            self.eqdsk_with_B_info["rgrid"],
            self.eqdsk_with_B_info["zgrid"],
            self.psirzNorm,
            levels=levels,
            colors="black",
        )
        ax.plot(self.eqdsk["rlim"], self.eqdsk["zlim"], color="red", linewidth=3)
        ax.plot(self.eqdsk["rbbbs"], self.eqdsk["zbbbs"], color="black", linewidth=3)
        font = FontProperties(size=fontsize)
        formatter = FuncFormatter(lambda x, _: f'{x:.1f}')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)  # If you want y-axis too
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        ax.set_xlabel('R [m]', font=font)
        ax.set_ylabel('Z [m]', font=font)
        ax.grid()
        if return_plot:
            return fig, ax

        plt.show()

    def load_mode_file(self, filename, name):
        """
        Parses a mode file with headers like 'n/m'.
        
        Returns:
            r (np.array): The radial grid (1D array)
            data (np.array): The mode data (2D array: rows=r, cols=modes)
            ns (list): List of toroidal mode numbers n
            ms (list): List of poloidal mode numbers m
        """
        
        # 1. Parse the header line
        with open(filename, 'r') as f:
            header_line = f.readline()

        # Regex to find patterns like "12/ 10" or "-5/-2" 
        # capturing the integer before and after the slash.
        # Pattern explanation:
        #  (-?\d+)  : Capture Group 1 (Integer, optional negative sign)
        #  \s*/\s* : A slash, optionally surrounded by whitespace
        #  (-?\d+)  : Capture Group 2 (Integer, optional negative sign)
        matches = re.findall(r'(-?\d+)\s*/\s*(-?\d+)', header_line)

        # Convert strings to integers
        ms = [int(x[0]) for x in matches]
        ns = [int(x[1]) for x in matches]

        # 2. Load the numerical data
        # skiprows=1 ignores the header we just parsed
        raw_data = np.loadtxt(filename, skiprows=1)

        # Split into radius (col 0) and mode values (cols 1 to end)
        r = raw_data[:, 0]
        data = raw_data[:, 1:]

        # Sanity check
        if data.shape[1] != len(ns):
            print(f"Warning: Found {len(ns)} headers but {data.shape[1]} data columns.")

        # produce interpolator over rho
        interpolators = []
        for i in range(len(ns)):
            interpolator = interp1d(r, data[:,i], kind='cubic')
            interpolators.append(interpolator)

        # store data 
        file_dict = {}
        file_dict['r'] = r
        file_dict['data'] = data
        file_dict['ns'] = ns
        file_dict['ms'] = ms
        file_dict['interpolators'] = interpolators
        self.data_dict[name] = file_dict

        print(f'File {filename} loaded. Access by key {name} from self.data_dict')

    def get_perturbed_RZ(self, R, Z, far3d_output_name, psi_norm_max=0.95, phase=0):
        psi_norm = self.getpsirzNorm(R,Z).item()
        if psi_norm < psi_norm_max:
            theta = np.mod(np.arctan2(Z - self.Zcenter, R - self.Rcenter), 2*np.pi)
            rho = psi_norm # TODO confirm its not np.sqrt(psi_norm)

            # load up the file dict to assess 
            file_dict = self.data_dict[far3d_output_name]
            ms = file_dict['ms']
            #print(np.where(np.array(ms) > 0))
            num_pos_ms = np.where(np.array(ms) > 0)[0].shape[0]
            mode_sum = 0

            for im in range(num_pos_ms):
                m = ms[im]
                print(f'm: {m}')
                # fr_cos_term = file_dict['interpolators'][im](rho)
                # fr_sin_term = file_dict['interpolators'][im + num_pos_ms](rho)
                fr_sin_term = file_dict['interpolators'][im](rho)
                fr_cos_term = file_dict['interpolators'][im + num_pos_ms](rho)
                mode_sum += fr_cos_term * np.cos(m*(theta + phase)) \
                + fr_sin_term * np.sin(m*(theta + phase))
            return mode_sum

        else:
            return 0
        
    def get_2d_mode_structure(self, Rarray, Zarray, far3d_output_name, psi_norm_max=0.95, phase=0):
        mode_2d_structure = np.zeros((Rarray.shape[0], Zarray.shape[0]))
        for iR in range(Rarray.shape[0]):
            for iZ in range(Zarray.shape[0]):
                R = Rarray[iR]
                Z = Zarray[iZ]
                mode_2d_structure[iR, iZ] = self.get_perturbed_RZ(R, 
                                                                  Z, 
                                                                  far3d_output_name, 
                                                                  psi_norm_max=psi_norm_max, 
                                                                  phase=phase)
        return mode_2d_structure
    
    def plot_2d_mode_structure(self, 
                               Rarray, 
                               Zarray, 
                               far3d_output_name, 
                               use_eqdsk_grids=True,
                               figsize=(10,10), 
                               psi_levels=6, 
                               fontsize=14, 
                               psi_norm_max=0.95, 
                               phase=0):
        
        if use_eqdsk_grids:
            Rarray = self.eqdsk_with_B_info["rgrid"]
            Zarray = self.eqdsk_with_B_info["zgrid"]

        fig, ax = self.plot_equilibrium(
            figsize=figsize, levels=psi_levels, fontsize=fontsize, return_plot=True
        )

        mode_2d_structure = self.get_2d_mode_structure(Rarray=Rarray, 
                                                       Zarray=Zarray, 
                                                       far3d_output_name=far3d_output_name, 
                                                       psi_norm_max=psi_norm_max, 
                                                       phase=phase)
        
        c1 = ax.contourf(Rarray, Zarray, mode_2d_structure.T, levels=300, cmap='seismic')
        fig.colorbar(
            c1, ax=ax, label=f"{far3d_output_name} magnitude"
        )


    def load_profile(self, profile_name, profile_array):
        if profile_name not in self.headers:
            raise ValueError(f'Name {profile_name} not recognized.')
        
        self.case_txt_dict[profile_name] = profile_array
        
    def setup_far3d_run_txt_file(self, out_txt_file_path):
        data = np.zeros((len(self.case_txt_dict['Rho(norml. sqrt. toroid. flux)']), len(self.headers)))

        i = 0
        for name in self.headers:
            if name not in self.case_txt_dict.keys():
                print(f'Warning. name {name} profile not found. Filling with zeros...')
                data[:,i] = 0
            else:
                data[:, i] = self.case_txt_dict[name]
            i += 1

        # now, save the txt file 


        # 1. Prepare your descriptive header block
        # Using a f-string makes it easy to inject variables if these change
        header_text = textwrap.dedent(f"""\
        PLASMA GEOMETRY 
        Vacuum Toroidal magnetic field at R=1.69550002m [Tesla]
            1.6621
        Geometric Center Major radius [m]
            1.69
        Minor radius [m]
            0.7936
        Avg. Elongation
            1.59
        Avg. Top/Bottom Triangularity
            0.36
        Main Contaminant Species
            12C
        Main Ion Species mass/proton mass
            2.0
        TRYING TO GET TO BETA(0)=0.011, Rmax=1.71

        {", ".join(self.headers)}""")

        # 2. Save using np.savetxt
        # 'data' is your 2D numpy array
        np.savetxt(
            out_txt_file_path, 
            data, 
            fmt="%.5f",           # Formats numbers to 5 decimal places to match your example
            header=header_text, 
            comments="",          # This prevents the default '#' from being added
            delimiter=" "         # Space delimited
        )
        
        

