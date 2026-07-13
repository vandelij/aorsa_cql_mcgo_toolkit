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
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit, root, least_squares
import os, sys
import netCDF4
import f90nml as f90
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from scipy.interpolate import UnivariateSpline
import re
# import John's toolkit area
import plasma
from plasma import equilibrium_process
import textwrap
import h5py
import json
from freeqdsk import geqdsk

# import Grant's eqdsk processor for getting B info
from process_eqdsk2 import getGfileDict

class Far3D_Analysis:
    """
    Class to post-process output from Far3d 
    """

    def __init__(self, eqdsk_file=None, species='d', fast_species='d'):
        self.eqdsk_file = eqdsk_file
        self.species = species
        self.fast_species = fast_species
        # load up eqdsk using john's methods
        if self.eqdsk_file is not None:
            self.process_eqdsk()     


        # assign mass 
        self.species_dict = {}
        self.species_dict['d'] = {'mass':3.343583e-27, 'charge':1.6022e-19}
        self.species_dict['dt-mix'] = {'mass':3.343583e-27*2.5/2, 'charge':1.6022e-19}
        self.species_dict['alpha'] = {'mass':3.343583e-27*2, 'charge':2*1.6022e-19}

        self.data_dict = {} # will hold all of the far3d output data
        self.case_txt_dict = {} # will hold the profiles for the case to aid in setup 

        # initialize lists to later hold fitted maxwellians for far3d. 
        self.nbulk = []
        self.Tbulk = []
        self.nNBI = []
        self.TNBI = []
        self.nRF = []
        self.TRF = []

        self.headers = [
                    "rho_e",           # 1. Normalized Rho
                    "qprof",           # 2. Safety factor q
                    "den_beam_e",      # 3. Beam Ion Density (10^20 m^-3)
                    "den_ion_e",       # 4. Bulk Ion Density (10^20 m^-3)
                    "den_elec_e",      # 5. Electron Density (10^20 m^-3)
                    "den_alpha_e",     # 6. RF Ion Density (10^20 m^-3) -> Using Alpha slot
                    "den_imp_e",       # 7. Impurity Density (10^20 m^-3)
                    "temp_beam_e",     # 8. Beam Ion Effective Temp (keV)
                    "temp_ion_e",      # 9. Bulk Ion Temp (keV)
                    "temp_elec_e",     # 10. Electron Temp (keV)
                    "temp_alpha_e",    # 11. RF Ion Effective Temp (keV) -> Using Alpha slot
                    "pres_beam_e",     # 12. Beam Pressure (kPa)
                    "pres_thermal_e",  # 13. Thermal Pressure (kPa)
                    "pres_equil_e",    # 14. Equilibrium Pressure (kPa)
                    "tor_rot_vel_e",   # 15. Toroidal Rotation (km/s)
                    "pol_rot_vel_e"    # 16. Poloidal Rotation (km/s)
                ]
        
        self.output_types = [
            "br",
            "bth",
            "phi",
            "pr",
            "psi",
            "uzt",
            "vr",
            "vth",
            "vthprlf",
            "nf"
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

        ax.scatter(self.Rcenter, self.Zcenter, marker='x', color='black')
        if return_plot:
            return fig, ax

        plt.show()

    def load_mode_file(self, filename, name):
        """
        Parses a mode file with headers like 'n/m'.
        Combines Real (R) and Imaginary (I) columns into a complex array.
        """
        import re
        import numpy as np
        from scipy.interpolate import interp1d

        # 1. Parse the header line
        with open(filename, 'r') as f:
            header_line = f.readline()

        # Regex finds all m/n patterns (both R and I halves)
        matches = re.findall(r'(-?\d+)\s*/\s*(-?\d+)', header_line)

        # Since it prints all Reals, then all Imaginaries, the true number of modes is half the matches
        num_modes = len(matches) // 2

        # Only grab the first half for our labels
        ms = [int(x[0]) for x in matches[:num_modes]]
        ns = [int(x[1]) for x in matches[:num_modes]]

        # 2. Load the numerical data
        raw_data = np.loadtxt(filename, skiprows=1)
        r = raw_data[:, 0]

        # 3. Reconstruct the Complex Array
        # Reals are in columns 1 through num_modes
        # Imaginaries are in columns num_modes+1 to the end
        real_part = raw_data[:, 1 : 1 + num_modes]
        imag_part = raw_data[:, 1 + num_modes : 1 + 2 * num_modes]
        
        # Combine into a single complex array
        complex_data = real_part + 1j * imag_part

        # produce interpolator over rho (interp1d handles complex numbers natively!)
        interpolators = []
        for i in range(num_modes):
            interpolator = interp1d(r, complex_data[:, i], kind='cubic')
            interpolators.append(interpolator)

        # store data 
        file_dict = {}
        file_dict['r'] = r
        file_dict['data'] = complex_data  # Now holds complex values!
        file_dict['ns'] = ns
        file_dict['ms'] = ms
        file_dict['interpolators'] = interpolators
        self.data_dict[name] = file_dict

        print(f'File {filename} loaded. {num_modes} complex modes found.')

    def read_all_profile_outputs_and_store(self, directory, run_indicator):
        for name in self.output_types:
            file_name = directory + f'/{name}_{run_indicator}'
            print(f'Loading {name}')
            self.load_mode_file(filename=file_name, name=name)

    def find_maximum_growth_rate_and_frequency(self, directory):
        # far3d_output_file = directory + f'/farprt{run_indicator}'
        far3d_output_file = directory + f'/farprt'
        best = {'gam': -1e30, 'om_r': 0.0}
        pattern = re.compile(
            r'^\s*(?:psi|phi|pr)\s*:\s*m=\s*\d+\s+n=\s*\d+\s*gam=\s*([0-9E+\-\.]+)\s*om_r=\s*([0-9E+\-\.]+)'
        )
        with open(far3d_output_file) as f:
            for line in f:
                m = pattern.match(line)
                if not m: continue
                gam, omr = float(m.group(1)), float(m.group(2))
                if gam > best['gam']:
                    best['gam'], best['om_r'] = gam, omr

        if best['gam'] < -1e29:
            raise RuntimeError("No mode lines found in farprt output")
        #print(f"[norm] Selected mode: γ = {best['gam']:.3e}  ω_r = {best['om_r']:.3e}  (τₐ⁻¹)")
        return best['gam'], best['om_r']
    
    def calculate_alfven_angular_freq(self):
        # read eqdsk 
        B0 = self.eqdsk['bcentr']
        R0 = self.eqdsk['rcentr']
        n_i0 = self.loaded_profile_txt['den_ion_e'][0] * 1e20 # convert back to real units 

        # Physics constants
        q_e = 1.602e-19       # Coulombs
        mu_0 = 4 * np.pi * 1e-7 # Vacuum permeability
        
        # 1. Alfvén velocity
        rho_mass = (n_i0) * (self.species_dict[self.species]['mass']) # Core mass density (kg/m^3)
        v_A0 = np.abs(B0) / np.sqrt(mu_0 * rho_mass)            # m/s
        
        # 2. Alfvén frequency
        omega_A0 = v_A0 / R0   # rad/s

        return omega_A0



    def plot_all_output_profiles(self, run_indicator, directory, wave_part='real', rotate=False, figsize=(20,20)):
        num_plots = len(self.output_types) - 1
        if num_plots % 2 == 0:
            num_rows = int(num_plots / 2)
        else:
            num_rows = int(num_plots / 2) + 1

        fig, axs = plt.subplots(num_rows, 2, figsize=figsize)
        axes = axs.flatten()

        # first, find the fastest growing mode and convert to kHz 
        # omega_A0 = self.calculate_alfven_angular_freq()
        # print(f'Found omega_A0:{omega_A0}')
        gamma, omega_r = self.find_maximum_growth_rate_and_frequency(directory=directory)
        # un_normalize = omega_A0  / (2 * np.pi * 1000) # to kHz
        with open(directory+'conversion_factor.txt', 'r') as f:
            un_normalize = float(f.read().strip())

        print(f'Conversion to KHz: {un_normalize}')
        gamma_kHz = gamma * un_normalize
        omega_r_kHz = omega_r * un_normalize

        # loop over data and make mass output plots 
        for i, name in enumerate(self.output_types):
            data_dict = self.data_dict[name]
            rgrid = data_dict['r']
            data = data_dict['data']
            ns = data_dict['ns']
            ms = data_dict['ms']
            
            # --- NEW GLOBAL ROTATION LOGIC ---
            if rotate:
                # 1. Find the 2D index of the absolute largest value across ALL modes
                # np.argmax on a 2D array flattens it, so we unravel it back to (row, col)
                max_idx = np.unravel_index(np.argmax(np.abs(data)), data.shape)
                
                # 2. Extract the phase of that single dominant point
                global_peak_phase = np.angle(data[max_idx])
            # ---------------------------------

            for imode in range(len(ns)):
                n = ns[imode]
                m = ms[imode]
                profile = data[:,imode]

                # now, rotate the wave using the SAME angle for every mode
                if rotate:
                    profile = profile * np.exp(-1j * global_peak_phase)

                if wave_part == 'real':
                    profile = np.real(profile)
                elif wave_part == 'imaginary':
                    profile = np.imag(profile)
                elif wave_part == 'mag':
                    profile = np.abs(profile)
                else:
                    raise ValueError('Wave plotting type wave_part not understood')
                
                label = f'm/n: {m}/{n}'
                axes[i].plot(rgrid, profile, label=label)
            
            axes[i].grid()
            axes[i].set_title(f'Name: {name}')
            axes[i].legend()

        title_string = r'$\gamma \tau_{A}$ ='+ f' {gamma:.3f}\n' + r'$\gamma$ = ' + f'{gamma_kHz:.3f} kHz\n' + r'$\omega_r$ =' + f' {omega_r_kHz:.3f} kHz' 

        fig.suptitle(title_string, fontsize=30, fontweight='bold')

    def get_perturbed_RZ(self, R, Z, far3d_output_name, psi_norm_max=0.95, phase=0, rotate=True):
        psi_norm = self.getpsirzNorm(R, Z).item()
        
        if psi_norm < psi_norm_max:
            # Calculate geometric coordinates
            theta = np.mod(np.arctan2(Z - self.Zcenter, R - self.Rcenter), 2 * np.pi)
            rho = np.sqrt(psi_norm) # Yes, rho is typically sqrt(psi_norm) in standard flux coordinates

            file_dict = self.data_dict[far3d_output_name]
            ms = file_dict['ms']
            data = file_dict['data']
            interpolators = file_dict['interpolators']

            # --- 1. Global Phase Unwrapping ---
            if rotate:
                # Find the dominant peak to anchor the phase (just like 1D plots)
                max_idx = np.unravel_index(np.argmax(np.abs(data)), data.shape)
                global_peak_phase = np.angle(data[max_idx])
            else:
                global_peak_phase = 0.0

            # --- 2. Complex Mode Summation ---
            mode_sum = 0.0 + 0.0j
            
            # Since the executable returns complex values, interpolators map 1-to-1 with ms.
            # We no longer need to split 'num_pos_ms' for separate cos/sin arrays.
            for im, m in enumerate(ms):
                # A_m(rho) is now a complex amplitude: X_real + i * X_imag
                A_m = interpolators[im](rho)
                
                # Spatial wave form: A_m * e^(i * m * theta)
                mode_sum += A_m * np.exp(1j * m * (theta + phase))

            # --- 3. Apply Rotation and Extract Physical Wave ---
            # Rotate the entire complex sum by the unwrapping phase, then take the Real part
            physical_perturbation = np.real(mode_sum * np.exp(-1j * global_peak_phase))
            
            return physical_perturbation

        else:
            return 0.0
        

    # def get_perturbed_RZ(self, R, Z, far3d_output_name, psi_norm_max=0.95, phase=0):
    #     psi_norm = self.getpsirzNorm(R,Z).item()
    #     if psi_norm < psi_norm_max:
    #         theta = np.mod(np.arctan2(Z - self.Zcenter, R - self.Rcenter), 2*np.pi)
    #         rho = np.sqrt(psi_norm) # TODO confirm its not np.sqrt(psi_norm)

    #         # load up the file dict to assess 
    #         file_dict = self.data_dict[far3d_output_name]
    #         ms = file_dict['ms']
    #         #print(np.where(np.array(ms) > 0))
    #         num_pos_ms = np.where(np.array(ms) > 0)[0].shape[0]
    #         mode_sum = 0

    #         for im in range(num_pos_ms):
    #             m = ms[im]
    #             #print(f'm: {m}')
    #             # fr_cos_term = file_dict['interpolators'][im](rho)
    #             # fr_sin_term = file_dict['interpolators'][im + num_pos_ms](rho)
    #             fr_sin_term = file_dict['interpolators'][im](rho)
    #             fr_cos_term = file_dict['interpolators'][im + num_pos_ms](rho)
    #             mode_sum += fr_cos_term * np.cos(m*(theta + phase)) \
    #             + fr_sin_term * np.sin(m*(theta + phase))
    #         return mode_sum

    #     else:
    #         return 0
        
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
                               ax=None, 
                               n=None,
                               colorbar_label='Fill me in',
                               use_eqdsk_grids=True,
                               figsize=(10,10), 
                               psi_levels=6, 
                               fontsize=14, 
                               psi_norm_max=0.95, 
                               phase=0,
                               use_range = False,
                               vmin=-1,
                               vmax=1,
                               ):
        
        if use_eqdsk_grids:
            Rarray = self.eqdsk_with_B_info["rgrid"]
            Zarray = self.eqdsk_with_B_info["zgrid"]

        if ax == None:
            fig, ax = self.plot_equilibrium(
                figsize=figsize, levels=psi_levels, fontsize=fontsize, return_plot=True
            )

        else: 
            fig = ax.get_figure()

        mode_2d_structure = self.get_2d_mode_structure(Rarray=Rarray, 
                                                       Zarray=Zarray, 
                                                       far3d_output_name=far3d_output_name, 
                                                       psi_norm_max=psi_norm_max, 
                                                       phase=phase)
        if use_range:
            c1 = ax.contourf(Rarray, Zarray, mode_2d_structure.T, levels=300, cmap='seismic', vmin=vmin, vmax=vmax)
        else:
            abs_max = np.max(np.abs(mode_2d_structure))
            c1 = ax.contourf(Rarray, Zarray, mode_2d_structure.T, levels=300, cmap='seismic', vmin=-abs_max, vmax=abs_max)
        cbar = fig.colorbar(
            c1, ax=ax
        )

        cbar.set_label(label=f"{colorbar_label} [arb. units]", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        if n != None:
            ax.set_title(f'n={n}', fontsize=fontsize)

        return ax

    def read_far3d_run_txt_file(self, in_txt_file_path):
        """
        Reads a formatted FAR3d profile text file and returns a dictionary
        mapping column names to 1D numpy arrays, identical to case_txt_dict.
        """
        import numpy as np
        
        # 1. Read the file to find where the header ends and data begins
        with open(in_txt_file_path, 'r') as f:
            lines = f.readlines()
            
        data_start_idx = 0
        for i, line in enumerate(lines):
            # Search for the column header line
            if line.strip().startswith("rho_e"):
                data_start_idx = i + 1
                break
                
        if data_start_idx == 0:
            raise ValueError(f"Could not find the 'rho_e' header line in {in_txt_file_path}")

        # 2. Load the numeric block into a 2D numpy array
        # skiprows skips all the text headers we just counted
        data = np.loadtxt(in_txt_file_path, skiprows=data_start_idx)
        
        # 3. Reconstruct the dictionary using your class's headers list
        loaded_dict = {}
        for i, name in enumerate(self.headers):
            # Extract the i-th column and assign it to the header name
            loaded_dict[name] = data[:, i]
            
        print(f"Successfully loaded profiles from {in_txt_file_path}")
        print(f"Loaded {len(loaded_dict['rho_e'])} radial grid points.")
        
        self.loaded_profile_txt = loaded_dict #TODO now you can calc the alfven frequency 



# switching to pre-processing functions for run setup. 

    def load_profile(self, profile_name, profile_array):
        if profile_name not in self.headers:
            raise ValueError(f'Name {profile_name} not recognized.')
        
        self.case_txt_dict[profile_name] = profile_array
        
    def setup_far3d_run_txt_file(self, out_txt_file_path, ion_mass_to_p_mass=2):
        data = np.zeros((len(self.case_txt_dict['rho_e']), len(self.headers)))

        i = 0
        for name in self.headers:
            if name not in self.case_txt_dict.keys():
                print(f'Warning. name {name} profile not found. Filling with zeros...')
                data[:,i] = 0
            else:
                data[:, i] = self.case_txt_dict[name]
            i += 1

        # calculate the minor radius 
        minor_radius = (max(self.eqdsk['rbbbs']) - min(self.eqdsk['rbbbs'])) / 2
        # now, save the txt file 


        # 1. Prepare your descriptive header block
        # Using a f-string makes it easy to inject variables if these change
        header_text = textwrap.dedent(f"""\
        PLASMA GEOMETRY 
        Vacuum Toroidal magnetic field at R={self.eqdsk['rcentr']:.3f}m [Tesla]
            {self.eqdsk['bcentr']:.3f}
        Geometric Center Major radius [m]
            {self.eqdsk['rcentr']:.3f}
        Minor radius [m]
            {minor_radius:.3f}
        Avg. Elongation
            1.59
        Avg. Top/Bottom Triangularity
            0.36
        Main Contaminant Species
            12C
        Main Ion Species mass/proton mass
            {ion_mass_to_p_mass:.1f}
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

    def calculate_effective_temperature(self, varray, farray, species, F_type='full'):
        """Calculates the effective temperature of a speed distribution.

        Parameters
        ----------
        varray : float array
            velocity array [m/s]. Must extend over desired part of distribution.  
        farray : float array
            distrubtuion vlaue. leading coefficients cancel out
        species : string
            species name for mass retreival 

        Returns
        -------
        float
            effective temperature [keV]
        """
        jouls_to_kev = 1/(1000 * 1.6022e-19)
        mass = self.species_dict[species]['mass']
        if F_type == 'full':
            return jouls_to_kev*mass*np.trapz(farray*varray**2, varray) / np.trapz(farray, varray)
        elif F_type == 'speed':
            return  jouls_to_kev*mass*np.trapz(farray, varray) / np.trapz(farray/varray**2, varray)
        

    def calculate_average_speed(self, varray, farray, F_type='full'):
        """Calculates the true average speed of a velocity or speed distribution.

        Parameters
        ----------
        varray : float array
            velocity array [m/s]. Must extend over desired part of distribution.  
        farray : float array
            distribution value. leading coefficients cancel out
        F_type : string
            'full' or 'speed' determining the distribution format

        Returns
        -------
        float
            average speed [m/s]
        """
        # We use np.abs(varray) because average speed must be strictly positive, 
        # even if the velocity array spans from negative to positive.
        speed_array = np.abs(varray)
        
        if F_type == 'full':
            # Expected value of |v|: integral of |v|*f(v) / integral of f(v)
            return np.trapz(farray * speed_array, varray) / np.trapz(farray, varray)
            
        elif F_type == 'speed':
            # Following your 'speed' convention: we lower the power of v by 1 
            # compared to your temperature function.
            return np.trapz(farray / speed_array, varray) / np.trapz(farray / varray**2, varray)
    
    def maxwell(self, v, n, T, species):
        """maxwellian distribution given v

        Parameters
        ----------
        v : float
            particle speed [m/s]
        n : float
            density [1/m^3]
        T : float
            temperature [keV]
        """
        T = T * 1000 * 1.6022e-19 # convert to J
        mass = self.species_dict[species]['mass']
        return n*(mass / (2*np.pi*T))**(3/2) * np.exp(-mass*v**2/(2*T))
    
    def maxwell_speed_distribution(self, s, n, T, species):
        return self.maxwell(v=s,n=n,T=T, species=species) * 4 * np.pi * s**2
    
    def dmaxwell_speed_distribution(self, s, n, T, species):
        T = T * 1000 * 1.6022e-19 # convert to J
        mass = self.species_dict[species]['mass']

        alpha = 4*np.pi*(mass/(2*np.pi))**(3/2)
        beta = mass/2

        return n*alpha*(1/T)**(3/2) * np.exp(-beta*s**2/T) * 2 * s * (1 - s**2*beta/T)

    
    def slowing_down_far3d(self, v, n, Te, species):
        me = 9.109e-31
        ve = np.sqrt(Te*1000*1.6022e-19*2/me)
        mass = self.species_dict[species]['mass']
        vc = (3 * np.sqrt(np.pi) * me/(4*mass))**(1/3) * ve
        return (n/(4*np.pi)) / (v**3 + vc**3)
    
    def slowing_down_speed_far3d(self, s, n, Te, species):
        return 4*np.pi*s**2 * self.slowing_down_far3d(v=s, n=n, Te=Te, species=species)
    
    def integrate_speed_distribution(self, v, F):
        return np.trapz(F, v) # density in m^-3
    
    def load_electron_temperature_profile(self, rho, Te):
        self.Te_interpolator_kev  = interp1d(rho, Te, kind='linear')

    def load_electron_density_profile(self, rho, ne):
        self.ne_interpolator_m3  = interp1d(rho, ne, kind='linear')

    def load_bulkion_density_profile(self, rho, ni):
        self.ni_interpolator_m3  = interp1d(rho, ni, kind='linear')

    def load_ion_temperature_profile(self, rho, Ti):
        self.Ti_interpolator_kev  = interp1d(rho, Ti, kind='linear')

    def load_bulk_interpolators_from_json(self, json_file):
        with open(json_file, 'r') as file:
            data_dict = json.load(file)

        rgrid = data_dict["rho"]
        Te = data_dict["Te [keV]"]
        TD = data_dict["TD [keV]"]
        nD = data_dict["nD [m^-3]"]
        ne = data_dict["ne [m^-3]"]
        self.load_electron_temperature_profile(rgrid, Te)
        self.load_ion_temperature_profile(rgrid, TD)
        self.load_electron_density_profile(rgrid, ne)
        self.load_bulkion_density_profile(rgrid, nD)

    def sum_of_maxwellians(self, v, *params):
        """Computes the sum of N maxwellians over the speed grid v

        Parameters
        ----------
        v : iterable of floats
            speed array
            *params expects densities. Additionally, self.maxwell_fit_temps needs to be set. 
        """
        y = np.zeros_like(v)
        for i in range(0, len(params)):
            n = params[i]
            T = self.maxwell_fit_temps[i]
            y += self.maxwell_speed_distribution(s=v, n=n, T=T, species=self.species)
        return y
    
    def single_maxwellain_to_fit(self, v, *params):
        n = params[0]
        T = params[1]
        return self.maxwell_speed_distribution(s=v, n=n, T=T, species=self.species)
    
    def log10_wrapper_single_maxwellain_to_fit(self, v, *params):
        n = params[0]
        T = params[1]
        return np.log10(self.maxwell_speed_distribution(s=v, n=n, T=T, species=self.species)+1e-50)
    
    def log10_wrapper_sum_maxwellians(self, v, *params):
        return np.log10(self.sum_of_maxwellians(v, *params) + 1e-50)
    
    def sum_of_maxwellians_bulk_nbi(self, v, *params):
        """Computes the sum of 2 maxwellians over the speed grid v

        Parameters
        ----------
        v : iterable of floats
            speed array
            *params expects densities. Additionally, self.maxwell_fit_temps needs to be set. 
        """
        Tbulk = self.T_bulk_opt
        nbulk = params[0]
        n_nbi = params[1]
        T_nbi = params[2]
        bulk = self.maxwell_speed_distribution(s=v, n=nbulk, T=Tbulk, species=self.species)
        nbi = self.maxwell_speed_distribution(s=v, n=n_nbi, T=T_nbi, species=self.species)
        return bulk + nbi
    
    def log10_wrapper_sum_maxwellians_bulk_nbi(self, v, *params):
        return np.log10(self.sum_of_maxwellians_bulk_nbi(v, *params) + 1e-50)

    def sum_of_maxwellians_rf(self, v, *params):
        """Computes the sum of 3 maxwellians over the speed grid v

        Parameters
        ----------
        v : iterable of floats
            speed array
            *params expects densities. Additionally, self.maxwell_fit_temps needs to be set. 
        """
        nbulk = self.opt1_results[0]
        Tbulk = self.opt1_results[1]
        n_nbi = self.opt1_results[2]
        T_nbi = self.opt1_results[3]
        nRF = params[0]
        TRF = params[1]

        bulk = self.maxwell_speed_distribution(s=v, n=nbulk, T=Tbulk, species=self.species)
        nbi = self.maxwell_speed_distribution(s=v, n=n_nbi, T=T_nbi, species=self.species)
        rf = self.maxwell_speed_distribution(s=v, n=nRF, T=TRF, species=self.species)
        return bulk + nbi + rf
    
    def log10_wrapper_sum_maxwellians_rf(self, v, *params):
        return np.log10(self.sum_of_maxwellians_rf(v, *params) + 1e-50)

    def fit_3_maxwellians_to_speed_distribution(self, v, F, E_NBI_kev, rho):
        E_NBI = E_NBI_kev * 1000 * 1.6022e-19
        v_NBI = np.sqrt(2*E_NBI / self.species_dict[self.species]['mass'])
        if v[-1] < v_NBI:
            raise ValueError(f'The velocity grid supplied does not extend to beam input energy {E_NBI_kev} keV')
        
        n_tot = self.integrate_speed_distribution(v=v, F=F)
        n_bulk_guess = n_tot

        Tbulk_kev = self.Ti_interpolator_kev(rho)
        Te_bulk_kev = self.Te_interpolator_kev(rho) 

        slowing_down_v = np.linspace(v[0], v_NBI, len(v))
        slowing_down_shape = self.slowing_down_far3d(v=slowing_down_v, n=1, Te=Te_bulk_kev, species=self.species)
        T_NBI = self.calculate_effective_temperature(varray=slowing_down_v, farray=slowing_down_shape, species=self.species)

        F_interp = interp1d(v, F, kind='linear')
        me = 9.109e-31
        ve = np.sqrt(Te_bulk_kev*1000*1.6022e-19*2/me)
        mass = self.species_dict[self.species]['mass']
        vc = (3 * np.sqrt(np.pi) * me/(4*mass))**(1/3) * ve
        F_at_v_NBI = F_interp(v_NBI)
        n_NBI_guess = F_at_v_NBI * (v_NBI**3 + vc**3) / v_NBI**2

        RF_v = np.linspace(v_NBI, v[-1], len(v))
        F_RF = F_interp(RF_v)
        T_RF = self.calculate_effective_temperature(varray=RF_v, farray=F_RF, species=self.species) #TODO this is not correct, F(s) vs f(v)
        n_RF_guess = self.integrate_speed_distribution(v=RF_v, F=F_RF)

        # set up optimization initial conditions
        print(f'temps found: {[float(Tbulk_kev), T_NBI, T_RF]}')
        self.maxwell_fit_temps = [float(Tbulk_kev), T_NBI, T_RF]

        initial_guess = [n_bulk_guess, n_NBI_guess, n_RF_guess]
        lower_bounds = [n_bulk_guess/2, 0.0, 0.0]
        upper_bounds = [n_bulk_guess*1.001, n_tot, n_tot] # certainly, the maximums will not be higher than the totol!
        print(initial_guess)
        # now, actually perform the optimization 
        popt, pcov = curve_fit(
            self.log10_wrapper_sum_maxwellians, 
            v, 
            np.log10(F+1e-50), 
            p0=initial_guess, 
            bounds=(lower_bounds, upper_bounds)
        )     

        return popt, self.maxwell_fit_temps
    
    def fit_3_maxwellians_to_speed_distribution_sequential(self, v, F, E_NBI_kev, rho):
        E_NBI = E_NBI_kev * 1000 * 1.6022e-19
        v_NBI = np.sqrt(2*E_NBI / self.species_dict[self.species]['mass'])

        if v[-1] < v_NBI:
            raise ValueError(f'The velocity grid supplied does not extend to beam input energy {E_NBI_kev} keV')
        
        n_tot = self.integrate_speed_distribution(v=v, F=F)
        n_bulk_guess = n_tot*0.99 # small factor to make sure the optimization is happy

        Tbulk_kev = self.Ti_interpolator_kev(rho)
        Te_bulk_kev = self.Te_interpolator_kev(rho) 

        slowing_down_v = np.linspace(v[0], v_NBI, len(v))
        slowing_down_shape = self.slowing_down_far3d(v=slowing_down_v, n=1, Te=Te_bulk_kev, species=self.species)
        T_NBI_guess = self.calculate_effective_temperature(varray=slowing_down_v, farray=slowing_down_shape, species=self.species)

        F_interp = interp1d(v, F, kind='linear')
        me = 9.109e-31
        ve = np.sqrt(Te_bulk_kev*1000*1.6022e-19*2/me)
        mass = self.species_dict[self.species]['mass']
        vc = (3 * np.sqrt(np.pi) * me/(4*mass))**(1/3) * ve
        F_at_v_NBI = F_interp(v_NBI)
        n_NBI_guess = F_at_v_NBI * (v_NBI**3 + vc**3) / v_NBI**2

        RF_v = np.linspace(v_NBI, v[-1], len(v))
        F_RF = F_interp(RF_v)
        T_RF_guess = T_NBI_guess #self.calculate_effective_temperature(varray=RF_v, farray=F_RF, species=self.species)
        n_RF_guess = self.integrate_speed_distribution(v=RF_v, F=F_RF)

        # set up optimization initial conditions for optimization 1:
        v_opt1 = np.linspace(v[0], v_NBI, len(v))
        F_opt_1 = F_interp(v_opt1)
        initial_guess = [n_bulk_guess, n_NBI_guess, T_NBI_guess]
        lower_bounds = [n_bulk_guess/2, 0.0, Te_bulk_kev]
        upper_bounds = [n_tot, n_tot, T_NBI_guess] # certainly, the maximums will not be higher than the totol for the density!
        self.T_bulk_opt = Tbulk_kev # set the tbulk for the optimizer to use
        print(f'initial guess for optimization 1 [n_bulk_guess, n_NBI_guess, T_NBI_guess]: {initial_guess}')

        # now, actually perform the optimization 1
        popt, pcov = curve_fit(
            self.log10_wrapper_sum_maxwellians_bulk_nbi, 
            v_opt1, 
            np.log10(F_opt_1+1e-50), 
            p0=initial_guess, 
            bounds=(lower_bounds, upper_bounds)
        )   

        # unwrap found answers and store for the next optimization 
        n_bulk = popt[0]
        n_NBI = popt[1]
        T_NBI = popt[2]
        print(f'Found: nbulk: {n_bulk}, nNBI: {n_NBI}, T_NBI: {T_NBI} kev')
        self.opt1_results = [n_bulk, Tbulk_kev, n_NBI, T_NBI]  

        # start second fit 
        initial_guess = [n_RF_guess, T_RF_guess]
        lower_bounds = [0, T_NBI]
        upper_bounds = [n_NBI, T_RF_guess*100] 
        print(f'initial guess for optimization 2: [n_RF_guess, T_RF_guess]: {initial_guess}')

        # now, actually perform the optimization 2
        popt2, pcov2 = curve_fit(
            self.log10_wrapper_sum_maxwellians_rf, 
            v, 
            np.log10(F+1e-50), 
            p0=initial_guess, 
            bounds=(lower_bounds, upper_bounds)
        )   

        # unpack the results 
        n_RF = popt2[0]
        T_RF = popt2[1]
        print(f'Results: n_bulk {n_bulk}, Tbulk_kev {Tbulk_kev}, n_NBI {n_NBI}, T_NBI {T_NBI}, n_RF {n_RF}, T_RF {T_RF}')
        return [n_bulk, Tbulk_kev, n_NBI, T_NBI, n_RF, T_RF]

    
    def fit_3_maxwellians_to_speed_distribution_NBI_moment(self, v, F, E_NBI_kev, rho):
        E_NBI = E_NBI_kev * 1000 * 1.6022e-19
        v_NBI = np.sqrt(2*E_NBI / self.species_dict[self.species]['mass'])

        if v[-1] < v_NBI:
            raise ValueError(f'The velocity grid supplied does not extend to beam input energy {E_NBI_kev} keV')
        
        n_tot = self.integrate_speed_distribution(v=v, F=F)
        n_bulk_guess = n_tot*0.99 # small factor to make sure the optimization is happy

        Tbulk_kev = self.Ti_interpolator_kev(rho)
        Te_bulk_kev = self.Te_interpolator_kev(rho) 

        v_thermal = np.sqrt(Tbulk_kev*1000*1.6022e-19*2/self.species_dict[self.species]['mass'])

        thermal_speeds = np.linspace(v[0], v_thermal*3, len(v)) # TODO assumes thermal distribution contribution is only 3x bulk 

        # slowing_down_v = np.linspace(v[0], v_NBI, len(v))
        # slowing_down_shape = self.slowing_down_far3d(v=slowing_down_v, n=1, Te=Te_bulk_kev, species=self.species)
        # T_NBI_guess = self.calculate_effective_temperature(varray=slowing_down_v, farray=slowing_down_shape, species=self.species)

        F_interp = interp1d(v, F, kind='linear')

        F_thermal_range = F_interp(thermal_speeds)

        # fit just the maxwellian
        initial_guess = [n_bulk_guess, Tbulk_kev]
        lower_bounds = [n_bulk_guess/2, Tbulk_kev/2]
        upper_bounds = [n_tot, Tbulk_kev*2]        

        # now, actually perform the optimization 1: fitting the thermal bulk 
        popt, pcov = curve_fit(
            self.single_maxwellain_to_fit, 
            thermal_speeds, 
            F_thermal_range, 
            p0=initial_guess, 
            bounds=(lower_bounds, upper_bounds)
        )   
        # unpack result 
        nbulk = popt[0]
        Tbulk = popt[1]

        # grab the thermal part, and subtract it from the F over the full range. 
        thermal_bulk_f_full = self.maxwell_speed_distribution(s=v, n=nbulk, T=Tbulk, species=self.species)
        F_minus_bulk = F - thermal_bulk_f_full
        F_minus_bulk_interp = interp1d(v, F_minus_bulk, kind='linear')

        # now, fit the RF tail on a log10 scale way out past vNBI: 1.2 times it 
        # rf_speeds = np.linspace(v_NBI*1.2, v[-1], len(v))
        rf_speeds = np.linspace(v_NBI*1.4, v[-1], len(v))
        rf_function_to_fit = F_minus_bulk_interp(rf_speeds)
        rf_function_to_fit[rf_function_to_fit < 0] = 0 # enforve non-negative 
        initial_guess = [nbulk/10, Tbulk]
        lower_bounds = [0.0, Tbulk]
        upper_bounds = [nbulk/2, Tbulk*200]

        #return rf_speeds, rf_function_to_fit

        popt, pcov = curve_fit(
            self.log10_wrapper_single_maxwellain_to_fit, 
            rf_speeds, 
            np.log10(rf_function_to_fit+1e-50), 
            p0=initial_guess, 
            bounds=(lower_bounds, upper_bounds)
        )  

        # unpack results 
        nRF = popt[0]
        TRF = popt[1]

        # finally, find the remainder and take moments of it to get the NBI contribution 
        v_slowing_down = np.linspace(0.01, v_NBI, len(v))
        F_remainder = F_minus_bulk_interp(v_slowing_down) - self.maxwell_speed_distribution(s=v_slowing_down, n=nRF, T=TRF, species=self.species)
        F_remainder[F_remainder < 0] = 0.0
        print(min(F_remainder))
        nNBI = self.integrate_speed_distribution(v=v_slowing_down, F=F_remainder)

        F_slowing = self.slowing_down_speed_far3d(s=v_slowing_down, n=nNBI, Te=Te_bulk_kev, species=self.species)
        TNBI =   self.calculate_effective_temperature(v_slowing_down, F_slowing, species=self.species, F_type='speed')

        return [nbulk, Tbulk, nNBI, TNBI, nRF, TRF]


    def fit_3_maxwellians_to_speed_distribution_only_moments(self, v, F, E_NBI_kev, rho):
        E_NBI = E_NBI_kev * 1000 * 1.6022e-19
        v_NBI = np.sqrt(2*E_NBI / self.species_dict[self.species]['mass'])

        if v[-1] < v_NBI:
            raise ValueError(f'The velocity grid supplied does not extend to beam input energy {E_NBI_kev} keV')
        
        n_tot = self.integrate_speed_distribution(v=v, F=F)
        n_bulk_guess = n_tot*0.99 # small factor to make sure the optimization is happy

        Tbulk_kev = self.Ti_interpolator_kev(rho)
        Te_bulk_kev = self.Te_interpolator_kev(rho) 

        v_thermal = np.sqrt(Tbulk_kev*1000*1.6022e-19*2/self.species_dict[self.species]['mass'])

        thermal_speeds = np.linspace(20000, v_NBI/np.sqrt(3), len(v))

        F_interp = interp1d(v, F, kind='linear')

        F_thermal_range = F_interp(thermal_speeds)

        nbulk = self.integrate_speed_distribution(v=thermal_speeds, F=F_thermal_range)
        Tbulk = self.calculate_effective_temperature(varray=thermal_speeds, farray=F_thermal_range, species=self.species, F_type='speed')

        # subtract of the bulk 
        F_rf_nbi = F - self.maxwell_speed_distribution(s=v, n=nbulk, T=Tbulk, species=self.species)
        F_rf_nbi_interp = interp1d(v, F_rf_nbi, kind='linear')

        NBI_speeds = np.linspace(20000, v_NBI*1.2, len(v))
        RF_speeds = np.linspace(v_NBI*1.2, v[-1], len(v))

        F_rf_nbi_nbi_speeds = F_rf_nbi_interp(NBI_speeds)
        F_rf_nbi_rf_speeds = F_rf_nbi_interp(RF_speeds)

        # nbi 
        nNBI = self.integrate_speed_distribution(v=NBI_speeds, F=F_rf_nbi_nbi_speeds)
        TNBI = self.calculate_effective_temperature(varray=NBI_speeds, farray=F_rf_nbi_nbi_speeds, species=self.species, F_type='speed')

        # RF 
        nRF = self.integrate_speed_distribution(v=RF_speeds, F=F_rf_nbi_rf_speeds)
        TRF = self.calculate_effective_temperature(varray=RF_speeds, farray=F_rf_nbi_rf_speeds, species=self.species, F_type='speed')
        return [nbulk, Tbulk, nNBI, TNBI, nRF, TRF]
        #return [nbulk, Tbulk, nNBI, TNBI, nRF, TRF], thermal_speeds, F_thermal_range

    def fit_3_maxwellians_to_speed_distribution_critical_speeds(self, v, F, E_NBI_kev, rho, num_iter_max=1000):
        E_NBI = E_NBI_kev * 1000 * 1.6022e-19
        v_NBI = np.sqrt(2*E_NBI / self.species_dict[self.species]['mass'])

        if v[-1] < v_NBI:
            raise ValueError(f'The velocity grid supplied does not extend to beam input energy {E_NBI_kev} keV')
        
        n_tot = self.integrate_speed_distribution(v=v, F=F)
        n_bulk_guess = n_tot*0.99 # small factor to make sure the optimization is happy

        Tbulk_kev = self.Ti_interpolator_kev(rho)
        Te_bulk_kev = self.Te_interpolator_kev(rho) 

        v_thermal = np.sqrt(Tbulk_kev*1000*1.6022e-19*2/self.species_dict[self.species]['mass'])

        # thermal_speeds = np.linspace(20000, v_NBI/np.sqrt(3), len(v))

        # F_interp = interp1d(v, F, kind='linear')

        # F_thermal_range = F_interp(thermal_speeds)

        # nbulk = self.integrate_speed_distribution(v=thermal_speeds, F=F_thermal_range)
        # Tbulk = self.calculate_effective_temperature(varray=thermal_speeds, farray=F_thermal_range, species=self.species, F_type='speed')
        thermal_speeds = np.linspace(v[0], v_thermal*3, len(v))
        F_interp = interp1d(v, F, kind='linear')

        F_thermal_range = F_interp(thermal_speeds)

        # fit just the maxwellian
        initial_guess = [n_bulk_guess, Tbulk_kev]
        lower_bounds = [n_bulk_guess/2, Tbulk_kev/2]
        upper_bounds = [n_tot, Tbulk_kev*2]        

        # now, actually perform the optimization 1: fitting the thermal bulk 
        popt, pcov = curve_fit(
            self.single_maxwellain_to_fit, 
            thermal_speeds, 
            F_thermal_range, 
            p0=initial_guess, 
            bounds=(lower_bounds, upper_bounds)
        )   
        # unpack result 
        nbulk = popt[0]
        Tbulk = popt[1]


        print(f'Ti at rho: {Tbulk_kev} keV')
        # subtract of the bulk 
        F_rf_nbi = F - self.maxwell_speed_distribution(s=v, n=nbulk, T=Tbulk, species=self.species)
        F_rf_nbi[F_rf_nbi < 0] = 0.0
        F_rf_nbi_interp = interp1d(v, F_rf_nbi, kind='linear')
        self.F_rf_nbi_interp = F_rf_nbi_interp # TODO for debug

        mu0 = 4*np.pi * 1e-7
        rho_m = nbulk * self.species_dict[self.species]['mass'] 
        vA = self.eqdsk['bcentr'] / np.sqrt(mu0*rho_m)

        RF_speeds = np.linspace(v_NBI*1.2, v[-1], len(v))
        F_rf_nbi_rf_speeds = F_rf_nbi_interp(RF_speeds)
        TRF = self.calculate_effective_temperature(varray=RF_speeds, farray=F_rf_nbi_rf_speeds, species=self.species, F_type='speed')

        nRF_guess = nbulk/10
        nRF_lower_bound = 0
        nRF_upper_bound = nbulk
        tol=1e-6
        error=1
        #num_iter_max = 1000
        i = 0
        while error > tol:
            Fguess = self.maxwell_speed_distribution(s=vA, n=nRF_guess, T=TRF, species=self.species)

            if Fguess > F_rf_nbi_interp(vA):
                nRF_upper_bound = nRF_guess
                nRF_guess = (nRF_lower_bound + nRF_guess)/2
            elif Fguess <=  F_rf_nbi_interp(vA):
                nRF_lower_bound = nRF_guess
                nRF_guess = (nRF_upper_bound + nRF_guess)/2       
            error = np.abs((Fguess - F_rf_nbi_interp(vA))/F_rf_nbi_interp(vA)) 
            i += 1
            if i > num_iter_max:
                raise ValueError(f'Maximum number of iterations reached for calculating nRF. error final: {error}')  

        nRF = nRF_guess 

        NBI_speeds = np.linspace(20000, v_NBI*1.2, len(v))
        # RF_speeds = np.linspace(v_NBI*1.2, v[-1], len(v))

        F_rf_nbi_nbi_speeds = F_rf_nbi_interp(NBI_speeds)
        
        # F_rf_nbi_rf_speeds = F_rf_nbi_interp(RF_speeds)

        # # nbi 
        # nNBI = self.integrate_speed_distribution(v=NBI_speeds, F=F_rf_nbi_nbi_speeds)
        TNBI = self.calculate_effective_temperature(varray=NBI_speeds, farray=F_rf_nbi_nbi_speeds, species=self.species, F_type='speed')

        nNBI_guess = nbulk/10
        nNBI_lower_bound = 0
        nNBI_upper_bound = nbulk
        tol=1e-6
        error=1
        #num_iter_max = 1000
        i = 0
        #print(f'F at vA/2: {F_rf_nbi_interp(vA/2):.4e}')
        #print(f'TNBI: {TNBI} kev')
        while error > tol:
            #print(nNBI_guess)
            Fguess = self.maxwell_speed_distribution(s=vA/2, n=nNBI_guess, T=TNBI, species=self.species)
            #print(f'{Fguess:.4e}')
            if Fguess > F_rf_nbi_interp(vA/2):
                nNBI_upper_bound = nNBI_guess
                nNBI_guess = (nNBI_lower_bound + nNBI_guess)/2
            elif Fguess <=  F_rf_nbi_interp(vA/2):
                nNBI_lower_bound = nNBI_guess
                nNBI_guess = (nNBI_upper_bound + nNBI_guess)/2       
            error = np.abs((Fguess - F_rf_nbi_interp(vA/2))/F_rf_nbi_interp(vA/2)) 
            i += 1
            if i > num_iter_max:
                raise ValueError(f'Maximum number of iterations, {num_iter_max}, reached for calculating NBI. error final: {error}. nNBI_guess: {nNBI_guess:.4e}')  

        nNBI = nNBI_guess 

        return [nbulk, Tbulk, nNBI, TNBI, nRF, TRF]
    
    def fit_3_maxwellians_to_speed_distribution_bulk_fit_NBI_RF_moments(self, v, F, E_NBI_kev, rho, 
                                                                        mode='cql3d', 
                                                                        mcgo_energy_filter_kev=30,
                                                                        return_speeds=False):
        """_summary_

        Parameters
        ----------
        v : _type_
            _description_
        F : _type_
            _description_
        E_NBI_kev : _type_
            _description_
        rho : _type_
            _description_
        mode : str, optional
            Can be set to 'cql3d' or 'mcgo', by default 'cql3d'. changes how the distribution function is processed since mcgo thermal bulk needs to be removed. 
        mcgo_energy_filter_kev : float
            Only used in mode = 'mcgo'. Should match the mcgo/p2f energy filter used. For truncating the NBI distribution to not count ash.  

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """

        F_interp = interp1d(v, F, kind='linear')


        E_NBI = E_NBI_kev * 1000 * 1.6022e-19
        v_NBI = np.sqrt(2*E_NBI / self.species_dict[self.species]['mass'])

        if v[-1] < v_NBI:
            raise ValueError(f'The velocity grid supplied does not extend to beam input energy {E_NBI_kev} keV')
        
        n_tot = self.integrate_speed_distribution(v=v, F=F)
        n_bulk_guess = n_tot*0.99 # small factor to make sure the optimization is happy
        nbulk_from_profile = self.ni_interpolator_m3(rho)

        Tbulk_kev = self.Ti_interpolator_kev(rho)
        Te_bulk_kev = self.Te_interpolator_kev(rho) 


        if mode == 'cql3d': 
            v_thermal = np.sqrt(Tbulk_kev*1000*1.6022e-19*2/self.species_dict[self.species]['mass'])

            thermal_speeds = np.linspace(20000, v_NBI/np.sqrt(3), len(v))


            F_thermal_range = F_interp(thermal_speeds)

            # fit just the maxwellian
            initial_guess = [n_bulk_guess, Tbulk_kev]
            lower_bounds = [n_bulk_guess/2, Tbulk_kev/2]
            upper_bounds = [n_tot, Tbulk_kev*2]        

            # now, actually perform the optimization 1: fitting the thermal bulk 
            popt, pcov = curve_fit(
                self.single_maxwellain_to_fit, 
                thermal_speeds, 
                F_thermal_range, 
                p0=initial_guess, 
                bounds=(lower_bounds, upper_bounds)
            )   
            # unpack result 
            nbulk = popt[0]
            Tbulk = popt[1]

        elif mode == 'mcgo':
            # for now, just assume the maxwellian due to the bulk is uneffected by NBI/RF. 
            nbulk = nbulk_from_profile.item()
            Tbulk = Tbulk_kev.item()
        else:
            raise ValueError(f"Mode {mode} not understood. Allowed modes are 'cql3d' and 'mcgo'.")

        # nbulk = self.integrate_speed_distribution(v=thermal_speeds, F=F_thermal_range)
        # Tbulk = self.calculate_effective_temperature(varray=thermal_speeds, farray=F_thermal_range, species=self.species, F_type='speed')

        # subtract off the bulk 
        F_rf_nbi = F - self.maxwell_speed_distribution(s=v, n=nbulk, T=Tbulk, species=self.species)
        # make sure negative values are ignored. 
        F_rf_nbi[F_rf_nbi < 0] = 0.0
        F_rf_nbi_interp = interp1d(v, F_rf_nbi, kind='linear')

        # calculate the speed at 5 eV to truncate noise near v = 0 inb the cql3d distribution. 
        E_5eV = 5*1.6022e-19
        mass = self.species_dict[self.species]['mass']
        v_5eV = np.sqrt(E_5eV * 2 / mass)

        E_cuttof_mcgo = mcgo_energy_filter_kev * 1000 *1.6022e-19
        v_mcgo_cuttoff = np.sqrt(E_cuttof_mcgo * 2 / mass)

        # NBI_speeds = np.linspace(v_5eV, v_NBI*1.2, len(v))
        # RF_speeds = np.linspace(v_NBI*1.2, v[-1], len(v))

        if mode == 'cql3d':
            NBI_speeds = np.linspace(v_5eV, v_NBI, len(v))
        elif mode == 'mcgo':
            NBI_speeds = np.linspace(v_mcgo_cuttoff, v_NBI, len(v))
                                     
        RF_speeds = np.linspace(v_NBI, v[-1], len(v))
        F_rf_nbi_nbi_speeds = F_rf_nbi_interp(NBI_speeds)
        F_rf_nbi_rf_speeds = F_rf_nbi_interp(RF_speeds)

        # nbi 
        nNBI = self.integrate_speed_distribution(v=NBI_speeds, F=F_rf_nbi_nbi_speeds)
        TNBI = self.calculate_effective_temperature(varray=NBI_speeds, farray=F_rf_nbi_nbi_speeds, species=self.species, F_type='speed')
        vNBI = self.calculate_average_speed(varray=NBI_speeds, farray=F_rf_nbi_nbi_speeds, F_type='speed')
        # RF 
        nRF = self.integrate_speed_distribution(v=RF_speeds, F=F_rf_nbi_rf_speeds)
        TRF = self.calculate_effective_temperature(varray=RF_speeds, farray=F_rf_nbi_rf_speeds, species=self.species, F_type='speed')
        vRF = self.calculate_average_speed(varray=RF_speeds, farray=F_rf_nbi_rf_speeds, F_type='speed')
        if return_speeds:
            return [nbulk, Tbulk, nNBI, TNBI, nRF, TRF, vNBI, vRF]
        else:
            return [nbulk, Tbulk, nNBI, TNBI, nRF, TRF]
    
    
    def system_of_equations(self, vars):
        n2, T2, n3, T3 = vars # unpack the variables. n2, n3 are per m^3 / 1e18

        # rescale to per m^3
        n2 = n2 * 1e18
        n3 = n3 * 1e18
        # grab these from saved state  
        n1 = self.N1
        T1 = self.T1
        F = self.F_of_v_interpolator
        dFdv = self.dF_dv_interpolator
        v_a = self.v_a
        v_b = self.v_b

        # list out the three equations
        eq1 = self.maxwell_speed_distribution(s=v_a, n=n1, T=T1, species=self.species) + \
              self.maxwell_speed_distribution(s=v_a, n=n2, T=T2, species=self.species) + \
              self.maxwell_speed_distribution(s=v_a, n=n3, T=T3, species=self.species) - \
              F(v_a)
        
        eq2 = self.maxwell_speed_distribution(s=v_b, n=n1, T=T1, species=self.species) + \
              self.maxwell_speed_distribution(s=v_b, n=n2, T=T2, species=self.species) + \
              self.maxwell_speed_distribution(s=v_b, n=n3, T=T3, species=self.species) - \
              F(v_b)
        
        eq3 = self.dmaxwell_speed_distribution(s=v_a, n=n1, T=T1, species=self.species) + \
              self.dmaxwell_speed_distribution(s=v_a, n=n2, T=T2, species=self.species) + \
              self.dmaxwell_speed_distribution(s=v_a, n=n3, T=T3, species=self.species) - \
              dFdv(v_a)
        
        eq4 = self.dmaxwell_speed_distribution(s=v_b, n=n1, T=T1, species=self.species) + \
              self.dmaxwell_speed_distribution(s=v_b, n=n2, T=T2, species=self.species) + \
              self.dmaxwell_speed_distribution(s=v_b, n=n3, T=T3, species=self.species) - \
              dFdv(v_b)
        
        return [eq1, eq2, eq3, eq4]
    

    def fit_3_maxwellians_to_speed_distribution_critical_speed_and_slope(self, v, F, E_NBI_kev, rho, mode='cql3d', mcgo_energy_filter_kev=30):
        """_summary_

        Parameters
        ----------
        v : _type_
            _description_
        F : _type_
            _description_
        E_NBI_kev : _type_
            _description_
        rho : _type_
            _description_
        mode : str, optional
            Can be set to 'cql3d' or 'mcgo', by default 'cql3d'. changes how the distribution function is processed since mcgo thermal bulk needs to be removed. 
        mcgo_energy_filter_kev : float
            Only used in mode = 'mcgo'. Should match the mcgo/p2f energy filter used. For truncating the NBI distribution to not count ash.  

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """

        F_interp = interp1d(v, F, kind='linear')
        self.F_of_v_interpolator = F_interp

        dF_dv = np.gradient(F, v)
        dFdv_interpolator = interp1d(v, dF_dv, kind='linear')
        self.dF_dv_interpolator = dFdv_interpolator


        E_NBI = E_NBI_kev * 1000 * 1.6022e-19
        v_NBI = np.sqrt(2*E_NBI / self.species_dict[self.species]['mass'])

        if v[-1] < v_NBI:
            raise ValueError(f'The velocity grid supplied does not extend to beam input energy {E_NBI_kev} keV')
        
        n_tot = self.integrate_speed_distribution(v=v, F=F)
        n_bulk_guess = n_tot*0.99 # small factor to make sure the optimization is happy
        nbulk_from_profile = self.ni_interpolator_m3(rho)

        Tbulk_kev = self.Ti_interpolator_kev(rho)
        Te_bulk_kev = self.Te_interpolator_kev(rho) 


        if mode == 'cql3d': 
            v_thermal = np.sqrt(Tbulk_kev*1000*1.6022e-19*2/self.species_dict[self.species]['mass'])

            thermal_speeds = np.linspace(20000, v_NBI/np.sqrt(3), len(v))


            F_thermal_range = F_interp(thermal_speeds)

            # fit just the maxwellian
            initial_guess = [n_bulk_guess, Tbulk_kev]
            lower_bounds = [n_bulk_guess/2, Tbulk_kev/2]
            upper_bounds = [n_tot, Tbulk_kev*2]        

            # now, actually perform the optimization 1: fitting the thermal bulk 
            popt, pcov = curve_fit(
                self.single_maxwellain_to_fit, 
                thermal_speeds, 
                F_thermal_range, 
                p0=initial_guess, 
                bounds=(lower_bounds, upper_bounds)
            )   
            # unpack result 
            nbulk = popt[0]
            Tbulk = popt[1]

        elif mode == 'mcgo':
            # for now, just assume the maxwellian due to the bulk is uneffected by NBI/RF. 
            nbulk = nbulk_from_profile.item()
            Tbulk = Tbulk_kev.item()
        else:
            raise ValueError(f"Mode {mode} not understood. Allowed modes are 'cql3d' and 'mcgo'.")
        
        # make the F interpolator 
            # done. 
        # make the dF/dv interpolator 
            # done.
        # load up bulk n, T
        self.N1 = nbulk
        self.T1 = Tbulk
        mu0 = 4*np.pi * 1e-7
        rho_m = nbulk * self.species_dict[self.species]['mass'] 
        vA = self.eqdsk['bcentr'] / np.sqrt(mu0*rho_m)
        self.v_a = vA/2
        self.v_b = vA
        # make initial guess for n2, T2, n3, T3
        initial_guess = [nbulk/10/1e18, Tbulk*5, nbulk/100/1e18, Tbulk*20]
        lower_bounds = [0, Tbulk*2, 0, Tbulk*2]
        upper_bounds = [nbulk, np.inf, nbulk, np.inf]

        # solve system of equations
        #result = root(self.system_of_equations, initial_guess)
        result = least_squares(
            self.system_of_equations, 
            initial_guess, 
            bounds=(lower_bounds, upper_bounds)
        )
        
        # unload result
        nNBI = result.x[0]*1e18
        TNBI = result.x[1]
        nRF = result.x[2]*1e18
        TRF = result.x[3]
        # return 

        # nbulk = self.integrate_speed_distribution(v=thermal_speeds, F=F_thermal_range)
        # Tbulk = self.calculate_effective_temperature(varray=thermal_speeds, farray=F_thermal_range, species=self.species, F_type='speed')

        # # subtract off the bulk 
        # F_rf_nbi = F - self.maxwell_speed_distribution(s=v, n=nbulk, T=Tbulk, species=self.species)
        # # make sure negative values are ignored. 
        # F_rf_nbi[F_rf_nbi < 0] = 0.0
        # F_rf_nbi_interp = interp1d(v, F_rf_nbi, kind='linear')

        # # calculate the speed at 5 eV to truncate noise near v = 0 inb the cql3d distribution. 
        # E_5eV = 5*1.6022e-19
        # mass = self.species_dict[self.species]['mass']
        # v_5eV = np.sqrt(E_5eV * 2 / mass)

        # E_cuttof_mcgo = mcgo_energy_filter_kev * 1000 *1.6022e-19
        # v_mcgo_cuttoff = np.sqrt(E_cuttof_mcgo * 2 / mass)

        # # NBI_speeds = np.linspace(v_5eV, v_NBI*1.2, len(v))
        # # RF_speeds = np.linspace(v_NBI*1.2, v[-1], len(v))

        # if mode == 'cql3d':
        #     NBI_speeds = np.linspace(v_5eV, v_NBI, len(v))
        # elif mode == 'mcgo':
        #     NBI_speeds = np.linspace(v_mcgo_cuttoff, v_NBI, len(v))
                                     
        # RF_speeds = np.linspace(v_NBI, v[-1], len(v))
        # F_rf_nbi_nbi_speeds = F_rf_nbi_interp(NBI_speeds)
        # F_rf_nbi_rf_speeds = F_rf_nbi_interp(RF_speeds)

        # # nbi 
        # nNBI = self.integrate_speed_distribution(v=NBI_speeds, F=F_rf_nbi_nbi_speeds)
        # TNBI = self.calculate_effective_temperature(varray=NBI_speeds, farray=F_rf_nbi_nbi_speeds, species=self.species, F_type='speed')

        # # RF 
        # nRF = self.integrate_speed_distribution(v=RF_speeds, F=F_rf_nbi_rf_speeds)
        # TRF = self.calculate_effective_temperature(varray=RF_speeds, farray=F_rf_nbi_rf_speeds, species=self.species, F_type='speed')
        return [nbulk, Tbulk, nNBI, TNBI, nRF, TRF]
        
        




    def plot_maxwellian_fit(self, v, F, dens, temps_kev, xlim, ylim, nm, Tm, figsize=(24,10)):

        # calculate the center alfven velocity energy
        mu0 = 4*np.pi * 1e-7
        rho_m = dens[0] * self.species_dict[self.species]['mass'] 
        vA = self.eqdsk['bcentr'] / np.sqrt(mu0*rho_m)
        print(f'Alfven velocity: {vA/1000:.2f} km/s')
        E_A = 0.5 * self.species_dict[self.species]['mass'] * vA**2
        E_A_over_two = 0.5 * self.species_dict[self.species]['mass'] * (vA/2)**2
        print(f'E_A: {E_A/1000 / 1.6022e-19} keV') 
        bulk_maxwellian = self.maxwell_speed_distribution(s=v, n=dens[0], T=temps_kev[0], species=self.species)
        nbi_maxwellian = self.maxwell_speed_distribution(s=v, n=dens[1], T=temps_kev[1], species=self.species)
        RF_maxwellian = self.maxwell_speed_distribution(s=v, n=dens[2], T=temps_kev[2], species=self.species)

        self.maxwell_fit_temps = temps_kev
        sum_maxwellians = self.sum_of_maxwellians(v, dens[0], dens[1], dens[2])

        fig, axs = plt.subplots(3, 1, figsize=figsize)
        egrid_kev = 0.5*self.species_dict[self.species]['mass']*v**2 / (1000 * 1.6022e-19)
        axs[0].plot(egrid_kev, bulk_maxwellian, color='red', label='Bulk')
        axs[0].plot(egrid_kev, nbi_maxwellian, color='purple', label='NBI')
        axs[0].plot(egrid_kev, RF_maxwellian, color='darkred', label='RF tail')
        axs[0].plot(egrid_kev, F, label='True F(v)', color='blue')
        axs[0].plot(egrid_kev, sum_maxwellians, color='black', linestyle='-.', label='sum')
        axs[0].grid()
        axs[0].set_ylim(ylim[0], ylim[1])
        axs[0].set_xlim(xlim[0], xlim[1])
        axs[0].axvline(x=(E_A/1000 / 1.6022e-19), color='darkgreen', linestyle='--', label=r'$v_{A}$')
        axs[0].axvline(x=(E_A_over_two/1000 / 1.6022e-19), color='green', linestyle='--', label=r'$v_{A}/2$')
        axs[0].legend()
        axs[0].set_xlabel('E [keV]')
        axs[0].set_ylabel(r'F(v) [m$^{-3}$/(m/s)]')

        axs[1].plot(egrid_kev, np.log10(bulk_maxwellian+1e-50), color='red', label='Bulk')
        axs[1].plot(egrid_kev, np.log10(nbi_maxwellian+1e-50), color='purple', label='NBI')
        axs[1].plot(egrid_kev, np.log10(RF_maxwellian+1e-50), color='darkred', label='RF tail')
        axs[1].plot(egrid_kev, np.log10(F+1e-50), label='True F(v)', color='blue')
        axs[1].plot(egrid_kev, np.log10(sum_maxwellians+1e-50), color='black', linestyle='-.', label='sum')
        axs[1].axvline(x=(E_A/1000 / 1.6022e-19), color='darkgreen', linestyle='--', label=r'$v_{A}$')
        axs[1].axvline(x=(E_A_over_two/1000 / 1.6022e-19), color='green', linestyle='--', label=r'$v_{A}/2$')
        axs[1].set_xlabel('E [keV]')
        axs[1].set_ylabel(r'log10 F(v) [m$^{-3}$/(m/s)]')
        
        axs[1].grid()
        axs[1].legend()
        #axs[1].set_xlim(xlim[0], xlim[1])
        axs[1].set_ylim(-20, None)
        axs[2].plot(egrid_kev, F - bulk_maxwellian, color='red', label='F - Thermal Bulk')
        axs[2].axvline(x=(E_A/1000 / 1.6022e-19), color='darkgreen', linestyle='--', label=r'$v_{A}$')
        axs[2].axvline(x=(E_A_over_two/1000 / 1.6022e-19), color='green', linestyle='--', label=r'$v_{A}/2$')
        axs[2].plot(egrid_kev, self.maxwell_speed_distribution(s=v, n=nm, T=Tm, species=self.species), label='manual maxwell')
        axs[2].grid()
        axs[2].legend()
        axs[2].set_ylim(ylim[0], ylim[1])
        axs[2].set_xlim(xlim[0], xlim[1])
        axs[2].set_xlabel('E [keV]')
        axs[2].set_ylabel(r'F(v) [m$^{-3}$/(m/s)]')
        #axs[1].set_xlim(xlim[0], xlim[1])

    def load_entire_h5_to_dict(self, filename):
        """
        Reads an entire HDF5 parameter scan file and returns it as a nested dictionary,
        safely handling any nested Groups or Datasets.
        """
        
        # We define a helper function to recursively unpack the file
        def _recurse(h5_obj):
            temp_dict = {}
            for key in h5_obj.keys():
                item = h5_obj[key]
                
                # If it's a dataset, unpack the data
                if isinstance(item, h5py.Dataset):
                    if item.shape == ():
                        temp_dict[key] = item[()]  # Unpack scalar
                    else:
                        temp_dict[key] = item[:].tolist()  # Unpack array
                        
                # If it's a group, dive into it recursively
                elif isinstance(item, h5py.Group):
                    temp_dict[key] = _recurse(item)
                    
            return temp_dict

        # Open the file and start the recursion from the root
        with h5py.File(filename, 'r') as f:
            master_dict = _recurse(f)
            
        return master_dict
    
    def smooth_mc_profile(self, rho_grid, choppy_array, s_multiplier=0.05):
        """
        Smooths choppy Monte Carlo radial profiles using a Univariate Spline.
        Automatically handles NaNs and Infs resulting from zero-density regions.
        """
        # 1. Splines require strictly increasing x-arrays. Sort just in case.
        sort_idx = np.argsort(rho_grid)
        rho_sorted = rho_grid[sort_idx]
        array_sorted = choppy_array[sort_idx]
        
        # 2. Mask out NaNs and Infs for the fit
        valid_mask = np.isfinite(array_sorted)
        rho_fit = rho_sorted[valid_mask]
        array_fit = array_sorted[valid_mask]
        
        # Fallback if the array is entirely NaNs
        if len(rho_fit) < 4:  # k=3 spline requires at least 4 valid points
            return np.zeros_like(rho_grid)
        
        # 3. Calculate a dynamic smoothing factor 's' on VALID data only. 
        variance = np.var(array_fit)
        s_val = len(rho_fit) * variance * s_multiplier
        
        # Ensure s_val is strictly >= 0 (in case of a perfectly flat profile where var=0)
        s_val = max(s_val, 0.0)
        
        # 4. Fit a cubic spline (k=3) on the valid data
        spline = UnivariateSpline(rho_fit, array_fit, k=3, s=s_val)
        
        # 5. Evaluate the smoothed spline on the ENTIRE sorted grid 
        # This naturally interpolates across the gaps where NaNs used to be
        smoothed_array = spline(rho_sorted)
        
        # 6. Physics check: clip negatives
        smoothed_array = np.clip(smoothed_array, a_min=1e-10, a_max=None)
        
        # 7. Revert to original array order
        unsort_idx = np.argsort(sort_idx)
        
        return smoothed_array[unsort_idx]

    def get_energetic_profiles(self, 
                               rho_array_F_of_v_indexable_by_rho=None, 
                               v_array_F_of_v_indexable_by_rho=None, 
                               F_of_v_indexable_by_rho=None, 
                               E_NBI_kev=80, 
                               num_iter_max=10000, 
                               return_arrays=False, 
                               mode='cql3d', 
                               mcgo_energy_filter_kev=30, 
                               mcgo_p2f_aorsa_h5_file=None,
                               h5_key=None,
                               s_density=0.05,
                               s_temp=0.05):


        # loop over rho. fit the profiles. 
        if mode == 'cql3d' or mode == 'mcgo':
            rho_array = rho_array_F_of_v_indexable_by_rho
            num_rhos = len(rho_array)

            # initialize the profiles arrays 
            Tbulks = np.zeros((num_rhos))
            nbulks = np.zeros((num_rhos))
            TNBIs = np.zeros((num_rhos))
            nNBIs = np.zeros((num_rhos))
            TRFs = np.zeros((num_rhos))
            nRFs = np.zeros((num_rhos))
            print('Starting profile fitting routine.')
            for ir in range(num_rhos):
                rhoi = rho_array[ir]
                #fit = self.fit_3_maxwellians_to_speed_distribution_critical_speeds(v=v_array, F=F_of_v_indexable_by_rho[ir, :], E_NBI_kev=E_NBI_kev, rho=rhoi, num_iter_max=num_iter_max)
                fit = self.fit_3_maxwellians_to_speed_distribution_bulk_fit_NBI_RF_moments(v=v_array_F_of_v_indexable_by_rho, F=F_of_v_indexable_by_rho[ir, :], E_NBI_kev=E_NBI_kev, rho=rhoi, mode=mode, mcgo_energy_filter_kev=mcgo_energy_filter_kev)
                # unpack 
                nbulks[ir] = fit[0]
                Tbulks[ir] = fit[1]

                nNBIs[ir] = fit[2]
                TNBIs[ir] = fit[3]

                nRFs[ir] = fit[4]
                TRFs[ir] = fit[5]

        elif mode == 'mcgo_p2f_aorsa':
            # instead, read directly in from a .h5 file 
            self.mcgo_p2f_aorsa_data_dict = self.load_entire_h5_to_dict(filename=mcgo_p2f_aorsa_h5_file)

            # load up rho grid 

            rho_array = self.mcgo_p2f_aorsa_data_dict[h5_key]['rho_mcgo_lfs']

            # assign nulk density and temperature profiles 
            nbulks = self.ni_interpolator_m3(rho_array)
            Tbulks = self.Ti_interpolator_kev(rho_array)

            # load up the other profiles directly to raw lists, but need to apply smothing to remove noise 
            nNBIs_raw = self.mcgo_p2f_aorsa_data_dict[h5_key]['NBI density [m^-3]']
            TNBIs_raw = self.mcgo_p2f_aorsa_data_dict[h5_key]['NBI Temp [keV]']

            nRFs_raw = self.mcgo_p2f_aorsa_data_dict[h5_key]['RF density [m^-3]']
            TRFs_raw = self.mcgo_p2f_aorsa_data_dict[h5_key]['RF Temp [keV]']

            # apply smoothing so we dont end up with crazy gradients 
            nNBIs = self.smooth_mc_profile(np.array(rho_array), np.array(nNBIs_raw), s_multiplier=s_density)
            nRFs = self.smooth_mc_profile(np.array(rho_array), np.array(nRFs_raw), s_multiplier=s_density)

            TNBIs = self.smooth_mc_profile(np.array(rho_array), np.array(TNBIs_raw), s_multiplier=s_temp)
            TRFs = self.smooth_mc_profile(np.array(rho_array), np.array(TRFs_raw), s_multiplier=s_temp)

        else: 
            raise ValueError(f'Mode type {mode} not understood.')


        # build interpolators for use later when loading up the profiles 
        self.nbulks_interp = interp1d(rho_array, nbulks, kind='linear', bounds_error=False, fill_value=(nbulks[0], nbulks[-1]))
        self.Tbulks_interp = interp1d(rho_array, Tbulks, kind='linear', bounds_error=False, fill_value=(Tbulks[0], Tbulks[-1]))

        self.nNBIs_interp = interp1d(rho_array, nNBIs, kind='linear', bounds_error=False, fill_value=(nNBIs[0], nNBIs[-1]))
        self.TNBIs_interp = interp1d(rho_array, TNBIs, kind='linear', bounds_error=False, fill_value=(TNBIs[0], TNBIs[-1]))

        self.nRFs_interp = interp1d(rho_array, nRFs, kind='linear', bounds_error=False, fill_value=(nRFs[0], nRFs[-1]))
        self.TRFs_interp = interp1d(rho_array, TRFs, kind='linear', bounds_error=False, fill_value=(TRFs[0], TRFs[-1]))

        if return_arrays:
            return nbulks, Tbulks, nNBIs, TNBIs, nRFs, TRFs
        
    # def load_F_of_v_indexable_by_rho(self, rho_array, v_array, F_of_v_indexable_by_rho):
    #     self.velocity_array_for_F = v_array
    #     self.rho_for_F_array = rho_array
    #     self.F_of_v_indexable_by_rho = F_of_v_indexable_by_rho

    def convert_cql3d_distribution_into_F_of_v_indexable_by_rho(self, cql_pp, index_to_cut=0):
        """
        index_to_cut allows for truncation at the edge of the rho grid. 
        """
        end = -index_to_cut if index_to_cut > 0 else None
        rya = cql_pp.rya[:end]
        enerkev = cql_pp.enerkev
        v_cql = np.sqrt(2*enerkev*1000*1.6022e-19 / self.species_dict['d']['mass'])

        F_of_v_indexable_by_rho = np.zeros((len(rya), len(v_cql)))

        for ir in range(len(rya)):
            f_integrated_over_pitch, ekev = cql_pp.integrate_distribution_over_pitch_angle(gen_species_index=0, rho_index=ir)
            F_of_v_indexable_by_rho[ir, :] = f_integrated_over_pitch

        return rya, v_cql, F_of_v_indexable_by_rho

    def convert_mcgo_distribution_into_F_of_v_indexable_by_rho(self, mcgo_pp):
        rya = mcgo_pp.rho_grid
        f = mcgo_pp.vdstb
        v = mcgo_pp.vbnd

        # grab the magnetic center 
        rmaxis = mcgo_pp.eqdsk['rmaxis']

        major_radius_grid = np.linspace(min(mcgo_pp.eqdsk['rbbbs']), max(mcgo_pp.eqdsk['rbbbs']), len(rya)) 

        # sign_hfs_vs_lfs = np.zeros(len(major_radius_grid))
        # for i in range(len(sign_hfs_vs_lfs)):
        #     if major_radius_grid[i] < rmaxis:
        #         sign_hfs_vs_lfs[i] = -1
        #     else:
        #         sign_hfs_vs_lfs[i] = 1

        # signed_major_radius_grid = major_radius_grid * sign_hfs_vs_lfs
        # num_pos_major_radii = len(signed_major_radius_grid[signed_major_radius_grid>0])
        
        # idx_of_lfs = np.where(signed_major_radius_grid > 0)[0]
        idx_of_lfs = np.where(major_radius_grid >= rmaxis)[0]
        num_lfs_major_radii = len(idx_of_lfs)
        # set up F storage matrix 
        F_of_v_idx_rho = np.zeros((num_lfs_major_radii, f.shape[1]))
        new_rhos = np.zeros(num_lfs_major_radii)
        for i in range(num_lfs_major_radii):
            rho_index = idx_of_lfs[i]
            new_rhos[i] = rya[rho_index]
            F_of_v_idx_rho[i, :] = mcgo_pp.f_integrated_over_pitch(rho_idx=rho_index)

        return new_rhos, v, F_of_v_idx_rho





    # def build_far3d_outfile(self, 
    #                         rho_array_for_far3d, 
    #                         mode='cql3d', 
    #                         cql_or_mcgo_pp=None, 
    #                         E_NBI_kev=80, 
    #                         num_iter_max=10000, 
    #                         index_to_cut=0, 
    #                         mcgo_energy_filter_kev=30, 
    #                         mcgo_p2f_aorsa_h5_file=None, 
    #                         s_density=0.05, 
    #                         s_temp=0.05):
    #     """
    #     Builds the FAR3D external profiles file using specified input mode data.
    #     Supported modes: 'cql3d', 'mcgo', 'mcgo_p2f_aorsa'
    #     """
    #     # --- Step 1: Mode-Specific Interpolator Setup ---
    #     if mode in ['cql3d', 'mcgo']:
    #         if mode == 'cql3d':
    #             rya, vs, F_of_v_indexable_by_rho = self.convert_cql3d_distribution_into_F_of_v_indexable_by_rho(
    #                 cql_or_mcgo_pp, index_to_cut=index_to_cut
    #             )
    #         else:  # mcgo
    #             rya, vs, F_of_v_indexable_by_rho = self.convert_mcgo_distribution_into_F_of_v_indexable_by_rho(
    #                 cql_or_mcgo_pp
    #             )

    #         # Build profile interpolators via kinetic distribution fitting
    #         self.get_energetic_profiles(
    #             rho_array_F_of_v_indexable_by_rho=rya, 
    #             v_array_F_of_v_indexable_by_rho=vs, 
    #             F_of_v_indexable_by_rho=F_of_v_indexable_by_rho, 
    #             E_NBI_kev=E_NBI_kev, 
    #             num_iter_max=num_iter_max, 
    #             return_arrays=False,
    #             mode=mode,
    #             mcgo_energy_filter_kev=mcgo_energy_filter_kev
    #         )

    #     elif mode == 'mcgo_p2f_aorsa':
    #         # Build profile interpolators directly from H5 macro-data
    #         self.get_energetic_profiles(
    #             return_arrays=False,
    #             mode='mcgo_p2f_aorsa',
    #             mcgo_p2f_aorsa_h5_file=mcgo_p2f_aorsa_h5_file,
    #             s_density=s_density,
    #             s_temp=s_temp
    #         )
    #     else:
    #         raise ValueError(f"Unknown mode: {mode}. Must be 'cql3d', 'mcgo', or 'mcgo_p2f_aorsa'.")

    #     # --- Step 2: Unified Profile Processing Pipeline (No More Duplication) ---
    #     kev_to_J = 1.6022e-19 * 1000

    #     # Evaluate core profiles onto the FAR3D radial grid
    #     nbulk_profile_for_far3d = self.nbulks_interp(rho_array_for_far3d)
    #     Tbulk_profile_for_far3d = self.Tbulks_interp(rho_array_for_far3d)
    #     p_kPa_bulk = nbulk_profile_for_far3d * Tbulk_profile_for_far3d * kev_to_J / 1000

    #     nNBI_profile_for_far3d = self.nNBIs_interp(rho_array_for_far3d)
    #     TNBI_profile_for_far3d = self.TNBIs_interp(rho_array_for_far3d)
    #     p_kPa_NBI = nNBI_profile_for_far3d * TNBI_profile_for_far3d * kev_to_J / 1000
    #     self.p_kPa_NBI = p_kPa_NBI

    #     nRF_profile_for_far3d = self.nRFs_interp(rho_array_for_far3d)
    #     TRF_profile_for_far3d = self.TRFs_interp(rho_array_for_far3d)
    #     p_kPa_RF = nRF_profile_for_far3d * TRF_profile_for_far3d * kev_to_J / 1000
    #     self.p_kPa_RF = p_kPa_RF

    #     # Electrons
    #     ne_profile_for_far3d = self.ne_interpolator_m3(rho_array_for_far3d)
    #     Te_profile_for_far3d = self.Te_interpolator_kev(rho_array_for_far3d)
    #     p_kPa_e = ne_profile_for_far3d * Te_profile_for_far3d * kev_to_J / 1000

    #     # Total Pressures 
    #     thermal_pressure = p_kPa_bulk + p_kPa_e
    #     equilibrium_pressure = thermal_pressure + p_kPa_NBI + p_kPa_RF

    #     # Load up profiles for export to FAR3D
    #     self.load_profile(profile_name="rho_e", profile_array=rho_array_for_far3d)

    #     # Bulk Ions / Electrons
    #     self.load_profile(profile_name="den_ion_e", profile_array=(nbulk_profile_for_far3d / 1e20))
    #     self.load_profile(profile_name="temp_ion_e", profile_array=Tbulk_profile_for_far3d)
    #     self.load_profile(profile_name="den_elec_e", profile_array=(ne_profile_for_far3d / 1e20))
    #     self.load_profile(profile_name="temp_elec_e", profile_array=Te_profile_for_far3d)   

    #     # Fast Species (NBI / RF)
    #     self.load_profile(profile_name="den_beam_e", profile_array=(nNBI_profile_for_far3d / 1e20))
    #     self.load_profile(profile_name="temp_beam_e", profile_array=TNBI_profile_for_far3d)   
    #     self.load_profile(profile_name="den_alpha_e", profile_array=(nRF_profile_for_far3d / 1e20))
    #     self.load_profile(profile_name="temp_alpha_e", profile_array=TRF_profile_for_far3d)   

    #     # Pressures 
    #     self.load_profile(profile_name="pres_thermal_e", profile_array=thermal_pressure)
    #     self.load_profile(profile_name="pres_beam_e", profile_array=p_kPa_NBI)
    #     self.load_profile(profile_name="pres_equil_e", profile_array=equilibrium_pressure)

    #     # Impurities (Assumed baseline value)
    #     impurity_density = np.ones_like(rho_array_for_far3d) * 0.01
    #     self.load_profile(profile_name='den_imp_e', profile_array=impurity_density)

    #     # Process and interpolate the safety factor (q) profile from EQDSK
    #     q_eqdsk = self.eqdsk['qpsi']
    #     psi_n = np.linspace(0, 1, self.eqdsk['nW'])
    #     phi = cumulative_trapezoid(q_eqdsk, psi_n, initial=0)
    #     phi_n = phi / phi[-1]
    #     rho_tor = np.sqrt(phi_n)
    #     q_interp = interp1d(rho_tor, q_eqdsk)
    #     q_far3d = q_interp(rho_array_for_far3d)
    #     self.load_profile(profile_name='qprof', profile_array=q_far3d)

    #     # Suppress rotation profiles for FAR3D baseline
    #     self.load_profile(profile_name='tor_rot_vel_e', profile_array=rho_array_for_far3d * 0.0)
    #     self.load_profile(profile_name='pol_rot_vel_e', profile_array=rho_array_for_far3d * 0.0)

    #     self.rho_array_for_far3d = rho_array_for_far3d

    def build_far3d_outfile(self, 
                            rho_array_for_far3d=None, # Made optional for fixed-file mode
                            mode='cql3d', 
                            cql_or_mcgo_pp=None, 
                            E_NBI_kev=80, 
                            num_iter_max=10000, 
                            index_to_cut=0, 
                            mcgo_energy_filter_kev=30, 
                            mcgo_p2f_aorsa_h5_file=None, 
                            h5_key=None,
                            s_density=0.05, 
                            s_temp=0.05,
                            fixed_profile_filepath=None,
                            qshift=0):
        """
        Builds or loads the FAR3D external profiles file using specified input mode data.
        Supported modes: 'cql3d', 'mcgo', 'mcgo_p2f_aorsa', 'fixed_file'
        """
        kev_to_J = 1.6022e-19 * 1000
        
        # --- NEW MODE: Handle Fixed Baseline Files (e.g., SPARC Case) ---
        if mode == 'fixed_file':
            if fixed_profile_filepath is None:
                raise ValueError("fixed_profile_filepath must be provided when mode is 'fixed_file'.")
            
            print(f"Reading fixed profiles from: {fixed_profile_filepath}")
            with open(fixed_profile_filepath, 'r') as f:
                lines = f.readlines()

            # Find the tabular data start
            header_idx = None
            for idx, line in enumerate(lines):
                if 'rho_e' in line:
                    header_idx = idx
                    break
            if header_idx is None:
                raise ValueError("Could not find the profile data header line ('rho_e') in the file.")

            # Parse Plasma Geometry Metadata (if needed for beta or tracking)
            self.geometry_metadata = {}
            i = 0
            while i < header_idx:
                line = lines[i].strip()
                if not line or line == "PLASMA GEOMETRY" or "TRYING TO GET TO" in line:
                    i += 1
                    continue
                if i + 1 < header_idx:
                    self.geometry_metadata[line] = lines[i+1].strip()
                    i += 2
                else:
                    i += 1

            # Extract column headers
            header_line = lines[header_idx].strip()
            columns = [col.strip() for col in header_line.split(',') if col.strip()]

            # Extract data rows
            data_rows = []
            for line in lines[header_idx + 1:]:
                cleaned_line = line.strip()
                if cleaned_line:
                    data_rows.append([float(val) for val in cleaned_line.split()])
            data_matrix = np.array(data_rows)

            # Map columns to a quick lookup index dictionary
            col_map = {name: idx for idx, name in enumerate(columns)}

            # Populate class attributes so downstream beta functions don't break
            if 'rho_e' in col_map:
                self.rho_array_for_far3d = data_matrix[:, col_map['rho_e']]
            if 'pres_beam_e' in col_map:
                self.p_kPa_NBI = data_matrix[:, col_map['pres_beam_e']]
            
            # Back-calculate RF pressure if its component pieces exist
            # if all(k in col_map for k in ['pres_equil_e', 'pres_thermal_e', 'pres_beam_e']):
            #     p_equil = data_matrix[:, col_map['pres_equil_e']]
            #     p_therm = data_matrix[:, col_map['pres_thermal_e']]
            #     p_nbi = data_matrix[:, col_map['pres_beam_e']]
            n_RF = data_matrix[:, col_map['den_alpha_e']]*1e20
            T_RF = data_matrix[:, col_map['temp_alpha_e']]
            self.p_kPa_RF = n_RF * T_RF * kev_to_J / 1000

            # Load every parsed profile column into FAR3D profile engine
            for col_name, col_idx in col_map.items():
                self.load_profile(profile_name=col_name, profile_array=data_matrix[:, col_idx])

            # build bulk density interpolator 
            ion_density = data_matrix[:, col_map['den_ion_e']] * 1e20
            #print('ion density', ion_density)
            self.nbulks_interp = interp1d(self.rho_array_for_far3d, ion_density, kind='linear', 
                                bounds_error=False, 
                                fill_value=(ion_density[0], ion_density[-1]))
            
            print("Profiles successfully loaded into FAR3D memory from fixed file.")
            return # Early exit! No need to run the analytical processing pipeline below.


        # --- Step 1: Mode-Specific Interpolator Setup (Kinetic / H5) ---
        if mode in ['cql3d', 'mcgo']:
            if mode == 'cql3d':
                rya, vs, F_of_v_indexable_by_rho = self.convert_cql3d_distribution_into_F_of_v_indexable_by_rho(
                    cql_or_mcgo_pp, index_to_cut=index_to_cut
                )
            else:  # mcgo
                rya, vs, F_of_v_indexable_by_rho = self.convert_mcgo_distribution_into_F_of_v_indexable_by_rho(
                    cql_or_mcgo_pp
                )

            self.get_energetic_profiles(
                rho_array_F_of_v_indexable_by_rho=rya, 
                v_array_F_of_v_indexable_by_rho=vs, 
                F_of_v_indexable_by_rho=F_of_v_indexable_by_rho, 
                E_NBI_kev=E_NBI_kev, 
                num_iter_max=num_iter_max, 
                return_arrays=False,
                mode=mode,
                mcgo_energy_filter_kev=mcgo_energy_filter_kev
            )

        elif mode == 'mcgo_p2f_aorsa':
            self.get_energetic_profiles(
                return_arrays=False,
                mode='mcgo_p2f_aorsa',
                mcgo_p2f_aorsa_h5_file=mcgo_p2f_aorsa_h5_file,
                h5_key=h5_key,
                s_density=s_density,
                s_temp=s_temp
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'cql3d', 'mcgo', 'mcgo_p2f_aorsa', or 'fixed_file'.")

        # --- Step 2: Unified Profile Processing Pipeline ---
        if rho_array_for_far3d is None:
            raise ValueError("rho_array_for_far3d must be provided for generative profile modes.")
            
        

        nbulk_profile_for_far3d = self.nbulks_interp(rho_array_for_far3d)
        Tbulk_profile_for_far3d = self.Tbulks_interp(rho_array_for_far3d)
        p_kPa_bulk = nbulk_profile_for_far3d * Tbulk_profile_for_far3d * kev_to_J / 1000

        nNBI_profile_for_far3d = self.nNBIs_interp(rho_array_for_far3d)
        TNBI_profile_for_far3d = self.TNBIs_interp(rho_array_for_far3d)
        p_kPa_NBI = nNBI_profile_for_far3d * TNBI_profile_for_far3d * kev_to_J / 1000
        self.p_kPa_NBI = p_kPa_NBI

        nRF_profile_for_far3d = self.nRFs_interp(rho_array_for_far3d)
        TRF_profile_for_far3d = self.TRFs_interp(rho_array_for_far3d)
        p_kPa_RF = nRF_profile_for_far3d * TRF_profile_for_far3d * kev_to_J / 1000
        self.p_kPa_RF = p_kPa_RF

        ne_profile_for_far3d = self.ne_interpolator_m3(rho_array_for_far3d)
        Te_profile_for_far3d = self.Te_interpolator_kev(rho_array_for_far3d)
        p_kPa_e = ne_profile_for_far3d * Te_profile_for_far3d * kev_to_J / 1000

        thermal_pressure = p_kPa_bulk + p_kPa_e
        equilibrium_pressure = thermal_pressure + p_kPa_NBI + p_kPa_RF

        self.load_profile(profile_name="rho_e", profile_array=rho_array_for_far3d)
        self.load_profile(profile_name="den_ion_e", profile_array=(nbulk_profile_for_far3d / 1e20))
        self.load_profile(profile_name="temp_ion_e", profile_array=Tbulk_profile_for_far3d)
        self.load_profile(profile_name="den_elec_e", profile_array=(ne_profile_for_far3d / 1e20))
        self.load_profile(profile_name="temp_elec_e", profile_array=Te_profile_for_far3d)   

        self.load_profile(profile_name="den_beam_e", profile_array=(nNBI_profile_for_far3d / 1e20))
        self.load_profile(profile_name="temp_beam_e", profile_array=TNBI_profile_for_far3d)   
        self.load_profile(profile_name="den_alpha_e", profile_array=(nRF_profile_for_far3d / 1e20))
        self.load_profile(profile_name="temp_alpha_e", profile_array=TRF_profile_for_far3d)   

        self.load_profile(profile_name="pres_thermal_e", profile_array=thermal_pressure)
        self.load_profile(profile_name="pres_beam_e", profile_array=p_kPa_NBI)
        self.load_profile(profile_name="pres_equil_e", profile_array=equilibrium_pressure)

        impurity_density = np.ones_like(rho_array_for_far3d) * 0.01
        self.load_profile(profile_name='den_imp_e', profile_array=impurity_density)

        q_eqdsk = self.eqdsk['qpsi']
        psi_n = np.linspace(0, 1, self.eqdsk['nW'])
        phi = cumulative_trapezoid(q_eqdsk, psi_n, initial=0)
        phi_n = phi / phi[-1]
        rho_tor = np.sqrt(phi_n)
        q_interp = interp1d(rho_tor, q_eqdsk)
        q_far3d = q_interp(rho_array_for_far3d) + qshift
        self.load_profile(profile_name='qprof', profile_array=q_far3d)

        self.load_profile(profile_name='tor_rot_vel_e', profile_array=rho_array_for_far3d * 0.0)
        self.load_profile(profile_name='pol_rot_vel_e', profile_array=rho_array_for_far3d * 0.0)

        self.rho_array_for_far3d = rho_array_for_far3d

    def plot_profiles(self, figsize=(12,24), from_saved_file=False):
            """
            Plots all 16 FAR3d profiles in a 4x4 grid against rho_array.
            """
            if from_saved_file:
                rho_array = self.loaded_profile_txt['rho_e']
            else:
                rho_array = self.rho_array_for_far3d

            # Create a 4x4 grid of subplots. sharex=True keeps the x-axis aligned.
            fig, axes = plt.subplots(9, 2, figsize=figsize, sharex=True)
            axes = axes.flatten()

            # A dictionary to give the y-axis labels proper formatting and units
            display_labels = {
                "rho_e": "Normalized Rho",
                "qprof": "Safety Factor (q)",
                "den_beam_e": "Beam Density (10^20 m^-3)",
                "den_ion_e": "Bulk Ion Density (10^20 m^-3)",
                "den_elec_e": "Electron Density (10^20 m^-3)",
                "den_alpha_e": "RF Ion Density (10^20 m^-3)",
                "den_imp_e": "Impurity Density (10^20 m^-3)",
                "temp_beam_e": "Beam Temp (keV)",
                "temp_ion_e": "Bulk Ion Temp (keV)",
                "temp_elec_e": "Electron Temp (keV)",
                "temp_alpha_e": "RF Ion Temp (keV)",
                "pres_beam_e": "Beam Pressure (kPa)",
                "pres_thermal_e": "Thermal Pressure (kPa)",
                "pres_equil_e": "Equil. Pressure (kPa)",
                "tor_rot_vel_e": "Toroidal Rot. (km/s)",
                "pol_rot_vel_e": "Poloidal Rot. (km/s)"
            }

            # Loop through the 16 headers and plot each one
            for i, header in enumerate(self.headers):
                ax = axes[i]
                
                if from_saved_file:
                    y_data = self.loaded_profile_txt[header]
                else:
                    y_data = self.case_txt_dict[header] 
                
                ax.plot(rho_array, y_data, linewidth=2, color='darkred')
                
                # Formatting to make it readable
                ax.set_title(header, fontsize=11, fontweight='bold')
                ax.set_ylabel(display_labels.get(header, ""), fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                ax.set_xlabel("Normalized Rho", fontsize=10)

            # after all the internal profiles, also plot the RF pressure profile 
            if from_saved_file == False:
                ax = axes[-2]
                ax.plot(rho_array, self.p_kPa_RF, linewidth=2, color='darkred')
                
                # Formatting to make it readable
                ax.set_title('RF pressure', fontsize=11, fontweight='bold')
                ax.set_ylabel('RF Pressure (kPa)', fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                ax.set_xlabel("Normalized Rho", fontsize=10)
            # Automatically adjust spacing so titles and labels don't overlap
            plt.tight_layout()
            plt.show()

    def calculate_normalized_cyclotron_freq(self, species):
        """
        Calculates the dimensionless cyclotron frequency (omcy) for FAR3d.
        
        B0: Vacuum magnetic field at axis (Tesla)
        R0: Major radius (m)
        n_i0: Core bulk ion density (10^20 m^-3) -> from your profile at rho=0
        m_fast_amu: Mass of the fast species in AMU (e.g., 2.0 for Deuterium)
        bulk_ion_amu: Mass of the bulk ions in AMU
        """
        # read eqdsk 
        B0 = np.abs(self.eqdsk['bcentr'])
        R0 = self.eqdsk['rcentr']
        n_i0 = self.nbulks_interp(self.rho_array_for_far3d[0])

        # Physics constants
        q_e = self.species_dict[species]['charge']    # Coulombs
        mu_0 = 4 * np.pi * 1e-7 # Vacuum permeability
        
        # 1. Alfvén velocity
        rho_mass = (n_i0) * (self.species_dict[self.species]['mass']) # Core mass density (kg/m^3)
        v_A0 = B0 / np.sqrt(mu_0 * rho_mass)            # m/s
        
        # 2. Alfvén frequency
        omega_A0 = v_A0 / R0                            # rad/s
        
        # 3. Fast particle cyclotron frequency
        m_fast_kg = self.species_dict[species]['mass']
        omega_c = (q_e * B0) / m_fast_kg                # rad/s
        
        # 4. Dimensionless FAR3d input
        omcy = omega_c / omega_A0
        
        #print(f"Calculated v_A0: {v_A0:.2e} m/s")
        #print(f"Calculated FAR3d omcy: {omcy:.2f}")
        
        return omcy
    
    def calculate_normalized_larmor_radius(self, species, temperature_header_name):
        temp_kev = self.case_txt_dict[temperature_header_name][0]
        # conver to jouls
        temp_jouls = temp_kev * 1.6022e-19 * 1000
        minor_radius = (max(self.eqdsk['rbbbs']) - min(self.eqdsk['rbbbs'])) / 2
        mass_kg = self.species_dict[species]['mass']

        velocity = np.sqrt(temp_jouls*2/mass_kg)
        B0 = np.abs(self.eqdsk['bcentr'])
        rL = mass_kg * velocity / (B0 * self.species_dict[species]['charge']) 

        if rL < 1e-6:
            rL = 0.0001 # floor 
        return  rL / minor_radius
    
    def calculate_betas(self):
        p_NBI_core = self.p_kPa_NBI[0] * 1000
        p_RF_core = self.p_kPa_RF[0] * 1000
        print('p_RF_core:', p_RF_core)
        mu_0 = 4 * np.pi * 1e-7 # Vacuum permeability
        magnetic_pressure_core = self.eqdsk['bcentr']**2 / (2*mu_0)

        return p_NBI_core/magnetic_pressure_core, p_RF_core/magnetic_pressure_core
        

    def write_far3d_parameters(self, filename="far3d_params.txt", 
                                eq_name="woutb",
                                ext_prof_name="external_profiles.txt",
                                ext_prof_len=100,
                                m_dynamic=[11, 10, 9, 8, 7],
                                n_dynamic=[10, -10],
                                leqdim=23,
                                include_alphas=True,
                                alpha_beta_amplifier=1):
            """
            Generates the main FAR3d parameters input file using the updated deck structure.
            Automatically formats Python lists for m and n modes into the 
            rigid Fortran repeat syntax (e.g., 5*10) and calculates total dimensions.
            """
            
            # --- MODE FORMATTING LOGIC ---
            # 1. Build Equilibrium Modes (0 to leqdim-1)
            m_eq = list(range(leqdim))
            mmeq_str = ",".join(map(str, m_eq))
            nneq_str = f"{leqdim}*0"
            
            # 2. Build Dynamic Modes
            mm_lines = []
            nn_parts = []

            # calculate cyclotron frequencies 
            omcy = self.calculate_normalized_cyclotron_freq(species=self.species)
            omcyalp = self.calculate_normalized_cyclotron_freq(species=self.fast_species) 

            # calculate larmor radii
            iflr = self.calculate_normalized_larmor_radius(species=self.species, temperature_header_name='temp_ion_e')
            r_epflr = self.calculate_normalized_larmor_radius(species=self.fast_species, temperature_header_name='temp_beam_e')
            r_epflralp = self.calculate_normalized_larmor_radius(species=self.fast_species, temperature_header_name='temp_alpha_e')

            # calculate the core betas
            betaNBI, betaRF = self.calculate_betas()
            print(f'Calculated beam beta: {betaNBI}')
            print(f'Calculated RF beta: {betaRF}')
            
            for n in n_dynamic:
                # Physics check: if n is negative, m must be negative to preserve helicity (q = m/n)
                sign = -1 if n < 0 else 1
                m_current = [m * sign for m in m_dynamic]
                
                # Format to Fortran strings
                mm_lines.append(",".join(map(str, m_current)))
                nn_parts.append(f"{len(m_current)}*{n}")
                
            # 3. Combine Dynamic and Equilibrium blocks
            mm_lines.append(mmeq_str)
            nn_parts.append(nneq_str)
            
            mm_str = "\n".join(mm_lines)
            nn_str = ",".join(nn_parts)
            
            # Calculate total dimension for ldim
            ldim = (len(m_dynamic) * len(n_dynamic)) + leqdim

            # calc the number of profiles per output file 
            lplots = len(m_dynamic) * len(n_dynamic)
            #{betaNBI:.4f} put me back in 
            # include alphas 
            if include_alphas:
                alphas_RF_included = 1
            else:
                alphas_RF_included = 0
            
            # --- TEMPLATE INJECTION ---
            template = f"""============================/ MAIM NPUT VARIABLES \===================================
!!!!!!!!!!! nstres: if 0 new run, if 1 the run is a continuation
0
!!!!!!!!!!! numrun: run number
0000
!!!!!!!!!!! numruno: name of the previous run output
0000z
!!!!!!!!!!! numvac: run number index
41503
!!!!!!!!!!! nonlin: linear run if 0, non linear run if 1 (no available yet) 
0
!!!!!!!!!!! ngeneq: equilibrium input (only VMEC available now) 
1
!!!!!!!!!!! eq_name: equilibrium name 
{eq_name}
!!!!!!!!!!! maxstp: simulation time steps 
1000
!!!!!!!!!!! dt0: simulation time step 
2
!!!!!!!!!!! ldim: total number of poloidal modes (equilibrium + dynamic) 
{ldim}
!!!!!!!!!!! leqdim: equilibrium poloidal modes 
{leqdim}
!!!!!!!!!!! jdim: number of radial points
1000
!!!!!!!!!!! ext_prof: include external profiles if 1
1
!!!!!!!!!!! ext_prof_name: external profile file name
{ext_prof_name}
!!!!!!!!!!! ext_prof_len: number of lines in the external profile
{ext_prof_len}
!!!!!!!!!!! iflr_on: activate thermal ion FLR damping effects if 1
0
!!!!!!!!!!! epflr_on: activate fast particle FLR damping effects if 1
0
!!!!!!!!!!! ieldamp_on: activate electron-ion Landau damping effect if 1
1
!!!!!!!!!!! twofl_on: activate two fluid effects if 1
0
!!!!!!!!!!! alpha_on: activate a 2nd fast particle species if 1
{alphas_RF_included}
!!!!!!!!!!! Trapped_on: activate correction for trapped 1st fast particle species if 1
1
!!!!!!!!!!! B_par_on: activate parallel magnetic field perturbation
0
!!!!!!!!!!! matrix_out: activate eigensolver output
.false. 
!!!!!!!!!!! m0dy: equilibrium modes as dynamic
0
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
==================================/ MODEL PARAMETERS \===================================
!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MODES INCLUDED IN THE MODEL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!! mm: poloidal dynamic and equilibrium modes                                                                               
{mm_str}                                                                                                   
!!!!!!!!!!! nn: toroidal dynamic and equilibrium modes
{nn_str}
!!!!!!!!!!! mmeq: poloidal equilibrium modes
{mmeq_str}                                                                                                    
!!!!!!!!!!! nneq: toroidal equilibrium modes
{nneq_str}
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PERTURBATION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!! ipert: different options to drive a perturbation in the equilibria
1
!!!!!!!!!!! widthi: size of the perturbation
1.e-140
!!!!!!!!!!! Auto_grid_on: auto grid spacing option
1
!!!!!!!!!!! ni: number of points interior to the island (if Auto_grid_on = 0)
499
!!!!!!!!!!! nis: number of points in the island (if Auto_grid_on = 0)
251
!!!!!!!!!!! ne: number of points exterior to the island (if Auto_grid_on = 0)
250
!!!!!!!!!!! delta: normalized width of the uniform fine grid (island)
0.1
!!!!!!!!!!! rc: center of the fine grid (island) along the normalized minor radius
0.625
!!!!!!!!!!! Edge_on: activates the VMEC data extrapolation
1
!!!!!!!!!!! edge_p: grid point from where the VMEC data is extrapolated
990
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PLASMA PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!! gamma: adiabatic index
0
!!!!!!!!!!! s: magnetic Lundquist number
5.e6
!!!!!!!!!!! betath_factor: thermal beta factor
1
!!!!!!!!!!! ietaeq: resistivity profile type (if 1 the electron temperature is used)
1
!!!!!!!!!!! spe1: species first EP population
2
!!!!!!!!!!! bet0_f: fast particle beta
{betaNBI:.4f}
!!!!!!!!!!! spe2: species second EP population
2
!!!!!!!!!!! bet0_falp: 2nd species fast particle beta
{alpha_beta_amplifier*betaRF:.4f}
!!!!!!!!!!! omcy: normalized fast particle cyclotron frequency
{omcy}
!!!!!!!!!!! omcyb: normalized helicaly trapped fast particle frequency
0.165
!!!!!!!!!!! rbound: normalized helicaly trapped bound length
0.7
!!!!!!!!!!! omcyalp: normalized 2nd species fast particle cyclotron frequency
{omcyalp}
!!!!!!!!!!! itime: time normalization option
2
!!!!!!!!!!! dpres: electron pressure normalized to the total pressure (two fluid effects)
0.35
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DIFFUSIVITIES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!! stdifp: thermal pressure eq. diffusivity
0
!!!!!!!!!!! stdifu: vorticity eq. diffusivity
0
!!!!!!!!!!! stdifv: thermal particle parallel velocity eq. diffusivity
0
!!!!!!!!!!! stdifnf: fast particle density eq. diffusivity
0
!!!!!!!!!!! stdifvf: fast particle parallel velocity eq. diffusivity
0
!!!!!!!!!!! stdifnfalp: fast particle density eq. diffusivity
0
!!!!!!!!!!! stdifvfalp: fast particle parallel velocity eq. diffusivity
0
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! LANDAU CLOSURE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!! LcA0: Landau closure 1
2.718
!!!!!!!!!!! LcA1: Landau closure 2
-1.311
!!!!!!!!!!! LcA2: correction to the fast particle beta
0.889
!!!!!!!!!!! LcA3: correction to the ratio between fast particle thermal velocity and Alfven velocity
1.077
!!!!!!!!!!! LcA0alp: Landau closure 1 2nd species
2.718
!!!!!!!!!!! LcA1alp: Landau closure 2 2nd species
-1.311
!!!!!!!!!!! LcA2alp: correction to the 2nd species fast particle beta
0.889
!!!!!!!!!!! LcA3alp: correction to the ratio between fast particle thermal velocity and Alfven velocity 2nd species
1.077
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DAMPINGS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!! omegar: eigenmode frequency without damping effects
0.7
!!!!!!!!!!! iflr: thermal ions larmor radius normalized to the minor radius
{iflr}
!!!!!!!!!!! r_epflr: energetic particle larmor radius normalized to the minor radius
{r_epflr}
!!!!!!!!!!! r_epflralp: 2nd species energetic particle larmor radius normalized to the minor radius
{r_epflralp}
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OUTPUT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!! lplots: number of eigenfunction modes in the output files
{lplots}
!!!!!!!!!!! nprint: number of step for an output in farprt file
100
!!!!!!!!!!! ndump: number of step for an output 
1000
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OTHER PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!! DIIID_u: turn on to use the same units than TRANSP output in the external profiles (cm not m)
0
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
================================/ SELF PROFILES PARAMETERS \============================
!!!!!!!!!!! EP_dens_on: user defined fast particle density profile (if 1)
1
!!!!!!!!!!! Adens: fast particle density profile flatness
8.5
!!!!!!!!!!! Bdens: location of the fast particle density profile gradient
0.4
!!!!!!!!!!! Alpha_dens_on: user defined 2nd species fast particle density profile (if 1)
1
!!!!!!!!!!! Adensalp: 2nd species fast particle density profile flatness
7
!!!!!!!!!!! Bdensalp: location of the 2nd species fast particle density profile gradient
0.4
!!!!!!!!!!! EP_vel_on: user defined fast particle vth/vA0 profile (if 1)
1
!!!!!!!!!!! Alpha_vel_on: user defined 2nd species fast particle vth/vA0 profile (if 1)
1
!!!!!!!!!!! q_prof_on: the safety factor profile of the external profile is used (if 1)
1
!!!!!!!!!!! Eq_vel_on: the equilibrium toroidal velocity profile of the external profile is used (if 1)
0
!!!!!!!!!!! Eq_velp_on: the equilibrium poloidal velocity profile of the external profile is used (if 1)
0
!!!!!!!!!!! Eq_Presseq_on: the equilibrium pressure profile of the external profile is used (if 1)
0
!!!!!!!!!!! Eq_Presstot_on: the equilibrium and EP pressure profiles of the external profile is used (if 1)
1
!!!!!!!!!!! deltaq: safety factor displacement (only tokamak eq.)
0
!!!!!!!!!!! deltaiota: iota displacement (only stellarator eq.)
0
!!!!!!!!!!! etascl: user defined constant resistivity (if ietaeq=2)
1
!!!!!!!!!!! eta0: user defined resistivity profile (if ietaeq=3)
1
!!!!!!!!!!! reta: user defined resistivity profile (if ietaeq=3)
0.5
!!!!!!!!!!! etalmb: user defined resistivity profile (if ietaeq=3)
0.5
!!!!!!!!!!! cnep: user defined thermal plasma density profile 
5.349622177242649E-01, -8.158145079082755E-01,  9.051313827341806E+00,                                                      
-9.908622794510921E+01,  5.436130071633405E+02, -1.683588662473988E+03,                                                      
3.184577674238863E+03, -3.789454655055126E+03,  2.786069409436922E+03,                                                       
-1.160818784526462E+03,  2.100700986027593E+02        
!!!!!!!!!!! ctep: user defined thermal electron plasma temperature profile
2.466297479423131E+00, -3.123519794977899E+00,  1.455166487731591E+01,
-8.288042333625850E+01,  3.310397776819081E+02, -9.891030935273041E+02,
2.133646774676442E+03, -3.082790956275732E+03,  2.757324727437074E+03,
-1.367187266111290E+03,  2.864529219889411E+02
!!!!!!!!!!! cnfp: user defined energetic particles density profile
1.5, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
!!!!!!!!!!! cvep: user defined thermal ions parallel velocity profile (only for thermal ion FLR effects)
5.349622177242649E-01, -8.158145079082755E-01,  9.051313827341806E+00,
-9.908622794510921E+01,  5.436130071633405E+02, -1.683588662473988E+03,
3.184577674238863E+03, -3.789454655055126E+03,  2.786069409436922E+03,
-1.160818784526462E+03,  2.100700986027593E+02
!!!!!!!!!!! cvfp: user defined energetic particles parallel velocity profile
0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
!!!!!!!!!!! cnfpalp: user defined 2nd species energetic particles density profile
5.349622177242649E-01, -8.158145079082755E-01,  9.051313827341806E+00,
-9.908622794510921E+01,  5.436130071633405E+02, -1.683588662473988E+03,
3.184577674238863E+03, -3.789454655055126E+03,  2.786069409436922E+03,
-1.160818784526462E+03,  2.100700986027593E+02
!!!!!!!!!!! cvfpalp: user defined 2nd species energetic particles parallel velocity profile
0.48, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
!!!!!!!!!!! eqvt: user defined equilibrium thermal toroidal velocity profile
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
!!!!!!!!!!! eqvp: user defined equilibrium thermal poloidal velocity profile
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

            with open(filename, 'w') as f:
                f.write(template)
                
            print(f"Successfully generated FAR3d parameters file: {filename}")
            print(f"-> Total Modes (ldim) Calculated: {ldim}")


#     def write_far3d_parameters(self, filename="far3d_params.txt", 
#                                eq_name="woutb",
#                                ext_prof_name="external_profiles.txt",
#                                ext_prof_len=100,
#                                m_dynamic=[12, 11, 10, 9, 8],
#                                n_dynamic=[10, -10],
#                                leqdim=23,
#                                include_alphas=True):
#         """
#         Generates the main FAR3d parameters input file.
#         Automatically formats Python lists for m and n modes into the 
#         rigid Fortran repeat syntax (e.g., 5*10) and calculates total dimensions.
#         """
        
#         # --- MODE FORMATTING LOGIC ---
#         # 1. Build Equilibrium Modes (0 to leqdim-1)
#         m_eq = list(range(leqdim))
#         mmeq_str = ",".join(map(str, m_eq))
#         nneq_str = f"{leqdim}*0"
        
#         # 2. Build Dynamic Modes
#         mm_lines = []
#         nn_parts = []

#         # calculate cyclotron frequencies 
#         omcy = self.calculate_normalized_cyclotron_freq()
#         omcyalp = omcy # these are the same species 

#         # calculate larmor radii
#         iflr = self.calculate_normalized_larmor_radius(temperature_header_name='temp_ion_e')
#         r_epflr = self.calculate_normalized_larmor_radius(temperature_header_name='temp_beam_e')
#         r_epflralp = self.calculate_normalized_larmor_radius(temperature_header_name='temp_alpha_e')

#         # calculate the core betas
#         betaNBI, betaRF = self.calculate_betas()
#         print(f'Calculated beam beta: {betaNBI}')
#         print(f'Calculated RF beta: {betaRF}')
        
#         for n in n_dynamic:
#             # Physics check: if n is negative, m must be negative to preserve helicity (q = m/n)
#             sign = -1 if n < 0 else 1
#             m_current = [m * sign for m in m_dynamic]
            
#             # Format to Fortran strings
#             mm_lines.append(",".join(map(str, m_current)))
#             nn_parts.append(f"{len(m_current)}*{n}")
            
#         # 3. Combine Dynamic and Equilibrium blocks
#         mm_lines.append(mmeq_str)
#         nn_parts.append(nneq_str)
        
#         mm_str = "\n".join(mm_lines)
#         nn_str = ",".join(nn_parts)
        
#         # Calculate total dimension for ldim
#         ldim = (len(m_dynamic) * len(n_dynamic)) + leqdim

#         # calc the number of profiles per output file 
#         lplots = len(m_dynamic) * len(n_dynamic)

#         # include alhpas 
#         if include_alphas:
#             alphas_RF_included = 1
#         else:
#             alphas_RF_included = 0
        
#         # --- TEMPLATE INJECTION ---
#         template = f"""============================/ MAIN INPUT VARIABLES \===================================
# !!!!!!!!!!! namelist_on: if 1 new run, namelist is used
# 0
# !!!!!!!!!!! nstres: if 0 new run, if 1 the run is a continuation
# 0
# !!!!!!!!!!! numrun: run number
# 0001
# !!!!!!!!!!! numruno: name of the previous run output
# 0884z
# !!!!!!!!!!! numvac: run number index
# 41503
# !!!!!!!!!!! nonlin: linear run if 0, non linear run if 1 
# 0
# !!!!!!!!!!! ngeneq: equilibrium input (only VMEC available now) 
# 1
# !!!!!!!!!!! eq_name: equilibrium name 
# {eq_name}
# !!!!!!!!!!! maxstp: simulation time steps 
# 1000
# !!!!!!!!!!! dt0: simulation time step 
# 2
# !!!!!!!!!!! ldim: total number of poloidal modes (equilibrium + dynamic) 
# {ldim}
# !!!!!!!!!!! leqdim: equilibrium poloidal modes 
# {leqdim}
# !!!!!!!!!!! jdim: number of radial points
# 1000
# !!!!!!!!!!! ext_prof: include external profiles if 1
# 1
# !!!!!!!!!!! ext_prof_name: external profile file name
# {ext_prof_name}
# !!!!!!!!!!! ext_prof_len: number of lines in the external profile
# {ext_prof_len}
# !!!!!!!!!!! iflr_on: activate thermal ion FLR damping effects if 1
# 0
# !!!!!!!!!!! epflr_on: activate fast particle FLR damping effects if 1
# 0
# !!!!!!!!!!! ieldamp_on: activate electron-ion Landau damping effect if 1
# 0
# !!!!!!!!!!! twofl_on: activate two fluid effects if 1
# 0
# !!!!!!!!!!! alpha_on: activate a 2nd fast particle species if 1
# {alphas_RF_included}
# !!!!!!!!!!! Trapped_on: activate correction for trapped 1st fast particle species if 1
# 1
# !!!!!!!!!!! matrix_out: activate eigensolver output
# .false. 
# !!!!!!!!!!! m0dy: equilibrium modes as dynamic
# 0
# !!!!!!!!!!! nopsievol_on: if 1, no nonlinear evolution of the poloidal flux
# 1
# !!!!!!!!!!! noprevol_on: if 1, no nonlinear evolution of the thermal pressure
# 1
# !!!!!!!!!!! nonfevol_on: no nonlinear evolution of the first EP population density
# 1
# !!!!!!!!!!! nonalpevol_on: no nonlinear evolution of the second EP population density
# 1
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ==================================/ MODEL PARAMETERS \===================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!! MODES INCLUDED IN THE MODEL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!! mm: poloidal dynamic and equilibrium modes                                                              
# {mm_str}
# !!!!!!!!!!! nn: toroidal dynamic and equilibrium modes
# {nn_str}
# !!!!!!!!!!! mmeq: poloidal equilibrium modes
# {mmeq_str}                                                                                                   
# !!!!!!!!!!! nneq: toroidal equilibrium modes
# {nneq_str}
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PERTURBATION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!! ipert: different options to drive a perturbation in the equilibria
# 1
# !!!!!!!!!!! widthi: size of the perturbation. 
# 1.e-140
# !!!!!!!!!!! Auto_grid_on: auto grid spacing option
# 1
# !!!!!!!!!!! ni: number of points interior to the island
# 499
# !!!!!!!!!!! nis: number of points in the island
# 251
# !!!!!!!!!!! ne: number of points exterior to the island
# 250
# !!!!!!!!!!! delta: normalized width of the uniform fine grid (island)
# 0.10
# !!!!!!!!!!! rc: center of the fine grid (island) along the normalized minor radius
# 0.625
# !!!!!!!!!!! Edge_on: activates the VMEC data extrapolation
# 1
# !!!!!!!!!!! edge_p: grid point from where the VMEC data is extrapolated
# 990
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PLASMA PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!! gamma: adiabatic index
# 1.37
# !!!!!!!!!!! s: magnetic Lundquist number
# 5.e6
# !!!!!!!!!!! betath_factor: thermal beta factor. Originally 1.
# 1
# !!!!!!!!!!! ietaeq: resistivity profile type (if 5 the electron temperature is used)
# 5
# !!!!!!!!!!! spe1: species first EP population
# 2
# !!!!!!!!!!! bet0_f: fast particle beta
# {betaNBI:.3f}
# !!!!!!!!!!! spe2: species second EP population
# 2
# !!!!!!!!!!! bet0_falp: 2nd species fast particle beta. 
# {betaRF:.3f}
# !!!!!!!!!!! omcy: normalized fast particle cyclotron frequency
# {omcy}
# !!!!!!!!!!! omcyb: normalized helicaly trapped fast particle frequency. 
# 0.165
# !!!!!!!!!!! rbound: normalized helicaly trapped bound length. 
# 0.7
# !!!!!!!!!!! omcyalp: normalized 2nd species fast particle cyclotron frequency. 
# {omcyalp}
# !!!!!!!!!!! itime: time normalization option
# 2
# !!!!!!!!!!! dpres: electron pressure normalized to the total pressure (two fluid effects)
# 0.35
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DIFFUSIVITIES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!! stdifp: thermal pressure eq. diffusivity
# 0
# !!!!!!!!!!! stdifu: vorticity eq. diffusivity
# 0
# !!!!!!!!!!! stdifv: thermal particle parallel velocity eq. diffusivity
# 0
# !!!!!!!!!!! stdifnf: fast particle density eq. diffusivity
# 0
# !!!!!!!!!!! stdifvf: fast particle parallel velocity eq. diffusivity
# 0
# !!!!!!!!!!! stdifnfalp: fast particle density eq. diffusivity
# 0
# !!!!!!!!!!! stdifvfalp: fast particle parallel velocity eq. diffusivity
# 0
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! LANDAU CLOSURE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!! LcA0: Landau closure 1
# 2.718
# !!!!!!!!!!! LcA1: Landau closure 2
# -1.311
# !!!!!!!!!!! LcA2: correction to the fast particle beta
# 0.889
# !!!!!!!!!!! LcA3: correction to the ratio between fast particle thermal velocity and Alfven velocity
# 1.077
# !!!!!!!!!!! LcA0alp: Landau closure 1 2nd species
# 2.718
# !!!!!!!!!!! LcA1alp: Landau closure 2 2nd species
# -1.311
# !!!!!!!!!!! LcA2alp: correction to the 2nd species fast particle beta
# 0.889
# !!!!!!!!!!! LcA3alp: correction to the ratio between fast particle thermal velocity and Alfven velocity 2nd species
# 1.077
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DAMPINGS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!! omegar: eigenmode frequency without damping effects. Alfven frequency is 764 kHz.
# 0.7
# !!!!!!!!!!! iflr: thermal ions larmor radius normalized to the minor radius. 
# {iflr}
# !!!!!!!!!!! r_epflr: energetic particle larmor radius normalized to the minor radius. 
# {r_epflr}
# !!!!!!!!!!! r_epflralp: 2nd species energetic particle larmor radius normalized to the minor radius.
# {r_epflralp}
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OUTPUT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!! lplots: number of eigenfunction modes in the output files
# {lplots}
# !!!!!!!!!!! nprint: number of step for an output in farprt file
# 100
# !!!!!!!!!!! ndump: number of step for an output 
# 1000
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! OTHER PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!! DIIID_u: turn on to use the same units than TRANSP output in the external profiles (cm not m)
# 0
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ================================/ SELF PROFILES PARAMETERS \============================
# !!!!!!!!!!! EP_dens_on: user defined fast particle density profile (if 1)
# 0
# !!!!!!!!!!! Adens: fast particle density profile flatness
# 8.5
# !!!!!!!!!!! Bdens: location of the fast particle density profile gradient
# 0.4
# !!!!!!!!!!! Alpha_dens_on: user defined 2nd species fast particle density profile (if 1)
# 0
# !!!!!!!!!!! Adensalp: 2nd species fast particle density profile flatness
# 7
# !!!!!!!!!!! Bdensalp: location of the 2nd species fast particle density profile gradient
# 0.4
# !!!!!!!!!!! EP_vel_on: user defined fast particle vth/vA0 profile (if 1). 
# 0
# !!!!!!!!!!! Alpha_vel_on: user defined 2nd species fast particle vth/vA0 profile (if 1)
# 0
# !!!!!!!!!!! q_prof_on: the safety factor profile of the external profile is used (is 1)
# 0
# !!!!!!!!!!! Eq_vel_on: the equilibrium toroidal velocity profile of the external profile is used (is 1)
# 0
# !!!!!!!!!!! Eq_velp_on: the equilibrium poloidal velocity profile of the external profile is used (if 1)
# 0
# !!!!!!!!!!! Eq_Presseq_on: the equilibrium pressure profile of the external profile is used (is 1)
# 0
# !!!!!!!!!!! Eq_Presstot_on: the equilibrium + fast particle pressure profiles of the external profile is used (is 1)
# 1
# !!!!!!!!!!! deltaq: safety factor displacement (only tokamak eq.)
# 0
# !!!!!!!!!!! deltaiota: iota displacement (only stellarator eq.)
# 0
# !!!!!!!!!!! etascl: user defined constant resistivity (if ietaeq=2)
# 1
# !!!!!!!!!!! eta0: user defined resistivity profile (if ietaeq=3)
# 1
# !!!!!!!!!!! reta: user defined resistivity profile (if ietaeq=3)
# 0.5
# !!!!!!!!!!! etalmb: user defined resistivity profile (if ietaeq=3)
# 0.5
# !!!!!!!!!!! cnep: user defined thermal plasma density profile 
# 5.349622177242649E-01, -8.158145079082755E-01,  9.051313827341806E+00,
# -9.908622794510921E+01,  5.436130071633405E+02, -1.683588662473988E+03,
# 3.184577674238863E+03, -3.789454655055126E+03,  2.786069409436922E+03,
# -1.160818784526462E+03,  2.100700986027593E+02
# !!!!!!!!!!! ctep: user defined thermal electron plasma temperature profile
# 2.466297479423131E+00, -3.123519794977899E+00,  1.455166487731591E+01,
# -8.288042333625850E+01,  3.310397776819081E+02, -9.891030935273041E+02,
# 2.133646774676442E+03, -3.082790956275732E+03,  2.757324727437074E+03,
# -1.367187266111290E+03,  2.864529219889411E+02
# !!!!!!!!!!! cnfp: user defined energetic particles density profile
# 1.5, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
# !!!!!!!!!!! cvep: user defined thermal ions parallel velocity profile (only for thermal ion FLR effects)
# 5.349622177242649E-01, -8.158145079082755E-01,  9.051313827341806E+00,
# -9.908622794510921E+01,  5.436130071633405E+02, -1.683588662473988E+03,
# 3.184577674238863E+03, -3.789454655055126E+03,  2.786069409436922E+03,
# -1.160818784526462E+03,  2.100700986027593E+02
# !!!!!!!!!!! cvfp: user defined energetic particles parallel velocity profile
# 0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
# !!!!!!!!!!! cnfpalp: user defined 2nd species energetic particles density profile
# 5.349622177242649E-01, -8.158145079082755E-01,  9.051313827341806E+00,
# -9.908622794510921E+01,  5.436130071633405E+02, -1.683588662473988E+03,
# 3.184577674238863E+03, -3.789454655055126E+03,  2.786069409436922E+03,
# -1.160818784526462E+03,  2.100700986027593E+02
# !!!!!!!!!!! cvfpalp: user defined 2nd species energetic particles parallel velocity profile
# 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
# !!!!!!!!!!! eqvt: user defined equilibrium thermal toroidal velocity profile
# 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
# !!!!!!!!!!! eqvp: user defined equilibrium thermal poloidal velocity profile
# 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

#         with open(filename, 'w') as f:
#             f.write(template)
            
#         print(f"Successfully generated FAR3d parameters file: {filename}")
#         print(f"-> Total Modes (ldim) Calculated: {ldim}")

    def grab_and_store_alfven_continuum(self, run_dir):
            # assumes you have run far3d in matrix mode and then ran xcontinuum_FAR3d
        """
        Parses the FAR3D Input_Model to extract dynamic m and n modes.
        Reads the far3d_continuum.dat file, groups the frequencies by mode index,
        and writes the data into an ALCON-style HDF5 dictionary.
        """
        # ---------------------------------------------------------
        # 1. PARSE THE INPUT_MODEL FOR MODES
        # ---------------------------------------------------------
        print("Parsing Input_Model for dynamic modes...")

        input_model_path = run_dir + 'Input_Model'
        continuum_data_path = run_dir + 'far3d_continuum.dat'
        output_hdf5_path = run_dir + 'combined_data.h5'

        with open(input_model_path, 'r') as f:
            lines = f.readlines()

        ldim = 0
        leqdim = 0
        mm_lines = []
        nn_lines = []
        
        current_section = None
        for i, line in enumerate(lines):
            # Extract the array dimensions
            if line.startswith('!!!!!!!!!!! ldim:'):
                ldim = int(lines[i+1].strip())
            elif line.startswith('!!!!!!!!!!! leqdim:'):
                leqdim = int(lines[i+1].strip())
                
            # Extract the arrays
            elif line.startswith('!!!!!!!!!!! mm:'):
                current_section = 'mm'
            elif line.startswith('!!!!!!!!!!! nn:'):
                current_section = 'nn'
            elif line.startswith('!!!!!!!!!!!'):
                if current_section in ['mm', 'nn']:
                    current_section = None  # Stop collecting at the next parameter flag
                    
            # Append lines to our array buffers
            elif current_section == 'mm':
                mm_lines.append(line.strip())
            elif current_section == 'nn':
                nn_lines.append(line.strip())

        num_dyn_modes = ldim - leqdim

        def parse_fortran_array(arr_lines):
            """Helper to parse Fortran multi-line, comma-separated, scalar-multiplied arrays (e.g. 4*1)."""
            vals = []
            for ln in arr_lines:
                # Clean up commas and split by spaces
                parts = ln.replace(',', ' ').split()
                for p in parts:
                    if '*' in p:
                        count, val = p.split('*')
                        vals.extend([int(val)] * int(count))
                    else:
                        vals.append(int(p))
            return vals
            
        # Parse and truncate the lists to ONLY include dynamic modes
        mm_vals = parse_fortran_array(mm_lines)[:num_dyn_modes]
        nn_vals = parse_fortran_array(nn_lines)[:num_dyn_modes]
        print(f"Detected {num_dyn_modes} dynamic modes.")

        # ---------------------------------------------------------
        # 2. PARSE THE CONTINUUM DATA
        # ---------------------------------------------------------
        print("Reading far3d_continuum.dat...")
        data = np.loadtxt(continuum_data_path)
        
        # Columns based on your output
        r_col = data[:, 0]
        freq_col = data[:, 1]
        idx_col = data[:, 2].astype(int)

        data_dict = {}
        
        # Group the data by mode index
        for i in range(num_dyn_modes):
            mode_idx = i + 1  # 1-based index matching Fortran loop
            m = mm_vals[i]
            n = nn_vals[i]
            
            # Filter data for this specific mode index
            mask = (idx_col == mode_idx)
            
            if not np.any(mask):
                continue
                
            mode_r = r_col[mask]
            mode_freq = freq_col[mask]
            
            # We classify all of these as 'a' (Alfvénic) since we disabled the acoustic branches
            mode_name = f"a_mode_n{n}_m{m}"
            data_dict[mode_name] = {
                'type': 'a',
                'n': n,
                'm': m,
                'r_grid': mode_r,
                'freqs': mode_freq
            }

        if not data_dict:
            print("No valid data found to process. Exiting.")
            return

        # ---------------------------------------------------------
        # 3. WRITE THE DATA DICT TO THE HDF5 FILE
        # ---------------------------------------------------------
        print(f"Writing to {output_hdf5_path}...")
        with h5py.File(output_hdf5_path, 'w') as f:
            for mode_key, mode_data in data_dict.items():
                mode_group = f.create_group(mode_key)
                for var_key, var_value in mode_data.items():
                    mode_group.create_dataset(var_key, data=var_value)
                    
        print(f"Success! Combined data for {len(data_dict)} modes saved to: {output_hdf5_path}")
        print('Loading into class object...')
        self.mode_dict = self.load_hdf5_to_dict(output_hdf5_path)

    def load_hdf5_to_dict(self, filepath):
        """
        Reads an ALCON/FAR3D HDF5 file and reconstructs the nested Python dictionary,
        loading all arrays into RAM.
        """
        restored_dict = {}
        with h5py.File(filepath, 'r') as f:
            for mode_key in f.keys():
                mode_group = f[mode_key]
                restored_dict[mode_key] = {}
                for var_key in mode_group.keys():
                    dataset = mode_group[var_key]
                    
                    if dataset.shape == (): 
                        val = dataset[()]
                        if isinstance(val, bytes):
                            val = val.decode('utf-8')
                        restored_dict[mode_key][var_key] = val
                    else:
                        restored_dict[mode_key][var_key] = dataset[:]
                        
        return restored_dict
    
    def plot_alfven_continuum(self, run_dir, load_mode_dict=False, color_points=False, return_fig=False, fig_size=(10,10), plot_acoustic=False, xlims=(0,1), ylims=(0,1), legend=True):
        if load_mode_dict:
            output_hdf5_path = run_dir + 'combined_data.h5'
            self.mode_dict = self.load_hdf5_to_dict(output_hdf5_path)

        # initialize the figure 
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        cmap_alfvenic = plt.get_cmap('jet')
        cmap_acoustic = plt.get_cmap('Greys')

        with open(run_dir+'conversion_factor.txt', 'r') as f:
                freq_to_kHz = float(f.read().strip())
        print(f'Found conversion to kHz: {freq_to_kHz}')

        num_a = 0
        num_s = 0

        data_dict = self.mode_dict

        for key in data_dict:
            if data_dict[key]['type'] == 'a':
                num_a += 1
            else:
                num_s += 1


        font_size=16
        i_a = 0
        i_s = 0
        for key in data_dict:
            freqs = data_dict[key]['freqs']
            m = data_dict[key]['m']
            n = data_dict[key]['n']
            rgrid = np.sqrt(data_dict[key]['r_grid'])
            #rgrid = data_dict[key]['r_grid']
            wave_type = data_dict[key]['type']
            if plot_acoustic:
                wave_type_long = 'Compressional'
                if color_points:
                    color = cmap_acoustic((i_s/num_s)/4+0.4)
                else:
                    color = 'gray'
                ax.scatter(rgrid, freqs*freq_to_kHz, s=2, marker='.', color=color, zorder=1)
                i_s +=1
            elif wave_type == 'a':
                wave_type_long = 'Alfvenic'
                if color_points:
                    color = cmap_alfvenic(i_a/num_a)
                else:
                    color = 'gray'
                
                i_a += 1
                label = f'n:{n}, m:{m}'
                if legend == False:
                    label=None
                if n > 0:
                    ax.scatter(rgrid, freqs*freq_to_kHz, label=label, marker='.', s=2, color=color, zorder=2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.tight_layout()
        if legend:
            ax.legend(fontsize=font_size*.75, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel('sqrt(far3d_continuum.dat radial grid) ', fontsize=font_size)
        ax.set_ylabel(r'$\omega$ [kHz]', fontsize=font_size)
        ax.tick_params(axis='both', labelsize=font_size)
        ax.set_ylim(ylims[0], ylims[1])
        ax.set_xlim(xlims[0], xlims[1])

        if return_fig:
            return fig, ax
        else:
            plt.show()


    def plot_profile_over_continuum(self, 
                                    profile_type,
                                    run_indicator,
                                    profile_run_dir, 
                                    matrix_run_dir, 
                                    load_mode_dict=False, 
                                    color_points=False, 
                                    return_fig=False, 
                                    fig_size=(10,10), 
                                    plot_acoustic=False, 
                                    xlims=(0,1), 
                                    ylims=(0,1),
                                    rotate=False,
                                    wave_part='real',
                                    scale_eigenmode=0.25,
                                    ):
        # grab the profile data 
        data_dict = self.data_dict[profile_type]
        rgrid = data_dict['r']
        data = data_dict['data']
        ns = data_dict['ns']
        ms = data_dict['ms']
        
        # --- NEW GLOBAL ROTATION LOGIC ---
        if rotate:
            # 1. Find the 2D index of the absolute largest value across ALL modes
            # np.argmax on a 2D array flattens it, so we unravel it back to (row, col)
            max_idx = np.unravel_index(np.argmax(np.abs(data)), data.shape)
            
            # 2. Extract the phase of that single dominant point
            global_peak_phase = np.angle(data[max_idx])
            global_peak_amplitude = np.abs(data[max_idx])
        # ---------------------------------     

        # now, grab the continuum plot  

        fig, ax = self.plot_alfven_continuum(run_dir=matrix_run_dir, 
                            load_mode_dict=load_mode_dict, 
                            color_points=False, 
                            return_fig=True, 
                            fig_size=fig_size, 
                            plot_acoustic=False, 
                            xlims=xlims, 
                            ylims=ylims,
                            legend=False)
        
        with open(profile_run_dir+'conversion_factor.txt', 'r') as f:
            freq_to_kHz = float(f.read().strip())
        print(f'Found conversion to kHz: {freq_to_kHz}')

        # get the fastest growing mode 
        gamma, omega_r = self.find_maximum_growth_rate_and_frequency(directory=profile_run_dir)
        
        fkHz = freq_to_kHz * omega_r # fastest growing mode frequency 

        
        for imode in range(len(ns)):
            n = ns[imode]
            m = ms[imode]
            profile = data[:,imode]

            # now, rotate the wave using the SAME angle for every mode
            if rotate:
                profile = profile * np.exp(-1j * global_peak_phase)

            if wave_part == 'real':
                profile = np.real(profile)
            elif wave_part == 'imaginary':
                profile = np.imag(profile)
            elif wave_part == 'mag':
                profile = np.abs(profile)
            else:
                raise ValueError('Wave plotting type wave_part not understood') 
            label = f'm/n: {m}/{n}'

            # shift the profile to the frequency, scale  
            profile = (profile/global_peak_amplitude)*scale_eigenmode*(ylims[1]-ylims[0]) + fkHz

            ax.plot(rgrid, profile, label=label) 

        ax.axhline(y=fkHz, color='black', linestyle='--')
        ax.legend()


    def plot_profile(self, 
                            ax,
                            profile_type,
                            case_dir,
                            rotate=False,
                            wave_part='real',
                            font_size=16,
                            legend_font_size=10,
                            cmap='viridis',
                            gamma_factor=1):
        # grab the profile data 
        data_dict = self.data_dict[profile_type]
        rgrid = data_dict['r']
        data = data_dict['data']
        ns = data_dict['ns']
        ms = data_dict['ms']
        
        # --- NEW GLOBAL ROTATION LOGIC ---
        if rotate:
            # 1. Find the 2D index of the absolute largest value across ALL modes
            # np.argmax on a 2D array flattens it, so we unravel it back to (row, col)
            max_idx = np.unravel_index(np.argmax(np.abs(data)), data.shape)
            
            # 2. Extract the phase of that single dominant point
            global_peak_phase = np.angle(data[max_idx])
            global_peak_amplitude = np.abs(data[max_idx])
        # ---------------------------------     

        # now, grab the continuum plot  
        
        with open(case_dir+'conversion_factor.txt', 'r') as f:
            freq_to_kHz = float(f.read().strip())
        print(f'Found conversion to kHz: {freq_to_kHz}')

        # get the fastest growing mode 
        gamma, omega_r = self.find_maximum_growth_rate_and_frequency(directory=case_dir)
        
        # fkHz = freq_to_kHz * omega_r # fastest growing mode frequency 
        cmap = plt.get_cmap(cmap)
        mode_colors = cmap(np.linspace(0, 1, len(ns)))
        ax.set_title(r'$\omega_r$= '+ f'{float(omega_r):.2f}' + r' $[\tau_A^{-1}]$,' + ' ' + r'$\gamma^*$= '+ f'{float(gamma*gamma_factor):.4f}' + r' $[\tau_A^{-1}]$', fontsize=font_size)
        
        for imode in range(len(ns)):
            n = ns[imode]
            m = ms[imode]
            profile = data[:,imode]

            # now, rotate the wave using the SAME angle for every mode
            if rotate:
                profile = profile * np.exp(-1j * global_peak_phase)

            if wave_part == 'real':
                profile = np.real(profile)
            elif wave_part == 'imaginary':
                profile = np.imag(profile)
            elif wave_part == 'mag':
                profile = np.abs(profile)
            else:
                raise ValueError('Wave plotting type wave_part not understood') 
            label = f'm/n: {m}/{n}'

            # shift the profile to the frequency, scale  
            # profile = (profile/global_peak_amplitude)*scale_eigenmode*(ylims[1]-ylims[0]) + fkHz
            line_color = mode_colors[imode]
            ax.plot(rgrid, profile, label=label, color=line_color) 

        # ax.axhline(y=fkHz, color='black', linestyle='--')
        ax.legend(fontsize=legend_font_size)
        return ax

    def get_q_profile(self, eqdsk_path, qshift=0):
        with open(eqdsk_path, 'r') as f:
            eqdsk_dict = geqdsk.read(f)
        nx = eqdsk_dict['nx']
        qpsi = eqdsk_dict['qpsi'] + qshift

        # 1. Create the normalized poloidal flux grid (0 to 1)
        # EQDSK 1D profiles are spaced uniformly in unnormalized poloidal flux.
        psi_norm = np.linspace(0, 1, nx)

        # 2. Calculate rho_pol
        rho_pol = np.sqrt(psi_norm)
        dq_drho = np.gradient(qpsi, rho_pol)
        return rho_pol, qpsi, dq_drho

    def get_applicable_m_numbers(self, n, eqdsk_path, maximum_rho, minimum_rho=0, qshift=0):
        rho_pol, qpsi, dq_drho = self.get_q_profile(eqdsk_path=eqdsk_path, qshift=qshift)

        rho_max_index = np.where(np.abs(rho_pol - maximum_rho) == np.min(np.abs(rho_pol - maximum_rho)))[0][0]
        rho_min_index = np.where(np.abs(rho_pol - minimum_rho) == np.min(np.abs(rho_pol - minimum_rho)))[0][0]

        q_applicable = qpsi[rho_min_index:(rho_max_index+1)]

        m_applicable = n*q_applicable

        m_min = np.min(m_applicable)
        m_max = np.max(m_applicable)

        return int(np.floor(m_min)), int(np.ceil(m_max)) 

    def save_mode_growth_rate_results(self, run_dir, save_file_path, mode_label='not labeled'):
        gamma, omega_r = self.find_maximum_growth_rate_and_frequency(directory=run_dir)

        with open(run_dir+'conversion_factor.txt', 'r') as f:
            freq_to_kHz = float(f.read().strip())
        print(f'Found conversion to kHz: {freq_to_kHz}')

        save_dict = {}
        save_dict['freq_to_kHz'] = freq_to_kHz
        save_dict['gamma'] = gamma
        save_dict['omega'] = omega_r

        # grab n. Assume that the user only ran with one n, and htey are all the same 
        data_dict = self.data_dict['br']
        ns = data_dict['ns']
        print(ns)
        n = ns[0]

        save_dict['n'] = n
        save_dict['Mode Label'] = mode_label


        if os.path.exists(save_file_path) and os.path.getsize(save_file_path) > 0:
            with open(save_file_path, 'r') as file:
                try:
                    data_list = json.load(file)
                except json.JSONDecodeError:
                    data_list = [] # Failsafe if the file exists but is corrupted/empty
        else:
            data_list = [] # File doesn't exist, start a fresh list

        # 2. Append the new dictionary to the Python list
        data_list.append(save_dict)

        # 3. Open the file in 'w' (write) mode to overwrite it with the updated list
        # (Using 'w' will automatically create the file if it doesn't exist)
        with open(save_file_path, 'w') as file:
            json.dump(data_list, file, indent=4)
        





        



        

        
        


         
         

















        





    


    
        
        

