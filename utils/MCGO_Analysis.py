##################################
#   Classes for working with both setup and post-proccesing of MCGO with P2F
#   Author: Jacob van de Lindt
#   Date: 6/12/2025
#################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm
from matplotlib import ticker, cm
from scipy.interpolate import interp1d, RectBivariateSpline, PchipInterpolator
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

# import Grant's eqdsk processor for getting B info
from process_eqdsk2 import getGfileDict

class MCGO_Post_Process:
    """
    Class to post-process output from MCGO including the .nc file and eventually the particle list
    """

    def __init__(self, mcgo_nc_file, eqdsk_file=None, species='d', particle_lists_on=False):
        self.eqdsk_file = eqdsk_file
        self.species = species
        self.particle_lists_on = particle_lists_on
        # load up eqdsk using john's methods
        if self.eqdsk_file is not None:
            self.process_eqdsk()     

        # read .nc file
        self.mcgo_nc = netCDF4.Dataset(mcgo_nc_file, "r")

        # parse the file 
        self.parse_mcgo_nc()

        # assign mass from supported mcgo species
        self.species_dict = {}
        self.species_dict['d'] = {'mass':3.343583e-27, 'charge':1.6022e-19}

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
        print('yeah')
        self.B_midplane_mag_interpolator = PchipInterpolator(
            self.rho_grid, self.getBStrength(self.R_f_grid, self.eqdsk['zmaxis'], grid=False)
        )

    def plot_equilibrium(self, figsize, levels=10, fontsize=20, return_plot=False):
        fig, ax = plt.subplots(figsize=figsize)
        # psizr = self.eqdsk["psizr"]
        # psi_mag_axis = self.eqdsk["simag"]
        # psi_boundary = self.eqdsk["sibry"]

        # ## THIS NEEDS TO BE TOROIDAL RHO
        # # normalize the psirz so that the norm is 1 on boundary and zero on axis
        # psirzNorm = (psizr - psi_mag_axis)/(psi_boundary-psi_mag_axis)
        # ax.axis("equal")
        # # img = ax.contour(self.eqdsk["r"], self.eqdsk["z"], psirzNorm.T, levels=levels, colors='black')
        ax.axis("equal")
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
        if return_plot:
            return fig, ax

        plt.show()
    
    def parse_mcgo_nc(self):
        self.rho_grid = self.mcgo_nc.variables['rho_sqpolflx'][:] #Norm-ed rho~sqrt(pol.flux) at radbnd
        self.R_f_grid = self.mcgo_nc.variables['radbnd'][:] # R bins corrisponding to self.rho_grid. 
        self.vdstb = self.mcgo_nc.variables['vdstb'][:] #Midplane distr.func aver over [tim_fdist_1;tim_fdist_2]
        self.vbnd = self.mcgo_nc.variables['vbnd'][:]   #Vel. grid [m/s] for distr. func.
        self.vmax = np.max(self.vbnd)
        self.ptchbnd = self.mcgo_nc.variables['ptchbnd'][:] #Pitch angle grid [rad] for distr. func.
        cosy = np.asmatrix(np.cos(self.ptchbnd)) #make a matrix (1,iptchbnd) {not same as vector}
        siny =np.asmatrix(np.sin(self.ptchbnd)) #make a matrix (1,iptchbnd) {not same as vector}
        xx =np.asmatrix(self.vbnd).T # make a matrix (1,ivbnd)   [m/s]
        self.X = np.dot(xx, cosy)   # (ivbnd, iptchbnd) matrix
        self.Y = np.dot(xx, siny)    # (ivbnd, iptchbnd) matrix

        if self.particle_lists_on:
            self.rend= self.mcgo_nc.variables['rend'][:] #[m] R-coord at t=tend
            self.zend= self.mcgo_nc.variables['zend'][:] #[m] Z-coord at t=tend
            self.vparend= self.mcgo_nc.variables['vparend'][:] #[m/s] Vpar at t=tend
            self.vperend= self.mcgo_nc.variables['vperend'][:] #[m/s] Vper at t=tend
            self.ivparini= self.mcgo_nc.variables['ivparini'][:] #sign of Vpar at t=0


    def get_rho_index(self, rho):
        """helper function to return the nearnest rho grid index for a particular rho

        Parameters
        ----------
        rho : float
            the desired rya for which to grab the nearest index

        Returns
        -------
        int
            index of the nearest rho point
        """
        return np.where(np.abs(self.rho_grid - rho) == min(np.abs(self.rho_grid - rho)))[0][0]
    

    def get_distribution_function_at_rho(self, rho_index):
        """grabs the distribution function for a specific general species

        Parameters
        ----------
        gen_species_index : int
            index of the desired general species

        Returns
        -------
        tuple of np arrays
            first index is distribution function of species gen_species_index at rho_index
            second index is mesh of parrallel velocities for each point in distribution function
            third index is mesh of perp velocities for each point in distribution function
        """
        f_s_rho = self.vdstb[rho_index, :, :]

        return (f_s_rho, np.asarray(self.X), np.asarray(self.Y))
    
    def cal_max_B_on_flux_surface(self, rho, tol=1e-3):
        self.build_B_midplane_mag_interpolator() # TODO self.rho_grid here is not strictly increasing like in cql, but 
        # is actually from 0.99 -> 0 -> 0.99 across all midplane R values. Need to fix this logic to handle this. 
        rho_target = rho
        Rmax = self.eqdsk['rmaxis']
        Rmin = min(self.eqdsk['rlim'])
        Zcentr = self.eqdsk['zmaxis']

        R = (Rmin+Rmax)/2
        rho_guess = np.sqrt(float(self.getpsirzNorm(R, Zcentr)))
        tola = np.abs(rho_target - rho_guess) / rho_target
        ticker = 0
        while tola > tol:
            if rho_guess >= rho_target:
                Rmin = R
                R = (Rmin+Rmax)/2
                rho_guess = np.sqrt(float(self.getpsirzNorm(R, Zcentr)))
            else:
                Rmax = R
                R = (Rmin+Rmax)/2
                rho_guess = np.sqrt(float(self.getpsirzNorm(R, Zcentr)))

            tola = np.abs(rho_target - rho_guess) / rho_target
            ticker += 1
            if ticker > 3000:
                print('Did not converge in 3000 iterations')
                break
            # print(f'tola: {tola}') 
            # print(f'rho_guess: ', rho_guess, 'rho_target: ', rho_target)              

        Bmax = float(self.getBStrength(R, Zcentr))
        return Bmax, R, Zcentr, rho_guess
    
    def plot_distribution_function_at_rho(
        self,
        rho_index,
        v_norm_over_v_max=0.015,
        log_scale_axis_multiple=1,
        log_clip_level=1e-6,
        figsize=(18, 6),
        cmap="viridis",
        num_energy_levels=6,
        energy_levels_linear=None,
        energy_levels_log=None,
        energy_color="red",
        return_plot=False,
        use_interpolated_rho=False,
        rho_to_interpolate_to=None,
        plot_trapped_passing=False,
    ):
        """Makes a plot of the linear and log scale distribution function for species gen_species_index
        versus vperp and vparallel, normalized to vnorm.

        Parameters
        ----------
        gen_species_index : int
            index of species s to plot
        rho_index : int
            index of the radial coordinate in rya array to plot
        v_norm_over_v_max : float, optional
            x and y v/vnorm scale maximum to plot, by default 0.015
        log_scale_axis_multiple : int, optional
            multiplier for log axis scales so more of the distribution function is visable, by default 1
        figsize : tuple, optional
            figure size, by default (18, 6)
        cmap : str, optional
            color map, by default "viridis"
        num_energy_levels : int, optional
            number of energy levels to plot, by default 6
        energy_levels_linear : list of floats, optional
            if set, these keV energy values will be set on the linear plot, by default None
        energy_levels_log : list of floats, optional
            if set, these keV energy values will be set on the log plot, by default None
        energy_color : str, optional
            color of energy contours, by default 'red'
        return_plot : bool, optional
            if true, returns fig and axs objects for user manipulation, by default False

        Returns
        -------
        matplotlib fig and ax objects
            The fig and ax objects for user manual manipulation
        """
        if use_interpolated_rho == False:
            f_s_rho, VPAR, VPERP = self.get_distribution_function_at_rho(rho_index)
        else:
            raise ValueError('Arb. rho not implimented yet')
            # f_s_rho, VPAR, VPERP = (
            #     self.get_species_distribution_function_at_arbitrary_rho(
            #         gen_species_index, rho=rho_to_interpolate_to
            #     )
            # )

        # calculate energy in keV of the ions
        mass_ion = self.species_dict[self.species]['mass']
        E_ion = (
            0.5
            * mass_ion
            * (VPAR**2 + VPERP**2)
            / 1.6022e-19
            / 1000
        )
        E_max_plot = (
            0.5
            * mass_ion
            * (v_norm_over_v_max*self.vmax) ** 2
            / 1.6022e-19
            / 1000
        )
        levels_linear_E = np.linspace(0, E_max_plot, num_energy_levels).tolist()
        levels_log_E = np.linspace(
            0, E_max_plot * log_scale_axis_multiple**2, num_energy_levels
        ).tolist()

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # linear scale subplot
        if use_interpolated_rho == False:
            axs[0].set_title(
                f"Distribution function"
                + r" at $\rho$"
                + f" = {self.rho_grid[rho_index]:.4f}\nR = {self.R_f_grid[rho_index]:.3f} m\nfor Species {self.species}"
            )
        else:
            axs[0].set_title(
                f"Distribution function (interpolated)"
                + r" at $\rho$"
                + f" = {rho_to_interpolate_to:.4f} for Species {self.species}"
            )

        axs[0].set_aspect("equal")
        axs[0].set_xlabel("$v_\parallel / v_{max}$")
        axs[0].set_ylabel("$v_\perp / v_{max}$")
        axs[0].set_xlim([-v_norm_over_v_max, v_norm_over_v_max])
        axs[0].set_ylim([0, v_norm_over_v_max])
        c1 = axs[0].contourf(VPAR/self.vmax, VPERP/self.vmax, f_s_rho, levels=400, cmap=cmap)
        axs[0].contour(VPAR/self.vmax, VPERP/self.vmax, f_s_rho, levels=20, colors='black', linewidths=0.5)
        if energy_levels_linear == None:
            contour_lines1 = axs[0].contour(
                VPAR/self.vmax, VPERP/self.vmax, E_ion, levels=levels_linear_E, colors=energy_color
            )
        else:
            contour_lines1 = axs[0].contour(
                VPAR/self.vmax, VPERP/self.vmax, E_ion, levels=energy_levels_linear, colors=energy_color
            )

        axs[0].clabel(
            contour_lines1, inline=True, fontsize=8, fmt=lambda x: f"{x:.1f} [keV]"
        )
        fig.colorbar(
            c1, ax=axs[0], label=r"$f_s$ [$\frac{v_{norm}^3}{(cm^3*(cm/sec)^3)}$]"
        )

        # log10 scale subplot
        if use_interpolated_rho == False:
            axs[1].set_title(
                f"LOG10 Distribution function"
                + r" at $\rho$"
                + f" = {self.rho_grid[rho_index]:.4f}\nR = {self.R_f_grid[rho_index]:.3f} m\nfor Species {self.species}"
            )
        else:
            axs[1].set_title(
                f"LOG10 Distribution function (interpolated)"
                + r" at $\rho$"
                + f" = {rho_to_interpolate_to:.4f} for Species {self.species}"
            )

        axs[1].set_aspect("equal")
        axs[1].set_xlabel("$v_\parallel / v_{norm}$")
        axs[1].set_ylabel("$v_\perp / v_{norm}$")
        axs[1].set_xlim(
            [
                -v_norm_over_v_max * log_scale_axis_multiple,
                v_norm_over_v_max * log_scale_axis_multiple,
            ]
        )
        axs[1].set_ylim([0, v_norm_over_v_max * log_scale_axis_multiple])
        if energy_levels_log == None:
            contour_lines2 = axs[1].contour(
                VPAR/self.vmax, VPERP/self.vmax, E_ion, levels=levels_log_E, colors=energy_color
            )
        else:
            contour_lines2 = axs[1].contour(
                VPAR/self.vmax, VPERP/self.vmax, E_ion, levels=energy_levels_log, colors=energy_color
            )

        axs[1].clabel(
            contour_lines2, inline=True, fontsize=8, fmt=lambda x: f"{x:.1f} [keV]"
        )
        c2 = axs[1].contourf(VPAR/self.vmax, VPERP/self.vmax, np.log10(np.clip(f_s_rho, log_clip_level, None)), levels=400, cmap=cmap)
        fig.colorbar(
            c2,
            ax=axs[1],
            label=r"LOG10($f_s$) [$\frac{v_{norm}^3}{(cm^3*(cm/sec)^3)}$]",
        )
        axs[1].contour(VPAR/self.vmax, VPERP/self.vmax, np.log10(np.clip(f_s_rho, log_clip_level, None)), levels=20, colors='black', linewidths=0.5)
        if use_interpolated_rho == False:
            rho = self.rho_grid[rho_index]
        else:
            rho=rho_to_interpolate_to
 
        if plot_trapped_passing:
            B_pi = self.cal_max_B_on_flux_surface(rho, tol=1e-3)[0]
            B0 = self.B_midplane_mag_interpolator(rho) 
            dB = (B_pi - B0) / B0
            slope = np.sqrt(1/dB)
            axs[0].axline( (0,0), None, slope=slope, color='r', linestyle='--') 
            axs[0].axline( (0,0), None, slope=-slope, color='r', linestyle='--')
            axs[1].axline( (0,0), None, slope=slope, color='r', linestyle='--') 
            axs[1].axline( (0,0), None, slope=-slope, color='r', linestyle='--')

            

        if return_plot == True:
            return fig, axs
        plt.show()   

    def plot_particle_end_RZ(self, figsize=(10,10), levels=10, fontsize=14, dotsize=2, return_plot=False):

        # plot the equilibrium
        fig, ax = self.plot_equilibrium(
            figsize=figsize, levels=levels, fontsize=fontsize, return_plot=True
        )

        kp=np.where(self.ivparini>0) # RED:  Vpar>0 at t=0
        kn=np.where(self.ivparini<0) # BLUE: Vpar<0 at t=0
        plt.plot(self.rend[kp],self.zend[kp],'r.',markersize=dotsize)  #Large arrays; consider stride
        plt.plot(self.rend[kn],self.zend[kn],'b.',markersize=dotsize)  #Large arrays; consider stride 
        ax.grid()
        if return_plot:
            return fig, ax

        plt.show()

    def update_fortran_input_deck(self,
        input_file,
        updates,
        out_file
        ):
        """Takes an mcgo input files and allows the user to update variables and profiles. 

        Parameters
        ----------
        input_file : path str
            path to mcgo input file, used as the template 
        updates : dict 
            a dictunary with string keys matching the mcgo input file variables, and values of what to replace them with.
            examples of dict pairs: 
           
            updates = {'mf': [120],
                    'rte': [1e5]*100,
                    'rni(1,1)': [2e5]*100},
                    "namei": ["a", "b", "z"],

        out_file : path str
            path to the output file where the new mcgo input file is written. 
        """

        # Load the deck from file
        with open(input_file, "r") as f:
            input_lines = f.readlines()

        output_lines = []
        i = 0

        while i < len(input_lines):
            line = input_lines[i].rstrip()
            leading_spaces = re.match(r"^(\s*)", line).group(1)
            indexed_match = re.match(r"^(\s*)(\w+\(\d+,\d+\))\s*=", line)
            if indexed_match:
                leading_ws, full_key = indexed_match.groups()
                if full_key in updates:
                    values = updates[full_key]
                    chunk_size = 5
                    chunks = [
                        leading_ws + '   ' + '   '.join(f"{v:.6E}" for v in values[i:i+chunk_size]) + '\n'
                        for i in range(0, len(values), chunk_size)
                    ]
                    # First line uses the assignment
                    output_lines.append(f"{leading_ws}{full_key}= {chunks[0].lstrip()}")
                    output_lines.extend(chunks[1:])
                    i += 1
                    assignment_start = re.compile(r"^\s*\w+(\([^\)]*\))?\s*=")
                    while i < len(input_lines) and not assignment_start.match(input_lines[i]):
                        i += 1
                    continue
                
            # Match varname = value [! comment]
            block_match = re.match(r"^(\w+)\s*=\s*([^!]*)(!?)(.*)", line.lstrip())
            if block_match:
                varname, existing_value, exclam, comment = block_match.groups()
                if varname in updates:
                    new_value = updates[varname]
                    comment_part = f" {exclam}{comment}" if exclam else ""

                    # Handle string or list-of-strings
                    if isinstance(new_value, str):
                        output_lines.append(leading_spaces + f"{varname}= '{new_value}'{comment_part}\n")
                    elif isinstance(new_value, list) and all(isinstance(v, str) for v in new_value):
                        joined = ','.join(f"'{v}'" for v in new_value)
                        output_lines.append(leading_spaces + f"{varname}= {joined}{comment_part}\n")
                    elif isinstance(new_value, list) and len(new_value) > 1:
                        # Float block array (e.g., te)
                        chunk_size = 5
                        chunks = [
                            '   ' + '   '.join(f"{v:.6E}" for v in new_value[i:i+chunk_size]) + '\n'
                            for i in range(0, len(new_value), chunk_size)
                        ]
                        output_lines.append(leading_spaces + f"{varname}= {chunks[0].strip()}\n")
                        output_lines.extend(chunks[1:])
                    else:
                        # Single float
                        val = new_value[0] if isinstance(new_value, list) else new_value
                        output_lines.append(leading_spaces + f"{varname}= {val}{comment_part}\n")

                    i += 1
                    while i < len(input_lines) and re.match(r"^\s*[\d.Ee\+\-\*]+", input_lines[i].strip()):
                        i += 1
                    continue

            # Default: preserve original
            output_lines.append(input_lines[i])
            i += 1

        # Save to a new file
        with open(out_file, "w") as f:
            f.writelines(output_lines)
    
    def load_p2f_output(self, p2f_out_file):
        self.p2f_out = netCDF4.Dataset(p2f_out_file, "r")
        # load in the 4D distribution and its grid once 
        self.p2f_f_vpar_vper_z_r = self.p2f_out.variables['f_rzvv'][:]
        self.p2f_rbin_centers = self.p2f_out.variables['R_binCenters'][:]
        self.p2f_zbin_centers = self.p2f_out.variables['z_binCenters'][:]
        self.p2f_vperp_bin_centers = self.p2f_out.variables['vPer_binCenters'][:]
        self.p2f_vpar_bin_centers = self.p2f_out.variables['vPar_binCenters'][:]

        self.p2f_vmax = np.max(np.union1d(self.p2f_vperp_bin_centers, self.p2f_vpar_bin_centers))
    
    def p2f_plot_density_RZ(self, figsize=(5,5), cmap='jet', return_plot=False):
        density = self.p2f_out.variables['density'][:]

        fig, ax = plt.subplots(figsize=(5,5))

        ax.axis('equal')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        ax.set_title('P2F Density')

        # convert from 1/cm^3 to 1/m^3
        contour = ax.contourf(self.p2f_rbin_centers, self.p2f_zbin_centers, density*1e6, levels=100, cmap=cmap)
        fig.colorbar(contour, ax=ax, label=r'Density [m$^{-3}$]')

        if return_plot:
            return fig, ax

        plt.show()     

    def p2f_get_distribution_function_at_RZ(self, indexR, indexZ):
        """Grabs the f(vpar, vperp) at radial bin center R[indexR], axial bin center Z[indexZ]
            creates meshgrid of velocity space and returns these meshes as well as the R and Z location in [m]. 

        Parameters
        ----------
        indexR : int
            R index 
        indexZ : int
            Z index

        Returns
        -------
        2d np array, 2d np array, 2d np array, float, float
            f(vpar, vperp), VPAR, VPER, R [m], Z [m]
        """
        VPAR, VPER = np.meshgrid(self.p2f_vpar_bin_centers, self.p2f_vperp_bin_centers, indexing='ij')
        return self.p2f_f_vpar_vper_z_r[:,:, indexZ, indexR], VPAR, VPER, self.p2f_rbin_centers[indexR], self.p2f_zbin_centers[indexZ]

    def p2f_plot_distribution_function_at_RZ(
        self,
        R_index,
        Z_index,
        v_norm_over_v_max=0.015,
        log_scale_axis_multiple=1,
        log_clip_level=1e-6,
        figsize=(18, 6),
        cmap="viridis",
        num_energy_levels=6,
        energy_levels_linear=None,
        energy_levels_log=None,
        energy_color="red",
        return_plot=False,
        use_interpolated_rho=False,
        rho_to_interpolate_to=None,
        plot_trapped_passing=False,
    ):
        """Makes a plot of the linear and log scale p2f distribution function for species
        versus vperp and vparallel, normalized to vnorm.

        Parameters
        ----------
        gen_species_index : int
            index of species s to plot
        R_index : int
            index of the radial coordinate in p2f Rbin centers array to plot
        Z_index : int
            index of the z coordinate in p2f zbin centers array to plot
        v_norm_over_v_max : float, optional
            x and y v/vnorm scale maximum to plot, by default 0.015
        log_scale_axis_multiple : int, optional
            multiplier for log axis scales so more of the distribution function is visable, by default 1
        figsize : tuple, optional
            figure size, by default (18, 6)
        cmap : str, optional
            color map, by default "viridis"
        num_energy_levels : int, optional
            number of energy levels to plot, by default 6
        energy_levels_linear : list of floats, optional
            if set, these keV energy values will be set on the linear plot, by default None
        energy_levels_log : list of floats, optional
            if set, these keV energy values will be set on the log plot, by default None
        energy_color : str, optional
            color of energy contours, by default 'red'
        return_plot : bool, optional
            if true, returns fig and axs objects for user manipulation, by default False

        Returns
        -------
        matplotlib fig and ax objects
            The fig and ax objects for user manual manipulation
        """
        # if use_interpolated_RZ == False:
        #     f_s_rho, VPAR, VPERP = self.get_distribution_function_at_rho(rho_index)
        # else:
        #     raise ValueError('Arb. rho not implimented yet')
        #     # f_s_rho, VPAR, VPERP = (
        #     #     self.get_species_distribution_function_at_arbitrary_rho(
        #     #         gen_species_index, rho=rho_to_interpolate_to
        #     #     )
        #     # )
        f_s_RZ, VPAR, VPERP, R, Z = self.p2f_get_distribution_function_at_RZ(indexR=R_index, indexZ=Z_index)

        # calculate energy in keV of the ions
        mass_ion = self.species_dict[self.species]['mass']
        E_ion = (
            0.5
            * mass_ion
            * (VPAR**2 + VPERP**2)
            / 1.6022e-19
            / 1000
        )
        E_max_plot = (
            0.5
            * mass_ion
            * (v_norm_over_v_max*self.vmax) ** 2
            / 1.6022e-19
            / 1000
        )
        levels_linear_E = np.linspace(0, E_max_plot, num_energy_levels).tolist()
        levels_log_E = np.linspace(
            0, E_max_plot * log_scale_axis_multiple**2, num_energy_levels
        ).tolist()

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # linear scale subplot
        axs[0].set_title(r'P2F f(v$_{||}$,v$_{perp}$)'+ f'\nR={R:.2f} m, Z={Z:.2f} m\n Species: {self.species}')

        axs[0].set_aspect("equal")
        axs[0].set_xlabel("$v_\parallel / v_{max}$")
        axs[0].set_ylabel("$v_\perp / v_{max}$")
        axs[0].set_xlim([-v_norm_over_v_max, v_norm_over_v_max])
        axs[0].set_ylim([0, v_norm_over_v_max])
        c1 = axs[0].contourf(VPAR/self.vmax, VPERP/self.vmax, f_s_RZ, levels=400, cmap=cmap)
        axs[0].contour(VPAR/self.vmax, VPERP/self.vmax, f_s_RZ, levels=20, colors='black', linewidths=0.5)
        if energy_levels_linear == None:
            contour_lines1 = axs[0].contour(
                VPAR/self.vmax, VPERP/self.vmax, E_ion, levels=levels_linear_E, colors=energy_color
            )
        else:
            contour_lines1 = axs[0].contour(
                VPAR/self.vmax, VPERP/self.vmax, E_ion, levels=energy_levels_linear, colors=energy_color
            )

        axs[0].clabel(
            contour_lines1, inline=True, fontsize=8, fmt=lambda x: f"{x:.1f} [keV]"
        )
        fig.colorbar(
            c1, ax=axs[0], label=r"$f_s$ [$\frac{v_{norm}^3}{(cm^3*(cm/sec)^3)}$]"
        )

        # log10 scale subplot

        axs[1].set_title(
            f"LOG10 P2F Distribution function "
            + r'f(v$_{||}$,v$_{perp}$)'+ f'\nR={R:.2f} m, Z={Z:.2f} m\n Species: {self.species}'
        )


        axs[1].set_aspect("equal")
        axs[1].set_xlabel("$v_\parallel / v_{max}$")
        axs[1].set_ylabel("$v_\perp / v_{max}$")
        axs[1].set_xlim(
            [
                -v_norm_over_v_max * log_scale_axis_multiple,
                v_norm_over_v_max * log_scale_axis_multiple,
            ]
        )
        axs[1].set_ylim([0, v_norm_over_v_max * log_scale_axis_multiple])
        if energy_levels_log == None:
            contour_lines2 = axs[1].contour(
                VPAR/self.vmax, VPERP/self.vmax, E_ion, levels=levels_log_E, colors=energy_color
            )
        else:
            contour_lines2 = axs[1].contour(
                VPAR/self.vmax, VPERP/self.vmax, E_ion, levels=energy_levels_log, colors=energy_color
            )

        axs[1].clabel(
            contour_lines2, inline=True, fontsize=8, fmt=lambda x: f"{x:.1f} [keV]"
        )
        c2 = axs[1].contourf(VPAR/self.vmax, VPERP/self.vmax, np.log10(np.clip(f_s_RZ, log_clip_level, None)), levels=400, cmap=cmap)
        fig.colorbar(
            c2,
            ax=axs[1],
            label=r"LOG10($f_s$) [$\frac{v_{norm}^3}{(cm^3*(cm/sec)^3)}$]",
        )
        axs[1].contour(VPAR/self.vmax, VPERP/self.vmax, np.log10(np.clip(f_s_RZ, log_clip_level, None)), levels=20, colors='black', linewidths=0.5)
        # if use_interpolated_rho == False:
        #     rho = self.rho_grid[rho_index]
        # else:
        #     rho=rho_to_interpolate_to
 
        # if plot_trapped_passing:
        #     B_pi = self.cal_max_B_on_flux_surface(rho, tol=1e-3)[0]
        #     B0 = self.B_midplane_mag_interpolator(rho) 
        #     dB = (B_pi - B0) / B0
        #     slope = np.sqrt(1/dB)
        #     axs[0].axline( (0,0), None, slope=slope, color='r', linestyle='--') 
        #     axs[0].axline( (0,0), None, slope=-slope, color='r', linestyle='--')
        #     axs[1].axline( (0,0), None, slope=slope, color='r', linestyle='--') 
        #     axs[1].axline( (0,0), None, slope=-slope, color='r', linestyle='--')


        if return_plot == True:
            return fig, axs
        plt.show()  

    def p2f_convert_mcgo_particle_list_to_p2f_particle_list(self, p2f_particle_list_filename):
        """
        Save 1D arrays of particle data to a NetCDF file for use with p2f.

        Parameters
        ----------
        p2f_particle_list_filename : str
            Output path to NetCDF file (e.g., "particles.nc").
        """
        # read in mcgo end particle lists
        r = self.mcgo_nc.variables['rend'][:] #[m] R-coord at t=tend
        z = self.mcgo_nc.variables['zend'][:] #[m] Z-coord at t=tend
        vpar= self.mcgo_nc.variables['vparend'][:] #[m/s] Vpar at t=tend
        vperp= self.mcgo_nc.variables['vperend'][:] #[m/s] Vper at t=tend
        weight = np.ones_like(r) # TODO for now, just assume the weights are one. Not sure how they should be set 

        
        assert all(len(arr) == len(r) for arr in [z, vperp, vpar, weight]), "All input arrays must be the same length"
        nP = len(r)

        with netCDF4.Dataset(p2f_particle_list_filename, "w", format="NETCDF4") as nc:
            # Define the dimension
            nc.createDimension("nP", nP)

            # Create variables that p2f expects for reading 
            r_var      = nc.createVariable("R", "f4", ("nP",))
            z_var      = nc.createVariable("z", "f4", ("nP",))
            vperp_var  = nc.createVariable("vPer", "f4", ("nP",))
            vpar_var   = nc.createVariable("vPar", "f4", ("nP",))
            weight_var = nc.createVariable("weight", "f4", ("nP",))

            # Write data
            r_var[:]      = r
            z_var[:]      = z
            vperp_var[:]  = vperp
            vpar_var[:]   = vpar
            weight_var[:] = weight

            # Optional: add metadata
            nc.title = "Particle list for p2f"
            nc.description = "Generated from MCGO output."
            nc.nP = nP         
        
        print(f'File saved to {p2f_particle_list_filename}\n num particles: {nP}')


