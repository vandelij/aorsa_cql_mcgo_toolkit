##################################
#   Classes for working with both setup and post-proccesing of MCGO
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
                + f" = {self.rho_grid[rho_index]:.4f} for Species {self.species}"
            )
        else:
            axs[0].set_title(
                f"Distribution function (interpolated)"
                + r" at $\rho$"
                + f" = {rho_to_interpolate_to:.4f} for Species {self.species}"
            )

        axs[0].set_aspect("equal")
        axs[0].set_xlabel("$v_\parallel / v_{norm}$")
        axs[0].set_ylabel("$v_\perp / v_{norm}$")
        axs[0].set_xlim([-v_norm_over_v_max*self.vmax, v_norm_over_v_max*self.vmax])
        axs[0].set_ylim([0, v_norm_over_v_max*self.vmax])
        c1 = axs[0].contourf(VPAR, VPERP, f_s_rho, levels=400, cmap=cmap)
        axs[0].contour(VPAR, VPERP, f_s_rho, levels=20, colors='black', linewidths=0.5)
        if energy_levels_linear == None:
            contour_lines1 = axs[0].contour(
                VPAR, VPERP, E_ion, levels=levels_linear_E, colors=energy_color
            )
        else:
            contour_lines1 = axs[0].contour(
                VPAR, VPERP, E_ion, levels=energy_levels_linear, colors=energy_color
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
                + f" = {self.rho_grid[rho_index]:.4f} for Species {self.species}"
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
                -v_norm_over_v_max*self.vmax * log_scale_axis_multiple,
                v_norm_over_v_max*self.vmax * log_scale_axis_multiple,
            ]
        )
        axs[1].set_ylim([0, v_norm_over_v_max*self.vmax * log_scale_axis_multiple])
        if energy_levels_log == None:
            contour_lines2 = axs[1].contour(
                VPAR, VPERP, E_ion, levels=levels_log_E, colors=energy_color
            )
        else:
            contour_lines2 = axs[1].contour(
                VPAR, VPERP, E_ion, levels=energy_levels_log, colors=energy_color
            )

        axs[1].clabel(
            contour_lines2, inline=True, fontsize=8, fmt=lambda x: f"{x:.1f} [keV]"
        )
        c2 = axs[1].contourf(VPAR, VPERP, np.log10(np.clip(f_s_rho, log_clip_level, None)), levels=400, cmap=cmap)
        fig.colorbar(
            c2,
            ax=axs[1],
            label=r"LOG10($f_s$) [$\frac{v_{norm}^3}{(cm^3*(cm/sec)^3)}$]",
        )
        axs[1].contour(VPAR, VPERP, np.log10(np.clip(f_s_rho, log_clip_level, None)), levels=20, colors='black', linewidths=0.5)
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
