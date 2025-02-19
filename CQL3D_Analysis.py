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
from plasma import equilibrium_process


class CQL3D_Post_Process:
    """
    Class to post-process the various output files from an Aorsa run
    """

    def __init__(
        self, cql3d_nc_file, cql3d_krf_file=None, eqdsk_file=None, cql_input_file=None
    ):
        self.cql3d_nc_file = cql3d_nc_file
        self.cql3d_krf_file = cql3d_krf_file
        self.eqdsk_file = eqdsk_file
        self.cql_input_file = cql_input_file

        # load up eqdsk using john's methods
        self.eqdsk, fig = plasma.equilibrium_process.readGEQDSK(
            eqdsk_file, doplot=False
        )

        self.R_wall = self.eqdsk["rlim"]
        self.Z_wall = self.eqdsk["zlim"]

        self.R_lcfs = self.eqdsk["rbbbs"]
        self.Z_lcfs = self.eqdsk["zbbbs"]

        # read .nc file and create usfull data
        self.cql_nc = netCDF4.Dataset(self.cql3d_nc_file, "r")

        if self.cql3d_krf_file != None:
            self.cqlrf_nc = netCDF4.Dataset(self.cql3d_krf_file, "r")

        if self.cql_input_file != None:
            with open(self.cql_input_file, "r", encoding="latin1") as file:
                self.cql_nml = f90.read(file)
                # TODO
            # make this more robust, i get what i need for now but f90 has trouble with preallocated arrays in the namelist

        # parse
        self.parse_cql_nc()

    def plot_equilibrium(self, figsize, levels):
        psizr = self.eqdsk["psizr"]
        plt.figure(figsize=figsize)
        plt.axis("equal")
        img = plt.contour(self.eqdsk["r"], self.eqdsk["z"], psizr.T, levels=levels)
        plt.plot(self.eqdsk["rlim"], self.eqdsk["zlim"], color="black", linewidth=3)
        plt.plot(self.eqdsk["rbbbs"], self.eqdsk["zbbbs"], color="black", linewidth=3)
        plt.colorbar(img)
        plt.show()

    def print_keys(self):
        print("The cql3d.nc file has keys")
        print(self.cql_nc.keys())
        if self.cql3d_krf_file != None:
            print("\n The cql3d_krf.nc file has keys")
            print(self.cqlrf.keys())

    def parse_cql_nc(self):
        self.rya = np.ma.getdata(self.cql_nc.variables["rya"][:])

        # pitch angles mesh at which f is defined in radians.
        # Note that np.ma.getdata pulls data through mask which
        # rejects bad data (NAN, etc)
        self.pitchAngleMesh = np.ma.getdata(self.cql_nc.variables["y"][:])

        # normalized speed mesh of f
        self.normalizedVel = self.cql_nc.variables["x"][:]

        self.enerkev = self.cql_nc.variables["enerkev"][:]

        # flux surface average energy per particle in keV
        self.energy = self.cql_nc.variables["energy"][:]

        self.ebkev = self.cql_nml["frsetup"]["ebkev"][0]
        # TODO for now just take the first beam energy

        self.f = self.cql_nc.variables["f"][:]  # get the ditrobution

        # convern vnorm to m/s
        self.vnorm = self.cql_nc.variables["vnorm"][:].data / 100

        # species masses
        self.species_mass = self.cql_nc.variables["fmass"][:] / 1000  # convert to kg

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
        return np.where(np.abs(self.rya - rho) == min(np.abs(self.rya - rho)))[0][0]

    def get_species_distrobution_function_at_rho(self, gen_species_index, rho_index):
        """grabs the distrobution function for a specific general species

        Parameters
        ----------
        gen_species_index : int
            index of the desired general species

        Returns
        -------
        tuple of np arrays
            first index is distrobution function of species gen_species_index at rho_index
            second index is mesh of parrallel velocities for each point in distrobution function
            third index is mesh of perp velocities for each point in distrobution function
        """
        f_s_rho = self.f[gen_species_index, rho_index, :, :]

        # grab and create velocity-pitch angle mesh
        V, THETA = np.meshgrid(
            self.normalizedVel, self.pitchAngleMesh[rho_index, :], indexing="ij"
        )
        VPAR = V * np.cos(THETA)
        VPERP = V * np.sin(THETA)

        return (f_s_rho, VPAR, VPERP)

        # double f(gen_species_dim, rdim, xdimf, ydimf) ;
        #         f:long_name = "Distribution function" ;
        #         f:units = "vnorm**3/(cm**3*(cm/sec)**3)" ;
        #         f:comment = "Additional dimension added for multi-species" ;

    def plot_species_distrobution_function_at_rho(
        self,
        gen_species_index,
        rho_index,
        v_norm_over_v_max=0.015,
        log_scale_axis_multiple=1,
        figsize=(10, 6),
        cmap="viridis",
        num_energy_levels=6,
        energy_levels_linear=None,
        energy_levels_log=None,
        energy_color='red',
        return_plot=False,
    ):
        f_s_rho, VPAR, VPERP = self.get_species_distrobution_function_at_rho(
            gen_species_index, rho_index
        )

        # calculate energy in keV of the ions
        mass_ion = self.species_mass[gen_species_index]
        E_ion = (0.5 * mass_ion * (VPAR**2 + VPERP**2) * self.vnorm**2 / 1.6022e-19 / 1000)
        E_max_plot = 0.5*mass_ion*(self.vnorm*v_norm_over_v_max)**2  / 1.6022e-19 / 1000
        levels_linear_E = np.linspace(0, E_max_plot, num_energy_levels).tolist()
        levels_log_E = np.linspace(0, E_max_plot*log_scale_axis_multiple**2, num_energy_levels).tolist()

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # linear scale subplot
        axs[0].set_title(f"Distribution function at r = {self.rya[gen_species_index]}")
        axs[0].set_aspect("equal")
        axs[0].set_xlabel("$v_\parallel / v_{norm}$")
        axs[0].set_ylabel("$v_\perp / v_{norm}$")
        axs[0].set_xlim([-v_norm_over_v_max, v_norm_over_v_max])
        axs[0].set_ylim([0, v_norm_over_v_max])
        c1 = axs[0].contourf(VPAR, VPERP, f_s_rho, levels=400, cmap=cmap)
        if energy_levels_linear == None:
            contour_lines1 = axs[0].contour(VPAR, VPERP, E_ion, levels=levels_linear_E, colors=energy_color)
        else:
            contour_lines1 = axs[0].contour(VPAR, VPERP, E_ion, levels=energy_levels_linear, colors=energy_color)

        axs[0].clabel(contour_lines1, inline=True, fontsize=8, fmt=lambda x: f'{x:.1f} [keV]')
        fig.colorbar(
            c1, ax=axs[0], label=r"$f_s$ [$\frac{v_{norm}^3}{(cm^3*(cm/sec)^3)}$]"
        )

        # log10 scale subplot
        axs[1].set_title(
            f"LOG10 Distribution function at r = {self.rya[gen_species_index]}"
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
            contour_lines2 = axs[1].contour(VPAR, VPERP, E_ion, levels=levels_log_E, colors=energy_color)
        else:
            contour_lines2 = axs[1].contour(VPAR, VPERP, E_ion, levels=energy_levels_log, colors=energy_color)

        axs[1].clabel(contour_lines2, inline=True, fontsize=8, fmt=lambda x: f'{x:.1f} [keV]')
        c2 = axs[1].contourf(VPAR, VPERP, np.log10(f_s_rho + 1), levels=400, cmap=cmap)
        fig.colorbar(
            c2,
            ax=axs[1],
            label=r"LOG10($f_s$+1) [$\frac{v_{norm}^3}{(cm^3*(cm/sec)^3)}$]",
        )

        if return_plot == True:
            return fig, axs
        fig.show()
