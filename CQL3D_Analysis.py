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
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.optimize import curve_fit
import os, sys
import netCDF4
import f90nml as f90

# import John's toolkit area
import plasma
from plasma import equilibrium_process

# import Grant's eqdsk processor for getting B info
from process_eqdsk2 import getGfileDict


class CQL3D_Post_Process:
    """
    Class to post-process the various output files from an Aorsa run
    """

    def __init__(
        self, gen_species_names, cql3d_nc_file, cql3d_krf_file=None, eqdsk_file=None, cql_input_file=None
    ):
        self.cql3d_nc_file = cql3d_nc_file
        self.cql3d_krf_file = cql3d_krf_file
        self.eqdsk_file = eqdsk_file
        self.cql_input_file = cql_input_file
        self.gen_species_names = gen_species_names

        # load up eqdsk using john's methods
        if eqdsk_file is not None:
            self.process_eqdsk()

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

        # build the powers map for use in powers functions 
        self.power_type_map = {} # from powers in ncdump -c cql3d.nc
        self.power_type_map["collisions with Maxw electrons"] = 0
        self.power_type_map["collisions with Maxw ions"] = 1
        self.power_type_map["Ohmic E.v"] = 2
        self.power_type_map["collisions with general spec"] = 3
        self.power_type_map["RF power"] = 4
        self.power_type_map["Ion particle source"] = 5 #(NB power for example)
        self.power_type_map["losses by lossmode"] = 6
        self.power_type_map["losses by torloss"] = 7
        self.power_type_map["Runaway losses"] = 8
        self.power_type_map["Synchrotron radiation losses"] = 9

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
            Bstrength = np.sqrt(np.square(B_zGrid) + np.square(B_TGrid) + np.square(B_rGrid))

            # create a function that can grab the B-feild magnitude at any r, z coordiante pair. 
            self.getBStrength = RectBivariateSpline(rgrid, zgrid, Bstrength)


    def plot_equilibrium(self, figsize, levels=10, return_plot=False):
        fig, ax = plt.subplots(figsize=figsize)
        psizr = self.eqdsk["psizr"]
        psi_mag_axis = self.eqdsk["simag"]
        psi_boundary = self.eqdsk["sibry"]
    
        ## THIS NEEDS TO BE TOROIDAL RHO
        # normalize the psirz so that the norm is 1 on boundary and zero on axis 
        psirzNorm = (psizr - psi_mag_axis)/(psi_boundary-psi_mag_axis) 
        ax.axis("equal")
        img = ax.contour(self.eqdsk["r"], self.eqdsk["z"], psirzNorm.T, levels=levels, colors='black')
        ax.plot(self.eqdsk["rlim"], self.eqdsk["zlim"], color="black", linewidth=3)
        ax.plot(self.eqdsk["rbbbs"], self.eqdsk["zbbbs"], color="black", linewidth=3)
        if return_plot:
            return fig, ax

        plt.show()

    def print_keys(self):
        print("The cql3d.nc file has keys")
        print(self.cql_nc.keys())
        if self.cql3d_krf_file != None:
            print("\n The cql3d_krf.nc file has keys")
            print(self.cqlrf.keys())

    def parse_cql_nc(self):
        # radial coordiante
        self.rya = np.ma.getdata(self.cql_nc.variables["rya"][:])

        # pitch angles mesh at which f is defined in radians.
        # Note that np.ma.getdata pulls data through mask which
        # rejects bad data (NAN, etc)
        self.pitchAngleMesh = np.ma.getdata(self.cql_nc.variables["y"][:])

        # normalized speed mesh of f v/vnorm
        self.normalizedVel = self.cql_nc.variables["x"][:]

        # energy mesh in keV
        self.enerkev = self.cql_nc.variables["enerkev"][:]

        # flux surface average energy per particle in keV
        self.energy = self.cql_nc.variables["energy"][:]

        try:
            self.ebkev = self.cql_nml["frsetup"]["ebkev"][0]
            # TODO for now just take the first beam energy
        except:
            pass

        # distribution function in units "vnorm**3/(cm**3*(cm/sec)**3)"
        self.f = self.cql_nc.variables["f"][:]  # get the ditrobution

        # vnorm in cm/s as it is in cql
        self.vnorm_cm_per_second = self.cql_nc.variables["vnorm"][:].data
        # convern vnorm to m/s (its cm/s in cql)
        self.vnorm_m_per_second = self.cql_nc.variables["vnorm"][:].data / 100

        # species masses
        self.species_mass = self.cql_nc.variables["fmass"][:] / 1000  # convert to kg

        # species charges
        self.species_charges = self.cql_nc["bnumb"][:].data*1.6022e-19

        # volume elements for each radial bin in cm^3
        self.dvols = self.cql_nc.variables['dvol'][:]

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

    def get_species_distribution_function_at_rho(self, gen_species_index, rho_index):
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

    def plot_species_distribution_function_at_rho(
        self,
        gen_species_index,
        rho_index,
        v_norm_over_v_max=0.015,
        log_scale_axis_multiple=1,
        figsize=(18, 6),
        cmap="viridis",
        num_energy_levels=6,
        energy_levels_linear=None,
        energy_levels_log=None,
        energy_color="red",
        return_plot=False,
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
        f_s_rho, VPAR, VPERP = self.get_species_distribution_function_at_rho(
            gen_species_index, rho_index
        )

        # calculate energy in keV of the ions
        mass_ion = self.species_mass[gen_species_index]
        E_ion = (
            0.5
            * mass_ion
            * (VPAR**2 + VPERP**2)
            * self.vnorm_m_per_second**2
            / 1.6022e-19
            / 1000
        )
        E_max_plot = (
            0.5
            * mass_ion
            * (self.vnorm_m_per_second * v_norm_over_v_max) ** 2
            / 1.6022e-19
            / 1000
        )
        levels_linear_E = np.linspace(0, E_max_plot, num_energy_levels).tolist()
        levels_log_E = np.linspace(
            0, E_max_plot * log_scale_axis_multiple**2, num_energy_levels
        ).tolist()

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # linear scale subplot
        axs[0].set_title(
            f"Distribution function"
            + r" at $\rho$"
            + f" = {self.rya[rho_index]:.3f} for Species {self.get_species_name(gen_species_index)}"
        )
        axs[0].set_aspect("equal")
        axs[0].set_xlabel("$v_\parallel / v_{norm}$")
        axs[0].set_ylabel("$v_\perp / v_{norm}$")
        axs[0].set_xlim([-v_norm_over_v_max, v_norm_over_v_max])
        axs[0].set_ylim([0, v_norm_over_v_max])
        c1 = axs[0].contourf(VPAR, VPERP, f_s_rho, levels=400, cmap=cmap)
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
        axs[1].set_title(
            f"LOG10 Distribution function"
            + r" at $\rho$"
            + f" = {self.rya[rho_index]:.3f} for Species {self.get_species_name(gen_species_index)}"
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
                VPAR, VPERP, E_ion, levels=levels_log_E, colors=energy_color
            )
        else:
            contour_lines2 = axs[1].contour(
                VPAR, VPERP, E_ion, levels=energy_levels_log, colors=energy_color
            )

        axs[1].clabel(
            contour_lines2, inline=True, fontsize=8, fmt=lambda x: f"{x:.1f} [keV]"
        )
        c2 = axs[1].contourf(VPAR, VPERP, np.log10(f_s_rho + 1), levels=400, cmap=cmap)
        fig.colorbar(
            c2,
            ax=axs[1],
            label=r"LOG10($f_s$+1) [$\frac{v_{norm}^3}{(cm^3*(cm/sec)^3)}$]",
        )

        if return_plot == True:
            return fig, axs
        plt.show()

    def integrate_distribution_over_pitch_angle(self, gen_species_index, rho_index):
        f_s_rho = self.get_species_distribution_function_at_rho(
            gen_species_index=gen_species_index, rho_index=rho_index
        )[0]

        # at this rho index, grab the grid of 2pi sin(theta) dtheta where theta = y is the pitch angle. 2pi comes from integrating over gyroangle already
        two_pi_siny_dy = np.ma.getdata(self.cql_nc.variables["cynt2"])[rho_index]

        # this converts f from "vnorm**3/(cm**3*(cm/sec)**3)"  and integrates over pitch and makes si units
        # for f_integrated_over_pitch in units of 1 / (m^3 m/s)
        f_integrated_over_pitch = (
            100**4
            * np.trapz(f_s_rho * two_pi_siny_dy, axis=1)
            * self.normalizedVel**2
            / self.vnorm_cm_per_second
        )  # SI units

        # should now be able to plot f_integrated_over_pitch vs self.enerkev to compare to a maxwellian distribution

        return f_integrated_over_pitch, self.enerkev
    
    def integrate_distribution_function_over_velocity_space(self, gen_species_index, rho_index):
        f_integrated_over_pitch = self.integrate_distribution_over_pitch_angle(gen_species_index, rho_index)[0]
        mass = self.cql_nc["fmass"][gen_species_index]/1000 # mass in grams
        velocity_grid = np.sqrt(2 * self.enerkev * 1e3 * 1.6022e-19 / mass)  # m/s
        density_per_m3 = np.trapz(f_integrated_over_pitch, velocity_grid)
        return density_per_m3
    
    def get_density_profile(self, gen_species_index):
        density = np.zeros_like(self.rya)
        for i in range(self.rya.shape[0]):
            density[i] = self.integrate_distribution_function_over_velocity_space(gen_species_index, i)
        return density, self.rya

    def plot_density_profile(self, gen_species_index, figsize=(10,5), color='red', return_plot=False):
        fig, ax = plt.subplots(figsize=figsize)
        density, rya = self.get_density_profile(gen_species_index)
        ax.plot(rya, density, color=color)
        ax.grid()
        ax.set_xlabel(r'$\rho$', fontsize=15)
        ax.set_ylabel(f'Density of species {self.get_species_name(gen_species_index)} '+r'[$m^{-3}$]', fontsize=15)
        ax.tick_params(axis="x", labelsize=12)  # For x-axis tick labels
        ax.tick_params(axis="y", labelsize=12)  # For y-axis tick labels
        if return_plot:
            return fig, ax
        plt.show()


    def make_maxwellian_on_enerkev_grid(self, n, T, mass):
        v_array = np.sqrt(2 * self.enerkev * 1e3 * 1.6022e-19 / mass)  # m/s

        T = T * 1e3 * 1.6022e-19  # J
        f_maxwell = (
            n
            * (mass / (2 * np.pi * T)) ** (3 / 2)
            * np.exp(-mass * v_array**2 / (2 * T))
        )

        # integrate over all angles
        maxwell_energy_distribution_function = 4 * np.pi * v_array**2 * f_maxwell
        return maxwell_energy_distribution_function, self.enerkev

    def get_species_name(self, gen_species_index):
        return self.gen_species_names[gen_species_index]
        

    def plot_pitch_integrated_distribution_function(
        self,
        gen_species_index,
        rho_index,
        Emax_keV,
        ylim=None,
        figsize=(10, 10),
        log10=False,
        color="blue",
        return_plot=False
    ):
        idx_max_kev = np.where(
            np.abs(self.enerkev - Emax_keV) == min(np.abs(self.enerkev - Emax_keV))
        )[0][
            0
        ]  # grab max index to plot to
        f_integrated_over_pitch = self.integrate_distribution_over_pitch_angle(
            gen_species_index, rho_index
        )[0]

        fig, ax = plt.subplots(figsize=figsize)
        ax.grid()
        if ylim is not None:
            ax.set_ylim(ylim)
        if log10:
            ax.plot(
                self.enerkev[:idx_max_kev],
                np.log10(f_integrated_over_pitch[:idx_max_kev] + 1),
                color=color,
            )
            ax.set_ylabel(r"log10(f(E)+1) $[1/m^3 m/s]$", fontsize=15)
        else:
            ax.plot(
                self.enerkev[:idx_max_kev],
                f_integrated_over_pitch[:idx_max_kev],
                color=color,
            )
            ax.set_ylabel(r"f(E) $[1/m^3 m/s]$", fontsize=15)

        ax.set_xlabel("Energy [keV]", fontsize=15)
        ax.tick_params(axis="x", labelsize=12)  # For x-axis tick labels
        ax.tick_params(axis="y", labelsize=12)  # For y-axis tick labels

        ax.set_title(
            f"Pitch-Integrated Distribution Function for Species {self.get_species_name(gen_species_index)}\n" + r" at $\rho$"
            + f" = {self.rya[rho_index]:.3f}",
            fontsize=15,
        )

        if return_plot is not False:
            return fig, ax
        plt.show()

    def plot_pitch_integrated_distribution_function_versus_maxwellian(
        self,
        nmax,
        Tmax,
        gen_species_index,
        rho_index,
        Emax_keV,
        ylim=None,
        figsize=(10, 10),
        log10=False,
        color="blue",
        maxwell_color="red",
    ):
        idx_max_kev = np.where(
            np.abs(self.enerkev - Emax_keV) == min(np.abs(self.enerkev - Emax_keV))
        )[0][0]
        # grab max index to plot to
        f_integrated_over_pitch = self.integrate_distribution_over_pitch_angle(
            gen_species_index, rho_index
        )[0]
        f_maxwellian_angle_integrated = self.make_maxwellian_on_enerkev_grid(
            nmax, Tmax, self.cql_nc["fmass"][gen_species_index]/1000 # convert grams to kg
        )[0]

        fig, ax = plt.subplots(figsize=figsize)
        ax.grid()
        if ylim is not None:
            ax.set_ylim(ylim)
        if log10:
            ax.plot(
                self.enerkev[:idx_max_kev],
                np.log10(f_integrated_over_pitch[:idx_max_kev] + 1),
                color=color, label='CQL3D'
            )
            ax.plot(
                self.enerkev[:idx_max_kev],
                np.log10(f_maxwellian_angle_integrated[:idx_max_kev] + 1),
                color=maxwell_color,
                label="Maxwellian",
                linestyle="--",
            )
            ax.set_ylabel(r"log10(f(E)+1) $[1/m^3 m/s]$", fontsize=15)
        else:
            ax.plot(
                self.enerkev[:idx_max_kev],
                f_integrated_over_pitch[:idx_max_kev],
                color=color, label='CQL3D'
            )
            ax.plot(
                self.enerkev[:idx_max_kev],
                f_maxwellian_angle_integrated[:idx_max_kev],
                color=maxwell_color,
                label="Maxwellian",
                linestyle="--",
            )
            ax.set_ylabel(r"f(E) $[1/m^3 m/s]$", fontsize=15)

        ax.set_xlabel("Energy [keV]", fontsize=15)
        ax.tick_params(axis="x", labelsize=12)  # For x-axis tick labels
        ax.tick_params(axis="y", labelsize=12)  # For y-axis tick labels
        ax.legend()

        ax.set_title(
            f"Pitch-Integrated Distribution Function for Species {self.get_species_name(gen_species_index)} \n" + r" at $\rho$"
            + f" = {self.rya[rho_index]:.3f}",
            fontsize=15,
        )

    def get_power_vs_rho(self, gen_species_index, power_type, time=-1):

        # see self.power_type_map for available power types 
        # units are W/cm^3 or equivilantly MW/m^3. 
        power = self.cql_nc.variables['powers'][time, gen_species_index, self.power_type_map[power_type], :]
        dvols_m3 = self.dvols*(1/100)**3 # convert volume elements to m^3
        total_power_MW = np.trapz(power*dvols_m3) # total power delvered to gen species in MW
        return power, total_power_MW, self.rya 
    
    def plot_powers_vs_rho(self, gen_species_index, power_types, time=-1, figsize=(10,5), colors=None, return_plot=False):

        fig, ax = plt.subplots(figsize=figsize)
        ax.grid()
        for power_type, color in zip(power_types, colors): 
            power, total_power_MW = self.get_power_vs_rho(gen_species_index, power_type, time)[:-1]
            ax.plot(self.rya, power, color=color, label=f'Power Type: {power_type} \n' + f'Total: {total_power_MW:.2f} MW')
        
        ax.set_xlabel(r"$\rho$", fontsize=15)
        ax.set_ylabel(r"Power Density [$MW/m^3$]", fontsize=15)
        ax.tick_params(axis="x", labelsize=12)  # For x-axis tick labels
        ax.tick_params(axis="y", labelsize=12)  # For y-axis tick labels
        ax.legend()

        ax.set_title(f'Net Powers to Species {self.get_species_name(gen_species_index)}')

        if return_plot == True:
            return fig, ax


    def plot_cyclotron_harmonics(self, frequency, harmonics, species_mass, species_charge, r_resolution, z_resolution, levels, figsize=(10,10), harmonic_color='blue', return_plot=False):
        """    frequency: launched wave fequency [Hz]
            harmonics: list of harmonics to plot (example: [1, 2, 3] will plot the 1st, second, and third cyclotron harmonics for species) 
            r_resolution: number of radial points to search over per z coord to plot the harmonic
            z_resolution: number of z coords. 
    
        """



        # plot the equilibrium
        fig, ax = self.plot_equilibrium(figsize=figsize, levels=levels, return_plot=True)

        # set up harmonics 
        w_wave = frequency*2*np.pi
        r_points = np.linspace(self.eqdsk_with_B_info["rgrid"][0], self.eqdsk_with_B_info["rgrid"][-1], r_resolution)
        z_points = np.linspace(self.eqdsk_with_B_info["zgrid"][0], self.eqdsk_with_B_info["zgrid"][-1], z_resolution)
        Bfield = self.getBStrength(r_points, z_points).T
 
        omega_j = species_charge*Bfield/species_mass
        print(np.max(omega_j))
        normalized_w_wave = w_wave / omega_j
        R, Z = np.meshgrid(r_points, z_points)
        print(np.max(normalized_w_wave))
        CS = ax.contour(R, Z, normalized_w_wave.T, levels=harmonics, colors=(harmonic_color,), linestyles=('--',))
        labels = ax.clabel(CS, fmt = '%2.1d', colors = harmonic_color, fontsize=20)

        for label in labels:
            label.set_rotation(0)

        if return_plot:
            return fig, ax
        plt.show()


# next, add tools for pitch-integrating and comparing to maxwellian
# add tools for plotting powers vs rya

# add tool for plotting eqdsk, rays, and harmonics from a genray run
