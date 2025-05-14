from scipy.interpolate import PchipInterpolator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import f90nml as f90
import os
from collections.abc import Iterable
import meshio
from plasma import equilibrium_process
# import helpers for eqdsk processing
import plasma


class Aorsa_Analysis():
    """
    Class to store tools for running analysis with aorsa on Permutter. 
    
    Author: Jacob van de Lindt
    12/1/2023
    """


    def __init__(self, rho_type):
        # set whether working in 'r/a', 'rho_tor', 'rho_pol'
        allowed_rhos = ['r/a', 'rho_tor', 'rho_pol']
        if rho_type not in allowed_rhos: 
            raise ValueError(f'rho_type = {rho_type} not an accepted rho type')

        self.rho_type = rho_type

        # directory set up if using multiple machines
        self.local_work_dir = '/~'
        self.remote_host='vandelij@perlmutter.nersc.gov'
        self.remote_work_dir = '/~'
        self.aorsa2din_template = 'aorsa2d_template.in'  # name of txt file containing bare bones aorsa2d.in  
        self.save_aorsa_file_name = 'aorsa2d_modified.in'
        self.species_list = [] # list for the names of species
        self.species_mass = {}  # dictunary with corrisponding species mass [kg]
        self.species_charge = {} # dictunary with corrisponding species charge [C]
        self.species_rho = {} # species rho profile for the following density and temp profiles.  
        self.species_density = {}  # species density over rho in m^-3
        self.species_temp = {}     # species temp over rho 
        self.species_ndist = {}

        self.rho_tor = ''
        self.rho_pol = ''
    
    def set_species(self, name, mass, charge, density, temp, rho, ndisti):
        """
        name: A name for the species. 'D', 'e', etc
        mass: species mass in kg
        charge: species charge in C
        density: profile over rho  in m^-3
        temp: profile over rho in KeV 
        rho: rho profile of type self.rho_type for the density and temp profiles
        ndisti: 0 for maxwellian treatment.  1 for non-maxwellian -> requires cql3d file. 
        """
        self.species_list.append(name)
        self.species_mass[name] = mass
        self.species_charge[name] = charge
        self.species_density[name] = density
        self.species_temp[name] = temp 
        self.species_rho[name] = rho
        self.species_ndist[name] = ndisti

        if self.species_list[0] != 'e' and self.species_list[0] != 'E':
            raise ValueError(f'The first species in self.species_list must be electrons, not "{self.species_list[0]}"')

    def convert(self, rho1,rho2,prof1):
        prof1fun = PchipInterpolator(rho1,prof1)
        rhoconvfun = PchipInterpolator(rho2,rho1)
        prof2 = prof1fun(rhoconvfun(rho2))
        prof2fun = PchipInterpolator(rho2,prof2)
        return prof2fun
    
    def map_profile_to_aorsa(self, rhogiven, profgiven, type):
        """ 
        rhogiven: rho profile of type self.rho_type
        profgiven: the corrisponding profile
        type: 'den' or 'temp' (density m^-3 or temperature in KeV)
        """

        # load up pol and toroidal grids TODO: couldnt get OMFIT to work
        # when it is, these should come directly from self.eqdsk 
        rhopol = np.loadtxt(self.local_work_dir+self.rho_pol)
        rhotor = np.loadtxt(self.local_work_dir+self.rho_tor)

        if type == 'den':
            S_NRHO_N = self.aorsa_nml['STATE']['S_NRHO_N']
            rho_aorsa = self.aorsa_nml['STATE']['S_RHO_N_GRID'][:S_NRHO_N]

        elif type == 'temp':
            S_NRHO_T = self.aorsa_nml['STATE']['S_NRHO_T']
            rho_aorsa = self.aorsa_nml['STATE']['S_RHO_T_GRID'][:S_NRHO_T]           
        
        # if the input density or temp arrays are over rhotor, simply interp to aorsa's grid
        if self.rho_type == 'rho_tor':
            return PchipInterpolator(rhogiven, profgiven)(rho_aorsa)
        
        # if the input density or temp arrays are over rhopol, first convert the profile 
        # the length to be converted, then interpolate to new flux grid. 
        elif self.rho_type == 'rho_pol':
            prof_grid = PchipInterpolator(rhogiven, profgiven)(rhopol)
            return self.convert(rhopol,rhotor,prof_grid)(rho_aorsa)
            
    
    def load_template(self):
        self.aorsa_nml = f90.read(self.local_work_dir + self.aorsa2din_template)

    def plot_density(self):
        S_NRHO_N = self.aorsa_nml['STATE']['S_NRHO_N']
        S_RHO_N_GRID = self.aorsa_nml['STATE']['S_RHO_N_GRID']
        S_N_S = self.aorsa_nml['STATE']['S_N_S']
        S_S_NAME = self.aorsa_nml['STATE']['S_S_NAME']
        plt.figure(figsize=(10,5))
        rho_aorsa = S_RHO_N_GRID[:S_NRHO_N]
        # plot the species 
        for i in range(len(S_S_NAME)):
            name = S_S_NAME[i]
            sn = S_N_S[i*181:(i*181 + S_NRHO_N)]
            plt.plot(rho_aorsa, sn, label=name)
        plt.legend()
        plt.title('AORSA input file density profiles')

        if self.rho_type == 'rho_tor':
            plt.xlabel(r'$\rho_{tor}$')
        elif self.rho_type == 'rho_pol':
            plt.xlabel(r'$\rho_{pol}$')
            
        plt.ylabel(r'n [$m^3$]')
        #plt.show()

    def plot_temperature(self):
        S_NRHO_T = self.aorsa_nml['STATE']['S_NRHO_T']
        S_RHO_T_GRID = self.aorsa_nml['STATE']['S_RHO_T_GRID']
        S_T_S = self.aorsa_nml['STATE']['S_T_S']
        S_S_NAME = self.aorsa_nml['STATE']['S_S_NAME']
        plt.figure(figsize=(10,5))
        rho_aorsa = S_RHO_T_GRID[:S_NRHO_T]
        # plot the species 
        for i in range(len(S_S_NAME)):
            name = S_S_NAME[i]
            stemp = S_T_S[i*181:(i*181 + S_NRHO_T)]
            plt.plot(rho_aorsa, stemp, label=name)
        plt.legend()
        plt.title('AORSA input file temperature profiles')

        if self.rho_type == 'rho_tor':
            plt.xlabel(r'$\rho_{tor}$')
        elif self.rho_type == 'rho_pol':
            plt.xlabel(r'$\rho_{pol}$')
            
        plt.ylabel(r'T [KeV]')
        #plt.show()

    def set_state(self):
        """
        Function which sets the aorsa namelist density profiles to those specified by 
        species created through self.set_species 
        """
        charge_list = []
        mass_list = []
        # set the species names to species_list
        self.aorsa_nml['STATE']['S_S_NAME'] = self.species_list
        nden = self.aorsa_nml['STATE']['S_NRHO_N']
        ntemp = self.aorsa_nml['STATE']['S_NRHO_T']

        # clear the density and temperature fields in the state
        self.aorsa_nml['STATE']['S_T_S'] = [0]*len(self.aorsa_nml['STATE']['S_T_S'])
        self.aorsa_nml['STATE']['S_N_S'] = [0]*len(self.aorsa_nml['STATE']['S_N_S'])

        print(f'self.aorsa_nml[STATE][S_T_S] after zerowing:', self.aorsa_nml['STATE']['S_N_S'])
        print(f'loading species {self.species_list} den, temp profiles, masses, charges in to aorsa namelist dictunary.')
        for i in range(len(self.species_list)):
            name = self.species_list[i]
            rho = self.species_rho[name]
            mass_list.append(self.species_mass[name]) # mass list to be written to nml
            charge_list.append(self.species_charge[name]) # charge list to be written to nml

            den = self.species_density[name]
            temp = self.species_temp[name]

            # get and save density 
            aorsa_den_to_save = self.map_profile_to_aorsa(rho, den, type='den')  # length of species density grid per species
            self.aorsa_nml['STATE']['S_N_S'][i*181:(i*181 + nden)] = aorsa_den_to_save

            # get and save temp
            aorsa_temp_to_save = self.map_profile_to_aorsa(rho, temp, type='temp')  # length of species density grid per species
            self.aorsa_nml['STATE']['S_T_S'][i*181:(i*181 + ntemp)] = aorsa_temp_to_save

        # update the charge and mass list 
        self.aorsa_nml['STATE']['S_Q_S'] = charge_list
        self.aorsa_nml['STATE']['S_M_S'] = mass_list
    
    def setup_antenna(self, power, freq, nstrap, i_antenna, R, Z, straplength,
                      strapwidth, dist_btwn_straps, npi_array, 
                      d_psi_ant=0.025, antlc=1.9, 
                      nphi_number=1, phase_array=[0., 180.0, 0., 180.0]):
        print('updating antenna perameters...')
        self.aorsa_nml['aorsa2din']['prfin'] = power #launched power [W]
        self.aorsa_nml['aorsa2din']['freqcy'] = freq # frequency [Hz]
        self.aorsa_nml['aorsa2din']['nstrap'] = nstrap # number of straps
        self.aorsa_nml['aorsa2din']['rant'] = R # location of the antenna in major radius R [m]
        self.aorsa_nml['aorsa2din']['yant'] = Z # location of the antenna in height vs midplane Z [m]
        self.aorsa_nml['aorsa2din']['antlen'] = straplength # length of antenna straps [m]
        self.aorsa_nml['aorsa2din']['xlt'] = strapwidth # width of straps [m]
        self.aorsa_nml['aorsa2din']['wd'] = dist_btwn_straps # distance between straps [m]
        self.aorsa_nml['aorsa2din']['dpsiant0'] = d_psi_ant # ??? 
        self.aorsa_nml['aorsa2din']['antlc'] = antlc # propagation constant of antenna c/vphase
        self.aorsa_nml['aorsa2din']['nphi_number'] = nphi_number # number of toroidal modes
        self.aorsa_nml['aorsa2din']['nphi_array'] = npi_array # !toroidal mode number
        self.aorsa_nml['aorsa2din']['phase_array'] = phase_array # list of phase on each antenna strap (deg)
        self.aorsa_nml['aorsa2din']['i_antenna'] = i_antenna #  ! i_antenna = flag determining which antenna model is used
                                         #if(i_antenna .eq. 0) antenna current is Gaussian 
                                         #if(i_antenna .eq. 1) antenna current is cos(ky * y)  (default)
                                         #where ky = omgrf / vphase = (omgrf / clight) * antlc = k0 * antlc
                                         #For constant current, set antlc = 0.0
    def setup_computational_box(self, psilim, ytop, ybottom, rwright, 
                                rwleft, n_prof_flux=1, iprofile=5):
        self.aorsa_nml['aorsa2din']['psilim'] = psilim # guess: limiting psi
        self.aorsa_nml['aorsa2din']['ytop'] = ytop # top of comp box in [m]
        self.aorsa_nml['aorsa2din']['ybottom'] = ybottom # bottom of comp box [m]
        self.aorsa_nml['aorsa2din']['rwright'] = rwright # major radius of the right conducting wall
        self.aorsa_nml['aorsa2din']['rwleft'] = rwleft # major radius of the left conducting wall
        self.aorsa_nml['aorsa2din']['n_prof_flux'] = n_prof_flux # 0 sqpolflx, 1 sqtorflx
        self.aorsa_nml['aorsa2din']['iprofile'] = iprofile # 1 guass, 2 parab, 3 fits, 5 numerical profiles

    def setup_resolution_and_proc_grid(self, nmodesx, nmodesy, nprow, npcol, lmax=3, lmaxe=1):
        self.aorsa_nml['aorsa2din']['nmodesx'] = nmodesx
        self.aorsa_nml['aorsa2din']['nmodesy'] = nmodesy
        self.aorsa_nml['aorsa2din']['nprow'] = nprow
        self.aorsa_nml['aorsa2din']['npcol'] = npcol
        self.aorsa_nml['aorsa2din']['lmax'] = lmax # highest order bessel function retained in ion plasma conductivity wdot
        self.aorsa_nml['aorsa2din']['lmaxe'] = lmaxe # heighest electron conductivity bessel function retained
        if lmax <= 3:
            print('be sure lmax has enough harmonics for the ion harmonic absorption you need') 

    def set_wdot_and_nonmax(self, enorm_factor, nuper=150, nupar=300, use_new_wdot=True,
                            nzeta_wdote=51, nzeta_wdoti=51):
        self.aorsa_nml['aorsa2din']['enorm_factor'] = enorm_factor #if (enorm_factor = 0.0) AORSA & CQL3D use same enorm (default)
                                                                   #if (enorm_factor > 0.0) AORSA enorm = enorm_factor x the maximum energy
        self.aorsa_nml['aorsa2din']['nuper'] = nuper # number of perp velocity grid points
        self.aorsa_nml['aorsa2din']['nupar'] = nupar # number of parallel velocity grid points
        self.aorsa_nml['aorsa2din']['use_new_wdot'] =  use_new_wdot #if (use_new_wdot .eq. .false.) use original wdote - resonant terms only (default)                                 
                                                                    #if (use_new_wdot .eq. .true. ) use new wdote - both resonant and non-resonant terms
        self.aorsa_nml['aorsa2din']['nzeta_wdote'] = nzeta_wdote # 0 no wdot calc for electrons, 1 wdot w/o interp, >=2 calc with interp 
        self.aorsa_nml['aorsa2din']['nzeta_wdoti'] = nzeta_wdoti # 0 no wdot calc for ions, 1 wdot w/o interp, >=2 calc with interp

    def species_specifications(self):
        # ndisti1 is a setting for ion species 1: 0 = maxwellian, 1 = non-maxwellian
        one_amu = 1.66054e-27 # kg
        proton_charge = 1.6022e-19 # C
        for i in range(1, len(self.species_list)):  # loop over only ions
            name = self.species_list[i]

            # deal with aorsa's poor coding practice of hard coded slots for species 
            if i==1:
                self.aorsa_nml['aorsa2din']['ndisti1'] = self.species_ndist[name]
                self.aorsa_nml['aorsa2din']['amu1'] = round(self.species_mass[name]/one_amu)
                self.aorsa_nml['aorsa2din']['z1'] = round(self.species_charge[name]/proton_charge)
                
            elif i ==2:
                self.aorsa_nml['aorsa2din']['ndisti2'] = self.species_ndist[name]
                self.aorsa_nml['aorsa2din']['amu2'] = round(self.species_mass[name]/one_amu)
                self.aorsa_nml['aorsa2din']['z2'] = round(self.species_charge[name]/proton_charge)

            elif i ==3:
                self.aorsa_nml['aorsa2din']['ndisti3'] = self.species_ndist[name]
                self.aorsa_nml['aorsa2din']['amu3'] = round(self.species_mass[name]/one_amu)
                self.aorsa_nml['aorsa2din']['z3'] = round(self.species_charge[name]/proton_charge)  

            elif i ==4:
                self.aorsa_nml['aorsa2din']['ndisti4'] = self.species_ndist[name]
                self.aorsa_nml['aorsa2din']['amu4'] = round(self.species_mass[name]/one_amu)
                self.aorsa_nml['aorsa2din']['z4'] = round(self.species_charge[name]/proton_charge) 

            elif i ==5:
                self.aorsa_nml['aorsa2din']['ndisti5'] = self.species_ndist[name]
                self.aorsa_nml['aorsa2din']['amu5'] = round(self.species_mass[name]/one_amu)
                self.aorsa_nml['aorsa2din']['z5'] = round(self.species_charge[name]/proton_charge) 

            elif i ==6:
                self.aorsa_nml['aorsa2din']['ndisti6'] = self.species_ndist[name]
                self.aorsa_nml['aorsa2din']['amu6'] = round(self.species_mass[name]/one_amu)
                self.aorsa_nml['aorsa2din']['z6'] = round(self.species_charge[name]/proton_charge)

            # TODO: user can add up to 6 of these i think, check aorsa file 
    
    def set_noise_control(self, z2_electron=1, upshift=1, xkperp_cutoff=0.5, damping=100.0, delta0=4.0e-05):
        self.aorsa_nml['aorsa2din']['z2_electron'] = z2_electron
        #if (z2_electron .eq. 0) use the original Z2 function for electrons (default)    
        #if (z2_electron .eq. 1) use the Z2 table for electrons with l = 0 (Taylor expansion along field line)
        #if (z2_electron .eq. 2) use Fourier expansion along field line for electrons with l = 0 (full orbits)
        self.aorsa_nml['aorsa2din']['upshift'] = upshift
        #upshift: if (upshift .ne.  0) upshift is turned on (default)
        #if (upshift .eq. -1) upshift is turned off for xkperp > xkperp_cutoff
        #if (upshift .eq. -2) don't allow k_parallel too close to zero
        #if (upshift .eq.  0) upshift is turned off always

        self.aorsa_nml['aorsa2din']['xkperp_cutoff'] = xkperp_cutoff
        #fraction of xkperp above which the electron conductivity (sig3) 
        #is enhanced to short out noise in E_parallel (default = 0.75)

        self.aorsa_nml['aorsa2din']['damping'] = damping
        #enhancement factor (default = 0.0) for the electron conductivity (sig3) 
        #applied above the fractional value of xkperp (xkperp_cutoff) to 
        #short out noise in E_parallel 

        self.aorsa_nml['aorsa2din']['delta0'] = delta0 #numerical damping for Bernstein wave:  about 1.e-04 (dimensionless)

    def save_aorsa2d_out(self):
        self.aorsa_nml.write(f'{self.local_work_dir}{self.save_aorsa_file_name}')
        print(f'Saved aorsa namelist to {self.local_work_dir}{self.save_aorsa_file_name}')

    def save_and_send_to_remote_host(self):
        print('Saving changes to namelist.')
        self.save_aorsa2d_out()
        print(f'Done. Sending namelist {self.save_aorsa_file_name} to:')
        print(f'{self.remote_host}:{self.remote_work_dir}')
        os.system(f'scp {self.local_work_dir}{self.save_aorsa_file_name} {self.remote_host}:{self.remote_work_dir}/aorsa2d_recieved.in')
        print(f'scp {self.local_work_dir}{self.save_aorsa_file_name} {self.remote_host}:{self.remote_work_dir}/aorsa2d_recieved.in')
        print('Done.')


# POST PROCESSING TOOLS
class Aorsa_Post_Process():
    """
    Class to post-process the various output files from an Aorsa run 
    """
    def __init__(self, vtk_file, aorsa2d_input_file, eqdsk_file):

    
        self.vtk_file = vtk_file # global path to the vtk file, or a list of paths if there are multiple


        self.aorsa2d_input_file = aorsa2d_input_file

        self.eqdsk, fig = plasma.equilibrium_process.readGEQDSK(eqdsk_file, doplot=False)
        self.aorsanml = f90.read(self.aorsa2d_input_file)

        self.R_wall = self.eqdsk['rlim']
        self.Z_wall = self.eqdsk['zlim']

        self.R_lcfs = self.eqdsk['rbbbs']
        self.Z_lcfs = self.eqdsk['zbbbs']

        self.read_mesh()

    def is_iterable(self, var):
        if isinstance(var, Iterable) and not isinstance(var, (str, bytes)):
            return True
        else:
            return False


    def plot_equilibrium(self, figsize, levels):
        psizr = self.eqdsk['psizr']
        plt.figure(figsize=figsize)
        plt.axis('equal')
        img = plt.contour( self.eqdsk['r'], self.eqdsk['z'], psizr.T, levels= levels)
        plt.plot(self.eqdsk['rlim'], self.eqdsk['zlim'], color='black', linewidth=3)
        plt.plot(self.eqdsk['rbbbs'], self.eqdsk['zbbbs'], color='black', linewidth=3)
        plt.colorbar(img)
        plt.show()

    def read_mesh(self):
        if self.is_iterable(self.vtk_file):
            self.mesh = []
            for path in self.vtk_file:
                self.mesh.append(meshio.read(path))

            # assumes the same RZ grid between aorsa files. T
            self.R_array = self.mesh[0].points[:,0]
            self.Z_array = self.mesh[0].points[:,1]
        else:
            self.mesh = meshio.read(self.vtk_file)
            self.R_array = self.mesh.points[:,0]
            self.Z_array = self.mesh.points[:,1]
        self.triangulation = tri.Triangulation(self.Z_array, self.R_array)

    def print_mesh_info(self):
        print('Mesh:')
        if self.is_iterable(self.vtk_file):
            for mesh in self.mesh:
                print(mesh)
                print('\nmesh.points.shape: ', mesh.points.shape)
        else:
            print(self.mesh)
            print('\nmesh.points.shape: ', self.mesh.points.shape) # note mesh.points has numpoints rows and 3 collumns, where col 0 is R, and col2 is Z

    def plot_result_2D(self, key, title, cbar_label, cmap='viridis', figsize=(3,6), logplot=False, return_plot=False, multifile_idx=None):
        fig, ax = plt.subplots(figsize=figsize)

        if logplot:
            if self.is_iterable(self.vtk_file):
                tcf=ax.tricontourf(self.R_array, self.Z_array, (np.abs(self.mesh[multifile_idx].point_data[key][:,0])+1), 400, cmap=cmap)
            else:
                tcf=ax.tricontourf(self.R_array, self.Z_array, (np.abs(self.mesh.point_data[key][:,0])+1), 400, cmap=cmap)
        else:
            if self.is_iterable(self.vtk_file):
                tcf=ax.tricontourf(self.R_array, self.Z_array, self.mesh[multifile_idx].point_data[key][:,0], 400, cmap=cmap)
            else:
                tcf=ax.tricontourf(self.R_array, self.Z_array, self.mesh.point_data[key][:,0], 400, cmap=cmap)

        cb = fig.colorbar(tcf)
        cb.set_label(cbar_label)
        ax.axis('equal')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        ax.set_title(title)
        ax.plot(self.R_lcfs, self.Z_lcfs, 'black')
        ax.plot(self.R_wall, self.Z_wall)
    
        if return_plot:
            return fig, ax
        
        plt.show()
        plt.close()
    
    def get_multifile_sum(self, key):
        if self.is_iterable(self.vtk_file):
            sum_mat = self.mesh[0].point_data[key][:,0].copy()

            for i in range(1,len(self.mesh)):
                sum_mat += self.mesh[i].point_data[key][:,0]
            
            return sum_mat
        
        else:
            raise ValueError('This helper function only is valid when multifile mode is being used.')

    def plot_Eplus_Eminus(self, cmap='viridis', figsize=(6,6), logplot=False, return_plot=False):
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # plot total abosorption 
        if logplot:
            if self.is_iterable(self.vtk_file):
                eminus_sum = self.get_multifile_sum('re_eminus')
                eplus_sum = self.get_multifile_sum('re_eplus')
                toplot0 = np.log(np.abs(eminus_sum)+1)
                toplot1 = np.log(np.abs(eplus_sum)+1)
            else:
                toplot0 = np.log(np.abs(self.mesh.point_data['re_eminus'][:,0])+1)
                toplot1 = np.log(np.abs(self.mesh.point_data['re_eplus'][:,0])+1)

            tcf0=axs[0].tricontourf(self.R_array, self.Z_array, toplot0, 400, cmap=cmap)
            axs[0].set_title(r'ln(|Re(E$_{-}$)| + 1)')
            
            tcf1=axs[1].tricontourf(self.R_array, self.Z_array, toplot1, 400, cmap=cmap)
            axs[1].set_title(r'ln(|Re(E$_{+}$)| + 1)')

        else:
            if self.is_iterable(self.vtk_file):
                eminus = self.get_multifile_sum('re_eminus')
                eplus = self.get_multifile_sum('re_eplus')

            else:
                eminus = self.mesh.point_data['re_eminus'][:,0]
                eplus = self.mesh.point_data['re_eplus'][:,0]

            tcf0=axs[0].tricontourf(self.R_array, self.Z_array, eminus, 400, cmap=cmap)
            tcf1=axs[1].tricontourf(self.R_array, self.Z_array, eplus, 400, cmap=cmap)
            axs[0].set_title(r'Re(E$_{-}$)')
            axs[1].set_title(r'Re(E$_{+}$)')


        cb0 = fig.colorbar(tcf0)
        cb0.set_label(r'arb. units')
        axs[0].axis('equal')
        axs[0].set_xlabel('R [m]')
        axs[0].set_ylabel('Z [m]')
        axs[0].plot(self.R_lcfs, self.Z_lcfs, 'black')
        axs[0].plot(self.R_wall, self.Z_wall)

        cb1 = fig.colorbar(tcf1)
        cb1.set_label(r'arb. units')
        axs[1].axis('equal')
        axs[1].set_xlabel('R [m]')
        axs[1].set_ylabel('Z [m]')
        axs[1].plot(self.R_lcfs, self.Z_lcfs, 'black')
        axs[1].plot(self.R_wall, self.Z_wall)

        if return_plot:
            return fig, axs
        plt.plot()
        #plt.close()
        
    def plot_species_absorption(self, figsize, return_fig=False):
        ion_names = self.aorsanml['STATE']['S_S_NAME'][1:]

        if len(ion_names) > 2:
            fig, axs = plt.subplots(3, 2, figsize=figsize)
        else:
            fig, axs = plt.subplots(2, 2, figsize=figsize)

        # plot total abosorption 
        if self.is_iterable(self.vtk_file):
            tcf0=axs[0,0].tricontourf(self.R_array, self.Z_array, self.get_multifile_sum('wdot_tot'), 400, cmap='hot')
        else:
            tcf0=axs[0,0].tricontourf(self.R_array, self.Z_array, self.mesh.point_data['wdot_tot'][:,0], 400, cmap='hot')

        cb0 = fig.colorbar(tcf0)
        cb0.set_label(r'W/$m^3$')
        axs[0,0].axis('equal')
        axs[0,0].set_xlabel('R [m]')
        axs[0,0].set_ylabel('Z [m]')
        axs[0,0].set_title(r'$\dot{w}_{tot}$')
        axs[0,0].plot(self.R_lcfs, self.Z_lcfs, 'black')
        axs[0,0].plot(self.R_wall, self.Z_wall)

        # plot electron absorption 
        if self.is_iterable(self.vtk_file):
            tcf1=axs[0,1].tricontourf(self.R_array, self.Z_array, self.get_multifile_sum('wdote'), 400, cmap='hot')
        else:
            tcf1=axs[0,1].tricontourf(self.R_array, self.Z_array, self.mesh.point_data['wdote'][:,0], 400, cmap='hot')

        cb1 = fig.colorbar(tcf1)
        cb1.set_label(r'W/$m^3$')
        axs[0,1].axis('equal')
        axs[0,1].set_xlabel('R [m]')
        axs[0,1].set_ylabel('Z [m]')
        axs[0,1].set_title(r'$\dot{w}_{e}$ Electron Absorption')
        axs[0,1].plot(self.R_lcfs, self.Z_lcfs, 'black')
        axs[0,1].plot(self.R_wall, self.Z_wall)

        for i in range(len(ion_names)):
            if i == 0:
                row = 1
                col= 0

            elif i == 1:
                row = 1
                col = 1

            elif i == 2:
                row = 2
                col = 0

            elif i == 3:
                row = 2
                col = 1
            
            key = 'wdoti' + str(i+1)
            if self.is_iterable(self.vtk_file):
                tcf=axs[row, col].tricontourf(self.R_array, self.Z_array,  self.get_multifile_sum(key), 400, cmap='hot')
            else:
                tcf=axs[row, col].tricontourf(self.R_array, self.Z_array, self.mesh.point_data[key][:,0], 400, cmap='hot')
            cb = fig.colorbar(tcf)
            cb.set_label(r'W/$m^3$')
            axs[row, col].axis('equal')
            axs[row, col].set_xlabel('R [m]')
            axs[row, col].set_ylabel('Z [m]')
            axs[row, col].set_title(r'$\dot{w}_i$ ' + f'Ion {str(i+1)} = '+ f'{ion_names[i]} Absorption')
            axs[row, col].plot(self.R_lcfs, self.Z_lcfs, 'black')
            axs[row, col].plot(self.R_wall, self.Z_wall)

        if return_fig:
            return fig, axs

        plt.show()
        plt.close()
