from scipy.interpolate import PchipInterpolator
import numpy as np
import matplotlib.pyplot as plt
import f90nml as f90
import os

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
        os.system(f'scp {self.local_work_dir}{self.save_aorsa_file_name} {self.remote_host}:{self.remote_work_dir}')
        print(f'scp {self.local_work_dir}{self.save_aorsa_file_name} {self.remote_host}:{self.remote_work_dir}')
        print('Done.')