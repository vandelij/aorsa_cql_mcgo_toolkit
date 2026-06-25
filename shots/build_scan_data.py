from aorsa_cql_mcgo_toolkit.utils.FAR3D_Analysis import Far3D_Analysis
from aorsa_cql_mcgo_toolkit.utils.MCGO_Analysis import MCGO_Post_Process
import numpy as np
import os
import h5py
import json
from freeqdsk import geqdsk
# want to point to the scan directory. 

mu0 = 4*np.pi*1e-7

def append_run_data(filename, run_name, run_dict):
    """Appends a new run containing a dictionary of floats and lists to the HDF5 file."""
    with h5py.File(filename, 'a') as f:
        # 1. Create the dictionary "key" for the run
        if run_name in f:
            raise ValueError(f"Run {run_name} already exists!")
            
        run_group = f.create_group(run_name)
        
        # 2. Iterate through your Python dict and save everything as a dataset
        for key, value in run_dict.items():
            # h5py natively understands how to store both single floats and lists
            run_group.create_dataset(key, data=value)

def get_run_dict(filename, run_name):
    """Reads an HDF5 group and reconstructs it into a standard Python dictionary."""
    reconstructed_dict = {}
    
    with h5py.File(filename, 'r') as f:
        run_group = f[run_name]
        
        for key in run_group.keys():
            dataset = run_group[key]
            
            # Check if it's a single float (shape is empty) or a list (shape > 0)
            if dataset.shape == ():
                # Unpack scalar
                reconstructed_dict[key] = dataset[()] 
            else:
                # Unpack array and convert back to standard Python list 
                # (it comes out as a numpy array by default)
                reconstructed_dict[key] = dataset[:].tolist() 
                
    return reconstructed_dict


def parse_mcgo_profiles(filepath):
    """Parses a MATLAB-style text data file from MCGO into a Python dictionary."""
    data = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    current_var = None
    current_data = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines or MATLAB comments
        if not line or line.startswith('%'):
            continue
            
        # Detect the start of a variable assignment
        if '=' in line and '[' in line:
            current_var = line.split('=')[0].strip()
            current_data = []
            
        # Detect the end of an array block
        elif '];' in line:
            if current_var is not None:
                # Convert the collected list to a NumPy array
                data[current_var] = np.array(current_data, dtype=float)
            current_var = None
            
        # If we are inside a block, grab the numbers
        elif current_var is not None:
            # Handle lines that might have multiple numbers or scientific notation
            parts = line.split()
            for p in parts:
                try:
                    current_data.append(float(p))
                except ValueError:
                    pass         
    return data


def build_run_dict(run_dir, iteration_index, profiles_json_file, num_mcgo=2):
    # need to store all the data in a dict:
    # rho_grid_mfile, rho_grid, RFpower, neutrons, fast_ion_density, betaRF, betaNBI, rhoL_over_a_RF,
    # rhoL_over_a_NBI, v_over_vA_NBI, v_over_vA_RF,
    E_NBI_kev = 80 # for now, this is constant 
    mcgo_energy_filter_kev = 30 # also set for now 
    # 
    data_dict = {} 

    # load up m file and read in rho_grid_mfile, RFpower, neutrons, summing if more than one file 
    mfile_data_list = []
    mcgo_nc_file_list = []

    for nmcgo in range(num_mcgo):
        mfile = run_dir + f'profiles_out_{nmcgo}_iteration_{iteration_index}.m'
        mdata = parse_mcgo_profiles(mfile)
        mfile_data_list.append(mdata)

        mcgo_nc_file = run_dir + f'mcgo_out_{nmcgo}_iteration_{iteration_index}.nc'
        mcgo_nc_file_list.append(mcgo_nc_file)


    rho_grid_mfile = mfile_data_list[0]['rho_sqrt_psi_bincent'] # assumes all mfiles have the same grid 
    RF_power = np.zeros_like(rho_grid_mfile)
    neutron_production = np.zeros_like(rho_grid_mfile)

    for mcgo_data in mfile_data_list:
        num_time_frames = int(mcgo_data['ntw'][0])
        num_radial_bins = int(mcgo_data['mf'][0])
        # grab the arrays at the final time step 
        RF_power += mcgo_data['pwr_rf'].reshape((num_time_frames, num_radial_bins))[-1, :]
        neutron_production += mcgo_data['ddneut'].reshape((num_time_frames, num_radial_bins))[-1, :]

    # load these m file data arrays to the data dict 
    data_dict['rho_sqrt_psi_mfile'] = rho_grid_mfile.tolist()
    data_dict['RF Power [MW_per_m^3]'] = RF_power.tolist()
    data_dict['DD Neutron Rate [n_per_s*cm^3]'] = neutron_production.tolist()

    # load in mcgo .nc files in mcgo_pp object
    mcgo_pp = MCGO_Post_Process(mcgo_nc_file=mcgo_nc_file_list, eqdsk_file=eqdsk_file, particle_lists_on=True)

    # generate far3d object for sole purpose of using distribution processing
    rho_far3d = np.linspace(0.01, 0.99, 100)
    far3d = Far3D_Analysis(eqdsk_file=eqdsk_file)

    # load up profiles from json file 
    with open(profiles_json_file, 'r') as f:
        # Load the JSON data back into a Python dictionary
        profile_data = json.load(f)
    

    far3d.load_electron_temperature_profile(profile_data['rho'], profile_data['Te [keV]'])
    far3d.load_ion_temperature_profile(profile_data['rho'], profile_data['TD [keV]'])
    far3d.load_electron_density_profile(profile_data['rho'], profile_data['ne [m^-3]'])
    far3d.load_bulkion_density_profile(profile_data['rho'], profile_data['nD [m^-3]'])

    far3d.build_far3d_outfile_from_cql3d(rho_array_for_far3d=rho_far3d, 
                                        cql_or_mcgo_pp=mcgo_pp, 
                                        E_NBI_kev=E_NBI_kev, 
                                        num_iter_max=10000, 
                                        index_to_cut=0,
                                        mode='mcgo',
                                        mcgo_energy_filter_kev=mcgo_energy_filter_kev)
    
    # call convert_mcgo_distribution_into_F_of_v_indexable_by_rho(self, mcgo_pp), spits out rho_grid, v, F_of_v_idx_rho
    mcgo_lfs_rhos, vgrid, F_of_v_idx_rho = far3d.convert_mcgo_distribution_into_F_of_v_indexable_by_rho(mcgo_pp)

    # initiate arrays to store fast_ion_density, betaRF, betaNBI, rhoL_over_a_RF,
    # rhoL_over_a_NBI, v_over_vA_NBI, v_over_vA_RF
    RF_density_list = []
    NBI_density_list = []
    RF_beta_list = []
    NBI_beta_list = []
    RF_rhoL_over_a_list = []
    NBI_rhoL_over_a_list = []
    RF_v_over_vA_list = []
    NBI_v_over_vA_list = []
    RF_temperature_list = []
    NBI_temperature_list = []


    # loop over rho and call fit_3_maxwellians_to_speed_distribution_bulk_fit_NBI_RF_moments to get the temps and densities. 
    # modify to also spit out the F averaged speed and the F averaged rhoL for RF and NBI populations. At each rho, 
    # append the lists initiated above. 

    # calculate central alfen speed and the central magnetif pressure 
    with open(eqdsk_file, 'r') as f:
        eq_dict = geqdsk.read(f)

    B0 = eq_dict['bcentr'] 
    ni0 = profile_data['nD [m^-3]'][0]
    mass = far3d.species_dict['d']['mass']
    charge = far3d.species_dict['d']['charge']
    vA =  B0 / np.sqrt(ni0*mass*mu0)
    magnetic_pressure = B0**2 / (2*mu0)
    minor_radius = (max(eq_dict['rbdry']) - min(eq_dict['rbdry'])) / 2

    convert_keV_per_m3_to_J_per_m3 = 1000*1.6022e-19

    for rho_i in range(mcgo_lfs_rhos.shape[0]):
        rhoi = mcgo_lfs_rhos[rho_i]
        fit = far3d.fit_3_maxwellians_to_speed_distribution_bulk_fit_NBI_RF_moments(v=vgrid, 
                                                                                    F=F_of_v_idx_rho[rho_i, :], 
                                                                                    E_NBI_kev=E_NBI_kev, 
                                                                                    rho=rhoi, 
                                                                                    mode='mcgo', 
                                                                                    mcgo_energy_filter_kev=mcgo_energy_filter_kev,
                                                                                    return_speeds=True)
        

        nNBI = fit[2]
        TNBI = fit[3]

        nRF = fit[4]
        TRF = fit[5]

        vNBI = fit[6]
        vRF = fit[7]

        betaNBI = nNBI * TNBI * convert_keV_per_m3_to_J_per_m3 / magnetic_pressure
        betaRF = nRF * TRF * convert_keV_per_m3_to_J_per_m3 / magnetic_pressure

        NBI_density_list.append(nNBI)
        NBI_temperature_list.append(TNBI)
        RF_density_list.append(nRF)
        RF_temperature_list.append(TRF)

        NBI_beta_list.append(betaNBI)
        RF_beta_list.append(betaRF)

        NBI_v_over_vA_list.append(vNBI/vA)
        RF_v_over_vA_list.append(vRF/vA)

        rhoL_NBI = mass*vNBI / (charge * B0)
        rhoL_RF = mass*vRF / (charge * B0)

        NBI_rhoL_over_a_list.append(rhoL_NBI/minor_radius)
        RF_rhoL_over_a_list.append(rhoL_RF/minor_radius)


    # build up the run_dir with key = list, key should have units. Also the Rf power from aorsa.powers.txt, and the B0, n, and vA used in beta and in v/vA
    data_dict['rho_mcgo_lfs'] = mcgo_lfs_rhos.tolist()
    data_dict['NBI density [m^-3]'] = NBI_density_list
    data_dict['RF density [m^-3]'] = RF_density_list
    data_dict['NBI Temp [keV]'] = NBI_temperature_list
    data_dict['RF Temp [keV]'] = RF_temperature_list
    data_dict['NBI beta'] = NBI_beta_list
    data_dict['RF beta'] = RF_beta_list
    data_dict['NBI rhoL_per_a'] = NBI_rhoL_over_a_list 
    data_dict['RF rhoL_per_a'] = RF_rhoL_over_a_list 
    data_dict['NBI v_per_vA'] = NBI_v_over_vA_list
    data_dict['RF v_per_vA'] = RF_v_over_vA_list
    data_dict['vA [m_per_s]'] = vA 
    data_dict['B0 [T]'] = B0
    data_dict['ni0 [m^-3]'] = ni0
    
    # load up aorsa powers and store 
    aorsa_powers_file = run_dir + f'aorsa_iteration{iteration_index}_powers.txt'
    aorsa_powers = np.loadtxt(aorsa_powers_file)
    data_dict['aorsa powers to ions [W]'] = aorsa_powers.tolist()

    return data_dict


if __name__ == "__main__":
    data_file = '/home/jacobvandelindt/aorsa_cql_mcgo_toolkit/shots/147634/paper2_scans/scan_data.h5'
    scan_dir = "/home/jacobvandelindt/aorsa_cql_mcgo_toolkit/shots/147634/paper2_scans/"
    eqdsk_file = "/home/jacobvandelindt/aorsa_cql_mcgo_toolkit/shots/147634/g147634.04525"
    profiles_json = '/home/jacobvandelindt/aorsa_cql_mcgo_toolkit/shots/147634/profiles.json'
    RF_powers_prefix_list = [300, 1900]
    NBI_powers_prefix_list = [10]
    iteration_index = 15

    for PRF in RF_powers_prefix_list:
        for PNBI in NBI_powers_prefix_list:
            print(f'Processing PNBI {PNBI}, PRF {PRF}...')
            run_dir = f'beamtot_{PNBI}_RFtot_{PRF}/'
            run_name = f'beamtot_{PNBI}_RFtot_{PRF}'
            run_dict = build_run_dict(run_dir=scan_dir+run_dir, 
                                      profiles_json_file=profiles_json,
                                      iteration_index=iteration_index)
            append_run_data(filename=data_file, run_dict=run_dict, run_name=run_name)

    print(f'Done building {data_file}')



