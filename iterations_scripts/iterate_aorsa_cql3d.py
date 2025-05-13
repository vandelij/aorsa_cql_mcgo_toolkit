import numpy as np
import os
import subprocess
import time 
import re
import netCDF4


#################### USER SETTINGS #####################
# Directory Setup (use absolute paths)

# The head directory in which the results are stored. lots of data, recommend pscratch. 
super_directory = "/pscratch/sd/v/vandelij/iteration_test_full_beta/"

# The eqdsk path. Wilkl be copied and renamed eqdsk into the run dirs
eqdsk_path = "/global/cfs/cdirs/m77/jacob_van_de_Lindt/aorsa_cql_iterate_templates_new/g147634.04525"

# the aorsa template directory with the initial case set up as well as the aorsa slurm job file.  
aorsa_clone_directory = "/global/cfs/cdirs/m77/jacob_van_de_Lindt/aorsa_cql_iterate_templates_new/aorsa_1gen/"

# the cql3d template directory with the initial case set up as well as the cql3d slurm job file.
cql3d_clone_directory = "/global/cfs/cdirs/m77/jacob_van_de_Lindt/aorsa_cql_iterate_templates_new/cql3d_1gen/"

# Simulation settings
num_iterations = 2 

# command to run the local cql3d slurm file
run_cql3d = "srun -n 60 --cpu-bind=cores -c 4 /global/common/software/atom/perlmutter/binaries/cql3d/default/xcql3d_mpi.perlmutter > log.cql3d 2>&1"

# command to launch aorsa
run_aorsa = "srun -n 1024 --cpu-bind=cores -c 2 /global/homes/v/vandelij/xaorsa2d_jacob_sam_john_feature_merge_3_kspecifix_HFS > log.aorsa 2>&1"

def ensure_and_change_directory(path):
    if not os.path.exists(path):  # Check if the directory exists
        os.makedirs(path)  # Create the directory (including parent directories if needed)
    os.chdir(path)  # Change to the directory
    print("Changed to directory:", os.getcwd())

def build_file_structure(num_iterations):
    for i in range(num_iterations):
        os.chdir(super_directory)
        # set up cql directories. i != 0 case setup handled later when building the input files
        cql3d_dir = super_directory + f"/cql3d_iteration_{i}"
        ensure_and_change_directory(cql3d_dir)
        # copy over required files
        os.system(f"cp {eqdsk_path} eqdsk")
        os.system(f"cp {cql3d_clone_directory}/cqlinput .")
        

        # change back to the super directory 
        os.chdir(super_directory)

        # set up aorsa directory
        aorsa_dir = super_directory + f"/aorsa_iteration_{i}"
        ensure_and_change_directory(aorsa_dir)
        # copy over required files 
        os.system(f"cp {eqdsk_path} eqdsk")
        os.system(f"cp {aorsa_clone_directory}/aorsa2d.in .")
        os.system(f"cp {aorsa_clone_directory}/Ztable .")
        os.system(f"cp {aorsa_clone_directory}/ZTABLE.TXT .")
        os.system(f"cp {aorsa_clone_directory}/grfont.dat .")

    os.chdir(super_directory)
    print('Done setting up directories.')


# define function which submits jobs
def submit_job(slurm_command_string, directory):
    """Submits the job using sbatch"""
    # wait a little bit for cp to finish executing
    os.sync()
    time.sleep(5)

    # now, call the subprocess. This is blocking. 
    subprocess.run(slurm_command_string, shell=True, cwd=directory)


# helper function for parsing cqlinput file 
def edit_cql3d_input(file_path_in, file_path_out, params, new_values):
    with open(file_path_in, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    line_num = 0

    for line in lines:
        match_found = False
        for i in range(len(params)):
            param = params[i]
            new_value = new_values[i]
            # Match parameter lines, handling both numerical and string values (quoted or unquoted)
            match = re.match(rf'(\s*{param}\s*=\s*)(["\']?)([^"\']*)(["\']?)(\s*!.*)?', line)
            if match:
                match_found = True
                # Preserve quotes if they existed; otherwise, insert quotes for string values
                quote = '"' if match.group(2) or match.group(4) else ''
                new_line = f"{match.group(1)}{quote}{new_value}{quote}{match.group(5) or ''}\n"
                updated_lines.append(new_line)
                print(line_num)

        if match_found == False: # only append the lines if the line wasnt already
            updated_lines.append(line)
        line_num += 1

    with open(file_path_out, 'w') as f:
        f.writelines(updated_lines)


# define two functions which handle the actual modification and exicution of aorsa and cql3d.
def cql3d_iteration_i(i):
    cql3d_dir = super_directory + f"/cql3d_iteration_{i}"
    os.chdir(cql3d_dir) # this should already exist from the build_file_structure function
    
    # TODO... done. add functionality to copy over out_coef1 over from aorsa i-1 if i != 0 as du0u01 (TODO make sure grid is correct), else just run on NBI
    # if i != 0, also copy over cql3d.nc from the previous cql run and set nlrestart to "enabled" to read it in. 
    # be cautious of overwriting it? maybe copy it twice with two names, nmuonic can maybe be cql3d_i.nc? 
    # # also, if i != 0, set rdcmod=format1, rdcfile='du0u0_input_1', rfread=text, 
    if i != 0:
        # copy over cql3d.nc from previous cql run and QL diffusion file from previous aorsa run
        previous_cql_dir = super_directory + f"/cql3d_iteration_{i-1}"
        previous_aorsa_dir = super_directory + f"/aorsa_iteration_{i-1}"

        os.system(f"cp {previous_cql_dir}/cql3d.nc .")
        os.system(f"cp {previous_aorsa_dir}/out_cql3d.coef1 du0u0_input_1")

        # convert the input file for restart and RF from QL diffusion
        print('Converting existing cqlinput for restart and RF QL diffusion...')
        edit_cql3d_input(file_path_in='cqlinput', file_path_out='cqlinput', params=['nlrestrt', 'rdcmod'], new_values=['ncdfdist','format1'])
        print('Done.')
        
    print(f"Submitting CQL3D iteration {i} from {cql3d_dir}, waiting for completion...")
    submit_job(slurm_command_string=run_cql3d, directory=cql3d_dir) # call the slurm job 

    print(f"CQL3D iteration {i} completed!")

# helper function for converting cql3d.nc files with 1 general species so aorsa can read them by giving them a gen_species index to f, wpar, wperp 
def add_gen_species_dim_to_cql_nc(input_nc_file, output_nc_file):
    cql_nc_new = netCDF4.Dataset(output_nc_file, 'w')
    cql_nc = netCDF4.Dataset(input_nc_file, "r")

    # Copy dimensions from the original file
    for name, dimension in cql_nc.dimensions.items():
        cql_nc_new.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

    for name, variable in cql_nc.variables.items():
        # extend the dimensions of the variables that aorsa expects to have a species dim
        if name == 'f' and cql_nc.dimensions['gen_species_dim'].size == 1:
            # update the dims
            list_dims = list(variable.dimensions)
            list_dims.insert(0, 'gen_species_dim')
            tuple_dims = tuple(list_dims)

            new_var = cql_nc_new.createVariable(name, variable.dtype, tuple_dims)

            # add the variable 
            new_var.setncatts(cql_nc.variables[name].__dict__)
            new_var[:] = np.expand_dims(cql_nc.variables[name][:], 0)


        elif name == 'wpar' and cql_nc.dimensions['gen_species_dim'].size == 1:
            # update the dims
            list_dims = list(variable.dimensions)
            list_dims.insert(1, 'gen_species_dim')
            tuple_dims = tuple(list_dims)

            new_var = cql_nc_new.createVariable(name, variable.dtype, tuple_dims)

            # add the variable 
            new_var.setncatts(cql_nc.variables[name].__dict__)
            new_var[:] = np.expand_dims(cql_nc.variables[name][:], 1)

        elif name == 'wperp' and cql_nc.dimensions['gen_species_dim'].size == 1:
            # update the dims
            list_dims = list(variable.dimensions)
            list_dims.insert(1, 'gen_species_dim')
            tuple_dims = tuple(list_dims)

            new_var = cql_nc_new.createVariable(name, variable.dtype, tuple_dims)

            # add the variable 
            new_var.setncatts(cql_nc.variables[name].__dict__)
            new_var[:] = np.expand_dims(cql_nc.variables[name][:], 1)

        else:
            new_var = cql_nc_new.createVariable(name, variable.dtype, variable.dimensions)
            new_var.setncatts(cql_nc.variables[name].__dict__)
            new_var[:] = cql_nc.variables[name][:]


    cql_nc_new.close()

def aorsa_iteration_i(i):
    aorsa_dir = super_directory + f"/aorsa_iteration_{i}"
    os.chdir(aorsa_dir)
    # copy over previous cql iteration cql3d.nc as cql3d.nc.i and cql3d.nc, and set netCDF_file1 = 'cql3d.nc'
    # enorm_factor= 0.0 allegedly forces cql3d and aorsa to have the same enorm 
    # set ndist1 = 1 for nonmaxwellian ions, also need to convert cql3d.nc to the version where f is indexed by gen_species

    previous_cql_dir =  super_directory + f"/cql3d_iteration_{i}"
    os.system(f"cp {previous_cql_dir}/cql3d.nc cql3d_{i}.nc")

    # now convert the cql3d.nc to f indexed by gen species. 
    print(f'Converting cql3d_{i}.nc to cql3d.nc with added gen species dim...')
    add_gen_species_dim_to_cql_nc(input_nc_file=f"cql3d_{i}.nc", output_nc_file="cql3d.nc")
    print('Done.')

    print(f"Submitting AORSA iteration {i} from {aorsa_dir}, waiting for completion...")
    submit_job(slurm_command_string=run_aorsa, directory=aorsa_dir) # call the slurm job 
    print(f"AORSA iteration {i} completed!")

# run the code
if __name__ == "__main__":
    # build the directory file structure
    build_file_structure(num_iterations)

    # enter the loop iterating the codes
    print('Entering iteration...') 
    for i_iter in range(num_iterations):
        cql3d_iteration_i(i=i_iter)
        aorsa_iteration_i(i=i_iter)
    
    print(f'Finished {num_iterations} cql3d/aorsa2d iterations. Iteration files saved to {super_directory}')








