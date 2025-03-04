import numpy as np
import os
import subprocess
import time 
import re


#################### USER SETTINGS #####################
# Directory Setup (use absolute paths)

# The head directory in which the results are stored. lots of data. 
super_directory = "/pscratch/sd/v/vandelij/iterate_test_paredown/"

# The eqdsk path. Wilkl be copied and renamed eqdsk into the run dirs
eqdsk_path = "/global/cfs/cdirs/m77/jacob_van_de_Lindt/aorsa_cql_iterate_templates/g147634.04525"

# the aorsa template directory with the initial case set up as well as the aorsa slurm job file.  
aorsa_clone_directory = "/global/cfs/cdirs/m77/jacob_van_de_Lindt/aorsa_cql_iterate_templates/aorsa_template_paredown/"

# the cql3d template directory with the initial case set up as well as the cql3d slurm job file.
cql3d_clone_directory = "/global/cfs/cdirs/m77/jacob_van_de_Lindt/aorsa_cql_iterate_templates/cql3d_template_paredown/"

# Simulation settings
num_iterations = 2

wait_time = 20 # time in secnods to wait in each loop after a job is submitted. 

# command to run the local cql3d slurm file
run_cql3d = "sbatch cql3d_slurm"

# command to run the local aorsa slurm file
run_aorsa = "sbatch aorsa_slurm"

def ensure_and_change_directory(path):
    if not os.path.exists(path):  # Check if the directory exists
        os.makedirs(path)  # Create the directory (including parent directories if needed)
    os.chdir(path)  # Change to the directory
    print("Changed to directory:", os.getcwd())

def build_file_structure(num_iterations):
    for i in range(num_iterations):
        os.chdir(super_directory)
        # set up cql directories. Special i = 0 case handled later when building the input files
        cql3d_dir = super_directory + f"/cql3d_iteration_{i}"
        ensure_and_change_directory(cql3d_dir)
        # copy over required files
        os.system(f"cp {eqdsk_path} eqdsk")
        os.system(f"cp {cql3d_clone_directory}/cql3d_slurm .")
        os.system(f"cp {cql3d_clone_directory}/cqlinput .")
        

        # change back to the super directory 
        os.chdir(super_directory)

        # set up aorsa directory
        aorsa_dir = super_directory + f"/aorsa_iteration_{i}"
        ensure_and_change_directory(aorsa_dir)
        # copy over required files 
        os.system(f"cp {eqdsk_path} eqdsk")
        os.system(f"cp {aorsa_clone_directory}/aorsa_slurm .")
        os.system(f"cp {aorsa_clone_directory}/aorsa2d.in .")
        os.system(f"cp {aorsa_clone_directory}/Ztable .")
        os.system(f"cp {aorsa_clone_directory}/ZTABLE.TXT .")
        os.system(f"cp {aorsa_clone_directory}/grfont.dat .")

    os.chdir(super_directory)
    print('Done setting up directories.')


# define two functions which submit jobs, capture output job id, and wait for completion of slurm job
def submit_job(slurm_command_string, directory):
    """Submits the job using sbatch and returns the job ID."""
    result = subprocess.run(slurm_command_string, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True, cwd=directory)
    output = result.stdout.strip()

    # Extract job ID from sbatch output (e.g., "Submitted batch job 123456")
    match = re.search(r"Submitted batch job (\d+)", output)
    if match:
        return match.group(1)
    else:
        raise RuntimeError("Failed to get job ID from sbatch output:\n" + output)

def wait_for_job(job_id):
    """Waits for the Slurm job to complete by checking squeue."""
    while True:
        result = subprocess.run(["squeue", "-j", job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if job_id not in result.stdout:  # Job has finished
            break
        time.sleep(wait_time)  # Wait for 10 seconds before checking again

# define two functions which handle the actual modification and exicution of aorsa and cql3d.
def cql3d_iteration_i(i):
    cql3d_dir = super_directory + f"/cql3d_iteration_{i}"
    os.chdir(cql3d_dir) # this should already exist from the build_file_structure function
    
    # TODO add functionality to copy over out_coef1 over from aorsa i-1 if i != 0 as du0u01 (TODO make sure grid is correct), else just run on NBI
    # if i != 0, also copy over cql3d.nc from the previous cql run and set nlrestart to "enabled" to read it in. 
    # be cautious of overwriting it? maybe copy it twice with two names, nmuonic can maybe be cql3d_i.nc? 
    # # also, if i != 0, set rdcmod=format1, rdcfile='du0u0_input_1', rfread=text, 

    job_id = submit_job(slurm_command_string=run_cql3d, directory=cql3d_dir) # call the slurm job 
    print(f"CQL3D iteration {i} job {job_id} submitted from {cql3d_dir}, waiting for completion...")
    wait_for_job(job_id)
    print(f"CQL3D iteration {i} job {job_id} completed!")


def aorsa_iteration_i(i):
    aorsa_dir = super_directory + f"/aorsa_iteration_{i}"
    os.chdir(aorsa_dir)
    # copy over previous cql iteration cql3d.nc as cql3d.nc.i and cql3d.nc, and set netCDF_file1 = 'cql3d.nc'
    # enorm_factor= 0.0 allegedly forces cql3d and aorsa to have the same enorm 
    # set ndist1 = 1 for nonmaxwellian ions 

    job_id = submit_job(slurm_command_string=run_aorsa, directory=aorsa_dir) # call the slurm job 
    print(f"AORSA iteration {i} job {job_id} submitted from {aorsa_dir}, waiting for completion...")
    wait_for_job(job_id)
    print(f"AORSA iteration {i} job {job_id} completed!")

# run the code
if __name__ == "__main__":
    # build the directory file structure
    build_file_structure(num_iterations)

    # enter the loop iterating the codes
    print('Entering iteration...') 
    for i_iter in range(num_iterations):
        cql3d_iteration_i(i=i_iter)
        aorsa_iteration_i(i=i_iter)
    
    print(f'Finished {num_iterations} cql3d/aorsa2d iterations.')








