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
import CQL3D_Analysis

cql_nc_new = netCDF4.Dataset('/home/jacobvandelindt/aorsa_cql_mcgo_toolkit/test.nc', 'w')
eqdsk_file = "/home/jacobvandelindt/aorsa_cql_mcgo_toolkit/shots/147634/g147634.04525"
cql3d_nc_file = "/home/jacobvandelindt/aorsa_cql_mcgo_toolkit/shots/147634/cql3d1gen.nc"
cql3d_krf_nc_file = "/home/jacobvandelindt/aorsa_cql_mcgo_toolkit/cql3d_krf001.nc"
cqlin_file = "/home/jacobvandelindt/aorsa_cql_mcgo_toolkit/shots/147634/cqlinput"

cql_pp = CQL3D_Analysis.CQL3D_Post_Process(
    gen_species_names=["D", "e"],
    cql3d_nc_file=cql3d_nc_file,
    cql3d_krf_file=cql3d_krf_nc_file,
    eqdsk_file=eqdsk_file,
    # cql_input_file=cqlin_file,
)


cql_nc = cql_pp.cql_nc

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