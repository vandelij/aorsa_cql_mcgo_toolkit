import numpy as np
import netCDF4

cql3d_nc_file = "/home/jacobvandelindt/aorsa_cql_mcgo_toolkit/iterations_scripts/cql3d_bb.nc"


output_file = "/home/jacobvandelindt/aorsa_cql_mcgo_toolkit/iterations_scripts/cql3d_bbp.nc"



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

add_gen_species_dim_to_cql_nc(input_nc_file=cql3d_nc_file, output_nc_file=output_file)