import os, sys
import time

shotnum = 147634
time_stamp = round(time.time())


#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

host = 'perlmutter'
username = 'vandelij'
parent_direc = '~/aorsa/test/diiid-aorsa-hires_copy'# '/global/homes/j/jwright/perlmutter-builds/aorsa/examples/DIIID-helicon'#'~/AORSA/DIIID-helicon/'  #'~/AORSA/DIIID-helicon/'
file_to_grab = 'aorsa2d.ps' #'aorsa2d_input_og_dont_tuch.in'
file_Efield_2D = 'Efield_2D.vtk'
target_direc = '/home/jacobvandelindt/aorsa_cql_mcgo_toolkit/shots/'
file_to_save = f'{shotnum}/aorsa{time_stamp}.ps'#f'{shotnum}/aorsa{time_stamp}.ps' 


os.system(f'scp {username}@{host}:{parent_direc}/{file_to_grab} {target_direc}{file_to_save}')
os.system(f'scp {username}@{host}:{parent_direc}/{file_Efield_2D} {target_direc}/{shotnum}/Efield_2D_new.vtk')
