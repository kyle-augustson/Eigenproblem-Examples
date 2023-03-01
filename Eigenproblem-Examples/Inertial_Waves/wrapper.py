import os
import numpy as np
from mpi4py import MPI
from Inertial_Waves import Inertial_Wave_Eigenproblem

CW = MPI.COMM_WORLD

Ek=-4e0
Ekstr="{:.2f}".format(Ek)
label='4'
data_dir = "ModelS_Fit_Ek" + Ekstr + '_' + label
exists  = os.path.isdir(data_dir)
myEk = 10**Ek
neigs = 128
guess = 0e0
myell = 512
mynn = 64
print(myEk,myell,mynn,guess,data_dir)

solver = Inertial_Wave_Eigenproblem(1,myell,mynn,myEk,False,False,False,neigs,guess,MPI.COMM_SELF,label)

