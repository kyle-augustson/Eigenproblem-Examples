import os
import time
import sys
import gc

import numpy as np
import dedalus.public as d3
from dedalus.core import operators
import h5py
import logging
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})

from matplotlib import cbook
from mpl_toolkits.axes_grid1 import ImageGrid


# Dedalus Ball Parameters
Mmax = 2
Lmax = 256
Nmax = 64
L_dealias = 3/2 #Dealias by x harmonics
N_dealias = 3/2 #Dealias by y orders

Lfull = int(L_dealias*Lmax)
Nfull = int(N_dealias*Nmax)

#Dedalus Timestepper & Parameters
ts = d3.SBDF2 #Timestepper choice
dtype = np.complex128 #Data type

#Problem Parameters
ri = 0.89
ro = 0.99    #Outer radius
t_end = 10
Pm = 1
Re = 10e0**7.0
Rm = Pm*Re #Magnetic Reynolds number
Rei = 1/Re
Rmi = 1/Rm

Le2 = 10e0**(-7.0)

#Define radii for coordinate system creators
radii = (ri,ro)

mesh = None

# Bases
c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor((c,), mesh=mesh, dtype=dtype)
b = d3.ShellBasis(c, (2*Mmax,Lmax,Nmax), radii=radii, dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias))
b_inner = b.S2_basis(radius=ri) #For inner sphere boundary conditions
b_outer = b.S2_basis(radius=ro) #For outer sphere boundary conditions

# Fields
uu = d.VectorField(c, bases=b, name='uu')
um = d.VectorField(c, bases=b)
aa = d.VectorField(c, bases=b, name='aa')
bb = d.VectorField(c, bases=b)

bb = d3.curl(aa)

prefix = '/nobackupp19/kaugusts/MRI_Sphere/ModelS/BalancedField/Re6_M-5.9_g0.08_f0.08_b14_bz0_pcpc/ModelS_Fit_Re6.00_M-5.90_b14.00_bz0.00_k0/'
#prefix = '/nobackupp19/kaugusts/MRI_Sphere/Modified_Dipole/Unstrat_Linear_Re6.00_M-4.50_k6/'
#prefix = '/nobackupp19/kaugusts/MRI_Sphere/Final_Sweep8/Unstrat_Linear_Re6.00_M-4.50/'
#prefix='Comparisons/Stratified_Exp/Unstrat_Linear_Re5.00_M-5.50/'
#prefix='../New6/start_up/Unstrat_Linear_Re6.00_M-4.50_k5/'

f1 = h5py.File(prefix+'checkpoints/checkpoints_s1.h5')

aa.load_from_hdf5(f1,0)
uu.load_from_hdf5(f1,0)

bt = bb.evaluate()
#bt.change_scales(4)
bt.require_grid_space()
uu.change_scales(3/2)
uu.require_grid_space()

um.change_scales(3/2)
um.require_grid_space()
phi, theta, r = b.local_grids((3/2, 3/2, 3/2)) #Get local coordinate arrays
streamfunction=True
if (streamfunction):
    r_rho=1.0
    um['g'] = (ri/(r_rho-ri))**(1.5)/(r/(r_rho-r))**(1.5)*uu['g']
    um['g'][0] = 0
    ep = d.VectorField(c,bases=(b.meridional_basis,), dtype=dtype)
    #ep = d.VectorField(c,bases=(b.radial_basis,), dtype=dtype)
    ep['g'][0] = 1
    ep['g'][1] = 0
    ep['g'][2] = 0
    #er = d.VectorField(c,bases=(b.radial_basis,), dtype=dtype)
    #er['g'][0] = 0
    #er['g'][1] = 0
    #er['g'][2] = 1

    psi = d.Field(bases=b, dtype=dtype,name='psi')
    #tau_s = d.Field(dtype=dtype,name='tau_s')
    tau_ri = d.Field(bases=(b_inner,),dtype=dtype,name='tau_ri')
    tau_ro = d.Field(bases=(b_outer,),dtype=dtype,name='tau_ro')
    
    lift_basis = b.clone_with(k=1) # First derivative basis
    lift1 = lambda A: d3.Lift(A, lift_basis, -1)
    #lift2 = lambda A: d3.Lift(A, b, -2)

    rvec = d.VectorField(c, bases=b.meridional_basis)
    #rvec = d.VectorField(c, bases=b.radial_basis)
    rvec.change_scales((3/2,3/2,3/2))
    rvec['g'][2] = r

    coef = d.Field(bases=b.meridional_basis)
    coef.change_scales((3/2,3/2,3/2))
    coef['g'] = r**2*np.sin(theta)**2

    rhs = (coef*d3.dot(ep,d3.curl(um))).evaluate()
    #rhs = (d3.dot(ep,d3.curl(um))).evaluate()

    #gpsi = d3.curl(psi*ep)
    #gpsi = d3.curl(psi*ep) - rvec*lift(tau_ri)
    #gpsi = d3.curl(psi*ep) #+ rvec*lift(tau_ri)
    gpsi = d3.grad(psi) + rvec*lift1(tau_ri)
    #bc_psi_ri = operators.interpolate(d3.dot(er,d3.curl(psi*ep)),r=ri)
    #bc_psi_ro = operators.interpolate(d3.dot(er,d3.curl(psi*ep)),r=ro)

    # Problem
    problem = d3.LBVP([psi,tau_ri,tau_ro],namespace=locals())

    problem.add_equation("-coef*div(gpsi) + psi  + lift1(tau_ro) = rhs")
    #problem.add_equation("dot(ep,curl(gpsi)) + lift(tau_ri)  + lift(tau_ro) = rhs",condition="ntheta!=0")
    #problem.add_equation("dot(er,psi*er) + lift1(tau_ri)  + lift2(tau_ro) = rhs",condition="ntheta!=0")
    #problem.add_equation("dot(ep,curl(gpsi))  + lift(tau_ro) = rhs",condition="ntheta!=0")
    #problem.add_equation("psi = 0",condition="ntheta==0")
    
    #problem.add_equation("integ(psi)=0")
    #problem.add_equation("bc_psi_ri=0")
    #problem.add_equation("bc_psi_ro=0")
    #problem.add_equation("tau_ri=0",condition="ntheta==0")
    #problem.add_equation("tau_ro=0",condition="ntheta==0")
    #problem.add_equation("psi(r=ri)=0",condition="ntheta!=0")
    #problem.add_equation("psi(r=ro)=0",condition="ntheta!=0")
    problem.add_equation("psi(r=ri)=0")
    problem.add_equation("psi(r=ro)=0")

    # Solver
    solver = problem.build_solver()
    solver.print_subproblem_ranks()
    solver.solve()

    psi.change_scales((1,4,3/2))
    psi.require_grid_space()

#phig, thetag, rg = b.global_grids((4, 4, 4)) #Get local coordinate arrays
phig, thetag, rg = b.global_grids((1, 4, 3/2)) #Get local coordinate arrays

thetas = np.array(thetag).flatten()
rads = np.array(rg).flatten()

uu.change_scales((1,4,3/2))
uu.require_grid_space()

aa.change_scales((1,4,3/2))
aa.require_grid_space()

bt.change_scales((1,4,3/2))
bt.require_grid_space()

uphi = np.real(uu['g'][0,0])
if (streamfunction):
    r_rho=1.0
    psig = np.real(psi['g'])*(rg/(r_rho-rg))**(1.5)/(ri/(r_rho-ri))**(1.5)
    psig = psig[0]
else:
    psig = np.real(uu['g'][0,0])
bphi = np.real(bt['g'][0,0])
aphi = np.real(aa['g'][0,0])

thetas = thetas.flatten()
nth=len(thetas)
nmid  = int(nth/2)
mytint = np.min(np.where(thetas <= np.pi*30.0/180.0))
print(mytint,thetas[0]/np.pi,thetas[nmid]/np.pi,thetas[-1]/np.pi)
x = thetas
y = rads
x = np.pi/2 - x
xx, yy = np.meshgrid(x, y)
    
fig = plt.figure(figsize=[8,6],dpi=600,constrained_layout=False,linewidth=2.0)
#data = [uphi,psig,bphi,aphi]
data = [aphi,bphi,psig,uphi]
#labels = [r'$u_\phi$',r'$\psi$',r'$b_\phi$',r'$a_\phi$']
labels = ['(a)','(b)','(c)','(d)']
labels2 = [r'$A_\phi$',r'$B_\phi$',r'$\Psi$',r'$u_\phi$']
#cmaps = ['viridis','plasma','cividis','bwr']
cmaps = ['PiYG','PRGn','RdYlGn','RdYlBu']
saturation = 0.99

#fig, axs = plt.subplots(figsize=[3,2], constrained_layout=True, subplot_kw=dict(polar=True), ncols=4)
for ii in np.arange(4):
    pos = [-0.21+ii*0.16,0.02,0.9,0.96]
    pb = fig.add_axes(pos,projection='polar')#,frame_on=False)
    #pb = axs[ii]
    pb.set_xticklabels([])
    pb.set_yticklabels([])
    
    field = data[ii]
    print(np.shape(field))
    vals = np.sort(np.abs(field.flatten()))
    vals = np.sort(vals)
    vmax = vals[int(saturation*len(vals))]
    vmin = -vmax
    print(vmin,vmax)
   
    meplot = pb.pcolormesh(xx, yy, field.T, cmap=cmaps[ii], vmin=vmin, vmax=vmax, rasterized=False, shading='auto')
    if (ii % 2 ==0):
        meplot2 = pb.contour(xx,yy,field.T,16,colors='xkcd:grey')
            
    pb.set_theta_offset(0)
    pb.set_thetalim([-np.pi/2,np.pi/2])#*60/180,0)
    pb.set_rlim(0.89,0.99)
    pb.set_rorigin(0)
    pb.set_aspect(1)
    pb.text(0.33,0.95,labels[ii],transform=pb.transAxes,fontsize=20)
    pb.text(0.62,0.03,labels2[ii],transform=pb.transAxes,fontsize=20)
    pb.grid(False)
        
fig.savefig('fig_for_geoff.png') #,bbox_inches='tight')

