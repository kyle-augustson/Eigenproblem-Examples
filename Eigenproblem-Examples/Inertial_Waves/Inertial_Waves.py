"""

         MRI spherical 2D eigensolver.

Usage:
         MRI_example.py [options]

Options:
         --nphi=<nphi>      [default: 1]
         --nth=<nth>        [default: 256]
         --nr=<nr>          [default: 96]
         --Ek=<Ek>          [default: 1e-4]
         --label=<label>    [default: None]
"""

import os
import numpy as np
import dedalus.public as d3
from dedalus.core import evaluator
import logging
logger = logging.getLogger(__name__)

#from docopt import docopt
#args = docopt(__doc__)
def Inertial_Wave_Eigenproblem(nphi,ntheta,nr,Ek,fit,stress_free,dense,neig,guess,comm,label):
    #Datatype
    dtype = np.complex128

    #Problem Parameters
    ri = 0.70    #Inner radius
    ro = 1.00    #Outer radius
    Ekv = Ek     #Ekman number

    # Set up output dir
    data_dir = 'Inertial_Waves_Ek{:.2f}'.format(np.log10(Ekv))+'_'+label

    #Define radii for coordinate system creators
    radii = (ri,ro)

    #For eigenproblems there's no MPI mesh for dedalus, but there can be for SLEPc.
    mesh = None
    # Bases
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist = d3.Distributor(coords, mesh=mesh, dtype=dtype,comm=comm)
    shell = d3.ShellBasis(coords, shape=(nphi,ntheta,nr), radii=radii, dtype=dtype)
    sphere = shell.outer_surface
    phi, theta, r = dist.local_grids(shell) #Get local coordinate arrays

    # Fields
    u = dist.VectorField(coords, bases=shell, name='u')
    v0 = dist.VectorField(coords, bases=shell.meridional_basis, name='v0')
    w0 = dist.VectorField(coords, bases=shell.meridional_basis, name='w0')
    ez = dist.VectorField(coords, bases=shell.meridional_basis, name='ez')
    p = dist.Field(bases=shell, name='p')

    # Eigenvalue
    omega = dist.Field(name='omega')

    #Tau boundaries
    tau_u_ri = dist.VectorField(coords, bases=sphere, name='tau_u_ri')
    tau_u_ro = dist.VectorField(coords,bases=sphere, name='tau_u_ro')
    tau_p = dist.Field(name='tau_p')

    dt = lambda A: -1j*omega*A

    lift_basis = shell.derivative_basis(1) # First derivative basis
    lift = lambda A: d3.Lift(A, lift_basis, -1)

    #First order formulation
    rvec = dist.VectorField(coords, bases=shell.meridional_basis, name='rvec')
    #Radial component, since chebyshev in radius
    rvec['g'][2] = r
    grad_u = d3.grad(u) + rvec*lift(tau_u_ri)

    #For stress-free boundary conditions
    strain_rate = d3.grad(u) + d3.transpose(d3.grad(u))

    #Vertical unit vector
    ez = dist.VectorField(coords, bases=shell.meridional_basis, name='ez')
    #Theta component
    ez['g'][1] = -np.sin(theta)
    #Radial component
    ez['g'][2] = np.cos(theta)

    #Azimuthal velocity vector and associated meridional vorticity in the rotating frame
    if (fit):
        #Radial fit (15 nccs)
        z = (r-ri)/(ro-ri)
        fz = 0.05*z-0.05*z**14
        fzp = (0.05 - 14e0*0.05*z**13)/(ro-ri)

        #Latitudinal fit (5 nccs)
        mu = np.cos(theta)
        gt = 1-0.145*mu**2-0.148*mu**4
        gtp = 2e0*np.sin(theta)*(0.145*mu + 2e0*0.148*mu**3)

        #Azimuthal velocity
        v0['g'][0] = 0.5*r*np.sin(theta)*fz*gt

        #Associated meridional vorticity
        w0['g'][1] = -0.5*np.sin(theta)*gt*(r*fzp + 2e0*fz)
        w0['g'][2] = 0.5*fz*(2e0*mu*gt+np.sin(theta)*gtp)
    else:
        #No differential rotation
        v0['g'][0] = 0e0
        w0['g'][0] = 0e0

    logger.info("Building problem.")

    # Problem
    problem = d3.EVP([p, u, tau_u_ri, tau_u_ro, tau_p], eigenvalue=omega, namespace=locals())

    #Equations of motion
    problem.add_equation("trace(grad_u) + tau_p = 0")
    
    problem.add_equation("dt(u) + grad(p) + cross(ez,u) + cross(curl(u),v0) + cross(w0,u) - Ekv*div(grad_u) + lift(tau_u_ro) = 0")
    
    # Pressure Gauge
    problem.add_equation("integ(p) = 0")

    #Boundary conditions
    if (stress_free):
        problem.add_equation("radial(u(r=ri)) = 0")
        problem.add_equation("angular(radial(strain_rate(r=ri),0),0) = 0")
        problem.add_equation("radial(u(r=ro)) = 0")
        problem.add_equation("angular(radial(strain_rate(r=ro),0),0) = 0")
    else:
        problem.add_equation("u(r=ri)=0")
        problem.add_equation("u(r=ro)=0")

    # Solver
    logger.info('Building solver.')
    solver = problem.build_solver()
    solver.ncc_cutoff=1e-10

    ss = solver.subproblems[0].subsystems[0]

    if (dense):
        solver.solve_dense(solver.subproblems[0])
    else:
        solver.solve_sparse(solver.subproblems[0], neig, guess)

    logger.info('Solved with eigenvalues:')
    logger.info(solver.eigenvalues)

    namespace = {}
    solver.evaluator = evaluator.Evaluator(solver.dist, namespace)

    logger.info('checkpointing in {}'.format(data_dir))

    if not os.path.exists('{:s}'.format(data_dir)):
        os.mkdir('{:s}'.format(data_dir))

    np.save(data_dir+'/eigenvalues.npy',solver.eigenvalues)
    path = data_dir + '/checkpoints'
    checkpoint = solver.evaluator.add_file_handler(path, max_writes=1)
    checkpoint.add_tasks(solver.state)

    for i in range(neig):
        solver.set_state(i, ss)
        solver.evaluator.evaluate_handlers([checkpoint],sim_time=1,wall_time=1,timestep=1,iteration=1)

    return solver
