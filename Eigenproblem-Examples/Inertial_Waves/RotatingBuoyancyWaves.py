
def RotatingBuoyancyWaves(ri,rb,S,Ek,Pr,Roc,nphi,ntheta,nr,target_m,neigs,guess,comm):
    #Let's get ourselves started with a few module imports.
    import os
    import numpy as np
    import dedalus.public as d3
    from dedalus.core import evaluator #currently necessary, will be obsolete soon.
    import logging
    logger = logging.getLogger(__name__)

    #Datatype
    dtype = np.complex128
    
    #Get Ra from Roc
    Ra = Pr*Roc/Ek**2    #Rayleigh number
    logger.info('Ra=',Ra)

    # Set up output directory for eigenmodes
    label='0'
    data_dir = 'RB_Waves_Roc{:.2f}_Ek{:.2f}'.format(Roc,np.log10(Ek))+'_'+label

    #Define radii for coordinate system creators
    radii = (ri,ro)

    #For eigenproblems there's no MPI mesh for dedalus, but there can be for SLEPc.
    mesh = None

    # Bases
    coords = d3.SphericalCoordinates('phi', 'theta', 'r') #Define the spherical coordinate names
    dist = d3.Distributor(coords, mesh=mesh, dtype=dtype, comm=comm) #Set up the internal communicators and initialize.
    #Define a shell basis (so Tensor spherical harmonics and a Chebyshev radial basis)
    shell = d3.ShellBasis(coords, shape=(nphi,ntheta,nr), radii=radii, dtype=dtype)
    sphere = shell.outer_surface #Define the bounding sphere coordinate systems (e.g. S_2)
    phi, theta, r = dist.local_grids(shell) #Get local coordinate arrays

    # Fields over the shell
    u = dist.VectorField(coords, bases=shell, name='u') #Vector velocity field
    p = dist.Field(bases=shell, name='p') #Gauge field.
    T = dist.Field(bases=shell, name='T') #Temperature field.
    grad_T0 = dist.Field(bases=shell.meridional_basis, name='T') #Temperature field.
    # Eigenvalue
    omega = dist.Field(name='omega')
    
    z = (r-rb)/(ro-ri)
    tnh = np.tanh(S*z)
    sch2 = 1-tnh**2
    grad_T0['g'] = -(1 + tnh + z*S*sch2)/(ro-rb)/2

    #Tau boundary mullifiers (dummy variables permitting the
    # our spectral representation of boundary conditions)
    tau_u_ri = dist.VectorField(coords, bases=sphere, name='tau_u_ri')
    tau_u_ro = dist.VectorField(coords,bases=sphere, name='tau_u_ro')
    tau_p = dist.Field(name='tau_p')

    tau_T_ri = dist.Field(bases=sphere)
    tau_T_ro = dist.Field(bases=sphere)

    #Substitutions
    dt = lambda A: -1j*omega*A #Our eigenvalue operator.

    #Here we lift the taus onto the shell derivative basis 
    #(So ChebyU in radius and the shifted Jacobi polynomials in latitude)
    lift_basis = shell.derivative_basis(1) # First derivative basis
    lift = lambda A: d3.Lift(A, lift_basis, -1)

    #First order formulation
    rvec = dist.VectorField(coords, bases=shell.meridional_basis, name='rvec')
    #Radial component, since chebyshev in radius
    rvec['g'][2] = r

    #Define the gradient tensor and the first lift for the lower boundary
    grad_u = d3.grad(u) + rvec*lift(tau_u_ri)
    grad_T = d3.grad(T) + rvec*lift(tau_T_ri)

    #For stress-free boundary conditions
    strain_rate = d3.grad(u) + d3.transpose(d3.grad(u))

    #Vertical unit vector in spherical coordinates
    #This guy is special: it is defined over a meridional basis 
    # (e.g. coupled in radius and theta, but not in phi).
    #This restricted basis is called the meridional_basis.
    ez = dist.VectorField(coords, bases=shell.meridional_basis, name='ez')
    #Theta component
    ez['g'][1] = -np.sin(theta)
    #Radial component
    ez['g'][2] = np.cos(theta)
    
    er = dist.VectorField(coords, bases=shell.meridional_basis, name='er')
    #Radial component
    er['g'][2] = 1

    kappa = Ek/Pr
    nu = Ek

    # Problem
    problem = d3.EVP([T, p, u, tau_T_ri, tau_T_ro, tau_u_ri, tau_u_ro, tau_p], eigenvalue=omega, namespace=locals())
    
    #Equations of motion
    #Continuity + a tau to control the gauge field.
    problem.add_equation("trace(grad_u) + tau_p = 0")
    #Momentum equation with the Coriolis force.
    problem.add_equation("dt(u) + grad(p) + cross(ez,u) - nu*div(grad_u) - Roc*T*er + lift(tau_u_ro) = 0")
    #Temperature equation
    problem.add_equation("dt(T) - dot(er*grad_T0,u) - kappa*div(grad_T) + lift(tau_T_ro) = 0")
    
    # Pressure Gauge
    problem.add_equation("integ(p) = 0")

    #Boundary conditions
    stress_free=True
    if (stress_free):
        problem.add_equation("radial(u(r=ri)) = 0") #Impenetrable lower boundary
        problem.add_equation("angular(radial(strain_rate(r=ri),0),0) = 0") #Stress-free lower boundary
        problem.add_equation("radial(u(r=ro)) = 0") #Impenetrable upper boundary
        problem.add_equation("angular(radial(strain_rate(r=ro),0),0) = 0") #Stress-free upper boundary
    else:
        problem.add_equation("u(r=ri)=0") #No-slip boundaries
        problem.add_equation("u(r=ro)=0")
    
    problem.add_equation("T(r=ri) = 0")
    problem.add_equation("T(r=ro) = 0")

    solver = problem.build_solver()
    subproblem = solver.subproblems_by_group[(target_m, None, None)]
    ss = subproblem.subsystems[0]
    
    #Let's use the sparse solver (waaaaay faster).
    dense=False

    if (dense):
        solver.solve_dense(subproblem)
    else:
        solver.solve_sparse(subproblem, neig, guess)
    
    namespace = {}
    solver.evaluator = evaluator.Evaluator(solver.dist, namespace)

    if not os.path.exists('{:s}'.format(data_dir)):
        os.mkdir('{:s}'.format(data_dir))

    np.save(data_dir+'/eigenvalues.npy',solver.eigenvalues)
    path = data_dir + '/checkpoints'
    checkpoint = solver.evaluator.add_file_handler(path, max_writes=1)
    checkpoint.add_tasks(solver.state)

    for i in range(neig):
        solver.set_state(i, ss)
        solver.evaluator.evaluate_handlers([checkpoint],sim_time=1,wall_time=1,timestep=1,iteration=1)

    
