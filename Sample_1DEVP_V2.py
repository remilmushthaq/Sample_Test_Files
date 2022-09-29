import numpy as np
import math as ma
import dedalus.public as de
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import time
start = time.time()

# Parameters
Ly = 1
Ny = 122
delta = 10
dtype = np.complex128

# Bases
ybasis = de.Chebyshev('y', Ny, interval=(0, Ly), dealias=1)
domain = de.Domain([ybasis], dtype)
y=domain.grid(0,scales=domain.dealias)

delta_field=domain.new_field(name='delta_field')
delta_field.meta['y']['constant'] = True
delta_field['g']=delta

c1 = domain.new_field(name='c1')
T_avg_init = np.genfromtxt("T_avg_90.csv", delimiter=',')
c1['g']=T_avg_init

# Problem
problem = de.EVP(domain, variables=['Pi', 'Piy'], eigenvalue='lam')
problem.parameters['c1']=c1
problem.parameters['delta']=delta_field
problem.add_equation("lam*Pi - c1*Pi + 1/delta**2*dy(c1)*Piy + 1/delta**2*c1*dy(Piy) = 0")
problem.add_equation("Piy - dy(Pi)= 0")
problem.add_equation("left(Piy) = 0")
problem.add_equation("right(Piy) = 0")

# Solve
solver = problem.build_solver()
solver.solve_sparse(solver.pencils[0], N=1, target=1.0)

end = time.time()
logger.info('Elapsed time = %.4f s, Eigenvalue = %.4f +/- %.4e j' %(end - start, np.real(solver.eigenvalues[0]), np.imag(solver.eigenvalues[0])))


order = np.argsort(solver.eigenvalues)
solver.eigenvalues = solver.eigenvalues[order]
solver.eigenvectors = solver.eigenvectors[:, order]

solver.set_state(0)

plt.figure(figsize=(6, 4))
temp=solver.state['Pi']['g'].real
plt.plot(y, temp/temp[0])
plt.ylabel(r"Fluctuating pressure pi hat")
plt.xlabel(r"y")
plt.tight_layout()
plt.savefig("first_eigenvector_p_D2.png", dpi=200)

plt.figure(figsize=(6, 4))
plt.plot(y, c1['g'].real)
plt.ylabel(r"Averaged temperature")
plt.xlabel(r"y")
plt.tight_layout()
plt.savefig("mean_temperature_profil_D2.png", dpi=200)
