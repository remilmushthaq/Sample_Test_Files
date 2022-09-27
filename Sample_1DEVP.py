import numpy as np
import math as ma
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

logger.info('Start')
# Domain Parameters
Lx = 2*np.pi
Ly = 1
Nx = 122
Ny = 122

# System Parameters
Re_no = 2500
Pr_no = 0.71
Sr_no = 100  ####################################
Pe_no = Re_no*Pr_no
eps = 1/Sr_no
# Important: Here delta is not tied to epsilon. Epsilon is sent to 0 and then delta is varied. 
delta = 7   #/ma.sqrt(eps)
Gamma = 0.3
gamma = 1.4
k = 1
Am = 2
dtype = np.float64
dealias = 3/2

# Bases
ycoord = d3.Coordinate('y')
dist = d3.Distributor(ycoord, dtype=dtype)
ybasis = d3.Chebyshev(ycoord, size=Ny, bounds=(0, Ly))
y = dist.local_grids(ybasis)

# Fields
Pi = dist.Field(name='Pi', bases=ybasis)
tau_1 = dist.Field(name='tau_1')
tau_2 = dist.Field(name='tau_2')
lam = dist.Field(name='lam')
c1 = dist.Field(name='c1', bases=ybasis)

# Substitutions
dy = lambda A: d3.Differentiate(A, ycoord)
lift_basis = ybasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
Piy = dy(Pi) + lift(tau_1) 
Piyy = dy(Piy) + lift(tau_2)

T_avg_init = np.genfromtxt('T_avg_90.csv', delimiter=',')

c1['g'] = T_avg_init #1 + Gamma*y  # 1D Array

# Problem
problem = d3.EVP([Pi, tau_1, tau_2], eigenvalue=lam, namespace=locals())
problem.add_equation("lam*Pi - k**2*c1*Pi + 1/delta**2*dy(c1)*Piy + 1/delta**2*c1*Piyy = 0")
problem.add_equation("Piy(y=0) = 0")
problem.add_equation("Piy(y=Ly) = 0")

# Solve
solver = problem.build_solver()
solver.solve_dense(solver.subproblems[0])

logger.info('End')