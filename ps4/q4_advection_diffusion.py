"""
Advection-diffusion solver (Section 4 of PS4).
Lax-Friedrich method is used for advection and an implicit method is used for
diffusion.
The problem is solved for two different diffusion coeffcients.

Author: Thomas Vandal
Feb 26th 2020
"""
import numpy as np
from matplotlib import pyplot as plt

import warnings


def getA_arr(n, beta):
    """Define multple 2D matrices for implciit method at once.
    The resulting 2D matrices are saved in a np.array along 3rd axis.
    Args:
        n (int): dimension for the nxn 2d matrix
        beta (array): beta factors for diffusion solver, will set the number
                      of A matrices returned
    Returns:
        A: Array of shape (len(beta), n, n) containing the nxn matrices
    """
    beta = np.array(beta)
    nreps = (beta.size, 1, 1)
    bmult = beta[:, None, None]  # adapted to multiply A array

    A = np.tile(np.eye(n), nreps) * (1.0 + 2.0 * bmult)
    A -= np.tile(np.eye(n, k=1), nreps) * bmult
    A -= np.tile(np.eye(n, k=-1), nreps) * bmult

    return A


# Defining constants
npts = 100
dx = 1.0
u = -0.1
dt = 0.2
D_arr = np.array([0.01, 0.5])  # save them in array to avoid looping
nsteps = 10000

# warning if CFL condition is not respected
if dt > np.abs(dx/u):
    msg = ('The CFL condition is not respected. '
           'Explicit methods will be unstable.')
    warnings.warn(msg, RuntimeWarning)

# Setting up the grid
x = np.linspace(0, npts*dx, num=npts, endpoint=False)

# Defining IC
f1 = x.copy()
f2 = x.copy()

# Defining advection constant factor
alpha = u*dt/(2*dx)

# Defining diffusion factors and matrices for each diffusion coeff.
beta_arr = D_arr * dt / dt**2       # 2 coefficients
A_arr = getA_arr(x.size, beta_arr)  # 2 matrices (see getA_arr above)

# Setting BC on A (no-slip on both ends)
A_arr[:, (0, -1), (0, -1)] = 1.0  # corners on diag for each matrix
A_arr[:, (0, -1), (1, -2)] = 0.0  # next to corner for each matrix

# Setting up figure
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[8.4, 4.8], sharey=True)
# Left panel
p1, = ax1.plot(x, f1)
ax1.set_title(r'Diffusion Coefficient $D = {}$'.format(D_arr[0]))
ax1.set_ylabel(r'Quantity $f$')
ax1.set_xlabel(r'Position $x$')
# Right panel
p2, = ax2.plot(x, f2)
ax2.set_title(r'Diffusion Coefficient $D = {}$'.format(D_arr[1]))
ax2.set_xlabel(r'Position $x$')
fig.canvas.draw()

print('Starting advection-diffusion...')
for i in range(nsteps):
    print('Completed {} steps.'.format(i), end='\r')
    # Calculate diffusion with implcit method (BC enforced by A)
    f1 = np.linalg.solve(A_arr[0], f1)
    f2 = np.linalg.solve(A_arr[1], f2)

    # Calculate advection with LF method (BC are enforced with indexing)
    f1[1:-1] = (0.5 * (f1[2:] + f1[:-2])
                - alpha * (f1[2:] - f1[:-2]))
    f2[1:-1] = (0.5 * (f2[2:] + f2[:-2])
                - alpha * (f2[2:] - f2[:-2]))

    # Update plots
    if not i % 100:
        p1.set_ydata(f1)
        p2.set_ydata(f2)
        fig.canvas.draw()
        plt.pause(0.001)

print('Completed {} steps.'.format(nsteps))
print('Advection-diffusion done.')
