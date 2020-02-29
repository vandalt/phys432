"""
Advection equation solver (Section 3 of PS4).
Using both the FTCS and the Lax-Friedrich methods.

Author: Thomas Vandal
Feb 25th 2020
"""
import numpy as np
from matplotlib import pyplot as plt

import warnings

# Defining constants
npts = 100      # number of points
dx = 1.0        # spacing between points
u = -0.1        # u set to a constant as required in the problem
dt = 0.2        # timestep size
nsteps = 10000  # number of timesteps to take

# warning if CFL condition is not respected
if dt > np.abs(dx/u):
    msg = ('The CFL condition is not respected. '
           'Explicit methods will be unstable.')
    warnings.warn(msg, RuntimeWarning)

# Setting up the grid
x = np.linspace(0, npts*dx, num=npts, endpoint=False)

# Defining initial condtion f(x, t=0) = x
f_ftcs = x.copy()  # FTCS scheme
f_lf = x.copy()    # Lax-Friedrich scheme

# Defining advection constant factor
alpha = u*dt/(2*dx)

# Setting up figure
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[8.4, 4.8], sharey=True)
# Left panel (FTCS)
p1, = ax1.plot(x, f_ftcs)
ax1.set_title('FTCS Method')
ax1.set_ylabel(r'Quantity $f$')
ax1.set_xlabel(r'Position $x$')
# Right panel (Lax-Friedrich)
p2, = ax2.plot(x, f_lf)
ax2.set_title('Lax-Friedrich Method')
ax2.set_xlabel(r'Position $x$')
fig.canvas.draw()

print('Starting advection...')
for i in range(nsteps):
    print('Completed {} steps.'.format(i), end='\r')
    # Calculate advection (BC are enforced with indexing)
    # FTCS Scheme
    f_ftcs[1:-1] -= alpha * (f_ftcs[2:] - f_ftcs[:-2])
    # LF Scheme
    f_lf[1:-1] = (0.5 * (f_lf[2:] + f_lf[:-2])
                  - alpha * (f_lf[2:] - f_lf[:-2]))

    # Update plots
    if not i % 100:
        p1.set_ydata(f_ftcs)
        p2.set_ydata(f_lf)
        fig.canvas.draw()
        plt.pause(0.001)

print('Completed {} steps.'.format(nsteps))
print('Advection done.')
